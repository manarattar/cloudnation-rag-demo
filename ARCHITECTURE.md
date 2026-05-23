# Enterprise RAG Architecture — National Tax Authority
### CloudNation Technical Assessment | Lead AI Engineer

---

## System Overview

An internal AI assistant for 500,000 documents (legislation, case law, policy, e-learning) that answers complex fiscal questions with **zero hallucination** — every claim carries an exact citation — while enforcing strict **role-based access control** and maintaining **TTFT < 1.5 seconds** at 20M+ vector chunks.

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Module 3: CRAG Orchestrator (LangGraph)                    │
│   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐  │
│   │ Query        │──►│ Hybrid       │──►│ Retrieval     │  │
│   │ Transform    │   │ Retrieval    │   │ Grader        │  │
│   │ (Decompose / │   │ (Module 2)   │   │ (LLM judge)   │  │
│   │  HyDE)       │   │              │   │               │  │
│   └──────────────┘   └──────────────┘   └───────┬───────┘  │
│                                                  │          │
│                         ┌────────────────────────┤          │
│                         │                        │          │
│                    RELEVANT               AMBIGUOUS /       │
│                         │                IRRELEVANT         │
│                         ▼                    ▼              │
│                   ┌──────────┐        ┌─────────────┐      │
│                   │ Generate │        │ Rewrite /   │      │
│                   │ + Cite   │        │ Expand +    │      │
│                   └──────────┘        │ Re-retrieve │      │
│                                       └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Semantic Cache check (Redis) → if hit: return cached answer
    │
    ▼
Final Answer with inline citations [Doc, Article, Paragraph]
```

---

## Module 1 — Ingestion & Knowledge Structuring

### 1.1 Chunking Strategy for Legal Documents

Standard recursive splitters (e.g., `RecursiveCharacterTextSplitter`) break at character count, destroying legislative hierarchy. A chunk that contains "bedraagt het bedrag van de in het kalenderjaar" without knowing it belongs to *Artikel 3.114, Lid 2* is useless for citation.

**Solution: Hierarchical chunking with metadata-carried legislative addresses.**

Every chunk payload stores its full legal address:

```python
{
  "doc_title":  "Wet Inkomstenbelasting 2024",
  "chapter":    "Hoofdstuk 3",
  "article":    "Artikel 3.114",
  "paragraph":  "Lid 2",
  "citation":   "Wet Inkomstenbelasting 2024, Hoofdstuk 3, Artikel 3.114, Lid 2",
  "classification": "public",
  "access_roles": ["*"]
}
```

**Chunk sizes (three-level hierarchy):**

| Level | Size | Purpose |
|-------|------|---------|
| Large (parent) | 2048 tokens | Section-level context for re-ranking |
| Medium (leaf) | 512 tokens | Primary retrieval unit |
| Small (child) | 128 tokens | Precision retrieval for dense legal prose |

The system uses **small-to-big retrieval**: retrieve small leaf nodes for precision, expand to parent nodes before passing to the LLM for full context.

**Case law (ECLI documents)** are split by structural section (Facts → Legal Consideration → Decision) before chunking. The section name is stored as `chapter` metadata, so the LLM knows whether it is reading factual background or a binding ruling.

See: `module1_ingestion/chunking_strategy.py`

---

### 1.2 Vector Database: Qdrant

**Selected: Qdrant (self-hosted)**

| Criterion | Decision |
|-----------|----------|
| Data sovereignty | Self-hosted on-premise — data never leaves the tax authority's network |
| RBAC enforcement | Native payload filtering applied *before* HNSW traversal (see Module 4) |
| Scale | Proven at 100M+ vectors in production benchmarks |
| Quantization | Built-in Scalar Quantization (SQ8): 4× memory reduction |
| Sparse vectors | Native SPLADE/BM25 support for hybrid search |

**HNSW Index Configuration:**

```python
HnswConfigDiff(
    m=16,             # 16 bidirectional links per node
    ef_construct=200, # candidate pool during index build
)
```

**Why m=16, ef_construct=200?**

- `m=8` → recall ~0.93 @ ef=100. In legal contexts, 7% miss rate means citing the wrong article.
- `m=16` → recall ~0.97 @ ef=100. Chosen balance of precision and RAM.
- `m=32` → recall ~0.99 but 2× RAM and 2× build time. Marginal gain over m=16.
- `ef_construct=200` over `ef_construct=100`: ~40% longer index build, but recall improves from ~0.95 to ~0.98. Index builds once; query precision matters permanently.

**Memory: Scalar Quantization (SQ8)**

| Configuration | Memory (20M × 1536-dim chunks) |
|---------------|-------------------------------|
| float32 (unquantized) | ~115 GB |
| SQ8 (int8, 4× compression) | **~29 GB** |
| Product Quantization (rejected) | ~14 GB but ~8% recall loss — too high |

SQ8 maps each float32 → int8 with 99th-percentile outlier clipping. Recall penalty: ~1-2%. Fits on a 64 GB production node with room for HNSW graph overhead (~5 GB) and payload (~3 GB).

See: `module1_ingestion/vector_db_config.py`

---

## Module 2 — Retrieval Strategy

### 2.1 Hybrid Search: BM25 + Dense with RRF

Legal queries are not uniform. They fall into two categories:

| Query type | Example | Best retrieval |
|------------|---------|----------------|
| Exact reference | `"ECLI:NL:HR:2023:123"` | Sparse (BM25) — dense compresses identifiers |
| Semantic concept | `"deductibility of home office expenses"` | Dense — BM25 misses paraphrase |

Running both in parallel and fusing with **Reciprocal Rank Fusion (RRF)** handles both without per-query tuning:

```
RRF(doc) = Σ  1 / (k + rank_i)    k = 60  (Cormack et al., 2009)
```

Documents that rank highly in *either* list are boosted. `k=60` is the empirically validated constant from the original paper — it down-weights low-ranked candidates without zeroing them.

**Why RRF over fixed alpha-weighting?**

Alpha weighting (`score = α·dense + (1-α)·sparse`) requires calibrating α for each query type. A fixed `α=0.5` under-serves both types simultaneously. RRF adapts implicitly: for an ECLI query, sparse gives rank 1 → large RRF contribution regardless of dense rank.

**Top-K parameters:**

```
Initial retrieval (per modality): top-50
After RRF fusion:                 top-50
After cross-encoder reranking:    top-8  → sent to LLM
```

### 2.2 Reranking: Cross-Encoder

The cross-encoder scores `(query, chunk)` pairs jointly, capturing fine-grained relevance that bi-encoders miss (e.g., distinguishing "Artikel 3.114 lid 1" from "lid 2" for a specific query).

**Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`**
- 6-layer MiniLM: ~300ms CPU inference for 50 candidates
- ms-marco trained on passage retrieval (nearest analogue to legal Q&A)
- Self-hostable — no API calls, no data egress

*Cloud alternative:* Cohere Rerank v3 (higher accuracy, ~15% better MRR on legal benchmarks, but requires API access and data leaves the network).

See: `module2_retrieval/hybrid_search.py`

---

## Module 3 — Agentic RAG & Self-Healing (CRAG)

### 3.1 Query Transformation

**Multi-part query decomposition:**

Complex tax questions ("Can I deduct home office AND childcare simultaneously while my spouse also claims them?") are decomposed into atomic sub-queries via an LLM call. Each sub-query retrieves independently; answers are aggregated before final generation. This prevents retrieval dilution where a single vector query represents multiple information needs poorly.

**HyDE (Hypothetical Document Embeddings):**

For vague conceptual queries where raw query embedding underperforms, the LLM generates a *hypothetical answer document* in legal register. That document is embedded and used as the search vector. Legal answers share vocabulary with legal source documents, dramatically improving recall on semantic queries.

HyDE activates on the `expand_context` fallback path (see §3.2).

### 3.2 CRAG State Machine (LangGraph)

The control loop prevents hallucination by refusing to generate when retrieval fails, rather than guessing.

```
                    ┌─────────────────────┐
              ┌────►│   transform_query   │
              │     │  (decompose / HyDE) │
              │     └──────────┬──────────┘
              │                │
              │     ┌──────────▼──────────┐
              │     │      retrieve       │◄──────────────┐
              │     │  (hybrid search +   │               │
              │     │   reranker)         │               │
              │     └──────────┬──────────┘               │
              │                │                          │
              │     ┌──────────▼──────────┐               │
              │     │   grade_retrieval   │               │
              │     │  (LLM relevance     │               │
              │     │   judge)            │               │
              │     └──────┬──────┬───────┘               │
              │            │      │         │              │
              │        RELEVANT AMBIGUOUS IRRELEVANT       │
              │            │      │         │              │
              │            │      ▼         ▼              │
              │            │  expand_   rewrite_      (retry < 2)
              │            │  context   query    ─────────┘
              │            │  (HyDE)    (rephrase)
              │            │
              │     ┌──────▼──────────────┐
              │     │      generate       │
              │     │  (cite every claim) │
              │     └──────────┬──────────┘
              │                │
              │         ┌──────▼──────┐
              └─────────┤    output   │
                        └─────────────┘
```

**Grader grades:**

| Grade | Condition | Action |
|-------|-----------|--------|
| `relevant` | Chunks directly address the question with legal basis | Generate answer |
| `ambiguous` | Chunks partially relevant; answer would require assumptions | Apply HyDE, expand context, re-retrieve |
| `irrelevant` | Chunks miss the topic entirely | Rewrite query (max 2 retries), then refuse |

**Safe refusal:** After 2 retries with irrelevant results, the system returns a structured refusal citing the closest available document rather than hallucinating. This is the correct behaviour for a zero-hallucination system — an honest "I cannot confirm this" is better than a fabricated article citation.

See: `module3_agentic_rag/crag_state_machine.py`

---

## Module 4 — Production Ops, Security & Evaluation

### 4.1 Semantic Cache (Redis)

**Similarity threshold: 0.97 cosine similarity**

This is deliberately higher than typical (0.90–0.95) because fiscal data is year- and version-sensitive:

- "Box 1 rate **2023**" vs "Box 1 rate **2024**" → cosine similarity ~0.94
- At threshold 0.95, the wrong year's answer is returned as a cache hit
- At threshold 0.97, a one-word temporal shift forces re-retrieval

The cost of a wrong cached tax rate is legal liability. The cost of a cache miss is one extra LLM call. The asymmetry justifies a strict threshold.

**TTL by document type:**

| Doc type | TTL | Rationale |
|----------|-----|-----------|
| Legislation | 24 hours | Rates and thresholds can change with yearly fiscal updates |
| Policy guides | 7 days | Operational procedures update periodically |
| Case law | 30 days | Verdicts are immutable once issued |

**Cache invalidation:** The ingestion pipeline emits an invalidation event whenever a document is updated. The cache flushes all entries derived from that `doc_type`, ensuring no stale answers survive a legislative update.

**Role isolation:** Cached answers are keyed by `(query_hash, sorted_roles)`. A helpdesk answer for "What is Box 1?" is not served to an inspector whose answer may include restricted policy details.

See: `module4_production/semantic_cache.py`

---

### 4.2 RBAC — Where Filtering *Must* Occur

**The answer: pre-retrieval payload filtering in Qdrant. Never at generation time.**

Three possible approaches and why only one is acceptable:

| Approach | Security guarantee | Risk |
|----------|-------------------|------|
| Ask LLM to ignore restricted docs | **None** | Prompt injection in a retrieved FIOD document can override system instruction |
| Python filter on returned results | **Weak** | Application-layer bug or code path bypass exposes data |
| **Qdrant payload pre-filter** | **Mathematical** | Restricted chunks never enter the HNSW traversal — they are invisible at the vector distance computation level |

**Implementation:**

```python
Filter(
    must=[
        FieldCondition(
            key="access_roles",
            match=MatchAny(any=user.allowed_classifications),
        )
    ]
)
```

This filter is injected into the `SearchRequest` itself — not applied afterwards. Qdrant evaluates it before computing any vector distances. A helpdesk employee cannot retrieve a FIOD document regardless of query phrasing, prompt engineering, or application logic.

**Role hierarchy:**

| Role | Accessible classifications |
|------|--------------------------|
| `helpdesk` | `public`, `internal` |
| `inspector` | `public`, `internal`, `restricted` |
| `legal` | `public`, `internal`, `restricted`, `legal_classified` |
| `fiod` | all including `fiod` |

Every document is assigned an explicit `classification` and `access_roles` list during ingestion. Documents with missing classification are **rejected** — the system fails closed, not open.

See: `module4_production/rbac_security.py`

---

### 4.3 CI/CD & Evaluation Gate

Before any new embedding model or LLM is promoted to production, the CI pipeline runs the full evaluation suite against a golden dataset of expert-annotated question/answer pairs.

**Metrics (DeepEval, primary):**

| Metric | Threshold | What it measures |
|--------|-----------|-----------------|
| **Faithfulness** | ≥ 0.95 | Every claim in the answer is supported by a retrieved chunk. Core zero-hallucination guard. |
| **Context Precision** | ≥ 0.80 | Relevant chunks are ranked higher than irrelevant ones by the retriever. |
| **Context Recall** | ≥ 0.75 | All information needed to answer the question is present in retrieved chunks. |
| **Answer Relevancy** | ≥ 0.85 | The answer addresses the question (no topic drift or off-topic verbosity). |

**Faithfulness** is the primary gate. A drop from 0.97 to 0.94 blocks promotion even if other metrics pass — in a zero-hallucination system, unsupported claims are the primary failure mode.

**Ragas** is run as a secondary cross-check on Faithfulness using an independent LLM judge. Two independent judges grading faithfulness reduces false negatives in either direction.

**Regression gate:** If any metric drops more than 0.03 below the production baseline (stored as `eval/production_baseline.json`), the promotion is blocked even if the absolute threshold is still met. This catches gradual drift.

**CI trigger points:**
- New embedding model candidate → full retrieval + generation eval
- New LLM version → generation eval only (faithfulness + answer relevancy)
- Weekly scheduled run → regression check

See: `module4_production/evaluation_pipeline.py`

---

## Infrastructure Summary

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Vector DB | Qdrant (self-hosted) | Data sovereignty, payload RBAC, SQ8 quantization |
| Embedding model | multilingual-e5-large (on-prem) | No data egress, Dutch/English legal text |
| LLM | GPT-4o or Llama-3.1-70B (on-prem) | Configurable; on-prem for classified queries |
| Sparse vectors | SPLADE / BM25 via fastembed | Exact legal reference retrieval |
| Reranker | ms-marco-MiniLM-L-6-v2 | Fast, self-hosted, good passage reranking |
| Orchestration | LangGraph | Native state machine, LangChain ecosystem |
| Cache | Redis + RedisVL | Sub-millisecond cache lookup, vector similarity search |
| Evaluation | DeepEval + Ragas | Two independent faithfulness judges |
| Infrastructure | Kubernetes + Helm | Horizontal scaling for ingestion workers |

## Latency Budget (TTFT < 1.5s)

| Step | Budget |
|------|--------|
| Semantic cache lookup (Redis) | ~5 ms |
| Query embedding | ~50 ms |
| Hybrid search (Qdrant, SQ8 + HNSW) | ~80 ms |
| Cross-encoder reranking (50 → 8) | ~300 ms |
| LLM generation (first token) | ~900 ms |
| **Total** | **~1,335 ms** ✓ |

Cache hit path (FAQs): ~5 ms lookup + ~50 ms embedding = **~55 ms TTFT**.

---

## Repository Structure

```
cloudnation assesment/
├── ARCHITECTURE.md                   ← this document
├── module1_ingestion/
│   ├── chunking_strategy.py          ← hierarchical legal chunking + metadata
│   └── vector_db_config.py           ← Qdrant HNSW + SQ8 config + memory estimates
├── module2_retrieval/
│   └── hybrid_search.py              ← BM25 + dense RRF fusion + cross-encoder reranking
├── module3_agentic_rag/
│   └── crag_state_machine.py         ← LangGraph CRAG: decompose/HyDE/grade/self-heal
└── module4_production/
    ├── semantic_cache.py             ← Redis semantic cache, threshold=0.97, TTL by type
    ├── rbac_security.py              ← Qdrant pre-filter RBAC, role hierarchy, audit log
    └── evaluation_pipeline.py        ← DeepEval CI gate: faithfulness, precision, recall
```
