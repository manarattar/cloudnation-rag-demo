# Enterprise RAG — Tax Authority Demo

## Quick start (5 minutes)

### 1. Install Ollama
Download from https://ollama.com and pull a model:
```bash
ollama pull llama3.2
```
Ollama runs as a background service automatically after install.

### 2. Install Python dependencies
```bash
cd "E:\manar\cloudnation assesment"
pip install -r requirements.txt
```

### 3. Run the demo
```bash
streamlit run demo/app.py
```

Opens at `http://localhost:8501`. On first run it auto-ingests 10 sample Dutch tax
documents. No Docker, no API key, no cloud required.

---

## What the demo shows

| Feature | How to see it |
|---------|--------------|
| **Grounded answers with citations** | Ask any tax question — every claim has `[Source, Article]` |
| **RBAC access control** | Switch role to `helpdesk`, ask about FIOD fraud indicators — returns nothing; switch to `fiod` — classified document appears |
| **CRAG self-healing** | Ask a vague question — watch grade badge show `🟡 Ambiguous` or `🔴 Irrelevant`, system rewrites and retries |
| **Semantic cache** | Ask the same question twice — second response shows `⚡ Cache Hit` at ~5 ms latency |
| **Retrieval transparency** | Expand "Retrieved chunks" to see scores, citations, and access roles per chunk |
| **Live model picker** | Sidebar shows all installed Ollama models — switch models mid-session |

---

## Suggested demo queries

```
What is the Box 1 tax rate for 2024?
Can I combine a home office deduction and childcare costs?
What does ECLI:NL:HR:2023:123 rule about home offices?
What are the VAT rates in the Netherlands?
Tell me about FIOD fraud investigation indicators.   <- blocked for helpdesk role
What is the minimum salary for a DGA in 2024?        <- restricted, inspector+ only
```

---

## Switching Ollama models

The sidebar automatically lists all models you have installed. To add more:
```bash
ollama pull qwen2.5:7b
ollama pull mistral
ollama pull llama3.1:8b
```

---

## Architecture modules

| Module | File |
|--------|------|
| Hierarchical legal chunking | `module1_ingestion/chunking_strategy.py` |
| Qdrant HNSW + SQ8 config | `module1_ingestion/vector_db_config.py` |
| Hybrid BM25 + dense + RRF + reranker | `module2_retrieval/hybrid_search.py` |
| CRAG LangGraph state machine | `module3_agentic_rag/crag_state_machine.py` |
| Semantic cache (Redis) | `module4_production/semantic_cache.py` |
| RBAC pre-filter security | `module4_production/rbac_security.py` |
| CI/CD evaluation gate | `module4_production/evaluation_pipeline.py` |
| Full architecture doc | `ARCHITECTURE.md` |

---

## Production deployment (optional Docker)

```bash
docker compose up -d
```

Then set in `.env`:
```
QDRANT_MODE=docker
QDRANT_HOST=localhost
QDRANT_PORT=6333
```
