"""
End-to-end RAG pipeline for the demo.
Uses Qdrant persistent HNSW + fakeredis + local sentence-transformers.
LLM: any OpenAI-compatible endpoint — Groq, Ollama, LM Studio, OpenAI, etc.

Pipeline (v3):
  embed → cache → hybrid_retrieve(BM25+dense+RRF) → grade
  → if irrelevant: rewrite → hybrid_retrieve → grade
  → if still irrelevant: HyDE → hybrid_retrieve → grade
  → if relevant/ambiguous: rerank(top30→top5) → generate(parent_text)
"""

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from typing import Optional

import fakeredis
import httpx
import numpy as np
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchAny,
                                  PointStruct, VectorParams)
from sentence_transformers import CrossEncoder, SentenceTransformer

from demo.config import (CACHE_THRESHOLD, CHUNK_OVERLAP, CHUNK_SIZE,
                         COLLECTION, EMBEDDING_MODEL, TOP_K_FINAL,
                         TOP_K_RETRIEVE)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_embedder: Optional[SentenceTransformer] = None
_qdrant: Optional[QdrantClient] = None
_redis = fakeredis.FakeRedis()
_reranker: Optional[CrossEncoder] = None
_bm25: Optional["_BM25Index"] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        from demo.config import QDRANT_PATH

        try:
            _qdrant = QdrantClient(path=QDRANT_PATH)
        except Exception:
            _qdrant = QdrantClient(":memory:")
    return _qdrant


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _make_llm_client() -> openai.OpenAI:
    """Create an OpenAI-compatible client from live config."""
    import demo.config as _cfg

    if not _cfg.LLM_API_KEY:
        raise ValueError(
            "No LLM API key configured. "
            "Enter your Groq API key in the sidebar (free at console.groq.com)."
        )
    return openai.OpenAI(
        base_url=_cfg.LLM_BASE_URL,
        api_key=_cfg.LLM_API_KEY,
        http_client=httpx.Client(),
    )


def _llm_chat(prompt: str, max_tokens: int = 300) -> str:
    import demo.config as _cfg

    client = _make_llm_client()
    response = client.chat.completions.create(
        model=_cfg.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Query expansion (English → Dutch cross-lingual terms)
# ---------------------------------------------------------------------------

_EN_NL = {
    "box 1": "box 1 belastbaar inkomen werk woning tarief",
    "box 2": "box 2 aanmerkelijk belang tarief",
    "box 3": "box 3 sparen beleggen vermogen",
    "income tax": "inkomstenbelasting",
    "tax rate": "belastingtarief",
    "vat": "btw omzetbelasting",
    "home office": "thuiswerkkosten werkruimte aftrek",
    "childcare": "kinderopvang toeslag",
    "dga": "directeur-grootaandeelhouder gebruikelijk loon",
    "minimum salary": "gebruikelijk loon minimumloon",
    "fraud": "fraude belastingfraude opsporing",
    "deduction": "aftrek persoonsgebonden",
}


def _expand_query(text: str) -> str:
    lower = text.lower()
    extras = [nl for en, nl in _EN_NL.items() if en in lower]
    return text + (" | " + " | ".join(extras) if extras else "")


def embed(text: str, expand: bool = False) -> list[float]:
    if expand:
        text = _expand_query(text)
    vec = get_embedder().encode(text, normalize_embeddings=True)
    return vec.tolist()


# ---------------------------------------------------------------------------
# BM25 sparse index (pure Python — no external dependency)
# ---------------------------------------------------------------------------


class _BM25Index:
    """BM25 sparse retrieval index. Built in-memory on ingest."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []
        self.tf: list[dict] = []
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, docs: list[dict]) -> None:
        self.docs = docs
        tokenized = [self._tokenize(d.get("text", "")) for d in docs]
        self.tf = [defaultdict(int) for _ in docs]
        df: dict[str, int] = defaultdict(int)
        total_len = 0
        for i, tokens in enumerate(tokenized):
            total_len += len(tokens)
            seen: set[str] = set()
            for t in tokens:
                self.tf[i][t] += 1
                if t not in seen:
                    df[t] += 1
                    seen.add(t)
        n = len(docs)
        self.avg_dl = total_len / n if n else 1.0
        self.idf = {
            t: math.log((n - cnt + 0.5) / (cnt + 0.5) + 1) for t, cnt in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        tokens = self._tokenize(query)
        dl = sum(self.tf[doc_idx].values())
        s = 0.0
        for t in tokens:
            if t not in self.idf:
                continue
            tf_val = self.tf[doc_idx].get(t, 0)
            num = tf_val * (self.k1 + 1)
            den = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            s += self.idf[t] * num / den
        return s

    def search(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Parent-child chunking
# ---------------------------------------------------------------------------


def _split_into_chunks(doc: dict) -> list[dict]:
    """Split long documents into overlapping child chunks for precise retrieval.

    Child chunks carry parent_text so generation can use the full context.
    Short documents (≤ CHUNK_SIZE) are returned as-is (single chunk).
    """
    text = doc["text"]
    if len(text) <= CHUNK_SIZE:
        return [doc]

    parent_text = text
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_doc = {
            **doc,
            "text": text[start:end],
            "parent_text": parent_text,
            "is_child": True,
        }
        chunks.append(chunk_doc)
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_documents(documents: list[dict]) -> int:
    global _bm25
    client = get_qdrant()
    dim = get_embedder().get_sentence_embedding_dimension()

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # Flatten documents into child chunks (parent-child model)
    all_chunks: list[dict] = []
    for doc in documents:
        all_chunks.extend(_split_into_chunks(doc))

    # Compute citations and upsert to Qdrant
    points = []
    bm25_docs: list[dict] = []
    for i, chunk in enumerate(all_chunks):
        citation = f"{chunk['doc_title']}, {chunk.get('article', '')}".rstrip(", ")
        payload = {**chunk, "citation": citation}
        points.append(PointStruct(id=i, vector=embed(chunk["text"]), payload=payload))
        bm25_docs.append({**chunk, "citation": citation})

    client.upsert(collection_name=COLLECTION, points=points)

    # Build BM25 index on the same chunks
    _bm25 = _BM25Index()
    _bm25.build(bm25_docs)

    return len(points)


def is_collection_ready() -> bool:
    try:
        info = get_qdrant().get_collection(COLLECTION)
        return info.points_count > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# RBAC filter
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS = {
    "helpdesk": ["public", "internal", "*"],
    "inspector": ["public", "internal", "restricted", "*"],
    "legal": ["public", "internal", "restricted", "legal_classified", "*"],
    "fiod": ["public", "internal", "restricted", "legal_classified", "fiod", "*"],
}


def build_filter(role: str) -> Filter:
    allowed = ROLE_PERMISSIONS.get(role, ["public", "*"])
    return Filter(
        must=[
            FieldCondition(
                key="access_roles",
                match=MatchAny(any=allowed),
            )
        ]
    )


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------


def _cache_key(query_vec: list[float], role: str) -> str:
    digest = hashlib.sha256((str(query_vec[:8]) + role).encode()).hexdigest()[:16]
    return f"cache:{digest}"


def cache_lookup(query_vec: list[float], role: str) -> Optional[dict]:
    pattern = b"cache:*"
    for key in _redis.scan_iter(pattern):
        raw = _redis.get(key)
        if not raw:
            continue
        entry = json.loads(raw)
        cached_vec = np.array(entry["vector"], dtype=np.float32)
        query_arr = np.array(query_vec, dtype=np.float32)
        similarity = float(np.dot(cached_vec, query_arr))
        if similarity >= CACHE_THRESHOLD and entry.get("role") == role:
            return {
                "answer": entry["answer"],
                "citations": entry["citations"],
                "cache_hit": True,
                "similarity": round(similarity, 4),
            }
    return None


def cache_store(
    query_vec: list[float], role: str, answer: str, citations: list[str]
) -> None:
    key = _cache_key(query_vec, role)
    entry = {
        "vector": query_vec,
        "role": role,
        "answer": answer,
        "citations": citations,
        "ts": time.time(),
    }
    _redis.setex(key, 86400, json.dumps(entry))


# ---------------------------------------------------------------------------
# Retrieval — dense (Qdrant), sparse (BM25), hybrid (RRF)
# ---------------------------------------------------------------------------


def retrieve(
    query_vec: list[float], role: str, top_k: int = TOP_K_RETRIEVE
) -> list[dict]:
    """Dense vector retrieval with RBAC pre-filter."""
    results = (
        get_qdrant()
        .query_points(
            collection_name=COLLECTION,
            query=query_vec,
            query_filter=build_filter(role),
            limit=top_k,
            with_payload=True,
        )
        .points
    )
    return [
        {
            "text": r.payload.get("text", ""),
            "parent_text": r.payload.get("parent_text", ""),
            "is_child": r.payload.get("is_child", False),
            "citation": r.payload.get("citation", ""),
            "doc_type": r.payload.get("doc_type", ""),
            "score": round(r.score, 4),
            "access_roles": r.payload.get("access_roles", []),
        }
        for r in results
    ]


def retrieve_sparse(
    query_text: str, role: str, top_k: int = TOP_K_RETRIEVE
) -> list[dict]:
    """BM25 sparse retrieval with RBAC filtering."""
    global _bm25
    if _bm25 is None:
        return []
    allowed = set(ROLE_PERMISSIONS.get(role, ["public", "*"]))
    results: list[dict] = []
    for idx, bm25_score in _bm25.search(query_text, top_k=top_k * 3):
        chunk = _bm25.docs[idx]
        roles = set(chunk.get("access_roles", []))
        if not (allowed & roles):
            continue
        results.append(
            {
                "text": chunk.get("text", ""),
                "parent_text": chunk.get("parent_text", ""),
                "is_child": chunk.get("is_child", False),
                "citation": chunk.get("citation", ""),
                "doc_type": chunk.get("doc_type", ""),
                "score": round(bm25_score, 4),
                "access_roles": chunk.get("access_roles", []),
            }
        )
        if len(results) >= top_k:
            break
    return results


def _rrf_fuse(dense: list[dict], sparse: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion — merges dense and sparse lists keyed by citation."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for rank, chunk in enumerate(dense):
        key = chunk["citation"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        meta[key] = chunk

    for rank, chunk in enumerate(sparse):
        key = chunk["citation"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        if key not in meta:
            meta[key] = chunk

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for citation, rrf_score in fused:
        chunk = dict(meta[citation])
        chunk["rrf_score"] = round(rrf_score, 6)
        result.append(chunk)
    return result


def hybrid_retrieve(
    query_vec: list[float],
    query_text: str,
    role: str,
    top_k: int = TOP_K_RETRIEVE,
) -> list[dict]:
    """Hybrid BM25 + dense retrieval fused with RRF (k=60)."""
    dense = retrieve(query_vec, role, top_k=top_k)
    sparse = retrieve_sparse(query_text, role, top_k=top_k)
    if not sparse:
        return dense
    return _rrf_fuse(dense, sparse)[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


def rerank(query: str, chunks: list[dict], top_n: int = TOP_K_FINAL) -> list[dict]:
    """Re-score chunks with cross-encoder blended with cosine similarity.

    The cross-encoder is English-only (ms-marco-MiniLM-L-6-v2), so for Dutch
    documents it can under-score highly relevant chunks.  Blending 60% normalised
    cross-encoder score with 40% normalised cosine score prevents high-similarity
    Dutch documents from being displaced entirely.
    """
    if not chunks:
        return chunks
    pairs = [(query, c["text"][:512]) for c in chunks]
    scores = list(get_reranker().predict(pairs))
    min_s, max_s = min(scores), max(scores)
    rng = max(max_s - min_s, 1e-9)
    cosines = [c.get("score", 0.0) for c in chunks]
    max_cos = max(cosines) if cosines else 1.0
    for chunk, raw_s, cos_s in zip(chunks, scores, cosines):
        norm_rerank = (raw_s - min_s) / rng
        norm_cosine = cos_s / max(max_cos, 1e-9)
        chunk["rerank_score"] = round(float(raw_s), 4)
        chunk["_blend"] = 0.6 * norm_rerank + 0.4 * norm_cosine
    ranked = sorted(chunks, key=lambda c: c["_blend"], reverse=True)
    for c in ranked:
        c.pop("_blend", None)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Retrieval grader
# ---------------------------------------------------------------------------


def grade_retrieval(query: str, chunks: list[dict]) -> str:
    """Score-based grader — no LLM call, instant response."""
    if not chunks:
        return "irrelevant"
    top_score = chunks[0].get("score", 0)
    if top_score >= 0.3:
        return "relevant"
    if top_score >= 0.08:
        return "ambiguous"
    return "irrelevant"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """Je bent een nauwkeurige assistent voor de Nederlandse Belastingdienst.
Beantwoord de vraag VOLLEDIG en DIRECT op basis van UITSLUITEND de verstrekte context.
Regels:
- Antwoord ALTIJD in het Nederlands.
- Schrijf decimale getallen met een komma (bijv. 36,97%, niet 36.97%).
- Begin met het belangrijkste feit dat de vraag direct beantwoordt.
- Behandel ALLE relevante tarieven, drempelwaarden en regels uit de context.
- Voeg na elke feitelijke bewering een inline-citaat toe: [Documenttitel, Artikel].
- Als de context geen antwoord bevat, zeg dat dan duidelijk — gok niet.

Context:
{context}

Vraag: {query}

Antwoord:"""

NO_ANSWER_TEMPLATE = (
    "Op basis van de beschikbare documenten kan deze vraag niet met zekerheid worden beantwoord. "
    "De meest relevante bron is: [{citation}]. "
    "Raadpleeg de relevante wetgeving rechtstreeks of neem contact op met een belastinginspecteur."
)


def _generation_text(c: dict) -> str:
    """Use full parent context for generation when the chunk is a child."""
    return c.get("parent_text") or c.get("text", "")


def generate_answer(query: str, chunks: list[dict]) -> tuple[str, list[str]]:
    context = "\n\n---\n\n".join(
        f"[{c['citation']}]\n{_generation_text(c)}" for c in chunks[:TOP_K_FINAL]
    )
    citations = [c["citation"] for c in chunks[:TOP_K_FINAL]]
    prompt = GENERATION_PROMPT.format(context=context, query=query)
    try:
        answer = _llm_chat(prompt, max_tokens=600)
        return answer, citations
    except Exception as e:
        return f"LLM error: {e}", citations


def rewrite_query(query: str) -> str:
    prompt = (
        f"Herschrijf deze Nederlandse belastingvraag met specifiekere juridische terminologie "
        f"of wetsartikelverwijzingen. Geef alleen de herschreven vraag terug.\n\nVraag: {query}"
    )
    try:
        return _llm_chat(prompt, max_tokens=100)
    except Exception:
        return query


def hyde_embed(query: str) -> list[float]:
    """Hypothetical Document Embedding — generate a hypothetical answer and embed it.

    Used as a last-resort self-heal when rewrite also fails (CRAG attempt 2).
    Falls back to the original query embedding on LLM error.
    """
    prompt = (
        f"Schrijf een beknopt antwoord (2-3 zinnen) op deze vraag als ware je een "
        f"belastingwetboek:\n\n{query}"
    )
    try:
        hypo_doc = _llm_chat(prompt, max_tokens=120)
        return embed(hypo_doc, expand=False)
    except Exception:
        return embed(query)


def generate_answer_stream(query: str, chunks: list[dict], out: dict):
    """Streaming answer generator. Yields token strings; populates out['citations']."""
    context = "\n\n---\n\n".join(
        f"[{c['citation']}]\n{_generation_text(c)[:400]}" for c in chunks[:TOP_K_FINAL]
    )
    out["citations"] = [c["citation"] for c in chunks[:TOP_K_FINAL]]
    prompt = GENERATION_PROMPT.format(context=context, query=query)

    import demo.config as _cfg

    try:
        client = _make_llm_client()
        stream = client.chat.completions.create(
            model=_cfg.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
            stream=True,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        yield f"LLM error: {e}"


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------


def query_streaming(
    user_query: str,
    role: str = "helpdesk",
    on_step=None,
) -> dict:
    """Streaming pipeline: hybrid retrieve → rerank → grade → generate."""

    def _noop(_):
        pass

    if on_step is None:
        on_step = _noop

    t0 = time.time()

    on_step("embed")
    query_vec = embed(user_query)
    expanded_vec = embed(user_query, expand=True)

    on_step("cache")
    cached = cache_lookup(query_vec, role)
    if cached:
        cached["grade"] = "cached"
        cached["latency_ms"] = round((time.time() - t0) * 1000)
        cached["stream_gen"] = None
        return cached

    on_step("retrieve")
    chunks = hybrid_retrieve(expanded_vec, user_query, role)

    on_step("grade")
    grade = grade_retrieval(user_query, chunks)

    out: dict = {"citations": []}

    if grade in ("relevant", "ambiguous") and chunks:
        chunks = rerank(user_query, chunks, top_n=TOP_K_FINAL)
        on_step("generate")
        stream_gen = generate_answer_stream(user_query, chunks, out)
    else:
        closest = chunks[0]["citation"] if chunks else "N/A"
        out["answer"] = NO_ANSWER_TEMPLATE.format(citation=closest)
        out["citations"] = [closest] if chunks else []
        stream_gen = None

    return {
        "stream_gen": stream_gen,
        "out": out,
        "grade": grade,
        "chunks": chunks[:TOP_K_FINAL],
        "cache_hit": False,
        "t0": t0,
        "query_vec": query_vec,
        "role": role,
    }


# ---------------------------------------------------------------------------
# Full CRAG pipeline (non-streaming)
# ---------------------------------------------------------------------------


def query(user_query: str, role: str = "helpdesk") -> dict:
    """Full pipeline: cache → hybrid retrieve → grade → self-heal → generate.

    Self-healing loop (max 2 attempts):
      Attempt 1: rewrite query with legal terminology
      Attempt 2: HyDE — embed a hypothetical answer document
    """
    t0 = time.time()

    query_vec = embed(user_query)
    expanded_vec = embed(user_query, expand=True)

    # 1. Semantic cache
    cached = cache_lookup(query_vec, role)
    if cached:
        cached["grade"] = "cached"
        cached["latency_ms"] = round((time.time() - t0) * 1000)
        return cached

    # 2. Hybrid retrieve (BM25 + dense + RRF)
    chunks = hybrid_retrieve(expanded_vec, user_query, role)

    # 3. Grade
    grade = grade_retrieval(user_query, chunks)

    # 4. Self-heal attempt 1: rewrite query
    if grade == "irrelevant":
        rewritten = rewrite_query(user_query)
        rewritten_vec = embed(rewritten, expand=True)
        chunks = hybrid_retrieve(rewritten_vec, rewritten, role)
        grade = grade_retrieval(rewritten, chunks)

    # 5. Self-heal attempt 2: HyDE (hypothetical document embedding)
    if grade == "irrelevant":
        hyde_vec = hyde_embed(user_query)
        chunks = hybrid_retrieve(hyde_vec, user_query, role)
        grade = grade_retrieval(user_query, chunks)

    # 6. Rerank top candidates before generation
    if grade in ("relevant", "ambiguous") and chunks:
        chunks = rerank(user_query, chunks, top_n=TOP_K_FINAL)

    # 7. Generate or refuse
    if grade in ("relevant", "ambiguous") and chunks:
        answer, citations = generate_answer(user_query, chunks)
    else:
        closest = chunks[0]["citation"] if chunks else "N/A"
        answer = NO_ANSWER_TEMPLATE.format(citation=closest)
        citations = [closest] if chunks else []

    # 8. Cache successful results only
    if not answer.startswith("LLM error"):
        cache_store(query_vec, role, answer, citations)

    return {
        "answer": answer,
        "citations": citations,
        "grade": grade,
        "chunks": chunks[:TOP_K_FINAL],
        "cache_hit": False,
        "latency_ms": round((time.time() - t0) * 1000),
    }
