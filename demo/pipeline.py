"""
End-to-end RAG pipeline for the demo.
Uses Qdrant in-memory (no Docker needed) + fakeredis + local sentence-transformers.
LLM: any OpenAI-compatible endpoint — Groq, Ollama, LM Studio, OpenAI, etc.
"""

import hashlib
import json
import time
from typing import Optional

import fakeredis
import httpx
import numpy as np
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchAny,
                                  PointStruct, VectorParams)
from sentence_transformers import SentenceTransformer

from demo.config import (CACHE_THRESHOLD, COLLECTION, EMBEDDING_MODEL,
                         TOP_K_FINAL, TOP_K_RETRIEVE)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_embedder: Optional[SentenceTransformer] = None
_qdrant: Optional[QdrantClient] = None
_redis = fakeredis.FakeRedis()


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


def _make_llm_client() -> openai.OpenAI:
    """Create an OpenAI-compatible client from live config. Works with Groq, Ollama, etc."""
    import demo.config as _cfg

    if not _cfg.LLM_API_KEY:
        raise ValueError(
            "No LLM API key configured. "
            "Enter your Groq API key in the sidebar (free at console.groq.com)."
        )
    return openai.OpenAI(
        base_url=_cfg.LLM_BASE_URL,
        api_key=_cfg.LLM_API_KEY,
        http_client=httpx.Client(),  # explicit client avoids httpx/proxies version bug
    )


def _llm_chat(prompt: str, max_tokens: int = 300) -> str:
    """Single entry point for all LLM calls via OpenAI-compatible API."""
    import demo.config as _cfg

    client = _make_llm_client()
    response = client.chat.completions.create(
        model=_cfg.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# Common English→Dutch tax term mappings for better cross-lingual retrieval
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
    """Append Dutch equivalents for known English tax terms to improve retrieval."""
    lower = text.lower()
    extras = [nl for en, nl in _EN_NL.items() if en in lower]
    return text + (" | " + " | ".join(extras) if extras else "")


def embed(text: str, expand: bool = False) -> list[float]:
    if expand:
        text = _expand_query(text)
    vec = get_embedder().encode(text, normalize_embeddings=True)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_documents(documents: list[dict]) -> int:
    client = get_qdrant()
    dim = get_embedder().get_sentence_embedding_dimension()

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    points = []
    for i, doc in enumerate(documents):
        vector = embed(doc["text"])
        citation = f"{doc['doc_title']}, {doc.get('article', '')}".rstrip(", ")
        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={**doc, "citation": citation},
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
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
    """Scan cache for a semantically similar prior query."""
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
# Retrieval
# ---------------------------------------------------------------------------


def retrieve(
    query_vec: list[float], role: str, top_k: int = TOP_K_RETRIEVE
) -> list[dict]:
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
            "citation": r.payload.get("citation", ""),
            "doc_type": r.payload.get("doc_type", ""),
            "score": round(r.score, 4),
            "access_roles": r.payload.get("access_roles", []),
        }
        for r in results
    ]


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


def generate_answer(query: str, chunks: list[dict]) -> tuple[str, list[str]]:
    context = "\n\n---\n\n".join(
        f"[{c['citation']}]\n{c['text']}" for c in chunks[:TOP_K_FINAL]
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


def generate_answer_stream(query: str, chunks: list[dict], out: dict):
    """Streaming answer generator. Yields token strings; populates out['citations']."""
    # Truncate each chunk to keep the prompt short and generation fast
    context = "\n\n---\n\n".join(
        f"[{c['citation']}]\n{c['text'][:400]}" for c in chunks[:TOP_K_FINAL]
    )
    out["citations"] = [c["citation"] for c in chunks[:TOP_K_FINAL]]
    prompt = GENERATION_PROMPT.format(context=context, query=query)

    import demo.config as _cfg  # read live

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


def query_streaming(
    user_query: str,
    role: str = "helpdesk",
    on_step=None,
) -> dict:
    """
    Streaming pipeline variant.
    Caller drains result['stream_gen'] then calls cache_store().

    Keys returned:
      stream_gen (generator | None), out (dict), grade, chunks,
      cache_hit, t0, query_vec, role.
    Cache-hit path returns: answer, citations, grade='cached', stream_gen=None.
    """

    def _noop(_):
        pass

    if on_step is None:
        on_step = _noop

    t0 = time.time()

    on_step("embed")
    # Cache lookup uses the plain query vec for exact match; retrieval uses expanded vec
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
    chunks = retrieve(expanded_vec, role)

    on_step("grade")
    grade = grade_retrieval(user_query, chunks)

    out: dict = {"citations": []}

    if grade in ("relevant", "ambiguous") and chunks:
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
# Full CRAG pipeline
# ---------------------------------------------------------------------------


def query(user_query: str, role: str = "helpdesk") -> dict:
    """
    Full pipeline: cache → retrieve → grade → (self-heal) → generate.
    Returns a dict with answer, citations, grade, cache_hit, latency_ms.
    """
    t0 = time.time()

    # Plain vec for cache (exact match); expanded vec for retrieval (cross-lingual)
    query_vec = embed(user_query)
    expanded_vec = embed(user_query, expand=True)

    # 1. Semantic cache check
    cached = cache_lookup(query_vec, role)
    if cached:
        cached["grade"] = "cached"
        cached["latency_ms"] = round((time.time() - t0) * 1000)
        return cached

    # 2. Retrieve
    chunks = retrieve(expanded_vec, role)

    # 3. Grade
    grade = grade_retrieval(user_query, chunks)

    # 4. Self-heal: rewrite once if irrelevant
    if grade == "irrelevant" and chunks:
        rewritten = rewrite_query(user_query)
        rewritten_vec = embed(rewritten)
        chunks = retrieve(rewritten_vec, role)
        grade = grade_retrieval(rewritten, chunks)

    # 5. Generate or refuse
    if grade in ("relevant", "ambiguous") and chunks:
        answer, citations = generate_answer(user_query, chunks)
    else:
        closest = chunks[0]["citation"] if chunks else "N/A"
        answer = NO_ANSWER_TEMPLATE.format(citation=closest)
        citations = [closest] if chunks else []

    # 6. Cache only successful results — don't propagate error strings
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
