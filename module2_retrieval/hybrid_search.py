"""
Module 2 — Retrieval Strategy: Hybrid Search + Reranking

Design decisions:
  Fusion:    Reciprocal Rank Fusion (RRF) over fixed alpha-weighting.
             Legal queries mix exact citations ("ECLI:NL:HR:2023:123") and
             semantic concepts ("home office deductibility"). RRF adapts
             per-query without manual alpha tuning.
  Sparse:    BM25 via Qdrant's sparse vector support (SPLADE or BM25 vectors).
             Catches exact article numbers, ECLI codes, and statutory references
             that dense embeddings compress away.
  Dense:     text-embedding-3-large (1536-dim) or multilingual-e5-large (1024-dim).
             Handles paraphrase, legal synonyms, and conceptual queries.
  Reranker:  cross-encoder/ms-marco-MiniLM-L-6-v2 as primary.
             Fast (CPU-feasible), 6-layer MiniLM, 300ms for top-50 candidates.
             Cohere Rerank v3 listed as cloud alternative for higher accuracy.
  Top-K:     Initial retrieval = 50 per modality → RRF → 50 fused → reranker → 8 final.
"""

import uuid
from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (FieldCondition, Filter, MatchAny, MatchValue,
                                  NamedSparseVector, NamedVector,
                                  QueryResponse, ScoredPoint, SearchRequest,
                                  SparseVector)
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# 1. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
#
# Why RRF over alpha weighting?
#   Alpha weighting (score = α*dense + (1-α)*sparse) requires calibrating α
#   per query type. For a legal corpus where some queries are pure keyword
#   ("Artikel 3.114") and others are semantic ("what expenses can I deduct
#   working from home?"), a fixed alpha always under-serves one type.
#
#   RRF(doc) = Σ 1 / (k + rank_i)   where k=60 is a smoothing constant.
#   Documents ranked highly in *either* list get boosted.
#   k=60 is the standard value from the original Cormack et al. paper.

RRF_K = 60
INITIAL_TOP_K = 50  # retrieved per modality before fusion
RERANKER_TOP_K = 8  # final candidates passed to the LLM context window


def reciprocal_rank_fusion(
    dense_results: list[ScoredPoint],
    sparse_results: list[ScoredPoint],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Fuses dense and sparse result lists using RRF.
    Returns list of (point_id, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}

    for rank, point in enumerate(dense_results):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    for rank, point in enumerate(sparse_results):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# 2. RBAC-aware hybrid retrieval
# ---------------------------------------------------------------------------


@dataclass
class UserContext:
    user_id: str
    roles: list[str]  # e.g. ["helpdesk"], ["inspector", "legal"], ["fiod"]
    department: str


def build_access_filter(user_context: UserContext) -> Filter:
    """
    Constructs a Qdrant payload filter that enforces RBAC.
    Only documents whose access_roles overlap with the user's roles are returned.

    This filter is applied BEFORE vector distance computation — Qdrant's
    pre-filtering guarantees that restricted chunks never enter the candidate set.
    A helpdesk user literally cannot receive FIOD document chunks regardless of
    query phrasing or LLM behavior. (See Module 4 for full security design.)
    """
    return Filter(
        must=[
            FieldCondition(
                key="access_roles",
                match=MatchAny(any=user_context.roles + ["*"]),
            )
        ]
    )


def hybrid_search(
    client: QdrantClient,
    query_dense_vector: list[float],
    query_sparse_vector: SparseVector,
    user_context: UserContext,
    collection_name: str = "tax_authority",
    top_k: int = INITIAL_TOP_K,
) -> list[ScoredPoint]:
    """
    Executes dense + sparse search in parallel using Qdrant's batch search API.
    Returns fused and deduplicated results before reranking.
    """
    access_filter = build_access_filter(user_context)

    # Batch both searches into a single network round-trip
    results = client.search_batch(
        collection_name=collection_name,
        requests=[
            # Dense (semantic) search
            SearchRequest(
                vector=NamedVector(name="dense", vector=query_dense_vector),
                filter=access_filter,
                limit=top_k,
                with_payload=True,
            ),
            # Sparse (BM25/SPLADE) search
            SearchRequest(
                vector=NamedSparseVector(
                    name="sparse",
                    vector=query_sparse_vector,
                ),
                filter=access_filter,
                limit=top_k,
                with_payload=True,
            ),
        ],
    )

    dense_hits, sparse_hits = results[0], results[1]

    # Fuse with RRF
    fused_ranked = reciprocal_rank_fusion(dense_hits, sparse_hits)

    # Reconstruct ScoredPoint objects preserving payload (needed for reranker)
    id_to_point: dict[str, ScoredPoint] = {
        str(p.id): p for p in dense_hits + sparse_hits
    }
    fused_points = [id_to_point[pid] for pid, _ in fused_ranked if pid in id_to_point]

    return fused_points[:top_k]


# ---------------------------------------------------------------------------
# 3. Cross-Encoder Reranking
# ---------------------------------------------------------------------------
#
# Cross-encoders score (query, document) pairs jointly — unlike bi-encoders
# which encode independently. This captures fine-grained relevance signals
# (e.g., distinguishing "Article 3.114 paragraph 1" from "paragraph 2") at
# the cost of O(n) forward passes. With top_k=50 candidates and MiniLM-L-6,
# this costs ~300ms on CPU — acceptable given we are post-fusion.
#
# Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - 6-layer MiniLM: fast, good for passage reranking
#   - ms-marco: trained on passage retrieval (closest to our Q&A use case)
#   - Self-hostable (critical for government deployment)
#
# Alternative: Cohere Rerank v3 (API call, higher accuracy, not self-hosted)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    return _reranker


def rerank_results(
    query: str,
    candidates: list[ScoredPoint],
    top_k: int = RERANKER_TOP_K,
) -> list[dict]:
    """
    Reranks candidate chunks using the cross-encoder.
    Returns top_k results with text, score, and citation metadata.
    """
    reranker = get_reranker()

    pairs = [(query, point.payload.get("text", "")) for point in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "text": point.payload.get("text", ""),
            "citation": point.payload.get("citation", "Unknown"),
            "score": float(score),
            "doc_type": point.payload.get("doc_type", ""),
            "article": point.payload.get("article", ""),
            "paragraph": point.payload.get("paragraph", ""),
            "access_roles": point.payload.get("access_roles", []),
        }
        for point, score in ranked[:top_k]
    ]


# ---------------------------------------------------------------------------
# 4. Full retrieval pipeline
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    query_dense_vector: list[float],
    query_sparse_vector: SparseVector,
    user_context: UserContext,
    client: QdrantClient,
) -> list[dict]:
    """
    End-to-end retrieval: hybrid search → RRF fusion → cross-encoder reranking.
    Returns top RERANKER_TOP_K chunks with citation metadata attached.
    """
    candidates = hybrid_search(
        client=client,
        query_dense_vector=query_dense_vector,
        query_sparse_vector=query_sparse_vector,
        user_context=user_context,
    )

    return rerank_results(query=query, candidates=candidates)


# ---------------------------------------------------------------------------
# 5. BM25 / sparse vector generation (pseudo-code)
# ---------------------------------------------------------------------------
#
# Qdrant supports SPLADE sparse vectors natively. During ingestion we generate
# sparse vectors alongside dense vectors and store them in the "sparse" named
# vector field. At query time we generate the sparse vector the same way.
#
# from fastembed import SparseTextEmbedding
#
# sparse_model = SparseTextEmbedding("Qdrant/bm25")
#
# def encode_sparse(text: str) -> SparseVector:
#     result = list(sparse_model.embed([text]))[0]
#     return SparseVector(indices=result.indices.tolist(),
#                         values=result.values.tolist())
