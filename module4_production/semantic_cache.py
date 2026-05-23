"""
Module 4 — Semantic Cache using Redis + vector similarity

Design:
  Redis with the RedisVL library stores (query_vector, answer, citations, ttl).
  On each request, we embed the query and check cosine similarity against cached
  queries. If similarity >= threshold, return the cached answer immediately —
  skipping retrieval and generation entirely.

Threshold: 0.97 cosine similarity
  Why so high (vs typical 0.90-0.95)?
  Tax and fiscal data has year-sensitivity and version-sensitivity.
  "Box 1 rate 2023" vs "Box 1 rate 2024" can have different answers but embed
  at ~0.94 similarity. A threshold of 0.95 would return the wrong year's answer.
  0.97 forces re-retrieval for any query where a one-word change shifts meaning.
  This is a deliberate precision-over-recall trade-off: cache misses cost compute,
  cache hits with wrong answers cost trust and legal liability.

TTL strategy:
  - Tax rates / statutory values: 24 hours (change on legislative updates)
  - Procedural guides: 7 days
  - Case law summaries: 30 days (verdicts are immutable once issued)
  On any document update event, the ingestion pipeline flushes affected cache keys.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema

SIMILARITY_THRESHOLD = 0.97

TTL_BY_DOC_TYPE = {
    "legislation": 86_400,  # 24 hours — tax rates change
    "policy": 604_800,  # 7 days
    "case_law": 2_592_000,  # 30 days — verdicts are immutable
    "elearning": 604_800,  # 7 days
    "default": 86_400,
}

CACHE_INDEX_SCHEMA = IndexSchema.from_dict(
    {
        "index": {"name": "tax_semantic_cache", "prefix": "cache:"},
        "fields": [
            {"name": "query_text", "type": "text"},
            {"name": "answer", "type": "text"},
            {"name": "citations", "type": "text"},
            {"name": "doc_type", "type": "tag"},
            {"name": "user_roles", "type": "tag"},
            {"name": "created_at", "type": "numeric"},
            {
                "name": "query_vector",
                "type": "vector",
                "attrs": {
                    "dims": 1536,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
        ],
    }
)


@dataclass
class CacheEntry:
    query_text: str
    answer: str
    citations: list[str]
    doc_type: str
    user_roles: list[str]
    query_vector: list[float]
    created_at: float = 0.0

    def cache_key(self) -> str:
        payload = f"{self.query_text}:{sorted(self.user_roles)}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class SemanticCache:
    """
    Semantic cache backed by Redis vector search.
    Role-aware: cached answers are only served to users with matching roles.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.client = redis.from_url(redis_url)
        self.index = SearchIndex(CACHE_INDEX_SCHEMA, redis_client=self.client)
        self.index.create(overwrite=False)

    def lookup(
        self,
        query_vector: list[float],
        user_roles: list[str],
        top_k: int = 1,
    ) -> Optional[dict]:
        """
        Searches the cache for a semantically similar prior query.
        Returns the cached answer if similarity >= threshold AND roles match.
        Returns None on cache miss.
        """
        q = VectorQuery(
            vector=query_vector,
            vector_field_name="query_vector",
            return_fields=[
                "query_text",
                "answer",
                "citations",
                "user_roles",
                "doc_type",
            ],
            num_results=top_k,
            return_score=True,
        )

        results = self.index.query(q)
        if not results:
            return None

        best = results[0]
        similarity = 1.0 - float(best.get("vector_distance", 1.0))

        if similarity < SIMILARITY_THRESHOLD:
            return None

        cached_roles = set(best.get("user_roles", "").split(","))
        request_roles = set(user_roles)
        if not request_roles.issubset(cached_roles):
            return None

        return {
            "answer": best["answer"],
            "citations": json.loads(best["citations"]),
            "cache_hit": True,
            "similarity": round(similarity, 4),
            "source": "semantic_cache",
        }

    def store(
        self,
        entry: CacheEntry,
        doc_type: str = "default",
    ) -> None:
        """Stores a new cache entry with appropriate TTL."""
        ttl = TTL_BY_DOC_TYPE.get(doc_type, TTL_BY_DOC_TYPE["default"])
        key = f"cache:{entry.cache_key()}"

        data = {
            "query_text": entry.query_text,
            "answer": entry.answer,
            "citations": json.dumps(entry.citations),
            "doc_type": doc_type,
            "user_roles": ",".join(sorted(entry.user_roles)),
            "created_at": time.time(),
            "query_vector": np.array(entry.query_vector, dtype=np.float32).tobytes(),
        }

        pipe = self.client.pipeline()
        pipe.hset(key, mapping=data)
        pipe.expire(key, ttl)
        pipe.execute()

    def invalidate_by_doc_type(self, doc_type: str) -> int:
        """
        Evicts all cached answers derived from a given doc_type.
        Called by the ingestion pipeline when a legislation update is detected.
        """
        pattern = "cache:*"
        keys_to_delete = []
        for key in self.client.scan_iter(pattern):
            cached_doc_type = self.client.hget(key, "doc_type")
            if cached_doc_type and cached_doc_type.decode() == doc_type:
                keys_to_delete.append(key)

        if keys_to_delete:
            self.client.delete(*keys_to_delete)
        return len(keys_to_delete)
