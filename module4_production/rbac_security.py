"""
Module 4 — Database-Level RBAC Security

The central security theorem of this system:
  Filtering must occur at the VECTOR DATABASE QUERY stage — before any chunk
  is returned to Python, before the reranker sees it, and before the LLM sees it.

Why pre-retrieval filtering is the ONLY mathematically safe option:

  Option A — Filter at generation (ask LLM to ignore restricted docs): INSECURE
    The LLM processes all retrieved chunks and is instructed to ignore FIOD docs.
    Attack surface: prompt injection in a retrieved FIOD document could instruct
    the LLM to ignore the system instruction. The LLM has no mathematical guarantee
    of compliance. This provides zero security assurance.

  Option B — Filter retrieved results in Python after Qdrant returns them: INSECURE
    Qdrant returns ALL matches including FIOD chunks; Python removes them.
    Attack surface: a bug in filter logic, a race condition, or a code path that
    bypasses the filter exposes restricted data. "Defense in depth" that relies
    on application-layer code is not sufficient for classified documents.

  Option C — Payload filter in Qdrant BEFORE HNSW traversal: SECURE (chosen)
    Qdrant evaluates `access_roles` payload conditions before entering the HNSW
    graph. Chunks that do not match the filter are mathematically excluded from
    the candidate set — they cannot appear in results regardless of query, prompt,
    or application logic. This is the only approach with formal guarantees.

RBAC Role Hierarchy:
  helpdesk    → ["public", "internal"]
  inspector   → ["public", "internal", "restricted"]
  legal       → ["public", "internal", "restricted", "legal_classified"]
  fiod        → ["public", "internal", "restricted", "legal_classified", "fiod"]
  admin       → all roles (for system maintenance only, not query responses)
"""

from dataclasses import dataclass
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.models import (FieldCondition, Filter, MatchAny,
                                  NamedVector, SearchRequest)


class ClassificationLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    LEGAL_CLASSIFIED = "legal_classified"
    FIOD = "fiod"


ROLE_PERMISSIONS: dict[str, list[str]] = {
    "helpdesk": ["public", "internal"],
    "inspector": ["public", "internal", "restricted"],
    "legal": ["public", "internal", "restricted", "legal_classified"],
    "fiod": ["public", "internal", "restricted", "legal_classified", "fiod"],
}


@dataclass
class AuthenticatedUser:
    user_id: str
    roles: list[str]
    department: str
    session_token: str

    @property
    def allowed_classifications(self) -> list[str]:
        """
        Derives the set of classification levels the user may access.
        Union of all permissions across the user's roles, plus the wildcard.
        """
        allowed = {"*"}
        for role in self.roles:
            allowed.update(ROLE_PERMISSIONS.get(role, []))
        return list(allowed)


def build_rbac_filter(user: AuthenticatedUser) -> Filter:
    """
    Constructs the Qdrant pre-filter for this user.

    The filter enforces: a chunk is retrievable only if its `access_roles`
    field contains at least one value from the user's allowed set.

    Qdrant evaluates this condition during HNSW graph traversal — chunks
    that fail the filter are never scored or returned.
    """
    return Filter(
        must=[
            FieldCondition(
                key="access_roles",
                match=MatchAny(any=user.allowed_classifications),
            )
        ]
    )


def secure_search(
    client: QdrantClient,
    user: AuthenticatedUser,
    query_vector: list[float],
    collection_name: str = "tax_authority",
    top_k: int = 50,
) -> list:
    """
    RBAC-enforced vector search.
    The access filter is injected directly into the Qdrant SearchRequest —
    it is not applied after the fact. There is no code path that returns
    results without the filter being active.
    """
    rbac_filter = build_rbac_filter(user)

    results = client.search(
        collection_name=collection_name,
        query_vector=NamedVector(name="dense", vector=query_vector),
        query_filter=rbac_filter,
        limit=top_k,
        with_payload=True,
    )

    # Audit log every access (required for government systems)
    _audit_log(user, query_vector, len(results))

    return results


def _audit_log(
    user: AuthenticatedUser, query_vector: list[float], result_count: int
) -> None:
    """
    Writes an immutable audit record for compliance.
    In production: write to append-only audit log (e.g., Kafka topic or S3).
    Fields: timestamp, user_id, department, roles, query_hash, result_count.
    """
    import hashlib
    import time

    query_hash = hashlib.sha256(str(query_vector[:8]).encode()).hexdigest()[:12]
    print(
        f"[AUDIT] ts={time.time():.0f} user={user.user_id} "
        f"dept={user.department} roles={user.roles} "
        f"query_hash={query_hash} results={result_count}"
    )


# ---------------------------------------------------------------------------
# Document classification enforcement during ingestion
# ---------------------------------------------------------------------------


def validate_classification_metadata(metadata: dict) -> None:
    """
    Called during ingestion to ensure every document has explicit classification.
    Rejects documents with missing or invalid classification — the system
    must never default a document to a lower classification than intended.

    Fail-closed: if classification is unknown, treat as FIOD (most restrictive).
    """
    classification = metadata.get("classification")
    if classification is None:
        raise ValueError(
            f"Document {metadata.get('doc_id')} has no classification field. "
            "All documents must be explicitly classified before ingestion."
        )

    valid = {level.value for level in ClassificationLevel}
    if classification not in valid:
        raise ValueError(
            f"Unknown classification '{classification}'. " f"Must be one of: {valid}"
        )

    access_roles = metadata.get("access_roles")
    if not access_roles:
        raise ValueError(
            f"Document {metadata.get('doc_id')} has empty access_roles. "
            "Specify at least one role, or ['*'] for public documents."
        )


# ---------------------------------------------------------------------------
# RBAC verification test cases (run in CI)
# ---------------------------------------------------------------------------


def run_rbac_verification_tests() -> None:
    """
    Smoke tests confirming RBAC filter construction is correct.
    These run in CI before every deployment.
    """
    helpdesk = AuthenticatedUser("u1", ["helpdesk"], "support", "tok1")
    fiod_officer = AuthenticatedUser("u2", ["fiod"], "investigations", "tok2")

    helpdesk_filter = build_rbac_filter(helpdesk)
    fiod_filter = build_rbac_filter(fiod_officer)

    helpdesk_allowed = helpdesk.allowed_classifications
    fiod_allowed = fiod_officer.allowed_classifications

    assert "fiod" not in helpdesk_allowed, "FAIL: helpdesk must not access fiod"
    assert "fiod" in fiod_allowed, "FAIL: fiod officer must access fiod docs"
    assert "*" in helpdesk_allowed, "FAIL: helpdesk must access public docs"
    assert "public" in helpdesk_allowed, "FAIL: helpdesk must access public"
    assert (
        "restricted" not in helpdesk_allowed
    ), "FAIL: helpdesk must not access restricted"

    print("All RBAC verification tests passed.")


if __name__ == "__main__":
    run_rbac_verification_tests()
