import os

# LLM — defaults to Groq (fast cloud inference).
# Switch to Ollama by setting LLM_BASE_URL=http://localhost:11434/v1
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")


def _load_api_key() -> str:
    """Load API key: env var → Streamlit secrets (cloud) → empty."""
    if key := os.getenv("LLM_API_KEY", ""):
        return key
    if key := os.getenv("GROQ_API_KEY", ""):
        return key
    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY", "") or st.secrets.get("LLM_API_KEY", "")
    except Exception:
        return ""


LLM_API_KEY = _load_api_key()
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# v3: parent-child chunking + hybrid BM25 retrieval + cross-encoder reranker
COLLECTION = "tax_authority_demo_v3"

# Persistent HNSW index — survives restarts, scales to 500K+ docs.
# Falls back to :memory: if the path is not writable (e.g. read-only container).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDRANT_PATH = os.path.join(_project_root, "qdrant_data")

TOP_K_RETRIEVE = 30  # candidates before reranking (BM25+dense combined)
TOP_K_FINAL = 5  # chunks passed to generation after reranking
CACHE_THRESHOLD = 0.97

# Parent-child chunking — long docs split into child chunks for retrieval;
# generation uses the full parent text for richer context.
CHUNK_SIZE = 400  # max chars per child chunk
CHUNK_OVERLAP = 50  # overlap between consecutive child chunks
