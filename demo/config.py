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
COLLECTION = "tax_authority_demo"
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 3
CACHE_THRESHOLD = 0.97
