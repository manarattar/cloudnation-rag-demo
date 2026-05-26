"""
Streamlit demo — Enterprise RAG for Tax Authority
Run: streamlit run demo/app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

import demo.config as _cfg  # noqa: E402
from demo.pipeline import get_qdrant  # noqa: E402
from demo.pipeline import (cache_store, ingest_documents, is_collection_ready,
                           query_streaming)
from demo.sample_documents import DOCUMENTS  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ollama_models() -> list[str]:
    try:
        import ollama

        from demo.config import LLM_MODEL as _default

        result = ollama.list()
        models = (
            result.models if hasattr(result, "models") else result.get("models", [])
        )
        names = []
        for m in models:
            name = m.model if hasattr(m, "model") else m.get("name") or m.get("model")
            if name:
                names.append(name)
        return names if names else [_default]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tax Authority AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS — professional legal / government aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
/* ── Base ── */
html, body, [data-testid="stApp"] { background: #F7F6F2 !important; }
[data-testid="stMain"]            { background: #F7F6F2 !important; }
[data-testid="stAppViewContainer"]{ background: #F7F6F2 !important; }

/* ── Sidebar — deep navy ── */
[data-testid="stSidebar"] {
    background: #1E3A5F !important;
    border-right: 3px solid #C4A14A !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] caption { color: #D0D8E8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #C4A14A !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #A8B8CC !important; }
[data-testid="stSidebar"] hr { border-color: rgba(196,161,74,0.35) !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #B8C8D8 !important; }

/* ── Hero banner ── */
.hero-banner {
    background: #1E3A5F;
    border-bottom: 4px solid #C4A14A;
    border-radius: 6px;
    padding: 1.5rem 2rem 1.2rem;
    margin-bottom: 1.4rem;
}
.hero-banner h1 {
    color: #FFFFFF;
    margin: 0;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: 0.01em;
}
.hero-banner p {
    color: #A8C0D8;
    margin: 0.35rem 0 0;
    font-size: 0.88rem;
}
.hero-pills { margin-top: 0.8rem; display: flex; gap: 0.45rem; flex-wrap: wrap; }
.hero-pill {
    background: rgba(196,161,74,0.15);
    color: #C4A14A;
    border: 1px solid rgba(196,161,74,0.4);
    padding: 2px 9px; border-radius: 3px; font-size: 0.72rem;
    font-weight: 500; letter-spacing: 0.04em;
}

/* ── Role badge pills (sidebar) ── */
.role-pill {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 3px;
    font-size: 0.76rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
}
.pill-helpdesk  { background: #0D3320; color: #5CDB8C; border: 1px solid #1A7A45; }
.pill-inspector { background: #3A2C00; color: #F0B840; border: 1px solid #C48A00; }
.pill-legal     { background: #3A1800; color: #F08040; border: 1px solid #C44800; }
.pill-fiod      { background: #3A0000; color: #F06060; border: 1px solid #C42020; }

/* ── Content cards (main area) ── */
.legal-card {
    background: #FFFFFF;
    border: 1px solid #D8D2C4;
    border-left: 4px solid #1E3A5F;
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(30,58,95,0.07);
}
.legal-card:hover { border-left-color: #C4A14A; box-shadow: 0 2px 8px rgba(30,58,95,0.12); }
.legal-card-title {
    font-size: 0.9rem; font-weight: 600;
    color: #1E3A5F; margin-bottom: 0.35rem;
}
.legal-card-body  { font-size: 0.8rem; color: #555550; line-height: 1.65; }
.legal-card-meta  {
    margin-top: 0.55rem; display: flex;
    gap: 0.45rem; flex-wrap: wrap; align-items: center;
}

/* ── Classification badges ── */
.cls-badge {
    display: inline-block; padding: 1px 7px; border-radius: 2px;
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.cls-public           { background: #E6F4ED; color: #1A6B3A; border: 1px solid #A8D8BC; }
.cls-internal         { background: #E6EEF8; color: #1A3A7A; border: 1px solid #A8C0DC; }
.cls-restricted       { background: #FBF3E0; color: #7A5800; border: 1px solid #D8B840; }
.cls-legal-classified { background: #FBE8DC; color: #7A3000; border: 1px solid #D88850; }
.cls-fiod             { background: #F8E0E0; color: #8B1A1A; border: 1px solid #D87070; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #E0DAD0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(30,58,95,0.05);
}

/* ── Expanders ── */
[data-testid="stExpander"] details {
    background: #FFFFFF !important;
    border: 1px solid #D8D2C4 !important;
    border-radius: 4px !important;
}

/* ── Metric labels ── */
[data-testid="stMetricLabel"] { color: #556677 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #1E3A5F !important; }

/* ── Tab styling ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600; color: #4A6A8A;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1E3A5F !important;
    border-bottom: 2px solid #C4A14A !important;
}

/* ── Buttons ── */
.stButton button {
    border: 1px solid #C8C0B0 !important;
    color: #1E3A5F !important;
    background: #FFFFFF !important;
    font-size: 0.82rem !important;
    border-radius: 3px !important;
}
.stButton button:hover {
    background: #EEE8DC !important;
    border-color: #1E3A5F !important;
}

/* ── Generating placeholder ── */
.gen-waiting {
    color: #4A6A8A;
    font-style: italic;
    padding: 0.6rem 0;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

if "ingested" not in st.session_state:
    with st.spinner("Loading knowledge base… (first run only)"):
        if not is_collection_ready():
            n = ingest_documents(DOCUMENTS)
            st.session_state["ingested"] = n
        else:
            st.session_state["ingested"] = len(DOCUMENTS)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "stats" not in st.session_state:
    st.session_state["stats"] = {"total": 0, "cache_hits": 0, "latencies": []}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚖️ Tax Authority RAG")
    st.caption("Enterprise RAG Demo — CloudNation Assessment")
    st.divider()

    st.subheader("Access Role (RBAC)")
    role = st.selectbox(
        "Your role",
        ["helpdesk", "inspector", "legal", "fiod"],
        index=0,
        help=(
            "Controls which documents you can retrieve.\n"
            "Helpdesk: public + internal only.\n"
            "FIOD: all including classified documents."
        ),
    )
    role_access = {
        "helpdesk": "Public · Internal",
        "inspector": "Public · Internal · Restricted",
        "legal": "Public · Internal · Restricted · Legal-classified",
        "fiod": "All documents including FIOD classified",
    }
    st.markdown(
        f'<span class="role-pill pill-{role}">{role.upper()}</span>'
        f"<br><small style='color:#A8B8CC;margin-top:6px;display:block'>"
        f"{role_access[role]}</small>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.subheader("Knowledge Base")
    st.metric("Documents loaded", st.session_state["ingested"])
    doc_types: dict = {}
    for d in DOCUMENTS:
        doc_types[d["doc_type"]] = doc_types.get(d["doc_type"], 0) + 1
    for dtype, count in doc_types.items():
        st.caption(f"  • {dtype}: {count}")

    st.divider()

    # ── Index & Scale ────────────────────────────────────────────────
    st.subheader("🏗️ Vector Index")
    try:
        _info = get_qdrant().get_collection(_cfg.COLLECTION)
        _vec_count = _info.points_count
        _vec_dim = _info.config.params.vectors.size
    except Exception:
        _vec_count = st.session_state.get("ingested", 0)
        _vec_dim = 384

    _ci1, _ci2 = st.columns(2)
    _ci1.metric("Vectors", f"{_vec_count:,}")
    _ci2.metric("Dimensions", _vec_dim)
    st.metric("Index", "HNSW · O(log n)")
    st.markdown(
        """
<div style="background:rgba(196,161,74,0.08);border:1px solid rgba(196,161,74,0.25);
border-radius:4px;padding:0.55rem 0.75rem;margin:0.3rem 0 0.5rem;
font-size:0.74rem;color:#A8B8CC;line-height:1.6">
<span style="color:#C4A14A;font-weight:600">Production scale</span><br>
500 K docs → ~1.1 GB index<br>
INT8 quantization → ~280 MB<br>
Search latency: &lt;20 ms<br>
Ingestion: one-time batch job
</div>""",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Rebuild index", help="Delete and re-ingest the vector index"):
        try:
            get_qdrant().delete_collection(_cfg.COLLECTION)
        except Exception:
            pass
        st.session_state.pop("ingested", None)
        st.rerun()

    st.divider()

    st.subheader("🤖 LLM Provider")
    provider = st.selectbox(
        "Provider",
        ["Groq (cloud — fast)", "Ollama (local)"],
        index=0,
        help="Groq runs the same model on LPU chips — sub-second responses. "
        "Ollama runs locally but is slower on CPU.",
    )

    if "Groq" in provider:
        _key_preconfigured = bool(_cfg.LLM_API_KEY)
        if _key_preconfigured:
            st.success("Groq configured ✓", icon="✅")
            groq_key = _cfg.LLM_API_KEY
        else:
            groq_key = st.text_input(
                "Groq API key",
                value="",
                type="password",
                placeholder="gsk_…",
                help="Free key at console.groq.com — 30 req/min on free tier.",
            )
            if groq_key:
                st.success("Groq configured ✓")
            else:
                st.warning("Enter your Groq API key above.")
        groq_model = st.selectbox(
            "Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ],
            index=0,
        )
        _cfg.LLM_BASE_URL = "https://api.groq.com/openai/v1"
        _cfg.LLM_API_KEY = groq_key
        _cfg.LLM_MODEL = groq_model
    else:
        ollama_models = _get_ollama_models()
        if ollama_models:
            selected_model = st.selectbox("Model", ollama_models, index=0)
            _cfg.LLM_BASE_URL = "http://localhost:11434/v1"
            _cfg.LLM_API_KEY = "ollama"
            _cfg.LLM_MODEL = selected_model
            st.caption("Endpoint: `http://localhost:11434`")
        else:
            st.error("Ollama not running — start with: `ollama serve`")
            st.caption("Then: `ollama pull llama3.2`")

    st.divider()

    st.subheader("📊 Session Stats")
    _s = st.session_state["stats"]
    _total = _s["total"]
    _hits = _s["cache_hits"]
    _lats = _s["latencies"]
    _hit_rate = f"{round(_hits / _total * 100)}%" if _total else "—"
    _avg_lat = f"{round(sum(_lats) / len(_lats))} ms" if _lats else "—"

    c1, c2 = st.columns(2)
    c1.metric("Queries", _total)
    c2.metric("Cache hits", _hits)
    st.metric("Cache hit rate", _hit_rate)
    st.metric("Avg latency", _avg_lat)

    if st.button("🗑️ Clear chat & stats"):
        st.session_state["messages"] = []
        st.session_state["stats"] = {"total": 0, "cache_hits": 0, "latencies": []}
        st.rerun()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_LABELS = {
    "embed": "🔢 Embedding query",
    "cache": "⚡ Checking semantic cache",
    "retrieve": "🔍 Retrieving relevant chunks",
    "grade": "📊 Grading retrieval quality",
    "generate": "🤖 Generating answer",
}

GRADE_BADGE = {
    "relevant": "🟢 Relevant",
    "ambiguous": "🟡 Ambiguous",
    "irrelevant": "🔴 Irrelevant",
    "cached": "⚡ Cache Hit",
}

SUGGESTED = [
    "What is the Box 1 tax rate for 2024?",
    "Can I combine a home office deduction and childcare costs?",
    "What does ECLI:NL:HR:2023:123 rule about home offices?",
    "What are the VAT rates in the Netherlands?",
    "Tell me about FIOD fraud investigation indicators.",
    "What is the minimum salary for a DGA in 2024?",
]

CLS_PRIORITY = ["fiod", "legal_classified", "restricted", "internal", "public"]
CLS_META = {
    "public": ("cls-public", "Public"),
    "internal": ("cls-internal", "Internal"),
    "restricted": ("cls-restricted", "Restricted"),
    "legal_classified": ("cls-legal-classified", "Legal Classified"),
    "fiod": ("cls-fiod", "FIOD Classified"),
}
DOCTYPE_ICON = {
    "legislation": "📜",
    "case_law": "⚖️",
    "policy": "📋",
    "e_learning": "🎓",
    "fiod_classified": "🔒",
}


def _top_cls(access_roles: list) -> str:
    for c in CLS_PRIORITY:
        if c in access_roles:
            return c
    return "public"


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_meta(meta: dict) -> None:
    grade = meta.get("grade", "")
    cache_hit = meta.get("cache_hit", False)
    latency = meta.get("latency_ms", 0)
    citations = meta.get("citations", [])
    chunks = meta.get("chunks", [])

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval grade", GRADE_BADGE.get(grade, grade))
    col2.metric("Cache", "HIT ⚡" if cache_hit else "MISS")
    col3.metric("Latency", f"{latency} ms")

    if citations:
        with st.expander("📚 Sources cited", expanded=False):
            for c in citations:
                st.markdown(f"- `{c}`")

    if chunks and not cache_hit:
        with st.expander("🔍 Retrieved chunks", expanded=False):
            for i, chunk in enumerate(chunks[:5], 1):
                score = chunk.get("score", 0)
                citation = chunk.get("citation", "")
                text = chunk.get("text", "")
                access = chunk.get("access_roles", [])
                st.markdown(
                    f"**Chunk {i}** — `{citation}` "
                    f"| score: `{score}` | roles: `{access}`"
                )
                st.caption(text[:300] + ("…" if len(text) > 300 else ""))
                st.divider()


def _kb_card(doc: dict) -> str:
    top = _top_cls(doc.get("access_roles", []))
    css_cls, label = CLS_META.get(top, ("cls-public", top))
    icon = DOCTYPE_ICON.get(doc.get("doc_type", ""), "📄")
    snippet = doc.get("text", "")[:190].replace("<", "&lt;").replace(">", "&gt;")
    if len(doc.get("text", "")) > 190:
        snippet += "…"
    title = doc.get("doc_title", "Untitled").replace("<", "&lt;")
    dtype = doc.get("doc_type", "").replace("_", " ").title()
    return (
        f'<div class="legal-card">'
        f'<div class="legal-card-title">{icon} {title}</div>'
        f'<div class="legal-card-body">{snippet}</div>'
        f'<div class="legal-card-meta">'
        f'<span class="cls-badge {css_cls}">{label}</span>'
        f'<span style="color:#888880;font-size:0.73rem;margin-left:4px">{dtype}</span>'
        f"</div></div>"
    )


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="hero-banner">
  <h1>⚖️ Belastingdienst AI Assistant</h1>
  <p>Enterprise RAG — RBAC · Semantic Cache · Corrective RAG · Live Vector Search</p>
  <div class="hero-pills">
    <span class="hero-pill">Qdrant in-memory</span>
    <span class="hero-pill">sentence-transformers</span>
    <span class="hero-pill">Groq LPU · Ollama</span>
    <span class="hero-pill">CRAG pipeline</span>
    <span class="hero-pill">semantic cache</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_kb = st.tabs(["🤖 Assistant", "📚 Knowledge Base"])

# ---------------------------------------------------------------------------
# Knowledge Base tab
# ---------------------------------------------------------------------------

with tab_kb:
    st.subheader("Indexed Documents")
    st.caption(
        "All documents are stored as 384-dim dense vectors (all-MiniLM-L6-v2). "
        "Classification level controls per-role retrieval via RBAC pre-filtering."
    )

    all_cls = sorted(
        {_top_cls(d.get("access_roles", [])) for d in DOCUMENTS},
        key=lambda x: CLS_PRIORITY.index(x) if x in CLS_PRIORITY else 99,
        reverse=True,
    )
    filter_cls = st.multiselect(
        "Filter by classification",
        options=all_cls,
        default=all_cls,
        format_func=lambda x: CLS_META.get(x, (None, x))[1],
    )

    visible = [
        d for d in DOCUMENTS if _top_cls(d.get("access_roles", [])) in filter_cls
    ]
    st.caption(f"Showing **{len(visible)}** of {len(DOCUMENTS)} documents")
    st.divider()

    col_a, col_b = st.columns(2, gap="medium")
    for i, doc in enumerate(visible):
        (col_a if i % 2 == 0 else col_b).markdown(_kb_card(doc), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Assistant tab
# ---------------------------------------------------------------------------

with tab_chat:
    # Guard: require API key before accepting queries
    _llm_ready = bool(_cfg.LLM_API_KEY)
    if not _llm_ready:
        st.warning(
            "**LLM not configured.** Enter your Groq API key in the sidebar to start. "
            "Get a free key at [console.groq.com](https://console.groq.com).",
            icon="🔑",
        )

    st.caption("💡 Try a suggested query:")
    cols = st.columns(3)
    for i, suggestion in enumerate(SUGGESTED):
        btn = cols[i % 3].button(
            suggestion,
            key=f"sug_{i}",
            use_container_width=True,
            disabled=not _llm_ready,
        )
        if btn:
            st.session_state["_pending_query"] = suggestion
            st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                _render_meta(msg["meta"])

    # Input
    _placeholder = (
        "Enter Groq API key in sidebar first…"
        if not _llm_ready
        else "Ask a fiscal or legal question…"
    )
    user_input = st.chat_input(_placeholder, disabled=not _llm_ready)

    if "_pending_query" in st.session_state:
        user_input = st.session_state.pop("_pending_query")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            # ── Pipeline status ──────────────────────────────────────────
            with st.status("Running CRAG pipeline…", expanded=True) as status_box:

                def _on_step(label: str) -> None:
                    st.write(STEP_LABELS.get(label, label))

                result = query_streaming(user_input, role=role, on_step=_on_step)

                if result.get("cache_hit"):
                    status_box.update(
                        label="⚡ Cache hit — answer ready",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status_box.update(
                        label="✅ Pipeline complete",
                        state="complete",
                        expanded=False,
                    )

            # ── Generate answer ──────────────────────────────────────────
            # llama3.1:8b on CPU takes ~20s TTFT; drain generator inside a
            # spinner so the UI stays animated instead of appearing frozen.
            if result.get("stream_gen") is not None:
                _provider_label = "Groq" if "groq" in _cfg.LLM_BASE_URL else "Ollama"
                with st.spinner(f"Generating answer via {_provider_label}…"):
                    tokens: list[str] = list(result["stream_gen"])
                answer = "".join(tokens).strip() or "*No response generated.*"
                citations = result["out"].get("citations", [])
                st.markdown(answer)

            elif result.get("cache_hit"):
                answer = result["answer"]
                citations = result.get("citations", [])
                st.markdown(answer)

            else:
                answer = result["out"].get("answer", "")
                citations = result["out"].get("citations", [])
                st.markdown(answer)

            # ── Cache result after streaming ─────────────────────────────
            if not result.get("cache_hit") and result.get("query_vec"):
                cache_store(result["query_vec"], result["role"], answer, citations)

            # ── Latency & metadata ───────────────────────────────────────
            latency_ms = (
                result.get("latency_ms", 0)
                if result.get("cache_hit")
                else round((time.time() - result["t0"]) * 1000)
            )

            full_result = {
                "answer": answer,
                "citations": citations,
                "grade": result.get("grade", ""),
                "chunks": result.get("chunks", []),
                "cache_hit": result.get("cache_hit", False),
                "latency_ms": latency_ms,
            }
            _render_meta(full_result)

        # ── Update session stats ─────────────────────────────────────────
        s = st.session_state["stats"]
        s["total"] += 1
        if result.get("cache_hit"):
            s["cache_hits"] += 1
        s["latencies"].append(latency_ms)

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "meta": full_result}
        )
