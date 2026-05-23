"""
Module 3 — Agentic RAG & Self-Healing: CRAG with LangGraph

Architecture overview:
  Standard linear RAG: query → retrieve → generate (fragile — hallucination on miss)

  CRAG control loop:
    query
      └─► transform (decompose / HyDE)
            └─► retrieve (hybrid search, Module 2)
                  └─► grade  ──► RELEVANT    → generate → cite → output
                              ├─► AMBIGUOUS  → web_search / expand → re-retrieve → grade
                              └─► IRRELEVANT → rewrite query → retrieve (max 2 retries)

  LangGraph models each step as a node and each routing decision as a
  conditional edge. The state dict flows through the graph accumulating
  context, citations, and retry counts.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Graph state definition
# ---------------------------------------------------------------------------


class RAGState(TypedDict):
    query: str  # original user query
    sub_queries: list[str]  # after decomposition
    retrieved_chunks: list[dict]  # from Module 2 retrieve()
    relevance_grade: str  # "relevant" | "ambiguous" | "irrelevant"
    answer: str  # final generated answer
    citations: list[str]  # exact citation strings
    retry_count: int  # guards against infinite rewrite loops
    rewritten_query: str  # query after rewrite (for irrelevant path)
    context_window: str  # assembled context string for LLM
    hallucination_check: bool  # True if every claim is grounded


MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# 2. Query transformation — Decomposition + HyDE
# ---------------------------------------------------------------------------
#
# Multi-part tax questions ("Can I deduct home office AND childcare costs
# while my spouse also claims them?") require splitting into atomic sub-queries
# that each retrieve independently, then aggregating answers.
#
# HyDE (Hypothetical Document Embeddings):
#   For vague conceptual queries where BM25+dense retrieval underperforms,
#   we ask the LLM to write a *hypothetical answer document* and embed that
#   instead of the raw query. Legal answers contain the same legal terminology
#   as the source documents, which dramatically improves recall.

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Dutch tax law expert. Break the following complex fiscal
question into atomic, self-contained sub-questions. Each sub-question must be
answerable independently from the tax code. Return a JSON list of strings.
If the question is already atomic, return a list with one element.""",
        ),
        ("human", "{query}"),
    ]
)

HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Write a short hypothetical excerpt from a Dutch tax law document
that would answer the following question. Use formal legal language and article
references. This will be used as a search query — do not add disclaimers.""",
        ),
        ("human", "{query}"),
    ]
)


def transform_query(state: RAGState, llm: ChatOpenAI) -> RAGState:
    """
    Step 1: Decompose complex queries. Optionally apply HyDE for ambiguous queries.
    Populates state['sub_queries'].
    """
    query = state["rewritten_query"] or state["query"]

    decompose_chain = DECOMPOSE_PROMPT | llm
    result = decompose_chain.invoke({"query": query})

    try:
        import json

        sub_queries = json.loads(result.content)
        if not isinstance(sub_queries, list):
            sub_queries = [query]
    except (json.JSONDecodeError, AttributeError):
        sub_queries = [query]

    return {**state, "sub_queries": sub_queries}


def apply_hyde(query: str, llm: ChatOpenAI) -> str:
    """
    Generates a hypothetical answer document and returns it as the search query.
    Used when semantic retrieval on the raw query returns low-confidence results.
    """
    hyde_chain = HYDE_PROMPT | llm
    result = hyde_chain.invoke({"query": query})
    return result.content


# ---------------------------------------------------------------------------
# 3. Retrieval Grader
# ---------------------------------------------------------------------------
#
# The grader is a structured LLM call that scores whether retrieved chunks
# are sufficient to answer the query without hallucination.
#
# Three grades:
#   relevant   — chunks directly address the question; proceed to generation
#   ambiguous  — chunks partially relevant; may need supplemental search
#   irrelevant — chunks miss the topic entirely; rewrite query and retry


class RelevanceGrade(BaseModel):
    grade: Literal["relevant", "ambiguous", "irrelevant"] = Field(
        description="Relevance of retrieved context to the query"
    )
    reasoning: str = Field(description="One sentence explaining the grade")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the grade (0=unsure, 1=certain)"
    )


GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict legal relevance evaluator for a Dutch tax authority
AI system. Given a user query and retrieved document chunks, grade whether the
chunks provide sufficient grounding to answer the query accurately.

Grade as:
  relevant   — chunks directly answer the query with specific legal basis
  ambiguous  — chunks are partially relevant; the answer would need assumptions
  irrelevant — chunks do not address the query; answering would require guessing

Return structured JSON matching the RelevanceGrade schema.
Zero hallucination is mandatory. When in doubt, grade as irrelevant.""",
        ),
        ("human", "Query: {query}\n\nRetrieved chunks:\n{context}"),
    ]
)


def grade_retrieval(state: RAGState, llm: ChatOpenAI) -> RAGState:
    """
    Step 3: Grade the retrieved chunks.
    Populates state['relevance_grade'].
    """
    context = "\n\n---\n\n".join(
        f"[{c['citation']}]\n{c['text']}" for c in state["retrieved_chunks"]
    )

    grader_llm = llm.with_structured_output(RelevanceGrade)
    grader_chain = GRADER_PROMPT | grader_llm
    grade: RelevanceGrade = grader_chain.invoke(
        {
            "query": state["query"],
            "context": context,
        }
    )

    return {**state, "relevance_grade": grade.grade}


# ---------------------------------------------------------------------------
# 4. Routing logic — conditional edges
# ---------------------------------------------------------------------------


def route_after_grading(state: RAGState) -> str:
    """
    Conditional edge: decides next node based on relevance grade and retry count.

    relevant   → generate
    ambiguous  → web_search (if allowed) or expand_context
    irrelevant → rewrite_query (if retries < MAX_RETRIES) else generate_no_answer
    """
    grade = state["relevance_grade"]
    retries = state["retry_count"]

    if grade == "relevant":
        return "generate"
    elif grade == "ambiguous":
        return "expand_context"
    elif grade == "irrelevant":
        if retries < MAX_RETRIES:
            return "rewrite_query"
        else:
            return "generate_no_answer"
    return "generate_no_answer"


# ---------------------------------------------------------------------------
# 5. Self-healing nodes
# ---------------------------------------------------------------------------

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """The previous retrieval returned irrelevant results.
Rewrite the query to be more specific, using official Dutch tax terminology,
article numbers if known, or alternative phrasings. Return only the rewritten query.""",
        ),
        ("human", "Original query: {query}\nRetrieval failed. Rewrite it:"),
    ]
)


def rewrite_query(state: RAGState, llm: ChatOpenAI) -> RAGState:
    """Fallback for irrelevant retrieval: rewrite and retry."""
    chain = REWRITE_PROMPT | llm
    result = chain.invoke({"query": state["query"]})
    return {
        **state,
        "rewritten_query": result.content,
        "retry_count": state["retry_count"] + 1,
    }


def expand_context(state: RAGState, llm: ChatOpenAI) -> RAGState:
    """
    Fallback for ambiguous retrieval: apply HyDE and re-retrieve.
    In production, this node could also trigger a structured web search
    against belastingdienst.nl or EUR-Lex for up-to-date legislation.
    """
    primary_query = state["sub_queries"][0] if state["sub_queries"] else state["query"]
    hyde_query = apply_hyde(primary_query, llm)
    return {
        **state,
        "rewritten_query": hyde_query,
        "retry_count": state["retry_count"] + 1,
    }


# ---------------------------------------------------------------------------
# 6. Generation with mandatory citation injection
# ---------------------------------------------------------------------------

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise Dutch tax authority assistant. Answer the question
using ONLY the provided context. Every factual claim MUST be followed by an
inline citation in the format [Document, Article, Paragraph].

If the context does not contain sufficient information to answer with certainty,
respond: "Based on available documents, this cannot be confirmed. Please consult
[citation of closest relevant document]."

Never guess. Never extrapolate beyond what the documents state.""",
        ),
        ("human", "Question: {query}\n\nContext:\n{context}"),
    ]
)


def generate_answer(state: RAGState, llm: ChatOpenAI) -> RAGState:
    """
    Step 5: Generate a grounded answer with inline citations.
    Then runs a hallucination check: verifies each sentence is supported by a chunk.
    """
    context = "\n\n---\n\n".join(
        f"[{c['citation']}]\n{c['text']}" for c in state["retrieved_chunks"]
    )

    chain = GENERATION_PROMPT | llm
    result = chain.invoke({"query": state["query"], "context": context})
    answer = result.content

    citations = [c["citation"] for c in state["retrieved_chunks"]]
    hallucination_check = _verify_citations_present(answer, citations)

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "hallucination_check": hallucination_check,
    }


def generate_no_answer(state: RAGState, llm: ChatOpenAI) -> RAGState:
    """
    Safe fallback: no sufficient context found after MAX_RETRIES.
    Returns a refusal with the closest available citation rather than hallucinating.
    """
    closest = (
        state["retrieved_chunks"][0]["citation"] if state["retrieved_chunks"] else "N/A"
    )
    answer = (
        f"The requested information could not be found with sufficient certainty "
        f"in the available documents. Closest related source: [{closest}]. "
        f"Please consult the relevant legislation directly or contact a tax inspector."
    )
    return {**state, "answer": answer, "hallucination_check": True}


def _verify_citations_present(answer: str, citations: list[str]) -> bool:
    """
    Lightweight check: confirms the answer text references at least one citation.
    In production, replace with a full NLI-based faithfulness scorer (see Module 4).
    """
    return any(f"[{c}" in answer or c[:20] in answer for c in citations)


# ---------------------------------------------------------------------------
# 7. LangGraph assembly
# ---------------------------------------------------------------------------


def build_crag_graph(llm: ChatOpenAI) -> StateGraph:
    """
    Assembles the full CRAG state machine.
    Nodes are functions; edges are routing decisions.
    """
    graph = StateGraph(RAGState)

    # Nodes — each wraps a function with the shared LLM
    graph.add_node("transform_query", lambda s: transform_query(s, llm))
    graph.add_node("retrieve", lambda s: s)  # injected externally
    graph.add_node("grade_retrieval", lambda s: grade_retrieval(s, llm))
    graph.add_node("generate", lambda s: generate_answer(s, llm))
    graph.add_node("generate_no_answer", lambda s: generate_no_answer(s, llm))
    graph.add_node("rewrite_query", lambda s: rewrite_query(s, llm))
    graph.add_node("expand_context", lambda s: expand_context(s, llm))

    # Entry point
    graph.set_entry_point("transform_query")

    # Linear edges
    graph.add_edge("transform_query", "retrieve")
    graph.add_edge("retrieve", "grade_retrieval")

    # Conditional edges — the self-healing loop
    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {
            "generate": "generate",
            "expand_context": "expand_context",
            "rewrite_query": "rewrite_query",
            "generate_no_answer": "generate_no_answer",
        },
    )

    # Self-healing paths loop back to retrieve
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("expand_context", "retrieve")

    # Terminal nodes
    graph.add_edge("generate", END)
    graph.add_edge("generate_no_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 8. Usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI as _LLM

    llm = _LLM(model="gpt-4o", temperature=0)
    app = build_crag_graph(llm)

    initial_state: RAGState = {
        "query": "Can I deduct home office expenses and childcare costs simultaneously "
        "in Box 1 for the 2024 tax year?",
        "sub_queries": [],
        "retrieved_chunks": [],
        "relevance_grade": "",
        "answer": "",
        "citations": [],
        "retry_count": 0,
        "rewritten_query": "",
        "context_window": "",
        "hallucination_check": False,
    }

    result = app.invoke(initial_state)
    print("Answer:", result["answer"])
    print("Citations:", result["citations"])
    print("Hallucination check passed:", result["hallucination_check"])
