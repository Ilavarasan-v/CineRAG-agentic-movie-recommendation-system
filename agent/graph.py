"""
agent/graph.py
──────────────
LangGraph state machine: plan → retrieve → check → generate

All LLM calls go through config.openai_client.chat() which:
  - tries OpenAI (gpt-4o-mini) first
  - automatically falls back to Groq (llama3-8b-8192) if OpenAI fails
"""

import json
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END
from config.openai_client import chat
from agent.tools import vector_search_tool, genre_filter_tool, web_search_tool
from retriever.reranker import rerank
from config.settings import MAX_AGENT_LOOPS


# ─────────────────────────────────────────────
# STATE SCHEMA
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    query:         str
    sub_questions: list[str]
    tool_calls:    list[dict]
    raw_docs:      Annotated[list, add]
    reranked_docs: list[dict]
    context_score: int
    loop_count:    int
    answer:        str
    sources:       list[str]


# ─────────────────────────────────────────────
# NODE 1: PLAN
# Breaks the user query into sub-questions and
# decides which tool to call for each one.
# Falls back to Groq automatically if OpenAI fails.
# ─────────────────────────────────────────────

def plan_node(state: AgentState) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a query planner for a movie recommendation RAG system. "
                "Given a user question, decompose it into 1-3 focused sub-questions "
                "and for each sub-question decide which tool to use.\n\n"
                "Available tools:\n"
                "  - vector_search: general semantic search\n"
                "  - genre_filter_search: search with genre/year filter (JSON input)\n"
                "  - web_search: fallback for rare or recent movies\n\n"
                "Respond ONLY as valid JSON — no extra text:\n"
                '{"sub_questions": ["q1"], '
                '"tool_calls": [{"tool": "vector_search", "input": "q1"}]}'
            )
        },
        {
            "role": "user",
            "content": f"User question: {state['query']}"
        }
    ]

    resp = chat(messages)   # OpenAI first → Groq fallback

    try:
        raw  = resp.strip().strip("```json").strip("```").strip()
        plan = json.loads(raw)
    except Exception:
        # If JSON parsing fails, default to a simple vector search
        plan = {
            "sub_questions": [state["query"]],
            "tool_calls":    [{"tool": "vector_search", "input": state["query"]}]
        }

    return {
        "sub_questions": plan.get("sub_questions", [state["query"]]),
        "tool_calls":    plan.get("tool_calls",    []),
        "loop_count":    state.get("loop_count", 0)
    }


# ─────────────────────────────────────────────
# NODE 2: RETRIEVE
# Calls the planned tools, deduplicates results,
# and passes them through the reranker.
# No LLM call here — pure retrieval.
# ─────────────────────────────────────────────

_TOOL_MAP = {
    "vector_search":       vector_search_tool,
    "genre_filter_search": genre_filter_tool,
    "web_search":          web_search_tool,
}

def retrieve_node(state: AgentState) -> dict:
    all_docs: list[dict] = []

    for call in state.get("tool_calls", []):
        tool_name  = call.get("tool", "vector_search")
        tool_input = call.get("input", state["query"])
        tool       = _TOOL_MAP.get(tool_name, _TOOL_MAP["vector_search"])

        # genre_filter_search expects a JSON string input
        if tool_name == "genre_filter_search" and isinstance(tool_input, dict):
            tool_input = json.dumps(tool_input)

        raw_result = tool.func(str(tool_input))

        try:
            docs = json.loads(raw_result)
            if isinstance(docs, list):
                all_docs.extend(docs)
        except Exception:
            pass

    # Deduplicate by movie title
    seen, unique_docs = set(), []
    for doc in all_docs:
        title = doc.get("title", "")
        if title not in seen:
            seen.add(title)
            unique_docs.append(doc)

    reranked = rerank(state["query"], unique_docs)

    return {
        "raw_docs":      unique_docs,
        "reranked_docs": reranked,
        "loop_count":    state.get("loop_count", 0) + 1
    }


# ─────────────────────────────────────────────
# NODE 3: CHECK
# LLM reads retrieved docs and scores relevance
# 0–10. Low score triggers a retry loop.
# Falls back to Groq automatically if OpenAI fails.
# ─────────────────────────────────────────────

def check_node(state: AgentState) -> dict:
    docs    = state.get("reranked_docs", [])
    context = "\n".join(
        f"- {d.get('title','?')} ({d.get('year','?')}): {d.get('plot','')[:120]}"
        for d in docs
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are evaluating whether retrieved movie documents are sufficient "
                "to answer the user's question. Score from 0 (completely useless) "
                "to 10 (perfectly answers the question). "
                "Reply ONLY with valid JSON — no extra text:\n"
                '{"score": <int 0-10>, "reason": "<one short sentence>"}'
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {state['query']}\n\n"
                f"Retrieved documents:\n{context or 'None retrieved'}"
            )
        }
    ]

    resp = chat(messages)   # OpenAI first → Groq fallback

    try:
        raw    = resp.strip().strip("```json").strip("```").strip()
        result = json.loads(raw)
        score  = int(result.get("score", 5))
    except Exception:
        score  = 5   # default to middle score if parsing fails

    return {"context_score": score}


# ─────────────────────────────────────────────
# CONDITIONAL EDGE
# Decides whether to retry retrieval or proceed
# to generate based on the context quality score.
# ─────────────────────────────────────────────

def should_retry(state: AgentState) -> str:
    score      = state.get("context_score", 10)
    loop_count = state.get("loop_count", 0)

    if score < 6 and loop_count < MAX_AGENT_LOOPS:
        return "retrieve"   # not good enough — try again
    return "generate"       # good enough — generate the answer


# ─────────────────────────────────────────────
# NODE 4: GENERATE
# Synthesises the final answer from reranked docs.
# Falls back to Groq automatically if OpenAI fails.
# ─────────────────────────────────────────────

def generate_node(state: AgentState) -> dict:
    docs    = state.get("reranked_docs", [])
    context = "\n\n".join(
        f"{i+1}. {d.get('title','?')} ({d.get('year','N/A')}) "
        f"[{', '.join(d.get('genres', []))}]\n   {d.get('plot','')}"
        for i, d in enumerate(docs)
    )
    sources = [d.get("title", "") for d in docs if d.get("title")]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert film critic and movie recommendation assistant with deep knowledge of cinema. "
                "Answer the user's question using ONLY the movies provided in the context.\n\n"
                "Structure your response as follows:\n"
                "1. Start with a short intro sentence about what you found.\n"
                "2. For EACH recommended movie write a detailed paragraph that covers:\n"
                "   - The movie title, year, and genre in bold\n"
                "   - A rich description of the plot and what makes it relevant to the query\n"
                "   - What specifically makes it worth watching (tone, themes, acting, direction)\n"
                "   - Who would enjoy it most\n"
                "3. End with a short summary ranking your top pick and why.\n\n"
                "Be enthusiastic, detailed, and specific. Write at least 3-4 sentences per movie. "
                "If the context does not contain a good match, say so honestly."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {state['query']}\n\n"
                f"Context movies:\n\n{context}\n\n"
                "Give a detailed, elaborate recommendation for each relevant movie."
            )
        }
    ]

    answer = chat(messages, temperature=0.3)   # OpenAI first → Groq fallback

    return {"answer": answer, "sources": sources}


# ─────────────────────────────────────────────
# BUILD THE LANGGRAPH GRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan",     plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check",    check_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan",     "retrieve")
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges("check", should_retry, {
        "retrieve": "retrieve",
        "generate": "generate"
    })
    graph.add_edge("generate", END)

    return graph.compile()


# Singleton — compiled once on import
rag_agent = build_graph()


def run_agent(query: str) -> dict:
    """
    Public entry point. Pass a user query, get back a dict:
      answer  - the final LLM response string
      sources - list of movie titles used as context
      loops   - how many retrieval iterations were needed
      score   - final context quality score (0-10)
    """
    initial: AgentState = {
        "query": query, "sub_questions": [], "tool_calls": [],
        "raw_docs": [], "reranked_docs": [], "context_score": 0,
        "loop_count": 0, "answer": "", "sources": []
    }
    final = rag_agent.invoke(initial)
    return {
        "answer":  final["answer"],
        "sources": final["sources"],
        "loops":   final["loop_count"],
        "score":   final["context_score"]
    }