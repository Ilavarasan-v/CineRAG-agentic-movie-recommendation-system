"""
agent/tools.py
──────────────
Wraps each retriever function as a LangChain Tool so the
LangGraph agent can call them by name based on the query.

Three tools are registered:
  1. vector_search_tool  — semantic search against MongoDB
  2. genre_filter_tool   — vector search + genre/year pre-filter
  3. web_search_tool     — live web search fallback via Tavily

The agent planner reads the tool descriptions to decide
which tool(s) to invoke for a given sub-question.
"""

import json
from langchain.tools import Tool
from retriever.vector_search import vector_search
from retriever.web_search import web_search


def _format_docs(docs: list[dict]) -> str:
    """Serialize retrieved docs to a JSON string the LLM can read."""
    return json.dumps(docs, indent=2, default=str)


# ── Tool 1: Plain vector search ────────────────────────────────────
def _vector_search_fn(query: str) -> str:
    docs = vector_search(query)
    return _format_docs(docs)

vector_search_tool = Tool(
    name="vector_search",
    func=_vector_search_fn,
    description=(
        "Search the MongoDB movies collection by semantic meaning. "
        "Use this for general movie queries like 'space exploration film' "
        "or 'heist with a twist'. Input: a natural language search string."
    )
)


# ── Tool 2: Genre + year filtered vector search ────────────────────
def _genre_filter_fn(query_json: str) -> str:
    """
    Accepts JSON input: {"query": "...", "genre": "Sci-Fi", "year_gte": 1990}
    All fields except "query" are optional.
    """
    try:
        params = json.loads(query_json)
    except json.JSONDecodeError:
        params = {"query": query_json}

    docs = vector_search(
        query    = params.get("query", ""),
        genre    = params.get("genre"),
        year_gte = params.get("year_gte")
    )
    return _format_docs(docs)

genre_filter_tool = Tool(
    name="genre_filter_search",
    func=_genre_filter_fn,
    description=(
        "Search movies filtered by genre and/or release year. "
        "Use when the user specifies a genre (e.g. 'Sci-Fi', 'Romance', 'Thriller') "
        "or a time period (e.g. '90s films', 'before 2000'). "
        'Input: JSON string with keys "query" (required), "genre" (optional), '
        '"year_gte" (optional int, e.g. 1990).'
    )
)


# ── Tool 3: Web search fallback ─────────────────────────────────────
def _web_search_fn(query: str) -> str:
    docs = web_search(query)
    if not docs:
        return "[]  # Web search unavailable or no results found."
    return _format_docs(docs)

web_search_tool = Tool(
    name="web_search",
    func=_web_search_fn,
    description=(
        "Search the web for current or niche movie information not in the database. "
        "Use as a fallback when vector_search returns low-scoring or irrelevant results. "
        "Input: a natural language search query."
    )
)


# ── Exported list of all tools ──────────────────────────────────────
ALL_TOOLS = [vector_search_tool, genre_filter_tool, web_search_tool]
