"""
retriever/web_search.py
───────────────────────
Web search fallback tool used by the agent when the local
MongoDB collection doesn't have enough relevant results.

Uses Tavily (tavily.com — free tier: 1000 searches/month).
If TAVILY_API_KEY is not set, the tool returns an empty list
and the agent falls back to what it already retrieved.

The agent decides when to call this — it's not called on every query.
"""

from config.settings import TAVILY_API_KEY

try:
    from tavily import TavilyClient
    _tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
except ImportError:
    _tavily = None


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Search the web for movie-related information.

    Args:
        query:       search query string
        max_results: max number of web results to return

    Returns:
        List of dicts with keys: title, plot (content), score (relevance)
        Shape matches vector_search() output so the agent handles both uniformly.
    """
    if not _tavily:
        return []

    response = _tavily.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=False
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title":  r.get("title", "Web result"),
            "plot":   r.get("content", ""),
            "genres": [],
            "year":   None,
            "score":  r.get("score", 0.0),
            "source": "web"
        })

    return results
