"""
retriever/reranker.py
─────────────────────
After vector search returns top-K candidates, the reranker
re-scores each document against the original query using a
cross-encoder model (much more accurate than cosine similarity).

Why rerank?
  Vector search uses approximate nearest neighbours — fast but noisy.
  The reranker reads the full (query, document) pair together, so it
  understands context that embedding similarity misses.

Fallback:
  If COHERE_API_KEY is not set, reranking is skipped and the
  original vector search order is returned unchanged.
"""

from config.settings import COHERE_API_KEY, RERANK_TOP_N

try:
    import cohere
    _cohere_client = cohere.Client(api_key=COHERE_API_KEY) if COHERE_API_KEY else None
except ImportError:
    _cohere_client = None


def rerank(query: str, docs: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Rerank a list of retrieved documents by relevance to query.

    Args:
        query:  the original user query
        docs:   list of dicts from vector_search() (must have "plot" key)
        top_n:  how many top docs to keep after reranking

    Returns:
        Filtered + sorted list of docs, best first.
        Each doc gets a "rerank_score" field added.
    """
    if not _cohere_client or not docs:
        # Fallback: return top_n from vector search order unchanged
        return docs[:top_n]

    passages = [
        f"{d.get('title', '')}. {d.get('plot', '')}"
        for d in docs
    ]

    response = _cohere_client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=passages,
        top_n=top_n
    )

    reranked = []
    for result in response.results:
        doc = docs[result.index].copy()
        doc["rerank_score"] = round(result.relevance_score, 4)
        reranked.append(doc)

    return reranked
