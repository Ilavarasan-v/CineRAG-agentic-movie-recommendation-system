"""
retriever/vector_search.py
──────────────────────────
Vector search using ChromaDB (fully local).

Note on genre filtering:
  ChromaDB does not support $contains on string fields.
  Genres are stored as a comma-separated string e.g. "Action, Thriller".
  So genre filtering is done AFTER retrieval in Python — we fetch
  more candidates (top_k * 3) and then filter down manually.
"""

import chromadb
from config.openai_client import embed
from config.settings import CHROMA_PATH, CHROMA_COLLECTION, TOP_K

chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)


def vector_search(
    query: str,
    top_k: int = TOP_K,
    genre: str | None = None,
    year_gte: int | None = None,
) -> list[dict]:
    """
    Search ChromaDB for movies semantically similar to the query.

    Args:
        query:    natural language search string
        top_k:    number of results to return
        genre:    optional genre string to filter (e.g. "Sci-Fi")
        year_gte: optional minimum release year (e.g. 1990)

    Returns:
        List of dicts — title, plot, genres, year, score (0–1)
    """
    query_vector = embed(query)

    # Build ChromaDB where filter — only year is supported natively
    # Genre filtering is done manually below in Python
    where = None
    if year_gte:
        where = {"year": {"$gte": year_gte}}

    # Fetch extra candidates when genre filtering is needed
    # so we have enough results after manual filtering
    fetch_k = top_k * 3 if genre else top_k

    results = chroma_collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    if not results or not results["ids"][0]:
        return docs

    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        score    = round(1 - (distance / 2), 4)

        genres_str = metadata.get("genres", "")
        genres_list = [g.strip() for g in genres_str.split(",") if g.strip()]

        docs.append({
            "title":  metadata.get("title", "Unknown"),
            "plot":   results["documents"][0][i],
            "genres": genres_list,
            "year":   metadata.get("year"),
            "score":  score
        })

    # ── Manual genre filter (post-retrieval) ──────────────────────
    # Check if the genre string appears anywhere in the genres list
    if genre:
        genre_lower = genre.lower()
        docs = [
            d for d in docs
            if any(genre_lower in g.lower() for g in d["genres"])
        ]

    return docs[:top_k]