"""
embedder/embed_movies.py
────────────────────────
One-time script: reads movies from local MongoDB, generates
embeddings, and stores them into ChromaDB.

Run ONCE (or re-run after deleting chroma_db):
    python -m embedder.embed_movies
"""

import time
import chromadb
from pymongo import MongoClient
from config.openai_client import embed
from config.settings import (
    MONGO_URI, DB_NAME, COLLECTION_NAME,
    CHROMA_PATH, CHROMA_COLLECTION
)

mongo         = MongoClient(MONGO_URI)
collection    = mongo[DB_NAME][COLLECTION_NAME]

chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)


def embed_and_index_movies(limit: int = 3000, batch_size: int = 50):
    """
    Fetch movies from MongoDB, embed plots, store in ChromaDB.
    Prioritises newer movies first (sorted by year descending).
    Skips movies with very short plots (less than 30 chars).
    """
    already_indexed = chroma_collection.count()
    if already_indexed >= limit:
        print(f"ChromaDB already has {already_indexed} movies — skipping.")
        print("To re-embed, delete the chroma_db folder first.")
        return

    print(f"Fetching movies from MongoDB (sorted newest first) ...")

    movies = list(
        collection.find(
            {
                "plot": {"$exists": True, "$ne": "", "$type": "string"},
                "year": {"$exists": True, "$type": "int"}
            },
            {"_id": 1, "title": 1, "plot": 1, "genres": 1, "year": 1}
        ).sort("year", -1).limit(limit)   # newest first
    )

    # Filter out movies with very short/useless plots
    movies = [m for m in movies if len(m.get("plot", "")) >= 30]

    if not movies:
        print("No movies found in MongoDB. Make sure sample_mflix is loaded.")
        return

    # Show year range so user knows what's in the dataset
    years = [m.get("year", 0) for m in movies if m.get("year")]
    print(f"Found {len(movies)} movies spanning {min(years)}–{max(years)}")
    print(f"Embedding into ChromaDB ...")

    ids, embeddings, documents, metadatas = [], [], [], []

    for i, movie in enumerate(movies):
        text = f"{movie.get('title', '')}. {movie.get('plot', '')}"
        vec  = embed(text)

        ids.append(str(movie["_id"]))
        embeddings.append(vec)
        documents.append(movie.get("plot", ""))
        metadatas.append({
            "title":  movie.get("title", "Unknown"),
            "year":   int(movie["year"]) if movie.get("year") else 0,
            "genres": ", ".join(movie.get("genres", []))
        })

        if len(ids) == batch_size:
            chroma_collection.upsert(
                ids=ids, embeddings=embeddings,
                documents=documents, metadatas=metadatas
            )
            print(f"  Indexed {i + 1}/{len(movies)} ... "
                  f"(latest: {movie.get('title','?')} {movie.get('year','')})")
            ids, embeddings, documents, metadatas = [], [], [], []
            time.sleep(0.1)

    if ids:
        chroma_collection.upsert(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas
        )

    total = chroma_collection.count()
    years_final = [m.get("year", 0) for m in movies if m.get("year")]
    print(f"\nDone! {total} movies in ChromaDB.")
    print(f"Year range: {min(years_final)} – {max(years_final)}")


if __name__ == "__main__":
    embed_and_index_movies(limit=3000)