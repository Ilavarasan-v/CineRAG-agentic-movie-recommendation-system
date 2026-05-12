"""
embedder/create_index.py
────────────────────────
For the fully local setup, ChromaDB builds its HNSW index
automatically when you upsert data in embed_movies.py.

You do NOT need to run this file separately.
It is kept here just to print a friendly reminder.

Run order:
  1. python -m embedder.embed_movies   ← does both embedding AND indexing
  2. python main.py                    ← or: streamlit run ui/app.py
"""

if __name__ == "__main__":
    print("No separate index creation needed!")
    print("ChromaDB builds its index automatically during embed_movies.py.")
    print()
    print("If you haven't already, run:")
    print("  python -m embedder.embed_movies")
