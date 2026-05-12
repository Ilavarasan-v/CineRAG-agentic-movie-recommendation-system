"""
embedder/fetch_tmdb.py
──────────────────────
Fetches modern movies from TMDB API and adds them into:
  1. Your local MongoDB (sample_mflix.movies collection)
  2. ChromaDB (for vector search)

Fetches movies from these categories:
  - Popular movies
  - Top rated movies
  - Movies by genre (Action, Sci-Fi, Horror, Romance, Animation etc.)
  - Movies from 2010 to 2026

Run after embed_movies.py:
    python -m embedder.fetch_tmdb

TMDB free API limits: 50 requests/second — we stay well under that.
"""

import time
import requests
import chromadb
from pymongo import MongoClient
from config.openai_client import embed
from config.settings import (
    MONGO_URI, DB_NAME, COLLECTION_NAME,
    CHROMA_PATH, CHROMA_COLLECTION
)
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE    = "https://api.themoviedb.org/3"
HEADERS      = {}   # v3 key passed as query param

mongo         = MongoClient(MONGO_URI)
collection    = mongo[DB_NAME][COLLECTION_NAME]

chroma_client     = chromadb.PersistentClient(path=CHROMA_PATH)
chroma_collection = chroma_client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

# TMDB genre ID map
GENRE_MAP = {
    28:    "Action",
    12:    "Adventure",
    16:    "Animation",
    35:    "Comedy",
    80:    "Crime",
    99:    "Documentary",
    18:    "Drama",
    10751: "Family",
    14:    "Fantasy",
    36:    "History",
    27:    "Horror",
    10402: "Music",
    9648:  "Mystery",
    10749: "Romance",
    878:   "Sci-Fi",
    53:    "Thriller",
    10752: "War",
    37:    "Western"
}


def fetch_page(endpoint: str, page: int = 1, extra_params: dict = {}) -> list[dict]:
    """Fetch one page of movies from TMDB."""
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page, **extra_params}
    try:
        resp = requests.get(
            f"{TMDB_BASE}/{endpoint}",
            headers=HEADERS,
            params=params,
            timeout=10
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"  TMDB fetch error: {e}")
        return []


def get_movie_details(tmdb_id: int) -> dict | None:
    """Fetch full details for a single movie including overview."""
    try:
        resp = requests.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            headers=HEADERS,
            params={"language": "en-US"},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def tmdb_to_mongo(movie: dict) -> dict | None:
    """Convert a TMDB movie dict to our MongoDB schema."""
    overview = movie.get("overview", "")
    title    = movie.get("title", "")

    if not overview or len(overview) < 30:
        return None

    year = None
    release = movie.get("release_date", "")
    if release and len(release) >= 4:
        try:
            year = int(release[:4])
        except ValueError:
            pass

    genre_ids = movie.get("genre_ids", [])
    genres    = [GENRE_MAP[g] for g in genre_ids if g in GENRE_MAP]

    return {
        "title":  title,
        "plot":   overview,
        "year":   year,
        "genres": genres,
        "source": "tmdb",
        "tmdb_id": movie.get("id")
    }


def already_exists(title: str, year: int | None) -> bool:
    """Check if movie already exists in MongoDB."""
    query = {"title": title}
    if year:
        query["year"] = year
    return collection.count_documents(query, limit=1) > 0


def save_and_embed(movies: list[dict]) -> int:
    """Save movies to MongoDB and embed into ChromaDB. Returns count added."""
    added = 0
    for movie in movies:
        if not movie:
            continue

        title = movie.get("title", "")
        year  = movie.get("year")

        # Skip if already in MongoDB
        if already_exists(title, year):
            continue

        # Save to MongoDB
        result = collection.insert_one(movie)
        doc_id = str(result.inserted_id)

        # Embed and store in ChromaDB
        text   = f"{title}. {movie.get('plot', '')}"
        vector = embed(text)

        chroma_collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[movie.get("plot", "")],
            metadatas=[{
                "title":  title,
                "year":   year or 0,
                "genres": ", ".join(movie.get("genres", []))
            }]
        )
        added += 1

    return added


def fetch_popular(pages: int = 10) -> int:
    """Fetch popular movies — pages * 20 movies."""
    print(f"\nFetching popular movies ({pages} pages) ...")
    total = 0
    for page in range(1, pages + 1):
        results = fetch_page("movie/popular", page=page)
        movies  = [tmdb_to_mongo(m) for m in results]
        count   = save_and_embed([m for m in movies if m])
        total  += count
        print(f"  Page {page}/{pages} — added {count} new movies")
        time.sleep(0.3)
    return total


def fetch_top_rated(pages: int = 10) -> int:
    """Fetch top rated movies."""
    print(f"\nFetching top rated movies ({pages} pages) ...")
    total = 0
    for page in range(1, pages + 1):
        results = fetch_page("movie/top_rated", page=page)
        movies  = [tmdb_to_mongo(m) for m in results]
        count   = save_and_embed([m for m in movies if m])
        total  += count
        print(f"  Page {page}/{pages} — added {count} new movies")
        time.sleep(0.3)
    return total


def fetch_by_year_range(start: int, end: int, pages_per_year: int = 3) -> int:
    """Fetch movies year by year for a given range."""
    print(f"\nFetching movies from {start}–{end} ...")
    total = 0
    for year in range(end, start - 1, -1):   # newest first
        for page in range(1, pages_per_year + 1):
            results = fetch_page("discover/movie", page=page, extra_params={
                "primary_release_year": year,
                "sort_by": "popularity.desc",
                "vote_count.gte": 50        # skip obscure movies
            })
            movies = [tmdb_to_mongo(m) for m in results]
            count  = save_and_embed([m for m in movies if m])
            total += count
            time.sleep(0.25)
        print(f"  {year} — running total: {total} new movies added")
    return total


def fetch_by_genres(pages: int = 5) -> int:
    """Fetch movies for key genres."""
    priority_genres = [878, 28, 27, 14, 80, 10749, 16, 35]  # Sci-Fi, Action, Horror etc.
    print(f"\nFetching movies by genre ...")
    total = 0
    for genre_id in priority_genres:
        genre_name = GENRE_MAP.get(genre_id, str(genre_id))
        for page in range(1, pages + 1):
            results = fetch_page("discover/movie", page=page, extra_params={
                "with_genres": genre_id,
                "sort_by": "popularity.desc",
                "vote_count.gte": 100
            })
            movies = [tmdb_to_mongo(m) for m in results]
            count  = save_and_embed([m for m in movies if m])
            total += count
            time.sleep(0.25)
        print(f"  {genre_name} — added {total} total so far")
    return total


def main():
    if not TMDB_API_KEY:
        print("ERROR: TMDB_API_KEY not found in .env file.")
        return

    print("=" * 60)
    print("CineRAG — TMDB Movie Fetcher")
    print("=" * 60)
    print(f"MongoDB: {DB_NAME}.{COLLECTION_NAME}")
    print(f"ChromaDB: {CHROMA_PATH}")
    print(f"Movies before fetch: {collection.count_documents({})}")
    print(f"ChromaDB before fetch: {chroma_collection.count()}")
    print("=" * 60)

    total = 0

    # 1. Popular movies (200 movies)
    total += fetch_popular(pages=10)

    # 2. Top rated movies (200 movies)
    total += fetch_top_rated(pages=10)

    # 3. Movies by genre (key genres, 5 pages each = ~100 per genre)
    total += fetch_by_genres(pages=5)

    # 4. Recent movies year by year (2015–2026)
    total += fetch_by_year_range(start=2015, end=2026, pages_per_year=3)

    print("\n" + "=" * 60)
    print(f"DONE! Added {total} new movies total.")
    print(f"MongoDB now has: {collection.count_documents({})} movies")
    print(f"ChromaDB now has: {chroma_collection.count()} vectors")
    print("=" * 60)
    print("\nRestart Streamlit to use the updated data:")
    print("  streamlit run ui/app.py")


if __name__ == "__main__":
    main()