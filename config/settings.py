"""
config/settings.py
──────────────────
Single source of truth for all config values.

Embeddings  → local sentence-transformers (all-MiniLM-L6-v2, 384-dim)
Chat        → OpenAI (primary) + Groq (fallback)
Vector DB   → ChromaDB (local)
Document DB → MongoDB (local)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB (local) ────────────────────────────────────────────────
MONGO_URI         = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME           = "sample_mflix"
COLLECTION_NAME   = "movies"

# ── ChromaDB (local vector store) ─────────────────────────────────
CHROMA_PATH       = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = "movie_plots"

# ── Embedding model (local — no API needed) ────────────────────────
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # sentence-transformers model
EMBEDDING_DIMS    = 384                   # output dimensions of MiniLM

# ── Chat models ────────────────────────────────────────────────────
OPENAI_CHAT_MODEL = "gpt-4o-mini"         # OpenAI primary
GROQ_CHAT_MODEL   = "llama-3.3-70b-versatile"     # Groq fallback

# ── Retrieval ──────────────────────────────────────────────────────
TOP_K             = 8    # candidates from ChromaDB
RERANK_TOP_N      = 4    # docs kept after reranking
MAX_AGENT_LOOPS   = 3    # max retrieval retries before giving up

# ── API Keys ───────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")   # for chat only
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")     # chat fallback
COHERE_API_KEY    = os.getenv("COHERE_API_KEY")   # optional reranker
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")   # optional web search
TMDB_API_KEY      = os.getenv("TMDB_API_KEY")     # TMDB movie fetcher