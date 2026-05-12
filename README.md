# CineRAG — Agentic Movie Recommendation System

CineRAG is a movie recommendation system built on top of Agentic RAG (Retrieval-Augmented Generation). You type a natural language query like "90s sci-fi movies about AI" or "something like Interstellar" and the system finds relevant movies from a database and gives you detailed recommendations with explanations.

The project uses MongoDB to store movie documents, ChromaDB as a local vector database for semantic search, and a LangGraph state machine to orchestrate the agent loop. For language models it uses OpenAI GPT-4o-mini as the primary and Groq LLaMA 3.3 as an automatic fallback if OpenAI fails or hits a rate limit. Embeddings are generated locally using sentence-transformers so there are no embedding API costs at all.

Movie data comes from two sources — the MongoDB sample_mflix dataset which covers classic films, and the TMDB API which we use to fetch modern movies up to 2026.

---

## How it differs from a normal RAG system

A normal RAG system works in a straight line — you embed the query, retrieve the top-K documents, stuff them into the LLM prompt, and return the answer. It does this once and stops regardless of whether the retrieved documents are actually relevant or not.

CineRAG adds an agentic loop on top of this. Instead of one retrieval step, the system has four stages:

**Plan** — the LLM first reads the query and breaks it into sub-questions, then decides which retrieval tool to use for each one. For example a query like "90s action heist movie" gets decomposed into separate searches for the action filter and the heist theme.

**Retrieve** — the agent calls the chosen tools, collects results from ChromaDB, deduplicates them, and passes them through a reranker that re-scores each document against the original query for better precision.

**Check** — this is the key difference from normal RAG. The LLM reads the retrieved documents and scores how well they answer the original query on a scale of 0 to 10. If the score is below 6 the agent goes back to the retrieve step with a refined query and tries again, up to 3 times.

**Generate** — only once the context quality is good enough does the LLM generate the final answer, citing specific movies with detailed explanations.

This self-evaluation and retry loop means the system catches bad retrievals instead of silently returning irrelevant results, which is the main failure mode of standard RAG.

---

## Tech Stack

- **MongoDB** — local document store for movie data
- **ChromaDB** — local vector database (no cloud needed)
- **Sentence Transformers** — local embeddings using all-MiniLM-L6-v2
- **LangGraph** — agent state machine
- **OpenAI GPT-4o-mini** — primary LLM
- **Groq LLaMA 3.3 70B** — automatic fallback LLM
- **TMDB API** — source for modern movies
- **Streamlit** — chat interface

---

## Project Structure

```
CineRAG/
├── config/
│   ├── settings.py          # constants and API keys
│   └── openai_client.py     # handles OpenAI + Groq fallback
├── embedder/
│   ├── embed_movies.py      # embeds movies into ChromaDB (run once)
│   └── fetch_tmdb.py        # fetches modern movies from TMDB API
├── retriever/
│   ├── vector_search.py     # ChromaDB semantic search
│   ├── reranker.py          # optional Cohere reranker
│   └── web_search.py        # optional Tavily web search fallback
├── agent/
│   ├── tools.py             # LangChain tool wrappers
│   └── graph.py             # LangGraph agent loop
├── ui/
│   └── app.py               # Streamlit chat UI
└── main.py                  # CLI entry point
```

---

## Setup

**Requirements** — Python 3.11, MongoDB running locally, sample_mflix dataset loaded.

```bash
git clone https://github.com/Ilavarasan-v/CineRAG-agentic-movie-recommendation-system.git
cd CineRAG-agentic-movie-recommendation-system

py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY and GROQ_API_KEY in .env

python -m embedder.embed_movies       # run once to build ChromaDB
python -m embedder.fetch_tmdb         # optional, fetches modern movies

streamlit run ui/app.py
```

---

## Environment Variables

```
MONGO_URI=mongodb://localhost:27017/
OPENAI_API_KEY=        # required
GROQ_API_KEY=          # required as fallback
TMDB_API_KEY=          # optional, for fetching modern movies
COHERE_API_KEY=        # optional, improves reranking
TAVILY_API_KEY=        # optional, web search fallback
```

---

## Architecture

```
                        User Query
                            │
                            ▼
                    ┌───────────────┐
                    │  PLAN NODE    │
                    │               │
                    │  LLM reads    │
                    │  the query,   │
                    │  breaks into  │
                    │  sub-questions│
                    │  picks tools  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ RETRIEVE NODE │
                    │               │
                    │  Calls tools: │◄──────────────┐
                    │               │               │
                    │ ┌───────────┐ │               │
                    │ │  ChromaDB │ │               │
                    │ │  vector   │ │               │
                    │ │  search   │ │               │
                    │ └───────────┘ │               │
                    │ ┌───────────┐ │               │
                    │ │  Genre /  │ │               │
                    │ │  year     │ │               │
                    │ │  filter   │ │               │
                    │ └───────────┘ │               │
                    │ ┌───────────┐ │               │
                    │ │   Web     │ │               │
                    │ │  search   │ │               │
                    │ │ fallback  │ │               │
                    │ └───────────┘ │               │
                    │               │               │
                    │  Deduplicates │               │
                    │  + Reranks    │               │
                    └───────┬───────┘               │
                            │                       │
                            ▼                       │
                    ┌───────────────┐               │
                    │  CHECK NODE   │               │
                    │               │               │
                    │  LLM scores   │  score < 6    │
                    │  context      ├───────────────┘
                    │  quality 0-10 │  (retry up to
                    │               │   3 times)
                    └───────┬───────┘
                            │ score ≥ 6
                            ▼
                    ┌───────────────┐
                    │ GENERATE NODE │
                    │               │
                    │  LLM writes   │
                    │  detailed     │
                    │  answer with  │
                    │  citations    │
                    └───────┬───────┘
                            │
                            ▼
                     Final Answer
                  + Sources + Score


Data Layer
──────────────────────────────────────────
  MongoDB (local)          ChromaDB (local)
  ├── sample_mflix         ├── 384-dim vectors
  │   (classic films)      │   (all-MiniLM-L6-v2)
  └── TMDB movies          └── cosine similarity
      (2015–2026)               search


LLM Layer
──────────────────────────────────────────
  Primary  →  OpenAI GPT-4o-mini
                    │ fails / rate limit
                    ▼
  Fallback →  Groq LLaMA 3.3 70B


Embedding Layer
──────────────────────────────────────────
  sentence-transformers (runs locally)
  model: all-MiniLM-L6-v2
  dims:  384
  cost:  free — no API calls
```
