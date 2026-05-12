# Agentic RAG — Movie Assistant (Fully Local)

MongoDB (local) + ChromaDB (local vector store) + LangGraph + OpenAI

## How it works

- **MongoDB** (local) stores the original movie documents
- **ChromaDB** (local, on disk) stores the vector embeddings and handles similarity search
- **LangGraph** powers the agentic plan → retrieve → check → generate loop
- **OpenAI** provides embeddings and chat completions

No Atlas subscription needed. Everything runs on your machine.

## Folder structure

```
agentic_rag_movies/
├── config/
│   └── settings.py          ← all config & API keys
├── embedder/
│   ├── embed_movies.py      ← one-time: embed movies into ChromaDB
│   └── create_index.py      ← informational only (Chroma auto-indexes)
├── retriever/
│   ├── vector_search.py     ← ChromaDB similarity search
│   ├── reranker.py          ← Cohere reranker (optional)
│   └── web_search.py        ← Tavily web fallback (optional)
├── agent/
│   ├── tools.py             ← LangChain Tool wrappers
│   └── graph.py             ← LangGraph state machine
├── ui/
│   └── app.py               ← Streamlit chat UI
├── main.py                  ← CLI entry point
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Make sure local MongoDB is running
mongod

# 2. Load the sample_mflix dataset into local MongoDB
# Download from: https://atlas-education.s3.amazonaws.com/sampledata.archive
# Then restore: mongorestore --archive=sampledata.archive --gzip

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in your keys
cp .env.example .env
# Only OPENAI_API_KEY is required

# 5. Embed movies into ChromaDB (run once — ~2 min for 500 movies)
python -m embedder.embed_movies

# 6a. Run via CLI
python main.py "best sci-fi movie about AI"

# 6b. Or run the Streamlit UI
streamlit run ui/app.py
```

## Requirements

- Local MongoDB running on port 27017
- sample_mflix dataset loaded into MongoDB
- OpenAI API key (for embeddings + chat)
- Cohere API key — optional, for reranking (free tier at cohere.com)
- Tavily API key — optional, for web search fallback (free tier at tavily.com)
