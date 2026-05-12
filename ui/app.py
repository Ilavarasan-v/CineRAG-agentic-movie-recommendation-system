"""
ui/app.py
─────────
Streamlit web UI for the Agentic RAG Movie Assistant.

Run with:
    streamlit run ui/app.py

Features:
  - Chat-style interface
  - Shows which tools were called
  - Shows how many retrieval loops the agent needed
  - Displays source movies with similarity scores
  - Context quality score badge
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import run_agent

st.set_page_config(
    page_title="Movie RAG Agent",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Agentic Movie RAG")
st.caption("Powered by MongoDB Atlas Vector Search + LangGraph")

# ── Session state ──────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar: example queries ───────────────────────────────────────
with st.sidebar:
    st.header("Example queries")
    examples = [
        "Best sci-fi movies about artificial intelligence",
        "A romantic film set in Paris with a happy ending",
        "90s action movies with a clever heist plot",
        "Movies similar to Interstellar about space and time",
        "Thriller with an unexpected twist ending",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_query = ex

# ── Chat history display ───────────────────────────────────────────
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander("Details"):
            col1, col2 = st.columns(2)
            col1.metric("Retrieval loops", turn["loops"])
            col2.metric("Context score", f"{turn['score']}/10")
            if turn["sources"]:
                st.write("**Sources used:**")
                for src in turn["sources"]:
                    st.write(f"  • {src}")

# ── Query input ────────────────────────────────────────────────────
pending = st.session_state.pop("pending_query", None)
query   = st.chat_input("Ask about a movie...") or pending

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking ..."):
            result = run_agent(query)

        st.write(result["answer"])

        with st.expander("Details"):
            col1, col2 = st.columns(2)
            col1.metric("Retrieval loops", result["loops"])
            col2.metric("Context score",   f"{result['score']}/10")
            if result["sources"]:
                st.write("**Sources used:**")
                for src in result["sources"]:
                    st.write(f"  • {src}")

    st.session_state.history.append({
        "query":  query,
        "answer": result["answer"],
        "loops":  result["loops"],
        "score":  result["score"],
        "sources": result["sources"]
    })
