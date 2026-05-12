"""
config/openai_client.py
───────────────────────
Central place for all AI API calls.

Embeddings  → sentence-transformers (local, free, no API needed)
Chat        → OpenAI first, Groq fallback
"""

from sentence_transformers import SentenceTransformer
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
from config.settings import (
    OPENAI_API_KEY, GROQ_API_KEY,
    OPENAI_CHAT_MODEL, GROQ_CHAT_MODEL
)

# ── Local embedding model ──────────────────────────────────────────
# Downloads once (~90MB), then cached locally forever
# No API key, no rate limits, no cost
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── OpenAI client (chat only) ──────────────────────────────────────
_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ── Groq client (chat fallback) ────────────────────────────────────
_groq = (
    OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    if GROQ_API_KEY else None
)

if not _openai and not _groq:
    raise ValueError(
        "No API key found. Set OPENAI_API_KEY or GROQ_API_KEY in your .env file."
    )


# ─────────────────────────────────────────────
# EMBED — fully local, no API call
# ─────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """
    Convert text to a 384-dim embedding vector using a local model.
    Runs on CPU, no internet needed after first download.
    Zero rate limits, zero cost.
    """
    text = text.replace("\n", " ").strip()
    vector = _embedder.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ─────────────────────────────────────────────
# CHAT — OpenAI first, Groq fallback
# ─────────────────────────────────────────────

def chat(
    messages: list[dict],
    temperature: float = 0.2
) -> str:
    """
    Run a chat completion. Tries OpenAI first, falls back to Groq.
    Returns the assistant reply as a plain string.
    """

    # ── Try OpenAI first ──────────────────────────────────────────
    if _openai:
        try:
            resp = _openai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=messages,
                temperature=temperature
            )
            return resp.choices[0].message.content

        except AuthenticationError:
            print("[OpenAI] Invalid API key — switching to Groq fallback ...")
        except RateLimitError:
            print("[OpenAI] Rate limit hit — switching to Groq fallback ...")
        except APIError as e:
            print(f"[OpenAI] API error ({e}) — switching to Groq fallback ...")

    # ── Try Groq fallback ─────────────────────────────────────────
    if not _groq:
        raise RuntimeError(
            "OpenAI failed and no GROQ_API_KEY is set. "
            "Add GROQ_API_KEY to your .env file."
        )

    try:
        resp = _groq.chat.completions.create(
            model=GROQ_CHAT_MODEL,
            messages=messages,
            temperature=temperature
        )
        print("[Groq] Response received from fallback.")
        return resp.choices[0].message.content

    except AuthenticationError:
        raise RuntimeError(
            "Groq key is also invalid. Check GROQ_API_KEY in your .env file."
        )
    except RateLimitError:
        raise RuntimeError(
            "Both OpenAI and Groq are rate limited. Please wait and retry."
        )
    except APIError as e:
        raise RuntimeError(f"Both OpenAI and Groq failed. Last error: {e}")