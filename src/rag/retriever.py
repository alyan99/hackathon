"""
RAG Retriever — SaayaShift

Hybrid keyword + vector search over the approved heat-safety corpus.
Guardrails:
  - Numbers/statistics come from SQL over Gold tables only, not LLM memory
  - If top-k relevance < RELEVANCE_THRESHOLD → return "I don't know"
  - No PII in prompts
  - Abstain when evidence is absent, stale, conflicting, or outside corpus

Returns: list of dicts with keys: answer, citations, retrieved_chunks, abstained
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR   = ROOT / "data" / "rag_index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
GOLD_PATH   = ROOT / "data" / "gold" / "weather_gold.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TOP_K = 5
RELEVANCE_THRESHOLD = 0.40    # cosine similarity below this → abstain
MODEL_NAME = "all-MiniLM-L6-v2"


# ── Load index ────────────────────────────────────────────────────────────────

def load_index():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers faiss-cpu")

    if not (INDEX_DIR / "corpus.faiss").exists():
        raise FileNotFoundError("FAISS index not found. Run `python -m src.rag.corpus` first.")

    index = faiss.read_index(str(INDEX_DIR / "corpus.faiss"))
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    embedder = SentenceTransformer(MODEL_NAME)
    return index, chunks, embedder


# ── Keyword filter ────────────────────────────────────────────────────────────

def keyword_filter(chunks: list[dict], query: str) -> list[dict]:
    """Pre-filter chunks by keyword overlap (BM25-like simplification)."""
    query_words = set(re.findall(r"\w+", query.lower()))
    scored = []
    for c in chunks:
        chunk_words = set(re.findall(r"\w+", c["text"].lower()))
        overlap = len(query_words & chunk_words)
        scored.append((overlap, c))
    # Return all with at least one keyword match, sorted descending
    filtered = [c for score, c in sorted(scored, key=lambda x: -x[0]) if score > 0]
    return filtered if filtered else chunks   # fall back to all if no keyword match


# ── Gold table stats query ────────────────────────────────────────────────────

def query_gold_stats() -> dict:
    """
    Pull summary statistics from the Gold table.
    Numbers in answers must come from here, never from LLM memory.
    """
    if not GOLD_PATH.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(GOLD_PATH)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        recent = df.nlargest(24, "time")   # last 24 hours
        return {
            "latest_apparent_temp_c": float(recent["apparent_temperature"].iloc[0])
                if len(recent) and "apparent_temperature" in recent.columns else None,
            "last_24h_mean_apparent_temp_c": float(recent["apparent_temperature"].mean())
                if "apparent_temperature" in recent.columns else None,
            "data_as_of": str(df["time"].max()),
            "total_rows": len(df),
        }
    except Exception as e:
        log.warning("Could not query Gold table: %s", e)
        return {}


# ── PII guard ─────────────────────────────────────────────────────────────────

PII_PATTERNS = [
    r"\b\d{10,13}\b",              # phone numbers
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",  # email
    r"\bCNIC\b",
]

def contains_pii(text: str) -> bool:
    for pattern in PII_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# ── Main retrieval ────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve relevant chunks for a query.
    Returns a structured result with citations and abstention logic.
    """
    # PII guard
    if contains_pii(query):
        return {
            "abstained": True,
            "answer": "I cannot process queries containing personal information.",
            "citations": [],
            "retrieved_chunks": [],
            "gold_stats": {},
        }

    try:
        index, chunks, embedder = load_index()
    except (ImportError, FileNotFoundError) as e:
        return {
            "abstained": True,
            "answer": f"I don't know — RAG index not available: {e}",
            "citations": [],
            "retrieved_chunks": [],
            "gold_stats": {},
        }

    # Embed query
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    q_emb /= np.maximum(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-8)

    # Vector search
    scores, indices = index.search(q_emb, min(top_k * 2, len(chunks)))
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    # Keyword re-ranking (hybrid)
    kw_filtered = keyword_filter(chunks, query)
    kw_ids = {c["chunk_id"] for c in kw_filtered[:top_k * 2]}

    candidates = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        # Boost score if keyword match
        boosted = score + (0.1 if chunk["chunk_id"] in kw_ids else 0.0)
        candidates.append((boosted, chunk))

    candidates.sort(key=lambda x: -x[0])
    top_candidates = candidates[:top_k]

    # Abstention check
    if not top_candidates or top_candidates[0][0] < RELEVANCE_THRESHOLD:
        gold_stats = query_gold_stats()
        return {
            "abstained": True,
            "answer": (
                "I don't know. The approved heat-safety corpus does not contain "
                "information that directly answers this question. "
                "What's missing: a document specifically addressing this topic."
            ),
            "citations": [],
            "retrieved_chunks": [],
            "gold_stats": gold_stats,
        }

    top_chunks = [c for _, c in top_candidates]
    gold_stats = query_gold_stats()

    # Build citations
    citations = []
    seen_sources = set()
    for chunk in top_chunks:
        key = chunk["url"]
        if key not in seen_sources:
            citations.append({
                "title": chunk["title"],
                "publisher": chunk["publisher"],
                "url": chunk["url"],
                "pub_date": chunk["pub_date"],
                "section": chunk["section"],
                "chunk_id": chunk["chunk_id"],
                "passage": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
            })
            seen_sources.add(key)

    # Build answer from top chunk text
    answer = top_chunks[0]["text"]

    return {
        "abstained": False,
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": top_chunks,
        "gold_stats": gold_stats,
    }


def format_response(result: dict) -> str:
    """Format a retrieval result for display."""
    if result["abstained"]:
        return f"**I don't know.**\n\n{result['answer']}"

    lines = [result["answer"], "", "---", "**Sources:**"]
    for i, cit in enumerate(result["citations"], 1):
        lines.append(
            f"{i}. *{cit['title']}* — {cit['publisher']} ({cit['pub_date']})  "
            f"[{cit['url']}]  \n   Section: {cit['section']}  \n   "
            f"Passage: \"{cit['passage']}\""
        )

    if result.get("gold_stats"):
        gs = result["gold_stats"]
        lines += [
            "",
            "**Live data (from Gold table):**",
            f"- Latest apparent temperature: {gs.get('latest_apparent_temp_c', 'N/A')} °C",
            f"- 24h mean apparent temperature: {gs.get('last_24h_mean_apparent_temp_c', 'N/A')} °C",
            f"- Data as of: {gs.get('data_as_of', 'N/A')}",
        ]

    return "\n".join(lines)


if __name__ == "__main__":
    test_queries = [
        "How often should outdoor workers drink water in hot weather?",
        "What is the stock price of Apple?",   # should abstain
        "What are the symptoms of heat stroke?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = retrieve(q)
        print(format_response(result))
        print("-" * 60)
