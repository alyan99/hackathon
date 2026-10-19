"""
RAG corpus preparation — SaayaShift

Loads openly licensed heat-safety reference documents,
chunks them, embeds with sentence-transformers,
and builds a FAISS vector index.

Every chunk preserves:
  title, publisher, url, pub_date, section, chunk_id, text

Only ingest documents whose reuse terms have been confirmed.
See docs/DATASET_DECLARATION.md Section 4 for the approved corpus list.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR  = ROOT / "data" / "rag_corpus"
INDEX_DIR   = ROOT / "data" / "rag_index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Approved reference documents ─────────────────────────────────────────────
# Only documents with confirmed reuse permission are listed here.
# Add new entries only after verifying their licence terms.
APPROVED_SOURCES = [
    {
        "title": "Protecting Workers from Heat Illness",
        "publisher": "OSHA (US Occupational Safety and Health Administration)",
        "url": "https://www.osha.gov/heat-exposure",
        "pub_date": "2023",
        "licence": "US Government work — public domain",
        "section": "General guidance",
        "text": (
            "Workers who are exposed to extreme heat or work in hot environments may be at risk "
            "of heat stress. Heat stress can result in heat stroke, heat exhaustion, heat cramps, "
            "or heat rashes. Heat stroke is the most serious heat-related disorder and occurs when "
            "the body can no longer control its temperature. When heat stroke occurs, body temperature "
            "can rise to 106°F or higher within 10 to 15 minutes. Heat stroke can cause death or permanent "
            "disability if emergency treatment is not given. "
            "Symptoms include hot, dry skin or profuse sweating, confusion, slurred speech, "
            "and loss of consciousness. "
            "Heat exhaustion is the body's response to an excessive loss of water and salt "
            "from sweating. Symptoms include heavy sweating, weakness, cold pale clammy skin, "
            "weak pulse, nausea or vomiting, and possible fainting. "
            "Prevention: Provide workers with water, rest, and shade. Allow new workers or workers "
            "returning after time away to gradually increase workloads and take more frequent breaks "
            "as they acclimatize to the heat. Modify work schedules and heat-stress exposures. "
            "Monitor workers for signs and symptoms of heat-related illness."
        ),
    },
    {
        "title": "Heat Stress — Technical Brief",
        "publisher": "International Labour Organization (ILO)",
        "url": "https://www.ilo.org/global/topics/safety-and-health-at-work/resources-library/publications/WCMS_211486/lang--en/index.htm",
        "pub_date": "2018",
        "licence": "ILO — reuse permitted for non-commercial purposes with attribution",
        "section": "Prevention and management",
        "text": (
            "Heat stress at work is a significant occupational safety concern, particularly for workers "
            "engaged in heavy outdoor or industrial work in tropical and subtropical climates. "
            "Acclimatization is the process by which the body adapts to heat over time, typically "
            "requiring 7 to 14 days of progressive heat exposure. "
            "Recommended controls: engineering controls such as shading work areas and providing cool "
            "rest areas; administrative controls such as scheduling heavy work during cooler times of "
            "the day, implementing a buddy system, and providing training; and personal protective "
            "equipment such as cooling vests where appropriate. "
            "Workers should have access to cool (15-22°C) drinking water and be encouraged to drink "
            "approximately 250 ml (one cup) every 20 minutes. Supervisors should monitor workers for "
            "symptoms of heat illness and should know how to provide first aid."
        ),
    },
    {
        "title": "Heat and Health — Fact Sheet",
        "publisher": "World Health Organization (WHO)",
        "url": "https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health",
        "pub_date": "2023",
        "licence": "CC BY-NC-SA 3.0 IGO",
        "section": "Health impacts and protective measures",
        "text": (
            "Extreme heat is one of the most dangerous natural hazards. Heat waves and high ambient "
            "temperatures cause heat stress and can lead to heat exhaustion and potentially fatal "
            "heat stroke. The elderly, young children, outdoor workers, and people with chronic "
            "illnesses are most vulnerable. "
            "Apparent temperature (also called the heat index or feels-like temperature) accounts for "
            "the combined effects of temperature and humidity and provides a more accurate indication "
            "of perceived heat stress than air temperature alone. High humidity reduces the body's "
            "ability to cool itself through sweat evaporation. "
            "Preventive measures include: staying in cool or shaded environments especially between "
            "10:00 and 16:00 local time, wearing loose light-coloured clothing, drinking water "
            "regularly even when not thirsty, avoiding strenuous outdoor activity during the hottest "
            "part of the day, and checking on vulnerable individuals. "
            "Cities should implement heat-health action plans including early warning systems, "
            "public cooling centres, and green infrastructure to reduce the urban heat island effect."
        ),
    },
    {
        "title": "Karachi Climate Overview",
        "publisher": "SaayaShift internal reference (public domain meteorological data)",
        "url": "internal",
        "pub_date": "2026",
        "licence": "CC0",
        "section": "Karachi heat context",
        "text": (
            "Karachi, Pakistan, experiences a semi-arid climate with very hot summers. "
            "Mean daily maximum temperatures in May to August frequently exceed 35°C, with heat "
            "waves pushing apparent temperatures above 45°C when combined with high humidity from "
            "the Arabian Sea. The period from late May to early September presents the highest "
            "occupational heat risk for outdoor workers. "
            "The pre-monsoon months (April–June) are characterised by hot, dry conditions; the "
            "monsoon months (July–September) bring increased humidity that elevates apparent "
            "temperature and heat stress risk even when air temperatures are slightly lower. "
            "Supervisors scheduling outdoor field crews should prefer early morning (before 08:00) "
            "and late afternoon (after 17:00) windows during peak summer months, consistent with "
            "WHO and ILO guidance to avoid the hottest part of the day."
        ),
    },
]


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def make_chunk_id(source_url: str, section: str, idx: int) -> str:
    raw = f"{source_url}::{section}::{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def iter_chunks() -> Iterator[dict]:
    """Yield all chunks with full metadata from approved sources."""
    for doc in APPROVED_SOURCES:
        sub_chunks = chunk_text(doc["text"])
        for i, text in enumerate(sub_chunks):
            yield {
                "chunk_id": make_chunk_id(doc["url"], doc["section"], i),
                "title": doc["title"],
                "publisher": doc["publisher"],
                "url": doc["url"],
                "pub_date": doc["pub_date"],
                "licence": doc["licence"],
                "section": doc["section"],
                "chunk_index": i,
                "text": text,
            }


# ── Embedding + index ─────────────────────────────────────────────────────────

def build_index() -> None:
    """Embed all chunks and save FAISS index + chunks JSON."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError:
        log.error("sentence-transformers or faiss-cpu not installed. Run: pip install sentence-transformers faiss-cpu")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks = list(iter_chunks())
    log.info("Chunking corpus: %d chunks from %d documents", len(chunks), len(APPROVED_SOURCES))

    texts = [c["text"] for c in chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product on L2-normalised vectors = cosine
    index.add(embeddings)

    # Save index and chunks
    faiss.write_index(index, str(INDEX_DIR / "corpus.faiss"))
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    log.info("FAISS index saved → %s (%d vectors, dim=%d)", INDEX_DIR / "corpus.faiss", len(chunks), dim)
    log.info("Chunks metadata saved → %s", CHUNKS_PATH)


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
