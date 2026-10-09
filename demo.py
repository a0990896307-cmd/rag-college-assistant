#!/usr/bin/env python
"""End-to-end demo: ingest a generated document and query it.

Runs with no database server, no API key and no network. This is the
smoke test a reviewer should be able to run before reading any code.

    python demo.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pymupdf

from app.chunking import Chunker
from app.embeddings import get_embedder
from app.ingestion import Ingestor
from app.retrieval import Retriever
from app.store import Store

BODY = 10.0
DOC = [
    ("UNIVERSITY HANDBOOK 2026", 16.0),
    ("1. FEES AND PAYMENT", 13.0),
    ("Tuition fees are EUR 2,400 per semester for EU citizens.", BODY),
    ("Non-EU citizens pay EUR 3,200 per semester.", BODY),
    ("1.1 Payment Deadlines", 11.5),
    ("The first instalment is due by 30 September each year.", BODY),
    ("A late payment fee of EUR 150 applies after the deadline.", BODY),
    ("2. ADMISSION REQUIREMENTS", 13.0),
    ("Applicants must hold a completed secondary certificate.", BODY),
    ("2.1 Entrance Examination", 11.5),
    ("The entrance examination consists of two written papers.", BODY),
    ("Each paper is graded on a scale from one to six.", BODY),
    ("3. LIBRARY SERVICES", 13.0),
    ("The library is open Monday to Friday from 08:00 to 22:00.", BODY),
    ("3.1 Borrowing", 11.5),
    ("Students may borrow up to ten books for a period of three weeks.", BODY),
]

QUESTIONS = [
    "When is the first instalment due?",
    "How much do non-EU citizens pay per semester?",
    "How many books can a student borrow?",
    "What time does the library close?",
]


def write_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 780
    for text, size in DOC:
        if text:
            page.insert_text((50, y), text, fontsize=size, fontname="helv")
        y -= size + 5
    doc.save(str(path))
    doc.close()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rag-demo-"))
    pdf = tmp / "handbook.pdf"
    write_pdf(pdf)

    store = Store(tmp / "demo.db")
    embedder = get_embedder()
    ing = Ingestor(store, embedder, Chunker(max_chars=320, min_chars=80))

    stats = ing.ingest_file(pdf)
    print(f"ingested {stats['doc']}: {stats['chunks']} chunks, {stats['headings']} headings")
    print(f"embedder: {'local ' + str(embedder.dimension) + 'd' if not embedder.degraded else 'hashing fallback'}")
    print()

    r = Retriever(store, embedder)
    for q in QUESTIONS:
        res = r.retrieve(q, top_k=2)
        print(f"Q: {q}")
        for h in res.hits:
            preview = h.text if len(h.text) <= 88 else h.text[:88] + "..."
            print(f"   score {h.score:.3f}  {h.citation}")
            print(f"            {preview}")
        if not res.hits:
            print("   (no matches)")
        print()

    store.close()
    print(f"temp files in {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())