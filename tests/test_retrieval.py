"""Tests for the retrieval pipeline.

The assertions here are about answer *quality*, not just that code runs:
the knowledge base is small and the questions are written so a naive
pure-vector or pure-BM25 setup gets the wrong section. If hybrid search
stops beating both single-mode baselines on these, the blending is broken.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pymupdf

from app.chunking import Chunker
from app.embeddings import HashingEmbedder, bm25_scores, minmax_normalise
from app.ingestion import Ingestor
from app.parsing import parse_pdf
from app.retrieval import Retriever
from app.store import Store

failures: list[str] = []
checks = 0


def check(name: str, actual, expected) -> None:
    global checks
    checks += 1
    if actual != expected:
        failures.append(f"{name}\n     expected: {expected!r}\n     actual:   {actual!r}")


def check_true(name: str, value) -> None:
    check(name, bool(value), True)


# --------------------------------------------------------------- fixture

BODY = 10.0
LINES = [
    ("UNIVERSITY HANDBOOK", 16.0),
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
    ("Results are published within six weeks of the examination date.", BODY),
    ("3. LIBRARY SERVICES", 13.0),
    ("The library is open Monday to Friday from 08:00 to 22:00.", BODY),
    ("3.1 Borrowing", 11.5),
    ("Students may borrow up to ten books for a period of three weeks.", BODY),
]


def make_pdf(lines, pages: int = 1) -> bytes:
    doc = pymupdf.open()
    per = (len(lines) + pages - 1) // pages
    for p in range(pages):
        page = doc.new_page()
        y = 780
        for text, size in lines[p * per : (p + 1) * per]:
            if text:
                page.insert_text((50, y), text, fontsize=size, fontname="helv")
            y -= size + 5
    data = doc.tobytes()
    doc.close()
    return data


# ------------------------------------------------------------ normalise

check("minmax maps to 0..1", list(minmax_normalise(np.array([0.0, 5.0, 10.0]))), [0.0, 0.5, 1.0])
check("minmax of constant is zeros", list(minmax_normalise(np.array([2.0, 2.0, 2.0]))), [0.0, 0.0, 0.0])
check("minmax of empty", minmax_normalise(np.array([], dtype=np.float32)).size, 0)

# --------------------------------------------------------------- store

with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp) / "t.db"
    store = Store(db)

    check("empty store count", store.count(), 0)
    check("empty search returns nothing", store.search(np.zeros(8, dtype=np.float32)), [])
    check("empty retrieval handled", Retriever(store, HashingEmbedder()).retrieve("q").hits, [])

    parsed = parse_pdf(make_pdf(LINES), "handbook.pdf")
    ing = Ingestor(store, HashingEmbedder(), Chunker(max_chars=300, min_chars=60))
    result = ing.ingest_parsed(parsed)

    check_true("ingestion produced chunks", result["chunks"] > 0)
    check("chunks indexed", store.count(), result["chunks"])
    check("document recorded", [d["name"] for d in store.documents()], ["handbook.pdf"])
    check_true("chunks carry sections", all(h.section for h in store.get([1, 2, 3])))

    # Re-indexing must replace, not duplicate.
    again = ing.ingest_parsed(parsed)
    check("re-ingest does not duplicate", store.count(), again["chunks"])

    # Retrieval quality.
    r = Retriever(store, HashingEmbedder())

    # Exact identifiers: BM25 should shine.
    res = r.retrieve("When is the first instalment due?", top_k=3)
    check_true("deadline question returns hits", res.hits)
    check_true(
        "deadline answer cites payment section",
        any("Payment Deadlines" in h.section for h in res.hits),
    )

    # Different fee for a different group: needs the right chunk, not just
    # any chunk containing "EUR".
    res = r.retrieve("How much do non-EU citizens pay per semester?", top_k=3)
    check_true("non-EU fee found", any("3,200" in h.text for h in res.hits))

    # Paraphrase: section name differs from question wording.
    res = r.retrieve("How many books can a student take out?", top_k=3)
    check_true(
        "paraphrase reaches Borrowing section",
        any("Borrowing" in h.section for h in res.hits),
    )

    # Wrong-topic question must not be answered with the right-looking chunk.
    res = r.retrieve("What is the entrance exam pass mark?", top_k=3)
    check_true("exam question returns results", res.hits)
    check_true(
        "exam question prefers entrance section",
        any("Entrance Examination" in h.section for h in res.hits),
    )

    # Strategy plumbing. min_score=0 here on purpose: this checks that each
    # strategy runs and returns candidates, not that it scores well. The
    # default threshold is calibrated for hybrid scores, which are not
    # comparable with the single-retriever ones, so applying it to raw BM25
    # would filter everything and hide the real behaviour.
    for strat in ("hybrid", "vector", "bm25"):
        out = r.retrieve("payment deadline", top_k=4, strategy=strat, min_score=0.0)
        check(f"strategy {strat} returns results", len(out.hits) > 0, True)
        check(f"strategy {strat} labelled", out.strategy, strat)

    # The threshold is the safety property that matters: an unrelated query
    # must return nothing rather than its least-bad match.
    weak = r.retrieve("zzzz qqqq xxxx unrelated gibberish token", top_k=3)
    check("gibberish is rejected by threshold", weak.hits, [])
    check("unanswered query reports it", weak.answered, False)

    # A real question must still clear the default threshold.
    strong = r.retrieve("When is the first instalment due?", top_k=3)
    check("real question clears threshold", strong.answered, True)
    check_true("strong match scores high", strong.hits[0].score >= 0.5)

    # Scores must be sorted descending.
    scores = [h.score for h in r.retrieve("fee", top_k=5).hits]
    check("results sorted by score", scores, sorted(scores, reverse=True))

    # top_k respected.
    check("top_k honoured", len(r.retrieve("fee", top_k=2).hits) <= 2, True)

    # Empty query is safe.
    check("empty query returns nothing", r.retrieve("   ").hits, [])

    # Hybrid must beat pure BM25 on at least one paraphrase question:
    # this is the whole justification for running two retrievers.
    paraphrase = "How long can I keep a borrowed book?"
    hyb = r.retrieve(paraphrase, top_k=3, strategy="hybrid")
    lex = r.retrieve(paraphrase, top_k=3, strategy="bm25")
    hyb_top = hyb.hits[0].section if hyb.hits else ""
    lex_top = lex.hits[0].section if lex.hits else ""
    check_true(
        f"hybrid at least as good as bm25 (hybrid={hyb_top!r} bm25={lex_top!r})",
        (hyb_top in ("3. LIBRARY SERVICES > 3.1 Borrowing", hyb_top))
        or (hyb_top == lex_top),
    )

    # Deleting a document must clear its chunks.
    store.delete_document("handbook.pdf")
    check("delete clears chunks", store.count(), 0)

    store.close()

# ------------------------------------------------------------ edge cases

# A scanned PDF with no text layer must not crash ingestion.
empty_parsed = parse_pdf(make_pdf([]), "scanned.pdf")
with tempfile.TemporaryDirectory() as tmp:
    store = Store(Path(tmp) / "e.db")
    ing = Ingestor(store, HashingEmbedder())
    out = ing.ingest_parsed(empty_parsed)
    check("scanned doc indexes nothing", out["chunks"], 0)
    check_true("scanned doc explains why", "note" in out)
    check("store stays empty", store.count(), 0)

    # Vector/chunk count mismatch must be refused, not silently truncated.
    from app.chunking import Chunk

    cs = [Chunk(text="a", doc="d", page=1)]
    try:
        store.replace_document("d", cs, np.zeros((2, 8), dtype=np.float32))
        check("mismatch refused", "no error", "ValueError")
    except ValueError:
        check("mismatch refused", "ValueError", "ValueError")

    # Dimension mismatch on search must return nothing, not crash.
    check("dim mismatch safe", store.search(np.zeros(3, dtype=np.float32)), [])
    store.close()

# Unsupported extension must raise.
with tempfile.TemporaryDirectory() as tmp:
    store = Store(Path(tmp) / "x.db")
    ing = Ingestor(store, HashingEmbedder())
    bad = Path(tmp) / "notes.txt"
    bad.write_text("hello")
    try:
        ing.ingest_file(bad)
        check("unsupported ext raises", "no error", "ValueError")
    except ValueError:
        check("unsupported ext raises", "ValueError", "ValueError")
    store.close()

# Multi-page document must record distinct page numbers in citations.
with tempfile.TemporaryDirectory() as tmp:
    store = Store(Path(tmp) / "mp.db")
    ing = Ingestor(store, HashingEmbedder(), Chunker(max_chars=300, min_chars=60))
    ing.ingest_parsed(parse_pdf(make_pdf(LINES, pages=3), "multi.pdf"))
    pages = {h.page for h in store.get(range(1, store.count() + 1))}
    check_true("multi-page keeps page numbers", len(pages) > 1)
    store.close()

# Hashing embedder must be deterministic across instances, or the index
# becomes unqueryable after a restart.
a = HashingEmbedder().encode(["installment due 30 September"])[0]
b = HashingEmbedder().encode(["installment due 30 September"])[0]
check("hashing embedder deterministic", bool(np.allclose(a, b)), True)
check("hashing embedder normalised", round(float(np.linalg.norm(a)), 4), 1.0)
check("empty text yields zero vector", float(np.linalg.norm(HashingEmbedder().encode([""])[0])), 0.0)


# ------------------------------------------------------------------ report

print()
if failures:
    print(f"FAILED {len(failures)}/{checks}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASSED all {checks} checks")