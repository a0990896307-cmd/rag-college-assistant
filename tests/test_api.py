"""Tests for the HTTP layer.

Uses the real app with a temporary database so the routes are exercised
through FastAPI itself rather than against stubbed functions. Ingestion,
search, error handling and the health contract are all covered.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TMP = tempfile.mkdtemp(prefix="rag-api-")
os.environ["RAG_DB"] = str(Path(TMP) / "api.db")

import pymupdf
from fastapi.testclient import TestClient

from app.api import app

failures: list[str] = []
checks = 0


def check(name: str, actual, expected) -> None:
    global checks
    checks += 1
    if actual != expected:
        failures.append(f"{name}\n     expected: {expected!r}\n     actual:   {actual!r}")


def check_true(name: str, value) -> None:
    check(name, bool(value), True)


BODY = 10.0
DOC = [
    ("POLICIES", 16.0),
    ("1. FEES", 13.0),
    ("Tuition fees are EUR 2,400 per semester for EU citizens.", BODY),
    ("1.1 Deadlines", 11.5),
    ("The first instalment is due by 30 September each year.", BODY),
    ("2. LIBRARY", 13.0),
    ("Students may borrow up to ten books for three weeks.", BODY),
]


def pdf_bytes() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 780
    for text, size in DOC:
        if text:
            page.insert_text((50, y), text, fontsize=size, fontname="helv")
        y -= size + 5
    data = doc.tobytes()
    doc.close()
    return data


client = TestClient(app)

# --------------------------------------------------------------- health

h = client.get("/health")
check("health status", h.status_code, 200)
check("health ok", h.json()["status"], "ok")
check("starts with no chunks", h.json()["chunks"], 0)
check_true("reports embedder", "embedder" in h.json())
check_true("reports dimension", h.json()["dimension"] > 0)

# ---------------------------------------------------------------- index

idx = client.get("/")
check("index serves html", idx.status_code, 200)
check_true("html content type", "text/html" in idx.headers["content-type"])

# --------------------------------------------------------------- ingest

bad = client.post("/ingest", files={"file": ("notes.txt", b"hello", "text/plain")})
check("rejects unsupported type", bad.status_code, 415)
check_true("rejection explains accepted types", "accepted" in bad.json()["detail"])

empty = client.post("/ingest", files={"file": ("empty.pdf", b"", "application/pdf")})
check("rejects empty file", empty.status_code, 400)

good = client.post(
    "/ingest",
    files={"file": ("policies.pdf", pdf_bytes(), "application/pdf")},
)
check("ingest accepted", good.status_code, 200)
check_true("ingest returned chunks", good.json()["chunks"] > 0)
check("ingest recorded doc", good.json()["doc"], "policies.pdf")

check("chunks counted", client.get("/health").json()["chunks"] > 0, True)
docs = client.get("/documents").json()["documents"]
check("document listed", [d["name"] for d in docs], ["policies.pdf"])

# Re-ingest must replace, not duplicate.
before = client.get("/health").json()["chunks"]
client.post("/ingest", files={"file": ("policies.pdf", pdf_bytes(), "application/pdf")})
check("re-ingest is idempotent", client.get("/health").json()["chunks"], before)

# --------------------------------------------------------------- search

r = client.post("/search", json={"question": "When is the first instalment due?"})
check("search ok", r.status_code, 200)
body = r.json()
check("grounded flag set", body["grounded"], True)
check_true("sources returned", len(body["sources"]) > 0)
check_true("answer cites a source", "[" in body["answer"])
check_true(
    "answer names the section",
    any("Deadlines" in s["section"] for s in body["sources"]),
)
check_true("source carries score", "score" in body["sources"][0])
check_true("source carries citation", bool(body["sources"][0]["citation"]))

for strat in ("hybrid", "vector", "bm25"):
    rr = client.post("/search", json={"question": "fees", "strategy": strat})
    check(f"strategy {strat} accepted", rr.status_code, 200)

check_true("top_k respected", len(client.post("/search", json={"question": "fees", "top_k": 1}).json()["sources"]) <= 1)

# Validation must reject nonsense rather than crash.
check("rejects empty question", client.post("/search", json={"question": ""}).status_code, 422)
check("rejects top_k 0", client.post("/search", json={"question": "x", "top_k": 0}).status_code, 422)
check("rejects bad strategy", client.post("/search", json={"question": "x", "strategy": "magic"}).status_code, 422)
check("rejects missing field", client.post("/search", json={}).status_code, 422)

# A question with no lexical or semantic match must return nothing rather
# than a confident wrong answer.
none_hits = client.post(
    "/search", json={"question": "zzzz qqqq xxxx unrelated gibberish token"}
).json()["sources"]
check_true("gibberish returns few or no sources", len(none_hits) <= 2)

# --------------------------------------------------------------- delete

d = client.delete("/documents/policies.pdf")
check("delete ok", d.status_code, 200)
check("delete clears chunks", client.get("/health").json()["chunks"], 0)
check("documents now empty", client.get("/documents").json()["documents"], [])


# ------------------------------------------------------------------ report

print()
if failures:
    print(f"FAILED {len(failures)}/{checks}")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"PASSED all {checks} checks")