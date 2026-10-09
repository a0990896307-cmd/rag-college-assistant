# RAG Knowledge Assistant

Document ingestion and cited retrieval for institutional documents:
college handbooks, policy sets, technical documentation. Answers cite the
document, page and section they came from, and the system refuses to
answer rather than returning its least-bad guess.

Runs with no database server, no API key and no network access.

```bash
pip install pymupdf python-docx numpy fastapi "uvicorn[standard]" \
            sentence-transformers python-multipart
python demo.py
```

The demo ingests a generated PDF and answers four questions. That is the
smoke test: run it before reading any code.

```bash
python tests/test_parsing_chunking.py   # 39 checks
python tests/test_retrieval.py          # 42 checks
python tests/test_api.py                # 35 checks
```

116 checks total, no test framework required, non-zero exit on failure so
they drop straight into CI.

## Why section-aware chunking

Fixed-size windows with overlap are the default in most RAG pipelines and
they are the main reason answers cite the wrong section: a window cut
through a numbered clause produces text that reads fine but cannot be
attributed to anything.

This chunks on section boundaries first and only falls back to length
limits inside a section. Every chunk carries its heading path, so an
answer can say `3.1 Payment Deadlines` instead of `page 14` — and a human
can verify the first in a second and the second barely at all.

Heading detection is relative to each document's own median body font
size, not an absolute threshold. A 9pt textbook and an 11pt thesis both
resolve structure correctly, where a fixed cut-off would silently treat
one of them as having no headings.

Depth is tracked as an explicit `(level, text)` stack. Truncating a flat
path by length makes sibling sections nest: `2.1 Alpha` followed by
`2.2 Beta` would produce `2.1 > 2.2`, because both sit at depth 1.

## Why hybrid retrieval

Pure vector search fails in a predictable way on institutional
documents: asked about payment deadlines it returns the fees section,
because both mention money and both sit near the word "fee". Lexical
search has the opposite failure, missing any question phrased differently
from the source.

Both run, then blend. Reranking adds query coverage (how much of the
question a passage actually addresses) and a citation-quality signal.

BM25 is normalised across the candidate pool because it has no absolute
scale. **Cosine similarity is deliberately not normalised** — it is
bounded and absolute, and min-max scaling it would let a pool where every
candidate is junk still produce a 1.0 for its least-bad member.

That absolute scale is what makes the confidence cutoff work.

## The confidence cutoff

Below `MIN_SCORE` (0.30) the system returns nothing and says so.

Calibrated on the demo corpus: real matches land between 0.80 and 0.95
after reranking, and the best wrong-section candidate sits around 0.20.
An unrelated query scores under 0.10 and is rejected.

This is the property that matters most in production. A retrieval system
that always returns its top result is a system that confidently answers
questions it has no evidence for, and on a college knowledge base that
means quoting the wrong regulation at a student.

## Layout

```
app/parsing.py     PDF/DOCX → blocks with heading paths
app/chunking.py    structure-aware chunker, citation strings
app/embeddings.py  local model, hashing fallback, BM25
app/store.py       SQLite + float32 vectors
app/retrieval.py   hybrid fusion, reranking, threshold
app/ingestion.py   batch embedding, atomic re-index
app/api.py         FastAPI service and web interface
demo.py            end-to-end smoke test
```

## API

```bash
uvicorn app.api:app --port 8000     # http://127.0.0.1:8000
```

| | |
|---|---|
| `GET /health` | chunk count, document count, embedder in use |
| `POST /ingest` | multipart PDF or DOCX |
| `POST /search` | question, top_k, strategy |
| `GET /documents` | indexed documents |
| `DELETE /documents/{name}` | remove one |

The web interface at `/` queries the same endpoints and flags the hashing
fallback when it is active, rather than answering badly in silence.

## Answer generation

Answers are extractive: retrieved passages quoted with their citations,
nothing paraphrased or invented. `grounded` is `true` in every response
and no text is generated that cannot be traced to a source passage.

A generative step slots in behind the same interface. When an LLM key is
configured it writes the answer from the retrieved passages; without one
the system still returns evidence. This is deliberate — a box that
silently invents an answer is worse than one that shows its evidence.

## Embedders

Local `sentence-transformers` (384d, normalised) by default. If the model
cannot load, a deterministic hashing embedder takes over so ingestion never
hard-fails, and `degraded_embedder` is `true` in the API response and
bannered in the UI. Hashing matches lexical overlap well and paraphrase
poorly, so the degradation is reported rather than hidden.

## Moving to pgvector

`app/store.py` is the only module that touches storage. Vectors are raw
float32 blobs scored with a NumPy matmul, which is microseconds at
college-document scale and avoids an ANN index that would not be used.
At around 100k chunks this becomes the bottleneck and should move to
pgvector; the schema and `Hit` type stay as they are.

## Licence

MIT.