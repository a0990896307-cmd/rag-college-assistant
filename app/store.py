"""Chunk store on SQLite with float32 vectors.

SQLite rather than pgvector so the project runs with one command and no
database server, which matters for a reviewer cloning the repo. The schema
and the vector access go through this module only, so moving to pgvector
later is a change to this file plus the DDL, not to the retrieval code.

Vectors are stored as raw little-endian float32 blobs and scored with a
NumPy matmul. At college-document scale (thousands of chunks) that is
microseconds and avoids pulling in an ANN index that would not be used.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .chunking import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    doc          TEXT NOT NULL,
    page         INTEGER NOT NULL,
    section      TEXT NOT NULL DEFAULT '',
    heading_path TEXT NOT NULL DEFAULT '[]',
    text         TEXT NOT NULL,
    citation     TEXT NOT NULL,
    vector       BLOB NOT NULL,
    dim          INTEGER NOT NULL,
    indexed_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc);

CREATE TABLE IF NOT EXISTS documents (
    name        TEXT PRIMARY KEY,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    char_count  INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class Hit:
    chunk_id: int
    text: str
    doc: str
    page: int
    section: str
    citation: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "citation": self.citation,
            "section": self.section,
            "doc": self.doc,
            "page": self.page,
            "score": round(self.score, 4),
            "text": self.text,
            "vector_score": round(self.vector_score, 4),
            "bm25_score": round(self.bm25_score, 4),
        }


class Store:
    def __init__(self, path: str | Path = "rag.db") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------ writing

    def replace_document(self, doc: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Index a document, replacing any previous version atomically.

        Re-ingesting must not leave a mix of old and new chunks: the old
        row for a section that was edited away would still be retrieved and
        would cite text that no longer exists.
        """
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError(
                f"vector count {len(vectors)} does not match chunk count {len(chunks)}"
            )
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE doc = ?", (doc,))
            rows = []
            for i, c in enumerate(chunks):
                vec = np.asarray(vectors[i], dtype=np.float32)
                rows.append(
                    (
                        doc,
                        c.page,
                        c.section,
                        json.dumps(c.heading_path, ensure_ascii=False),
                        c.text,
                        c.citation,
                        vec.tobytes(),
                        int(vec.shape[0]),
                    )
                )
            self.conn.executemany(
                "INSERT INTO chunks (doc, page, section, heading_path, text, citation,"
                " vector, dim) VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            self.conn.execute(
                "INSERT INTO documents (name, chunk_count, char_count) VALUES (?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET chunk_count=excluded.chunk_count,"
                " char_count=excluded.char_count, indexed_at=datetime('now')",
                (doc, len(chunks), sum(len(c.text) for c in chunks)),
            )
        return len(chunks)

    def delete_document(self, doc: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE doc = ?", (doc,))
            self.conn.execute("DELETE FROM documents WHERE name = ?", (doc,))

    # ------------------------------------------------------------ reading

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"])

    def documents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT name, chunk_count, char_count, indexed_at FROM documents"
            " ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]

    def all_texts(self) -> list[str]:
        rows = self.conn.execute("SELECT text FROM chunks ORDER BY id").fetchall()
        return [r["text"] for r in rows]

    def _matrix(self) -> tuple[np.ndarray, list[sqlite3.Row]]:
        rows = self.conn.execute(
            "SELECT id, text, doc, page, section, citation, vector FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), rows
        dim = int(rows[0]["vector"].__len__() // 4)
        mat = np.vstack(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        )
        return mat.astype(np.float32), rows

    def search(self, query_vec: np.ndarray, top_k: int = 8) -> list[Hit]:
        """Pure vector search. The retrieval module blends this with BM25."""
        mat, rows = self._matrix()
        if mat.size == 0 or mat.shape[0] == 0:
            return []
        if query_vec.shape[0] != mat.shape[1]:
            # Dimension mismatch means the index was built with a different
            # model. Returning nothing is safer than scoring garbage.
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn == 0:
            return []
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        sims = (mat @ q / qn / norms).astype(np.float32)
        k = min(top_k, sims.shape[0])
        order = np.argpartition(-sims, k - 1)[:k]
        order = order[np.argsort(-sims[order])]
        hits: list[Hit] = []
        for idx in order:
            r = rows[int(idx)]
            hits.append(
                Hit(
                    chunk_id=int(r["id"]),
                    text=r["text"],
                    doc=r["doc"],
                    page=int(r["page"]),
                    section=r["section"],
                    citation=r["citation"],
                    vector_score=float(sims[int(idx)]),
                )
            )
        return hits

    def get(self, chunk_ids: Iterable[int]) -> list[Hit]:
        ids = list(chunk_ids)
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, text, doc, page, section, citation FROM chunks"
            f" WHERE id IN ({marks})",
            ids,
        ).fetchall()
        by_id = {int(r["id"]): r for r in rows}
        out: list[Hit] = []
        for cid in ids:
            r = by_id.get(cid)
            if r is None:
                continue
            out.append(
                Hit(
                    chunk_id=cid,
                    text=r["text"],
                    doc=r["doc"],
                    page=int(r["page"]),
                    section=r["section"],
                    citation=r["citation"],
                )
            )
        return out