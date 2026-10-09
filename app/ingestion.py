"""Ingestion: files in, cited chunks in the store."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .chunking import Chunker
from .embeddings import Embedder
from .parsing import parse
from .store import Store

SUPPORTED = {".pdf", ".docx"}


class Ingestor:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        chunker: Chunker | None = None,
        batch_size: int = 32,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or Chunker()
        self.batch_size = batch_size

    def ingest_file(self, path: str | Path) -> dict:
        path = Path(path)
        if path.suffix.lower() not in SUPPORTED:
            raise ValueError(
                f"unsupported file type {path.suffix!r}; expected one of {sorted(SUPPORTED)}"
            )
        parsed = parse(str(path))
        return self.ingest_parsed(parsed)

    def ingest_bytes(self, name: str, data: bytes) -> dict:
        parsed = parse(name, data=data)
        return self.ingest_parsed(parsed)

    def ingest_parsed(self, parsed) -> dict:
        chunks = self.chunker.chunk(parsed)
        if not chunks:
            # An empty parse is a real outcome (scanned PDF with no text
            # layer), not an error. Recording it keeps the caller honest
            # instead of reporting a successful index of nothing.
            return {
                "doc": parsed.name,
                "chunks": 0,
                "headings": parsed.heading_count,
                "note": "no text extracted; likely a scanned document needing OCR",
            }

        vectors = self._embed_all([c.text for c in chunks])
        self.store.replace_document(parsed.name, chunks, vectors)
        return {
            "doc": parsed.name,
            "chunks": len(chunks),
            "headings": parsed.heading_count,
            "characters": sum(len(c.text) for c in chunks),
        }

    def ingest_dir(self, directory: str | Path, recursive: bool = True) -> list[dict]:
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        out: list[dict] = []
        for p in sorted(directory.glob(pattern)):
            if p.is_file() and p.suffix.lower() in SUPPORTED:
                try:
                    out.append(self.ingest_file(p))
                except Exception as exc:
                    out.append({"doc": p.name, "chunks": 0, "error": str(exc)})
        return out

    def _embed_all(self, texts: list[str]) -> np.ndarray:
        """Embed in batches: one long batch is a memory spike, and a failure
        halfway through wastes the work already done."""
        if not texts:
            return np.zeros((0, self.embedder.dimension), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            out.append(self.embedder.encode(batch))
        return np.vstack(out).astype(np.float32)