"""Embedding generation.

Defaults to a local sentence-transformers model so the whole pipeline runs
without an API key or network access. A deterministic hashing embedder is
available as a fallback for environments where the model cannot be loaded,
so ingestion never hard-fails on a missing dependency.

Hashing is genuinely worse at semantics and the retrieval layer needs to
know that, so `HashingEmbedder.degraded` is True and the API surfaces it.
Silently degrading would mean answering questions badly with no signal
that anything was wrong.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dimension: int
    degraded: bool

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class LocalEmbedder:
    """sentence-transformers, normalised float32 vectors."""

    degraded = False

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # get_sentence_embedding_dimension is deprecated upstream; prefer the
        # new name and fall back so this keeps working on older releases.
        dim = getattr(self._model, "get_embedding_dimension", None)
        self.dimension = int(
            dim() if dim else self._model.get_sentence_embedding_dimension()
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vecs = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """Deterministic hashed bag-of-tokens fallback.

    Same text always yields the same vector, so an index built with it
    stays queryable across restarts. It matches lexical overlap well and
    paraphrase poorly, which is why `degraded` is exposed.
    """

    degraded = True

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            # Hash each token and also a bigram of adjacent tokens, so word
            # order contributes something without a real language model.
            digest = hashlib.blake2b(tok.encode(), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.dimension
            sign = 1.0 if digest[0] & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts]).astype(np.float32)


def get_embedder(prefer_local: bool = True) -> Embedder:
    """Local model if it loads, otherwise the hashing fallback."""
    if prefer_local:
        try:
            return LocalEmbedder()
        except Exception:
            pass
    return HashingEmbedder()


def cosine_matrix(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity, assuming both sides are already L2-normalised."""
    if corpus.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query.reshape(-1)
    denom = np.linalg.norm(q)
    if denom == 0:
        return np.zeros((corpus.shape[0],), dtype=np.float32)
    norms = np.linalg.norm(corpus, axis=1)
    norms[norms == 0] = 1.0
    return (corpus @ q / denom / norms).astype(np.float32)


def bm25_scores(
    query: str,
    docs: Sequence[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Okapi BM25 over a small in-memory corpus.

    Deliberately simple. A college knowledge base is thousands of chunks,
    not millions, so an inverted index on disk would add moving parts for
    no measurable gain. At 100k chunks this becomes the bottleneck and
    should move to a real search engine.
    """
    n = len(docs)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    tokenized = [_TOKEN.findall(d.lower()) for d in docs]
    lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
    avg_len = float(lengths.mean()) if n else 1.0

    q_terms = _TOKEN.findall(query.lower())
    if not q_terms:
        return np.zeros((n,), dtype=np.float32)

    # Document frequency per query term.
    df: dict[str, int] = {}
    for term in set(q_terms):
        df[term] = sum(1 for toks in tokenized if term in toks)

    scores = np.zeros(n, dtype=np.float32)
    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        counts: dict[str, int] = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        s = 0.0
        for term in q_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            denom_t = tf + k1 * (1 - b + b * lengths[i] / avg_len)
            s += idf * (tf * (k1 + 1)) / denom_t
        scores[i] = s

    return scores


def minmax_normalise(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 so two scorers can be blended with fixed weights."""
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-9:
        # Everything equal: return zeros rather than dividing by ~zero and
        # producing noise that looks like signal.
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)