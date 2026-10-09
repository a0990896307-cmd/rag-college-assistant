"""Hybrid retrieval: BM25 plus vector search, with a reranking pass.

Pure vector retrieval fails in a predictable way on institutional
documents: it returns a chunk about the wrong fee when asked about
deadlines, because both mention money and both sit near the word "fee".
Lexical search has the opposite failure, missing any question phrased
differently from the source.

Both are run and blended. Reranking then re-scores the fused candidates,
which is where the real gain comes from: the two retrievers disagree
about what matters, and the reranker uses features neither of them had.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder, bm25_scores, minmax_normalise
from .store import Hit, Store

# Vector carries more weight because questions are usually paraphrases of
# the source rather than copies of it. BM25 is not decoration though: it is
# what keeps exact identifiers like "EUR 2,400" or "15 August" findable,
# which is most of what a college knowledge base is asked about.
W_VECTOR = 0.65
W_BM25 = 0.35


# Scores below this are treated as "nothing matched" rather than returned as
# weak answers. Calibrated on the demo corpus: real matches land between 0.80
# and 0.95 after reranking, and the best wrong-section candidate sits around
# 0.20 to 0.30. Returning a weak passage with low confidence is how a RAG
# system ends up confidently wrong, which is worse than admitting ignorance.
MIN_SCORE = 0.30


@dataclass
class Retrieved:
    hits: list[Hit]
    query: str
    strategy: str
    degraded: bool = False
    threshold: float = MIN_SCORE

    @property
    def answered(self) -> bool:
        return bool(self.hits)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "degraded_embedder": self.degraded,
            "threshold": self.threshold,
            "answered": self.answered,
            "results": [h.to_dict() for h in self.hits],
        }


def _rerank(hits: list[Hit], query_tokens: set[str]) -> None:
    """Re-score in place using query coverage and citation quality.

    Coverage: how much of the query's distinctive vocabulary a candidate
    addresses. A passage answering half the question should lose to one
    answering all of it even if that one matched fewer keywords.

    Citation quality: at equal relevance, prefer a chunk with a section
    path, because a human can verify "3.1 Payment Deadlines" instantly
    and "page 14" barely at all.
    """
    if not hits:
        return
    for h in hits:
        text_tokens = set(h.text.lower().split())
        coverage = (
            len(query_tokens & text_tokens) / len(query_tokens)
            if query_tokens
            else 0.0
        )
        h.score = float(
            0.75 * h.score + 0.20 * coverage + (0.05 if h.section else 0.0)
        )


class Retriever:
    def __init__(self, store: Store, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def _ids_and_texts(self) -> tuple[list[int], list[str]]:
        rows = self.store.conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        return [int(r["id"]) for r in rows], [r["text"] for r in rows]

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        candidate_k: int = 24,
        strategy: str = "hybrid",
        min_score: float = MIN_SCORE,
    ) -> Retrieved:
        q = query.strip()
        degraded = bool(getattr(self.embedder, "degraded", False))
        if not q:
            return Retrieved([], q, strategy, degraded, min_score)

        ids, texts = self._ids_and_texts()
        if not ids:
            return Retrieved([], q, strategy, degraded, min_score)

        # --- candidate pool -------------------------------------------------
        pool: list[int] = []
        vector_scores: dict[int, float] = {}
        bm25_raw = np.zeros(len(ids), dtype=np.float32)

        if strategy in ("hybrid", "vector"):
            qv = self.embedder.encode([q])[0]
            for h in self.store.search(qv, top_k=candidate_k):
                vector_scores[h.chunk_id] = max(float(h.vector_score), 0.0)
                if h.chunk_id not in pool:
                    pool.append(h.chunk_id)

        if strategy in ("hybrid", "bm25"):
            bm25_raw = bm25_scores(q, texts)
            order = np.argsort(-bm25_raw)[:candidate_k]
            for pos in order:
                if bm25_raw[int(pos)] <= 0:
                    continue
                cid = ids[int(pos)]
                if cid not in pool:
                    pool.append(cid)

        if not pool:
            return Retrieved([], q, strategy, degraded, min_score)

        # --- fetch any candidate neither search returned as a Hit ---------
        have = set(vector_scores)
        bm25_by_id = {ids[i]: float(bm25_raw[i]) for i in range(len(ids))}
        missing = [cid for cid in pool if cid not in have]
        fetched = {h.chunk_id: h for h in self.store.get(missing)}

        # --- normalise then blend ------------------------------------------
        # BM25 has no absolute scale: the same query gives different raw
        # scores depending on corpus size, so it must be normalised across
        # the pool or the blend weights are meaningless.
        #
        # Cosine similarity is different. It is bounded and absolute: 1.0
        # is identical, 0.0 is orthogonal. Min-max scaling it against the
        # pool would be actively harmful, because a pool where every
        # candidate is junk would still produce a 1.0 for its least-bad
        # member and sail past the threshold. The absolute scale is exactly
        # what makes the confidence cutoff possible, so it is preserved.
        b_vals = np.array([bm25_by_id.get(c, 0.0) for c in pool], dtype=np.float32)
        b_norm = minmax_normalise(b_vals)
        v_vals = np.array([vector_scores.get(c, 0.0) for c in pool], dtype=np.float32)

        hits: list[Hit] = []
        for i, cid in enumerate(pool):
            h = fetched.get(cid) or next((x for x in self.store.get([cid])), None)
            if h is None:
                continue
            h.vector_score = float(v_vals[i])
            h.bm25_score = float(b_vals[i])
            if strategy == "vector":
                h.score = float(v_vals[i])
            elif strategy == "bm25":
                h.score = float(b_norm[i])
            else:
                h.score = float(W_VECTOR * v_vals[i] + W_BM25 * b_norm[i])
            hits.append(h)

        # --- rerank ---------------------------------------------------------
        query_tokens = {w for w in q.lower().split() if len(w) > 2}
        _rerank(hits, query_tokens)
        hits.sort(key=lambda h: h.score, reverse=True)
        kept = [h for h in hits if h.score >= min_score][:top_k]

        return Retrieved(
            hits=kept,
            query=q,
            strategy=strategy,
            degraded=degraded,
            threshold=min_score,
        )