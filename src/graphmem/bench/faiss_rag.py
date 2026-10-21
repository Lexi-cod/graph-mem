"""Baseline: dense retrieval over raw conversation turns (FAISS, same embedder)."""

from __future__ import annotations

import time

import numpy as np

from ..embedder import Embedder
from ..locomo import Conversation


class FaissRAG:
    def __init__(self, conv: Conversation, embedder: Embedder):
        import faiss

        self.docs: list[tuple[str, str]] = []  # (dia_id, rendered text with session date)
        for sess in conv.sessions:
            for t in sess.turns:
                self.docs.append((t.dia_id, f"[{sess.time:%d %B %Y}] {t.render()}"))
        self.embedder = embedder
        vecs = np.asarray(embedder.embed([d for _, d in self.docs]), dtype="float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # vectors are L2-normalized: IP = cosine
        self.index.add(vecs)

    def retrieve(self, query: str, k: int) -> dict:
        t0 = time.perf_counter()
        q = np.asarray(self.embedder.embed([query]), dtype="float32")
        t1 = time.perf_counter()
        scores, idx = self.index.search(q, k)
        t2 = time.perf_counter()
        hits = [self.docs[i] for i in idx[0] if i >= 0]
        return {
            "context": "\n".join(f"- {text}" for _, text in hits),
            "turn_ids": [d for d, _ in hits],
            "search_ms": (t2 - t1) * 1000,
            "embed_ms": (t1 - t0) * 1000,
        }
