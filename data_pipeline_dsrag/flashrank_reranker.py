"""FlashRank reranker adapter for dsRAG.

FlashRank is a small cross-encoder that runs locally — no API key, no
network call at rerank time. We wrap it in dsRAG's Reranker interface
and apply the same beta.cdf score transform CohereReranker uses so
Relevant Segment Extraction sees score distributions it's tuned for.
"""

from __future__ import annotations

from scipy.stats import beta

from dsrag.reranker import Reranker


class FlashRankReranker(Reranker):
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        # Lazy-import so the module is importable without flashrank installed
        # (e.g. when dsRAG reconstructs other component configs from disk).
        from flashrank import Ranker, RerankRequest

        self.model_name = model_name
        self._ranker = Ranker(model_name=model_name)
        self._RerankRequest = RerankRequest

    def transform(self, x: float) -> float:
        # Matches CohereReranker's shape (a=b=0.4) — reshapes [0,1] sigmoid
        # scores toward a more uniform distribution for RSE thresholding.
        return float(beta.cdf(x, 0.4, 0.4))

    def rerank_search_results(self, query: str, search_results: list) -> list:
        passages = [
            {
                "id": i,
                "text": (
                    f"{r['metadata']['chunk_header']}\n\n"
                    f"{r['metadata']['chunk_text']}"
                ),
            }
            for i, r in enumerate(search_results)
        ]
        req = self._RerankRequest(query=query, passages=passages)
        ranked = self._ranker.rerank(req)
        reordered = []
        for item in ranked:
            r = search_results[item["id"]]
            r["similarity"] = self.transform(float(item["score"]))
            reordered.append(r)
        return reordered

    def to_dict(self):
        base = super().to_dict()
        base.update({"model_name": self.model_name})
        return base
