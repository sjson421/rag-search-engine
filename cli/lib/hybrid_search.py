import os
from collections import defaultdict

from constants.constants import BM25_B, BM25_K1
from keyword_search_cli import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from loader import load_movies


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.chunked_semantic_search = ChunkedSemanticSearch()
        self.chunked_semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.INDEX_PATH):
            movies = load_movies()
            self.idx.build(movies)
            self.idx.save()

    def normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []

        max_score = float("-inf")
        min_score = float("inf")

        for score in scores:
            min_score = min(min_score, score)
            max_score = max(max_score, score)
        diff = max_score - min_score
        if diff == 0:
            return [1.0] * len(scores)

        new_scores = []
        for score in scores:
            new_scores.append((score - min_score) / diff)
        return new_scores

    def _bm25_search(self, query: str, limit: int) -> list:
        self.idx.load()
        return self.idx.bm25_search(query, limit, BM25_K1, BM25_B)

    def _get_hybrid_score(self, alpha, keyword_score, semantic_score):
        return alpha * keyword_score + (1 - alpha) * semantic_score

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25s = self._bm25_search(query, limit * 500)
        sems = self.chunked_semantic_search.search_chunks(query, limit * 500)

        scores = defaultdict(dict)
        for doc in self.documents:
            scores[doc["id"]]["document"] = doc
        norm_bm25s = self.normalize([score for _, score in bm25s])
        for (doc_id, _), score in zip(bm25s, norm_bm25s):
            scores[doc_id]["bm25"] = score
        norm_sems = self.normalize([search_res["score"] for search_res in sems])
        for search_res, score in zip(sems, norm_sems):
            scores[search_res["id"]]["semantic"] = score

        for doc_id, score_dict in scores.items():
            score_dict["hybrid"] = self._get_hybrid_score(
                alpha, score_dict.get("bm25", 0.0), score_dict.get("semantic", 0.0)
            )
        return sorted(scores.values(), key=lambda x: x["hybrid"], reverse=True)[:limit]

    def rrf_search(self, query: str, k: int = 60, limit: int = 10) -> list[dict]:
        """
        Keys: document, bm25, semantic, rrf
        """
        bm25s = self._bm25_search(query, limit * 500)
        sems = self.chunked_semantic_search.search_chunks(query, limit * 500)
        scores = defaultdict(dict)

        for doc in self.documents:
            scores[doc["id"]]["document"] = doc
        for i, (doc_id, _) in enumerate(bm25s):
            scores[doc_id]["bm25"] = i
        for i, search_res in enumerate(sems):
            scores[search_res["id"]]["semantic"] = i

        for doc_id, scores_dict in scores.items():
            bm25_rrf = rrf_score(scores_dict.get("bm25", k))
            sem_rrf = rrf_score(scores_dict.get("semantic", k))
            scores_dict["rrf"] = bm25_rrf + sem_rrf
        return sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)[:limit]


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)
