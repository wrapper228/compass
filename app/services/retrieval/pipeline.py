from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import anyio
import numpy as np
from sentence_transformers import CrossEncoder

from app.core.config import Settings, get_settings
from app.services.llm_gateway import chat_completion
from app.services.retrieval.index_manager import HybridIndexManager
import structlog


@dataclass
class RetrievalCandidate:
    chunk_id: int
    dense_score: float
    bm25_score: float
    rrf_score: float
    cross_score: float = 0.0


class HybridRetrievalPipeline:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.indices = HybridIndexManager(self.settings)
        self._cross_encoder: Optional[CrossEncoder] = None
        self._rrf_k = 60
        self._logger = structlog.get_logger(__name__)

    async def retrieve(self, query: str, top_k: int) -> List[dict]:
        if not self.settings.retrieval_enabled:
            return []

        top_k = max(1, top_k)
        started = time.perf_counter()
        primary_results, meta = await self._run_once(query, top_k)
        best_results = primary_results
        best_meta = meta
        best_score = best_results[0]["score"] if best_results else 0.0

        threshold = self.settings.self_check_threshold
        iterations = self.settings.self_check_iterations

        if best_score < threshold:
            for _ in range(iterations):
                refined = await self._refine_query(query, best_results)
                if not refined or refined.strip().lower() == query.strip().lower():
                    break
                alt_results, alt_meta = await self._run_once(refined, top_k)
                if not alt_results:
                    continue
                alt_score = alt_results[0]["score"]
                if alt_score > best_score:
                    best_results = alt_results
                    best_meta = alt_meta
                    best_score = alt_score
                if best_score >= threshold:
                    break

        duration = time.perf_counter() - started
        self._logger.info(
            "retrieval.completed",
            query=query,
            top_k=top_k,
            results=len(best_results),
            score=best_score,
            duration_ms=int(duration * 1000),
            refined=best_meta.get("search_query") != query,
        )

        for item in best_results:
            meta_info = item.setdefault("meta", {})
            meta_info.setdefault("search_query", best_meta.get("search_query"))
            if best_meta.get("hyde"):
                meta_info.setdefault("hyde_snippet", best_meta["hyde"][:500])
        return best_results[:top_k]

    async def _run_once(self, query: str, top_k: int) -> Tuple[List[dict], Dict[str, str]]:
        hyde_text = await self._generate_hyde(query)

        search_queries = [query]
        if hyde_text:
            search_queries.append(hyde_text)

        query_embeddings = await anyio.to_thread.run_sync(self.indices.encode_queries, search_queries)
        base_query_embedding = query_embeddings[0]

        dense_runs: List[List[Tuple[int, float]]] = []
        for emb in query_embeddings:
            dense_runs.append(self.indices.dense_search(emb, self.settings.dense_index_top_k))

        bm25_runs: List[List[Tuple[int, float]]] = [
            self.indices.bm25_search(query, self.settings.bm25_top_k)
        ]
        if hyde_text:
            bm25_runs.append(self.indices.bm25_search(hyde_text, self.settings.bm25_top_k))

        candidates = self._combine_candidates(dense_runs, bm25_runs)
        if not candidates:
            return [], {"search_query": query, "hyde": hyde_text or ""}

        candidates_sorted = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
        rerank_limit = min(self.settings.rerank_top_k, len(candidates_sorted))
        candidates_top = candidates_sorted[:rerank_limit]

        chunk_payloads = await anyio.to_thread.run_sync(
            self.indices.fetch_chunk_payloads, [c.chunk_id for c in candidates_top]
        )
        if not chunk_payloads:
            return [], {"search_query": query, "hyde": hyde_text or ""}

        available_candidates = [c for c in candidates_top if c.chunk_id in chunk_payloads]
        if not available_candidates:
            return [], {"search_query": query, "hyde": hyde_text or ""}

        cross_scores = await self._cross_rerank(query, available_candidates, chunk_payloads)
        for candidate in available_candidates:
            if candidate.chunk_id in cross_scores:
                candidate.cross_score = cross_scores[candidate.chunk_id]

        doc_texts = [chunk_payloads[c.chunk_id]["text"] for c in available_candidates]
        doc_embeddings = await anyio.to_thread.run_sync(self.indices.encode_queries, doc_texts)

        mmr_indices = self._apply_mmr(
            base_query_embedding,
            doc_embeddings,
            lambda_=self.settings.mmr_lambda,
            top_n=min(top_k, len(available_candidates)),
        )

        final_results: List[dict] = []
        for idx in mmr_indices:
            candidate = available_candidates[idx]
            payload = chunk_payloads.get(candidate.chunk_id)
            if not payload:
                continue
            final_results.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "text": payload["text"],
                    "path": payload["path"],
                    "sha": payload["sha"],
                    "start_line": payload["start_line"],
                    "end_line": payload["end_line"],
                    "ordinal": payload["ordinal"],
                    "score": candidate.cross_score,
                    "score_details": {
                        "rerank": candidate.cross_score,
                        "rrf": candidate.rrf_score,
                        "dense": candidate.dense_score,
                        "bm25": candidate.bm25_score,
                    },
                    "meta": {"search_query": query, "hyde": hyde_text or ""},
                }
            )

        return final_results, {"search_query": query, "hyde": hyde_text or ""}

    async def _generate_hyde(self, query: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": self.settings.hyde_prompt},
            {"role": "user", "content": query},
        ]
        try:
            return await chat_completion(messages, is_complex=False)
        except Exception:
            self._logger.warning("retrieval.hyde_failed", query=query)
            return None

    async def _refine_query(self, query: str, current_results: List[dict]) -> Optional[str]:
        top_fragments = "\n\n".join(
            f"- {item.get('text','')[:200]}" for item in current_results[:2]
        )
        prompt = (
            "Ты переформулируешь поисковой запрос для более точного нахождения фактов в базе знаний. "
            "Сохрани смысл, сделай запрос более точным и добавь ключевые термины, если их не хватает. "
            "Ответь только новым запросом без пояснений."
        )
        user_msg = f"Исходный запрос:\n{query}\n\nДоступный контекст:\n{top_fragments}"
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            refined = await chat_completion(messages, is_complex=False)
            if refined:
                return refined.strip()
        except Exception:
            self._logger.warning("retrieval.refine_failed", query=query)
            return None
        return None

    def _combine_candidates(
        self,
        dense_runs: List[List[Tuple[int, float]]],
        bm25_runs: List[List[Tuple[int, float]]],
    ) -> List[RetrievalCandidate]:
        scores: Dict[int, RetrievalCandidate] = {}

        def ensure_candidate(chunk_id: int) -> RetrievalCandidate:
            if chunk_id not in scores:
                scores[chunk_id] = RetrievalCandidate(
                    chunk_id=chunk_id, dense_score=0.0, bm25_score=0.0, rrf_score=0.0
                )
            return scores[chunk_id]

        for run in dense_runs:
            for rank, (chunk_id, score) in enumerate(run):
                candidate = ensure_candidate(chunk_id)
                candidate.dense_score = max(candidate.dense_score, score)
                candidate.rrf_score += 1.0 / (self._rrf_k + rank + 1)

        for run in bm25_runs:
            for rank, (chunk_id, score) in enumerate(run):
                candidate = ensure_candidate(chunk_id)
                candidate.bm25_score = max(candidate.bm25_score, score)
                candidate.rrf_score += 1.0 / (self._rrf_k + rank + 1)

        return list(scores.values())

    async def _cross_rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        payloads: Dict[int, dict],
    ) -> Dict[int, float]:
        model = self._get_cross_encoder()
        filtered_candidates = [c for c in candidates if c.chunk_id in payloads]
        pairs = [(query, payloads[c.chunk_id]["text"]) for c in filtered_candidates]
        if not pairs:
            return {}
        scores = await anyio.to_thread.run_sync(lambda: model.predict(pairs, convert_to_numpy=True))
        return {cand.chunk_id: float(score) for cand, score in zip(filtered_candidates, scores)}

    def _get_cross_encoder(self) -> CrossEncoder:
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.settings.rerank_model_name, max_length=512)
        return self._cross_encoder

    def _apply_mmr(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        lambda_: float,
        top_n: int,
    ) -> List[int]:
        if doc_embeddings.size == 0 or top_n == 0:
            return []
        selected: List[int] = []
        candidates = list(range(doc_embeddings.shape[0]))
        query_vec = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)

        doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12
        normalized_docs = doc_embeddings / doc_norms

        while candidates and len(selected) < top_n:
            mmr_scores: Dict[int, float] = {}
            for idx in candidates:
                relevance = float(np.dot(normalized_docs[idx], query_vec))
                redundancy = 0.0
                if selected:
                    redundancy = max(
                        float(np.dot(normalized_docs[idx], normalized_docs[sel])) for sel in selected
                    )
                mmr_scores[idx] = lambda_ * relevance - (1 - lambda_) * redundancy
            best_idx = max(mmr_scores, key=mmr_scores.get)
            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected


_pipeline_instance: Optional[HybridRetrievalPipeline] = None


def get_pipeline() -> HybridRetrievalPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = HybridRetrievalPipeline()
    return _pipeline_instance

