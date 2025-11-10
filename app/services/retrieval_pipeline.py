from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import structlog

from app.core.config import get_settings
from app.services.embeddings import get_embedding
from app.services.indexes import IndexRepository
from app.services.rerankers import CrossEncoderReranker
from app.services.llm_gateway import chat_completion

logger = structlog.get_logger(__name__)


@dataclass
class PipelineOptions:
    top_k: int
    filters: Optional[Dict] = None
    max_iterations: Optional[int] = None


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    text: str
    citation: str
    path: str
    metadata: Dict[str, int]


@dataclass
class RetrievalResult:
    hits: List[RetrievalHit]
    iterations: int
    query_variants: List[str]


class RetrievalPipeline:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        base_path = Path(settings.retrieval_index_path)
        self.index_repo = IndexRepository(base_path)
        self.reranker = CrossEncoderReranker()

    async def retrieve(self, query: str, options: PipelineOptions) -> RetrievalResult:
        max_iterations = options.max_iterations or self.settings.retrieval_self_check_max_iterations
        query_variants: List[str] = []
        all_hits: List[RetrievalHit] = []
        iterations = 0
        augment = ""
        while iterations < max_iterations:
            iterations += 1
            augmented_query = query if not augment else f"{query}\n\nКонтекст: {augment}"
            query_variants.append(augmented_query)
            hits = await self._retrieve_once(augmented_query, options.top_k)
            all_hits = hits
            if len(hits) >= options.top_k or not hits:
                break
            augment = " \n".join(hit.text[:200] for hit in hits)
        return RetrievalResult(hits=all_hits, iterations=iterations, query_variants=query_variants)

    async def _retrieve_once(self, query: str, top_k: int) -> List[RetrievalHit]:
        hypothesis = await self._generate_hypothesis(query)
        dense_query_vec = await get_embedding(hypothesis)
        dense_hits, sparse_hits = self.index_repo.search(
            query_vector=dense_query_vec,
            query_text=f"{query}\n{hypothesis}",
            top_k_dense=self.settings.retrieval_dense_top_k,
            top_k_sparse=self.settings.retrieval_sparse_top_k,
        )
        fused = self._rrf_fusion(dense_hits, sparse_hits)
        if not fused:
            return []
        candidate_ids = [cid for cid, _ in fused[: max(top_k * 3, 20)]]
        chunks = [self.index_repo.fetch_chunk(cid) for cid in candidate_ids]
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            return []
        rerank_scores = self.reranker.score(query, [c.text for c in chunks])
        combined_scores = {}
        for chunk, rerank in zip(chunks, rerank_scores):
            base = next((score for cid, score in fused if cid == chunk.chunk_id), 0.0)
            combined_scores[chunk.chunk_id] = base + float(rerank)
        mmr_order = self._mmr_order(candidate_ids, combined_scores, top_k)
        hits: List[RetrievalHit] = []
        for cid in mmr_order[:top_k]:
            chunk = self.index_repo.fetch_chunk(cid)
            if not chunk:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=combined_scores.get(chunk.chunk_id, 0.0),
                    text=chunk.text,
                    citation=chunk.short_citation,
                    path=chunk.path,
                    metadata={
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "token_count": chunk.token_count,
                    },
                )
            )
        return hits

    async def _generate_hypothesis(self, query: str) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Ты помогаешь подготовить поисковый запрос. Дан текст пользователя. "
                    "Сформулируй короткий абстрактный ответ (1-2 предложения), "
                    "который мог бы встретиться в базе знаний."
                ),
            },
            {
                "role": "user",
                "content": f"Запрос: {query}\n\nОтвет:",
            },
        ]
        try:
            hypo = await chat_completion(prompt, is_complex=False)
            if hypo:
                return hypo.strip()
        except Exception as exc:  # pragma: no cover - network error fallback
            logger.warning("retrieval.hyde_failed", error=str(exc))
        return query

    def _rrf_fusion(
        self,
        dense_hits: Sequence[Tuple[str, float]],
        sparse_hits: Sequence[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        k = self.settings.retrieval_rrf_k
        scores: Dict[str, float] = {}
        for rank, (cid, _) in enumerate(dense_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        for rank, (cid, _) in enumerate(sparse_hits):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    def _mmr_order(self, candidates: List[str], scores: Dict[str, float], top_k: int) -> List[str]:
        lambda_coeff = self.settings.retrieval_mmr_lambda
        selected: List[str] = []
        remaining = list(dict.fromkeys(candidates))
        while remaining and len(selected) < top_k:
            best_id = None
            best_score = float("-inf")
            for cid in remaining:
                relevance = scores.get(cid, 0.0)
                diversity = 0.0
                if selected:
                    vec = self.index_repo.fetch_vector(cid) or []
                    diversity = max(
                        self._cosine(vec, self.index_repo.fetch_vector(sel) or [])
                        for sel in selected
                    )
                mmr_score = lambda_coeff * relevance - (1 - lambda_coeff) * diversity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_id = cid
            if best_id is None:
                break
            selected.append(best_id)
            remaining.remove(best_id)
        return selected

    @staticmethod
    def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            if len(vec_a) < len(vec_b):
                vec_a = list(vec_a) + [0.0] * (len(vec_b) - len(vec_a))
            else:
                vec_b = list(vec_b) + [0.0] * (len(vec_a) - len(vec_b))
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


