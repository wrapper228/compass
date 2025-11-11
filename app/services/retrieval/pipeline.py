from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import anyio
import numpy as np
import re
import structlog
import time
from sentence_transformers import CrossEncoder

from app.core.config import Settings, get_settings
from app.services.llm_gateway import chat_completion
from app.services.knowledge import KnowledgeRepository
from app.services.retrieval.index_manager import HybridIndexManager

_slug_pattern = re.compile(r"[^a-z0-9\-]+")


def _slugify(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    normalized = _slug_pattern.sub("-", normalized)
    normalized = normalized.strip("-")
    return normalized or value.strip().lower()


@dataclass
class RetrievalCandidate:
    chunk_id: int
    dense_score: float
    bm25_score: float
    rrf_score: float
    cross_score: float = 0.0


@dataclass
class FolderFilter:
    folder: str
    dataset: Optional[str] = None


@dataclass
class RetrievalIntent:
    original_query: str
    cleaned_query: str
    dataset_filters: List[str]
    folder_filters: List[FolderFilter]
    mode: Literal["standard", "folder_summary"]


class HybridRetrievalPipeline:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.indices = HybridIndexManager(self.settings)
        self.repo = KnowledgeRepository()
        self._cross_encoder: Optional[CrossEncoder] = None
        self._rrf_k = 60
        self._logger = structlog.get_logger(__name__)

    async def retrieve(self, query: str, top_k: int) -> List[dict]:
        intent = self._analyze_intent(query)
        if not self.settings.retrieval_enabled and intent.mode != "folder_summary":
            return []
        top_k = max(1, top_k)

        if intent.mode == "folder_summary":
            results = await self._retrieve_folder_summary(intent)
            self._logger.info(
                "retrieval.completed",
                query=query,
                top_k=top_k,
                results=len(results),
                score=1.0 if results else 0.0,
                duration_ms=0,
                refined=False,
                mode="folder_summary",
            )
            return results

        return await self._retrieve_standard(intent, top_k)

    def refresh_all(self) -> None:
        self.indices.refresh()
        self.repo.refresh()

    async def _retrieve_standard(self, intent: RetrievalIntent, top_k: int) -> List[dict]:
        started = time.perf_counter()
        best_results, best_meta = await self._run_standard_once(intent, top_k)
        best_score = best_results[0]["score"] if best_results else 0.0

        baseline_query = intent.cleaned_query or intent.original_query
        threshold = self.settings.self_check_threshold
        iterations = self.settings.self_check_iterations

        if best_score < threshold:
            for _ in range(iterations):
                refined_query = await self._refine_query(intent, best_results)
                if not refined_query:
                    break
                if refined_query.strip().lower() == baseline_query.strip().lower():
                    break
                refined_intent = RetrievalIntent(
                    original_query=refined_query,
                    cleaned_query=refined_query,
                    dataset_filters=intent.dataset_filters,
                    folder_filters=intent.folder_filters,
                    mode="standard",
                )
                alt_results, alt_meta = await self._run_standard_once(refined_intent, top_k)
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
        refined_flag = best_meta.get("search_query") != baseline_query.strip()
        self._logger.info(
            "retrieval.completed",
            query=intent.original_query,
            top_k=top_k,
            results=len(best_results),
            score=best_score,
            duration_ms=int(duration * 1000),
            refined=refined_flag,
            mode="standard",
        )

        hyde_snippet = best_meta.get("hyde") or ""
        for item in best_results:
            meta_info = item.setdefault("meta", {})
            meta_info.setdefault("search_query", best_meta.get("search_query"))
            if hyde_snippet:
                meta_info.setdefault("hyde_snippet", hyde_snippet[:500])
        if best_results:
            return best_results[:top_k]

        dataset_overview = self._dataset_overview(intent)
        if dataset_overview:
            self._logger.info(
                "retrieval.dataset_overview",
                query=intent.original_query,
                dataset=dataset_overview[0]["meta"].get("dataset") if dataset_overview else None,
                results=len(dataset_overview),
            )
            return dataset_overview
        return []

    async def _run_standard_once(
        self, intent: RetrievalIntent, top_k: int
    ) -> Tuple[List[dict], Dict[str, str]]:
        base_query = (intent.cleaned_query or intent.original_query).strip()
        augmented_query = self._augment_query(base_query, intent)
        hyde_text = await self._generate_hyde(augmented_query)

        search_queries = [augmented_query]
        if hyde_text:
            search_queries.append(hyde_text)

        query_embeddings = await anyio.to_thread.run_sync(self.indices.encode_queries, search_queries)
        base_query_embedding = query_embeddings[0]

        dense_runs: List[List[Tuple[int, float]]] = [
            self.indices.dense_search(emb, self.settings.dense_index_top_k) for emb in query_embeddings
        ]

        bm25_runs: List[List[Tuple[int, float]]] = [
            self.indices.bm25_search(augmented_query, self.settings.bm25_top_k)
        ]
        if hyde_text:
            bm25_runs.append(self.indices.bm25_search(hyde_text, self.settings.bm25_top_k))

        candidates = [
            candidate
            for candidate in self._combine_candidates(dense_runs, bm25_runs)
            if self._candidate_allowed(intent, candidate.chunk_id)
        ]
        if not candidates:
            return [], {"search_query": augmented_query, "hyde": hyde_text or ""}

        candidates_sorted = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)
        rerank_limit = min(self.settings.rerank_top_k, len(candidates_sorted))
        candidates_top = candidates_sorted[:rerank_limit]

        chunk_payloads = await anyio.to_thread.run_sync(
            self.indices.fetch_chunk_payloads, [c.chunk_id for c in candidates_top]
        )
        if not chunk_payloads:
            return [], {"search_query": augmented_query, "hyde": hyde_text or ""}

        available_candidates = [
            c for c in candidates_top if c.chunk_id in chunk_payloads and self._candidate_allowed(intent, c.chunk_id)
        ]
        if not available_candidates:
            return [], {"search_query": augmented_query, "hyde": hyde_text or ""}

        cross_scores = await self._cross_rerank(augmented_query, available_candidates, chunk_payloads)
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
            dataset_slug = payload.get("dataset") or ""
            folder = payload.get("folder") or ""
            final_results.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "text": payload["text"],
                    "path": payload["path"],
                    "dataset": dataset_slug,
                    "folder": folder,
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
                        "dataset": dataset_slug,
                    },
                    "meta": {
                        "search_query": augmented_query,
                        "hyde": hyde_text or "",
                        "dataset": dataset_slug,
                        "folder": folder,
                        "document": payload.get("document_name"),
                    },
                }
            )

        return final_results, {"search_query": augmented_query, "hyde": hyde_text or ""}

    async def _retrieve_folder_summary(self, intent: RetrievalIntent) -> List[dict]:
        results: List[dict] = []
        seen_docs: set[int] = set()
        folder_filters = intent.folder_filters
        if not folder_filters:
            return results

        max_items = max(3, self.settings.dense_index_top_k // 2)

        for folder_filter in folder_filters:
            docs = self.repo.documents_in_folder(folder_filter.dataset, folder_filter.folder)
            for doc in docs:
                if doc.id in seen_docs:
                    continue
                seen_docs.add(doc.id)
                tail = doc.tail_text or doc.last_chunk_text or doc.summary_text
                if not tail:
                    continue
                results.append(
                    {
                        "chunk_id": -doc.id,
                        "text": tail,
                        "path": doc.path,
                        "dataset": doc.dataset_slug,
                        "folder": doc.folder,
                        "score": 1.0,
                        "score_details": {"mode": "folder_summary"},
                        "meta": {
                            "search_query": intent.cleaned_query or intent.original_query,
                            "dataset": doc.dataset_slug,
                            "folder": doc.folder,
                            "document": doc.name,
                            "summary": doc.summary_text,
                        },
                        "start_line": doc.last_start_line,
                        "end_line": doc.last_end_line,
                    }
                )
                if len(results) >= max_items:
                    return results
        return results

    def _dataset_overview(self, intent: RetrievalIntent) -> List[dict]:
        dataset_slug = intent.dataset_filters[0] if intent.dataset_filters else self.repo.latest_dataset_slug()
        documents = self.repo.recent_documents(dataset_slug, limit=5)
        if not documents:
            return []
        overview: List[dict] = []
        for doc in documents:
            text = doc.summary_text or doc.tail_text or doc.last_chunk_text
            if not text:
                continue
            overview.append(
                {
                    "chunk_id": -doc.id,
                    "text": text,
                    "path": doc.path,
                    "dataset": doc.dataset_slug,
                    "folder": doc.folder,
                    "score": 0.8,
                    "meta": {
                        "mode": "dataset_overview",
                        "dataset": doc.dataset_slug,
                        "folder": doc.folder,
                        "document": doc.name,
                    },
                }
            )
        return overview

    def _analyze_intent(self, query: str) -> RetrievalIntent:
        cleaned = query
        lowered = query.lower()
        dataset_filters: List[str] = []
        folder_filters: List[FolderFilter] = []

        dataset_pattern = re.compile(r"(?:dataset|датасет):([^\s,;]+)", re.IGNORECASE)
        folder_pattern = re.compile(r"(?:folder|папка):([^\s,;]+)", re.IGNORECASE)

        explicit_datasets = dataset_pattern.findall(query)
        explicit_folders = folder_pattern.findall(query)

        cleaned = dataset_pattern.sub(" ", cleaned)
        cleaned = folder_pattern.sub(" ", cleaned)

        if explicit_datasets:
            dataset_filters.extend(_slugify(ds) for ds in explicit_datasets)

        inferred_datasets = self.repo.match_datasets(query)
        for slug in inferred_datasets:
            if slug not in dataset_filters:
                dataset_filters.append(slug)

        inferred_folders = self.repo.match_folders(query)
        for info in inferred_folders:
            folder_filters.append(FolderFilter(folder=info.folder, dataset=info.dataset_slug))

        for folder_token in explicit_folders:
            # try to align with known folders; if none, keep dataset unknown
            matched = False
            token_lower = folder_token.lower()
            for info in inferred_folders:
                if info.folder.lower().endswith(token_lower):
                    matched = True
                    break
            if not matched:
                folder_filters.append(FolderFilter(folder=folder_token))

        # Deduplicate folder filters
        unique_filters: Dict[tuple[str, str], FolderFilter] = {}
        for ff in folder_filters:
            key = ((ff.dataset or "").lower(), ff.folder.lower())
            if key not in unique_filters:
                unique_filters[key] = ff
        folder_filters = list(unique_filters.values())

        cleaned_query = " ".join(cleaned.split())
        if not cleaned_query:
            cleaned_query = query.strip()

        folder_summary_hints = ("конце", "финал", "заключ", "послед", "законч", "summary", "обзор", "подытож")
        mode: Literal["standard", "folder_summary"] = "standard"
        if folder_filters and any(hint in lowered for hint in folder_summary_hints):
            mode = "folder_summary"

        return RetrievalIntent(
            original_query=query,
            cleaned_query=cleaned_query,
            dataset_filters=dataset_filters,
            folder_filters=folder_filters,
            mode=mode,
        )

    def _candidate_allowed(self, intent: RetrievalIntent, chunk_id: int) -> bool:
        if not intent.dataset_filters and not intent.folder_filters:
            return True
        meta = self.indices.metadata_for_id(chunk_id)
        if not meta:
            return not intent.dataset_filters and not intent.folder_filters
        dataset_slug = (meta.get("dataset") or "").lower()
        folder = (meta.get("folder") or "").lower()
        path = (meta.get("path") or "").lower()
        if intent.dataset_filters and dataset_slug not in [ds.lower() for ds in intent.dataset_filters]:
            return False
        if not intent.folder_filters:
            return True
        for folder_filter in intent.folder_filters:
            if self._folder_matches(folder_filter, dataset_slug, folder, path):
                return True
        return False

    def _folder_matches(
        self,
        folder_filter: FolderFilter,
        dataset_slug: str,
        folder: str,
        path: str,
    ) -> bool:
        if folder_filter.dataset and dataset_slug and dataset_slug != folder_filter.dataset.lower():
            return False
        target = folder_filter.folder.lower()
        if not target:
            return False
        if folder.endswith(target) or folder == target:
            return True
        if target in path:
            return True
        return False

    def _augment_query(self, base_query: str, intent: RetrievalIntent) -> str:
        parts = [base_query]
        if intent.folder_filters:
            parts.append(" ".join(f.folder for f in intent.folder_filters if f.folder))
        if intent.dataset_filters:
            parts.append(" ".join(intent.dataset_filters))
        augmented = " ".join(part for part in parts if part).strip()
        return augmented or base_query

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

    async def _refine_query(self, intent: RetrievalIntent, current_results: List[dict]) -> Optional[str]:
        top_fragments = "\n\n".join(
            f"- {item.get('text','')[:200]}" for item in current_results[:2]
        )
        prompt = (
            "Ты переформулируешь поисковой запрос для более точного нахождения фактов в базе знаний. "
            "Сохрани смысл, сделай запрос более точным и добавь ключевые термины, если их не хватает. "
            "Ответь только новым запросом без пояснений."
        )
        filters_hint = ""
        if intent.dataset_filters:
            filters_hint += f"\nОграничения по датасетам: {', '.join(intent.dataset_filters)}"
        if intent.folder_filters:
            filters_hint += "\nОграничения по папкам: " + ", ".join(ff.folder for ff in intent.folder_filters)
        user_msg = (
            f"Исходный запрос:\n{intent.cleaned_query or intent.original_query}"
            f"{filters_hint}\n\nДоступный контекст:\n{top_fragments}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            refined = await chat_completion(messages, is_complex=False)
            if refined:
                return refined.strip()
        except Exception:
            self._logger.warning(
                "retrieval.refine_failed", query=intent.original_query
            )
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

