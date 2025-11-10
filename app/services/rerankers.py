from __future__ import annotations

from typing import Iterable, List

import structlog

from app.core.config import get_settings

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Wrapper around ``sentence-transformers`` cross-encoders with graceful fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = None
        self.available = False
        model_name = settings.cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:  # pragma: no cover - optional heavy dependency
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            self.available = True
        except Exception as exc:
            logger.warning("reranker.unavailable", error=str(exc))
            self.model = None
            self.available = False

    def score(self, query: str, passages: Iterable[str]) -> List[float]:
        docs = list(passages)
        if not docs:
            return []
        if self.available and self.model is not None:  # pragma: no cover - heavy path
            pairs = [[query, doc] for doc in docs]
            scores = self.model.predict(pairs)
            return [float(s) for s in scores]
        # Lightweight heuristic fallback: reward token overlap and brevity.
        query_terms = set(query.lower().split())
        results: List[float] = []
        for doc in docs:
            tokens = doc.lower().split()
            overlap = len(query_terms.intersection(tokens))
            length_penalty = 1.0 / (1 + len(tokens))
            results.append(overlap + length_penalty)
        return results


