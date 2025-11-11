from __future__ import annotations

from typing import List

from .pipeline import get_pipeline, HybridRetrievalPipeline


async def retrieve(query: str, top_k: int = 6) -> List[dict]:
    pipeline = get_pipeline()
    return await pipeline.retrieve(query, top_k)


def get_pipeline_instance() -> HybridRetrievalPipeline:
    return get_pipeline()


def refresh_indices() -> None:
    """Reload cached index state inside the shared pipeline (used after ingestion)."""
    pipeline = get_pipeline()
    pipeline.refresh_all()


