from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from app.core.config import get_settings
from app.schemas.chat import RetrievalOptions
from app.services.context_builder import build_context
from app.services.retrieval_pipeline import PipelineOptions, RetrievalPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline()


async def retrieve_for_text(text: str, options: RetrievalOptions | None) -> Dict:
    settings = get_settings()
    if not settings.retrieval_enabled:
        return {"hits": [], "context_prompt": "", "clusters": [], "iterations": 0, "query_variants": []}
    pipeline = get_pipeline()
    top_k = options.top_k if options else 6
    max_iterations = options.max_iterations if options else None
    result = await pipeline.retrieve(
        text,
        PipelineOptions(
            top_k=top_k,
            filters=options.filters if options else None,
            max_iterations=max_iterations,
        ),
    )
    context = await build_context(result.hits)
    return {
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "text": hit.text,
                "citation": hit.citation,
                "path": hit.path,
                "metadata": hit.metadata,
            }
            for hit in result.hits
        ],
        "context_prompt": context.prompt,
        "clusters": [
            {
                "path": cluster.path,
                "summary": cluster.summary,
                "quotes": [
                    {"text": quote.text, "citation": quote.citation}
                    for quote in cluster.quotes
                ],
            }
            for cluster in context.clusters
        ],
        "iterations": result.iterations,
        "query_variants": result.query_variants,
    }


