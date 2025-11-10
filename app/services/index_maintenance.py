from __future__ import annotations

from pathlib import Path

import structlog

from app.core.config import get_settings
from app.services.embeddings import embed_batch_sync
from app.services.indexes import DenseVector, IndexRepository

logger = structlog.get_logger(__name__)


def rebuild_all_indexes() -> None:
    settings = get_settings()
    repo = IndexRepository(Path(settings.retrieval_index_path))
    chunks = repo.list_chunks()
    if not chunks:
        logger.info("index_maintenance.no_chunks")
        return
    logger.info("index_maintenance.rebuild_start", chunks=len(chunks))
    repo.dense_index.reset()
    embeddings = embed_batch_sync([chunk.text for chunk in chunks])
    repo.ensure_dense_vectors(
        DenseVector(chunk_id=chunk.chunk_id, vector=embeddings[idx])
        for idx, chunk in enumerate(chunks)
    )
    repo.ensure_sparse_index()
    logger.info("index_maintenance.rebuild_done", chunks=len(chunks))


def schedule_monthly_rebuild(scheduler) -> None:
    def job() -> None:
        try:
            rebuild_all_indexes()
        except Exception as exc:  # pragma: no cover - scheduler safety
            logger.warning("index_maintenance.rebuild_failed", error=str(exc))

    scheduler.add_job(job, "cron", day=1, hour=2, minute=0)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    rebuild_all_indexes()


