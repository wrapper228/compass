import io
import zipfile

import pytest

from app.core.config import get_settings
from app.db import models
from app.db.session import SessionLocal
from app.services.ingestion import process_zip
from app.services.retrieval import retrieve


def _build_chat_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "chat_history_with_different_agents/agent1.md",
            "Диалог с агентом 1.\n...\nФинал: мы договорились продолжить завтра.",
        )
        z.writestr(
            "chat_history_with_different_agents/agent2.md",
            "Диалог с агентом 2.\n...\nВ конце я принял решение выйти из проекта.",
        )
    return buf.getvalue()


@pytest.mark.asyncio
async def test_folder_summary(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("INDICES_PATH", str(tmp_path / "indices"))
    monkeypatch.setenv("RETRIEVAL_ENABLED", "false")

    dataset_slug = "chat-history"
    job_id = "job-folder-test"
    zip_bytes = _build_chat_zip()

    process_zip(job_id, zip_bytes, tmp_path, dataset_slug=dataset_slug, dataset_title="Chat history")

    query = "посмотри мои диалоги в chat_history_with_different_agents и объясни, чем они закончились"
    hits = await retrieve(query, top_k=5)
    assert hits, "retrieval returned no results for folder summary"
    assert any("chat_history_with_different_agents" in (item.get("folder") or "") for item in hits)

    session = SessionLocal()
    session.query(models.KnowledgeDataset).filter_by(slug=dataset_slug).delete()
    session.commit()
    session.close()

