import io
import zipfile

from app.services.ingestion import process_zip
from app.db.session import SessionLocal
from app.db import models
from app.core.config import get_settings


def _build_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(
            "note.txt",
            "Первая строка\nВторая строка\n\nЧетвёртая строка продолжает тему.",
        )
    return buf.getvalue()


def test_process_zip_persists_documents(tmp_path, monkeypatch):
    get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setenv("RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("INDICES_PATH", str(tmp_path / "indices"))

    job_id = "job-meta-test"
    zip_bytes = _build_zip()

    result = process_zip(job_id, zip_bytes, tmp_path)
    assert result["documents"] == 1
    assert result["chunks"] >= 1

    session = SessionLocal()
    try:
        doc = session.query(models.KnowledgeDocument).filter_by(job_id=job_id).one()
        assert doc.path == "note.txt"
        assert doc.file_size > 0
        assert doc.chunks
        chunk = doc.chunks[0]
        assert chunk.start_line == 1
        assert chunk.end_line >= 2
    finally:
        session.query(models.KnowledgeDocument).filter_by(job_id=job_id).delete()
        session.commit()
        session.close()

