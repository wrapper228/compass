import io
import time
import zipfile

from fastapi.testclient import TestClient

from app.db.session import SessionLocal
from app.db import models
from app.main import app


def make_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('docs/note.txt', 'hello world')
    return buf.getvalue()


def test_ingest_zip_done():
    client = TestClient(app)
    files = {"zip_file": ("test.zip", make_zip_bytes(), "application/zip")}
    r = client.post(
        "/api/files/upload",
        data={"dataset": "test-ingest", "title": "Test ingest dataset"},
        files=files,
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # ждем завершения фона
    ok = False
    for _ in range(20):
        s = client.get(f"/api/ingest/{job_id}?dataset=test-ingest").json()
        if s["status"] == "done":
            ok = True
            break
        time.sleep(0.1)
    assert ok

    datasets = client.get("/api/datasets").json()
    assert any(d["slug"] == "test-ingest" for d in datasets)

    documents = client.get("/api/datasets/test-ingest/documents").json()
    assert len(documents) == 1
    assert documents[0]["path"] == "docs/note.txt"
    assert documents[0]["chunk_count"] >= 1

    session = SessionLocal()
    session.query(models.KnowledgeDataset).filter_by(slug="test-ingest").delete()
    session.commit()
    session.close()


