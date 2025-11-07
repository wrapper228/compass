import io
import time
import zipfile

from fastapi.testclient import TestClient

from app.main import app


def make_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        z.writestr('docs/note.txt', 'hello world')
    return buf.getvalue()


def test_ingest_zip_done():
    client = TestClient(app)
    files = {"zip_file": ("test.zip", make_zip_bytes(), "application/zip")}
    r = client.post("/api/files/upload", files=files)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # ждем завершения фона
    ok = False
    for _ in range(20):
        s = client.get(f"/api/ingest/{job_id}").json()
        if s["status"] == "done":
            ok = True
            break
        time.sleep(0.1)
    assert ok


