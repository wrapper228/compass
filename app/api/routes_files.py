import json
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.ingestion import process_zip


router = APIRouter(prefix="/api")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/files/upload")
async def upload_files(zip_file: UploadFile = File(...), background: BackgroundTasks = None) -> dict:
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Ожидается zip-архив")

    job_id = str(uuid.uuid4())
    content = await zip_file.read()

    # Сохраняем оригинал и ставим фоновую задачу на парсинг/чанкинг
    target_path = UPLOAD_DIR / f"{job_id}.zip"
    target_path.write_bytes(content)

    workspace = Path("data/tmp")
    workspace.mkdir(parents=True, exist_ok=True)
    if background is not None:
        background.add_task(process_zip, job_id, content, workspace)
    else:
        process_zip(job_id, content, workspace)

    return {"job_id": job_id}


@router.get("/ingest/{job_id}")
def ingest_status(job_id: str) -> dict:
    zip_path = UPLOAD_DIR / f"{job_id}.zip"
    workspace = Path("data/tmp") / job_id
    result_path = workspace / "ingest_result.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"job_id": job_id}
        payload.update({"status": "done"})
        return payload
    exists = zip_path.exists()
    return {"status": "queued" if exists else "not_found", "job_id": job_id}


