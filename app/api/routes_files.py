import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks

from app.db import models
from app.db.session import SessionLocal
from app.services.ingestion import process_zip, slugify_dataset
from app.services.retrieval import get_pipeline_instance

router = APIRouter(prefix="/api")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = Path("data/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/files/upload")
async def upload_files(
    background: BackgroundTasks,
    dataset: str = Form(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    zip_file: UploadFile = File(...),
) -> dict:
    if not zip_file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Ожидается zip-архив")

    job_id = str(uuid.uuid4())
    slug = slugify_dataset(dataset)
    content = await zip_file.read()

    dataset_upload_dir = UPLOAD_DIR / slug
    dataset_upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = dataset_upload_dir / f"{job_id}.zip"
    target_path.write_bytes(content)

    background.add_task(process_zip, job_id, content, TMP_DIR, slug, title, description)

    return {"job_id": job_id, "dataset": slug}


@router.get("/ingest/{job_id}")
def ingest_status(job_id: str, dataset: Optional[str] = None) -> dict:
    result_path = _resolve_result_path(job_id, dataset)
    if result_path and result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"job_id": job_id}
        payload.update({"status": "done"})
        return payload

    queued = _resolve_upload_path(job_id, dataset)
    return {"status": "queued" if queued and queued.exists() else "not_found", "job_id": job_id}


def _resolve_result_path(job_id: str, dataset: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if dataset:
        candidates.append(TMP_DIR / dataset / job_id / "ingest_result.json")
    candidates.append(TMP_DIR / job_id / "ingest_result.json")  # legacy layout
    for dataset_dir in TMP_DIR.iterdir():
        if dataset_dir.is_dir():
            candidates.append(dataset_dir / job_id / "ingest_result.json")
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _resolve_upload_path(job_id: str, dataset: Optional[str]) -> Optional[Path]:
    candidates: list[Path] = []
    if dataset:
        candidates.append(UPLOAD_DIR / dataset / f"{job_id}.zip")
    candidates.append(UPLOAD_DIR / f"{job_id}.zip")
    for dataset_dir in UPLOAD_DIR.iterdir():
        if dataset_dir.is_dir():
            candidates.append(dataset_dir / f"{job_id}.zip")
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


@router.get("/datasets")
def list_datasets() -> list[dict]:
    session = SessionLocal()
    try:
        rows = (
            session.query(
                models.KnowledgeDataset.slug,
                models.KnowledgeDataset.title,
                models.KnowledgeDataset.description,
                models.KnowledgeDataset.total_documents,
                models.KnowledgeDataset.total_chunks,
                models.KnowledgeDataset.total_tokens,
                models.KnowledgeDataset.last_ingested_at,
            )
            .order_by(models.KnowledgeDataset.updated_at.desc())
            .all()
        )
    finally:
        session.close()

    def iso(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    return [
        {
            "slug": row[0],
            "title": row[1],
            "description": row[2],
            "documents": row[3] or 0,
            "chunks": row[4] or 0,
            "tokens": row[5] or 0,
            "last_ingested_at": iso(row[6]),
        }
        for row in rows
    ]


@router.get("/datasets/{slug}/documents")
def list_documents(slug: str) -> list[dict]:
    session = SessionLocal()
    try:
        rows = (
            session.query(
                models.KnowledgeDocument.id,
                models.KnowledgeDocument.path,
                models.KnowledgeDocument.name,
                models.KnowledgeDocument.folder,
                models.KnowledgeDocument.chunk_count,
                models.KnowledgeDocument.token_count,
                models.KnowledgeDocumentSummary.summary_text,
                models.KnowledgeDocumentSummary.tail_text,
                models.KnowledgeDocument.updated_at,
            )
            .join(models.KnowledgeDataset, models.KnowledgeDocument.dataset_id == models.KnowledgeDataset.id)
            .outerjoin(
                models.KnowledgeDocumentSummary,
                models.KnowledgeDocumentSummary.document_id == models.KnowledgeDocument.id,
            )
            .filter(models.KnowledgeDataset.slug == slug)
            .order_by(models.KnowledgeDocument.name)
            .all()
        )
    finally:
        session.close()

    def iso(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    return [
        {
            "id": row[0],
            "path": row[1],
            "name": row[2],
            "folder": row[3],
            "chunk_count": row[4] or 0,
            "token_count": row[5] or 0,
            "summary": row[6] or "",
            "tail": row[7] or "",
            "updated_at": iso(row[8]),
        }
        for row in rows
    ]


@router.post("/datasets/rebuild")
def rebuild_indices() -> dict:
    pipeline = get_pipeline_instance()
    pipeline.indices.rebuild(source="manual")
    pipeline.repo.refresh()
    return {"status": "ok"}

