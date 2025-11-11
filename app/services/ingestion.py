from __future__ import annotations

import csv as _csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import structlog
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import get_settings
from app.db import models
from app.db.session import SessionLocal
from app.services.retrieval.chunker import Chunk, chunk_document, deduplicate_chunks
from app.services.retrieval.index_manager import ChunkIndexInput, HybridIndexManager
from app.services.retrieval import refresh_indices

logger = structlog.get_logger(__name__)


@dataclass
class SourceDocument:
    path: str
    text: str
    sha256: str
    file_size: int


def process_zip(job_id: str, zip_bytes: bytes, workspace: Path) -> dict:
    settings = get_settings()
    base_dir = workspace / job_id
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    result_path = base_dir / "ingest_result.json"
    legacy_marker = base_dir / "chunks.json"

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        archive.extractall(raw_dir)

    documents = list(_collect_documents(raw_dir))
    if not documents:
        result = {"job_id": job_id, "documents": 0, "chunks": 0, "indexed": 0}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        legacy_marker.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("ingest.no_documents", job_id=job_id)
        return result

    manager = HybridIndexManager(settings)
    session = SessionLocal()
    indexed_inputs: List[ChunkIndexInput] = []
    chunk_total = 0

    logger.info("ingest.started", job_id=job_id, documents=len(documents))

    try:
        for doc in documents:
            chunks = _chunk_document(doc, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
            if not chunks:
                continue

            # Унифицируем ординалы после возможной дедупликации
            for idx, chunk in enumerate(chunks):
                chunk.ordinal = idx

            # Заменяем предыдущие версии документа по пути
            removed_chunk_ids = _delete_existing_document(session, doc.path)
            if removed_chunk_ids and settings.retrieval_enabled:
                manager.remove_chunks(removed_chunk_ids, source=f"job:{job_id}", session=session)

            doc_row = models.KnowledgeDocument(
                job_id=job_id,
                path=doc.path,
                sha256=doc.sha256,
                file_size=doc.file_size,
            )
            session.add(doc_row)
            session.flush()

            chunk_rows: List[tuple[models.KnowledgeChunk, Chunk]] = []
            for chunk in chunks:
                chunk_row = models.KnowledgeChunk(
                    document_id=doc_row.id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    token_count=chunk.token_count,
                    sha256=chunk.sha256,
                )
                session.add(chunk_row)
                chunk_rows.append((chunk_row, chunk))

            session.flush()
            for row, chunk in chunk_rows:
                indexed_inputs.append(
                    ChunkIndexInput(
                        chunk_id=row.id,
                        text=chunk.text,
                        document_path=doc.path,
                        document_sha=doc.sha256,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                    )
                )
            chunk_total += len(chunk_rows)

        session.flush()

        indexed = 0
        if indexed_inputs and settings.retrieval_enabled:
            indexed = manager.index_chunks(indexed_inputs, source=f"job:{job_id}", session=session)

        session.commit()
        logger.info(
            "ingest.completed",
            job_id=job_id,
            documents=len(documents),
            chunks=chunk_total,
            indexed=indexed,
        )
    except Exception:
        session.rollback()
        if settings.retrieval_enabled:
            try:
                manager.rebuild(source="rollback")
            except Exception:
                logger.exception("ingest.rebuild_failed", job_id=job_id)
        logger.exception("ingest.failed", job_id=job_id)
        raise
    finally:
        session.close()

    result = {"job_id": job_id, "documents": len(documents), "chunks": chunk_total, "indexed": indexed}
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    result_path.write_text(payload, encoding="utf-8")
    legacy_marker.write_text(payload, encoding="utf-8")
    try:
        refresh_indices()
    except Exception:
        logger.exception("ingest.refresh_failed", job_id=job_id)
    return result


def _collect_documents(raw_dir: Path) -> Iterable[SourceDocument]:
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(raw_dir).as_posix()
        text = _extract_text(path)
        if not text:
            continue
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        yield SourceDocument(path=rel_path, text=text, sha256=sha, file_size=path.stat().st_size)


def _extract_text(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    try:
        if ext in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if ext == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return json.dumps(data, ensure_ascii=False, indent=2)
        if ext == ".csv":
            rows: List[str] = []
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                reader = _csv.reader(fh)
                for row in reader:
                    rows.append(", ".join(row))
            return "\n".join(rows)
        if ext in {".html", ".htm"}:
            html = path.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "lxml")
            return soup.get_text("\n", strip=True)
        if ext == ".docx":
            doc = DocxDocument(path)
            return "\n".join(para.text for para in doc.paragraphs)
        if ext == ".pdf":
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
    except Exception:
        return None
    return None


def _chunk_document(doc: SourceDocument, chunk_size: int, overlap: int) -> List[Chunk]:
    chunks = chunk_document(doc.text, chunk_size=chunk_size, overlap=overlap)
    return deduplicate_chunks(chunks)


def _delete_existing_document(session: Session, path: str) -> List[int]:
    stmt = (
        select(models.KnowledgeDocument)
        .options(selectinload(models.KnowledgeDocument.chunks))
        .where(models.KnowledgeDocument.path == path)
    )
    doc = session.execute(stmt).scalar_one_or_none()
    if doc is None:
        return []
    chunk_ids = [chunk.id for chunk in doc.chunks]
    session.delete(doc)
    session.flush()
    return chunk_ids

