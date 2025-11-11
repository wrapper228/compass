from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import func

from app.db import models
from app.db.session import SessionLocal


@dataclass
class DatasetInfo:
    slug: str
    title: str
    description: Optional[str]
    document_count: int
    last_ingested_at: Optional[datetime]


@dataclass
class FolderInfo:
    dataset_slug: str
    folder: str
    document_count: int


@dataclass
class DocumentDetail:
    id: int
    path: str
    name: str
    folder: str
    dataset_slug: str
    summary_text: str
    tail_text: Optional[str]
    last_chunk_text: Optional[str]
    last_start_line: Optional[int]
    last_end_line: Optional[int]


class KnowledgeRepository:
    def __init__(self) -> None:
        self._datasets: List[DatasetInfo] = []
        self._folders: List[FolderInfo] = []
        self.refresh()

    @property
    def datasets(self) -> List[DatasetInfo]:
        return self._datasets

    @property
    def folders(self) -> List[FolderInfo]:
        return self._folders

    def refresh(self) -> None:
        session = SessionLocal()
        try:
            dataset_rows = (
                session.query(
                    models.KnowledgeDataset.slug,
                    models.KnowledgeDataset.title,
                    models.KnowledgeDataset.description,
                    func.count(models.KnowledgeDocument.id),
                    models.KnowledgeDataset.last_ingested_at,
                )
                .outerjoin(models.KnowledgeDocument, models.KnowledgeDocument.dataset_id == models.KnowledgeDataset.id)
                .group_by(
                    models.KnowledgeDataset.slug,
                    models.KnowledgeDataset.title,
                    models.KnowledgeDataset.description,
                    models.KnowledgeDataset.last_ingested_at,
                )
                .order_by(models.KnowledgeDataset.updated_at.desc())
                .all()
            )
            folder_rows = (
                session.query(
                    models.KnowledgeDataset.slug,
                    models.KnowledgeDocument.folder,
                    func.count(models.KnowledgeDocument.id),
                )
                .join(models.KnowledgeDocument, models.KnowledgeDocument.dataset_id == models.KnowledgeDataset.id)
                .filter(models.KnowledgeDocument.folder != "")
                .group_by(models.KnowledgeDataset.slug, models.KnowledgeDocument.folder)
                .all()
            )
        finally:
            session.close()

        self._datasets = [
            DatasetInfo(
                slug=row[0],
                title=row[1],
                description=row[2],
                document_count=row[3] or 0,
                last_ingested_at=row[4],
            )
            for row in dataset_rows
        ]
        self._folders = [
            FolderInfo(dataset_slug=row[0], folder=row[1], document_count=row[2] or 0)
            for row in folder_rows
            if row[1]
        ]

    def match_datasets(self, text: str) -> List[str]:
        lowered = text.lower()
        slugs: List[str] = []
        for dataset in self._datasets:
            if dataset.slug.lower() in lowered or dataset.title.lower() in lowered:
                slugs.append(dataset.slug)
        return slugs

    def match_folders(self, text: str) -> List[FolderInfo]:
        lowered = text.lower()
        matches: List[FolderInfo] = []
        seen: set[tuple[str, str]] = set()
        for info in self._folders:
            folder_lower = info.folder.lower()
            short = folder_lower.split("/")[-1]
            if folder_lower in lowered or (short and short in lowered):
                key = (info.dataset_slug, info.folder)
                if key not in seen:
                    matches.append(info)
                    seen.add(key)
        return matches

    def documents_in_folder(self, dataset_slug: Optional[str], folder: str) -> List[DocumentDetail]:
        session = SessionLocal()
        try:
            query = (
                session.query(
                    models.KnowledgeDocument.id,
                    models.KnowledgeDocument.path,
                    models.KnowledgeDocument.name,
                    models.KnowledgeDocument.folder,
                    models.KnowledgeDataset.slug,
                    models.KnowledgeDocumentSummary.summary_text,
                    models.KnowledgeDocumentSummary.tail_text,
                )
                .join(models.KnowledgeDataset, models.KnowledgeDocument.dataset_id == models.KnowledgeDataset.id)
                .outerjoin(
                    models.KnowledgeDocumentSummary,
                    models.KnowledgeDocumentSummary.document_id == models.KnowledgeDocument.id,
                )
                .filter(models.KnowledgeDocument.folder == folder)
            )
            if dataset_slug:
                query = query.filter(models.KnowledgeDataset.slug == dataset_slug)
            documents = query.order_by(models.KnowledgeDocument.name).all()

            details: List[DocumentDetail] = []
            for doc in documents:
                chunk_row = (
                    session.query(
                        models.KnowledgeChunk.text,
                        models.KnowledgeChunk.start_line,
                        models.KnowledgeChunk.end_line,
                    )
                    .filter(models.KnowledgeChunk.document_id == doc.id)
                    .order_by(models.KnowledgeChunk.ordinal.desc())
                    .limit(1)
                    .one_or_none()
                )
                details.append(
                    DocumentDetail(
                        id=doc.id,
                        path=doc.path,
                        name=doc.name,
                        folder=doc.folder,
                        dataset_slug=doc.slug,
                        summary_text=doc.summary_text or "",
                        tail_text=doc.tail_text,
                        last_chunk_text=chunk_row[0] if chunk_row else None,
                        last_start_line=chunk_row[1] if chunk_row else None,
                        last_end_line=chunk_row[2] if chunk_row else None,
                    )
                )
            return details
        finally:
            session.close()

    def recent_documents(self, dataset_slug: Optional[str], limit: int = 5) -> List[DocumentDetail]:
        session = SessionLocal()
        try:
            query = (
                session.query(
                    models.KnowledgeDocument.id,
                    models.KnowledgeDocument.path,
                    models.KnowledgeDocument.name,
                    models.KnowledgeDocument.folder,
                    models.KnowledgeDataset.slug,
                    models.KnowledgeDocumentSummary.summary_text,
                    models.KnowledgeDocumentSummary.tail_text,
                )
                .join(models.KnowledgeDataset, models.KnowledgeDocument.dataset_id == models.KnowledgeDataset.id)
                .outerjoin(
                    models.KnowledgeDocumentSummary,
                    models.KnowledgeDocumentSummary.document_id == models.KnowledgeDocument.id,
                )
                .order_by(models.KnowledgeDocument.updated_at.desc())
            )
            if dataset_slug:
                query = query.filter(models.KnowledgeDataset.slug == dataset_slug)
            documents = query.limit(limit).all()

            details: List[DocumentDetail] = []
            for doc in documents:
                details.append(
                    DocumentDetail(
                        id=doc.id,
                        path=doc.path,
                        name=doc.name,
                        folder=doc.folder,
                        dataset_slug=doc.slug,
                        summary_text=doc.summary_text or "",
                        tail_text=doc.tail_text,
                        last_chunk_text=None,
                        last_start_line=None,
                        last_end_line=None,
                    )
                )
            return details
        finally:
            session.close()

    def latest_dataset_slug(self) -> Optional[str]:
        if not self._datasets:
            return None
        sorted_ds = sorted(
            self._datasets,
            key=lambda ds: ds.last_ingested_at or datetime.min,
            reverse=True,
        )
        return sorted_ds[0].slug if sorted_ds else None

