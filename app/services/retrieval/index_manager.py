from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import structlog

from app.core.config import Settings, get_settings
from app.db import models
from app.db.session import SessionLocal


@dataclass
class ChunkIndexInput:
    chunk_id: int
    text: str
    document_path: str
    document_sha: str
    start_line: int
    end_line: int


class HybridIndexManager:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.base_path = Path(self.settings.indices_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.dense_index_path = self.base_path / "dense.index"
        self.metadata_path = self.base_path / "metadata.json"
        self.bm25_path = self.base_path / "bm25.json"
        self._logger = structlog.get_logger(__name__)

        self._metadata: Dict[int, dict] = {}
        self._bm25_tokens: Dict[int, List[str]] = {}
        self._bm25_model: Optional[BM25Okapi] = None
        self._bm25_ids: List[int] = []
        self._bm25_corpus: List[List[str]] = []

        self._dense_index: Optional[faiss.Index] = None
        self._dense_dim: Optional[int] = None

        self._model: Optional[SentenceTransformer] = None
        self._token_pattern = re.compile(r"[0-9A-Za-zА-Яа-я_]+", re.UNICODE)

        self._load_state()

    # ------------------------------------------------------------------ public API
    def index_chunks(self, chunks: Iterable[ChunkIndexInput], source: Optional[str] = None) -> int:
        if not self.settings.retrieval_enabled:
            return 0

        chunk_list = [c for c in chunks]
        if not chunk_list:
            return 0

        model = self._ensure_model()
        embeddings = model.encode(
            [c.text for c in chunk_list],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self._ensure_dense_index(dim)

        ids = np.array([c.chunk_id for c in chunk_list], dtype=np.int64)

        # Удаляем дубликаты (переиндексация)
        existing_ids = [cid for cid in ids if cid in self._metadata]
        if existing_ids:
            self._remove_from_dense(existing_ids)

        self._dense_index.add_with_ids(embeddings, ids)

        for chunk in chunk_list:
            self._metadata[chunk.chunk_id] = {
                "path": chunk.document_path,
                "sha": chunk.document_sha,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
            self._bm25_tokens[chunk.chunk_id] = self._tokenize(chunk.text)

        self._persist_dense_index()
        self._persist_metadata()
        self._persist_bm25_tokens()
        self._rebuild_bm25_model()
        self._record_version(source or "update")
        self._logger.info(
            "indices.upsert",
            added=len(chunk_list),
            dense_total=self._dense_index.ntotal if self._dense_index else 0,
            bm25_docs=len(self._bm25_tokens),
        )

        return len(chunk_list)

    def remove_chunks(self, chunk_ids: Iterable[int], source: Optional[str] = None) -> int:
        ids = [int(cid) for cid in chunk_ids if cid in self._metadata]
        if not ids:
            return 0

        self._remove_from_dense(ids)

        for cid in ids:
            self._metadata.pop(cid, None)
            self._bm25_tokens.pop(cid, None)

        self._persist_dense_index()
        self._persist_metadata()
        self._persist_bm25_tokens()
        self._rebuild_bm25_model()
        self._record_version(source or "delete")
        self._logger.info(
            "indices.delete",
            removed=len(ids),
            dense_total=self._dense_index.ntotal if self._dense_index else 0,
            bm25_docs=len(self._bm25_tokens),
        )

        return len(ids)

    def rebuild(self, source: Optional[str] = None) -> None:
        session = SessionLocal()
        try:
            rows = (
                session.query(
                    models.KnowledgeChunk.id,
                    models.KnowledgeChunk.text,
                    models.KnowledgeChunk.start_line,
                    models.KnowledgeChunk.end_line,
                    models.KnowledgeDocument.path,
                    models.KnowledgeDocument.sha256,
                )
                .join(models.KnowledgeDocument, models.KnowledgeChunk.document_id == models.KnowledgeDocument.id)
                .all()
            )
        finally:
            session.close()

        inputs = [
            ChunkIndexInput(
                chunk_id=row.id,
                text=row.text,
                start_line=row.start_line,
                end_line=row.end_line,
                document_path=row.path,
                document_sha=row.sha256,
            )
            for row in rows
        ]

        self._reset_state()
        if inputs:
            self.index_chunks(inputs, source=source or "rebuild")
        else:
            self._persist_dense_index(clear=True)
            self._persist_metadata()
            self._persist_bm25_tokens()
            self._record_version(source or "rebuild-empty")
        self._logger.info(
            "indices.rebuild",
            chunks=len(inputs),
            dense_total=self._dense_index.ntotal if self._dense_index else 0,
        )

    # ------------------------------------------------------------------ search helpers
    def dense_search(self, query_embedding: np.ndarray, top_k: int) -> List[tuple[int, float]]:
        if self._dense_dim is None:
            self._dense_dim = query_embedding.shape[0]
        self._ensure_dense_index(self._dense_dim)
        if self._dense_index is None or self._dense_index.ntotal == 0:
            return []
        scores, indices = self._dense_index.search(query_embedding.reshape(1, -1).astype(np.float32), top_k)
        results: List[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results

    def bm25_search(self, query: str, top_k: int) -> List[tuple[int, float]]:
        if not self._bm25_model:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25_model.get_scores(tokens)
        pairs = list(zip(self._bm25_ids, scores))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return [(cid, float(score)) for cid, score in pairs[:top_k] if score > 0]

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def metadata_for_id(self, chunk_id: int) -> Optional[dict]:
        return self._metadata.get(chunk_id)

    def fetch_chunk_payloads(self, chunk_ids: Iterable[int]) -> Dict[int, dict]:
        ids = {int(cid) for cid in chunk_ids}
        if not ids:
            return {}
        session = SessionLocal()
        try:
            rows = (
                session.query(
                    models.KnowledgeChunk.id,
                    models.KnowledgeChunk.text,
                    models.KnowledgeChunk.ordinal,
                    models.KnowledgeChunk.start_line,
                    models.KnowledgeChunk.end_line,
                    models.KnowledgeDocument.path,
                    models.KnowledgeDocument.sha256,
                )
                .join(models.KnowledgeDocument, models.KnowledgeChunk.document_id == models.KnowledgeDocument.id)
                .filter(models.KnowledgeChunk.id.in_(ids))
                .all()
            )
        finally:
            session.close()
        payloads: Dict[int, dict] = {}
        for row in rows:
            payloads[row.id] = {
                "text": row.text,
                "path": row.path,
                "sha": row.sha256,
                "start_line": row.start_line,
                "end_line": row.end_line,
                "ordinal": row.ordinal,
            }
        return payloads

    # ------------------------------------------------------------------ internal helpers
    def _load_state(self) -> None:
        if self.metadata_path.exists():
            data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            chunks = data.get("chunks", {})
            self._metadata = {int(k): v for k, v in chunks.items()}
            self._dense_dim = data.get("dense", {}).get("dim")
        if self.bm25_path.exists():
            data = json.loads(self.bm25_path.read_text(encoding="utf-8"))
            tokens = data.get("tokens", {})
            self._bm25_tokens = {int(k): list(v) for k, v in tokens.items()}
            self._rebuild_bm25_model()

        # ленивое чтение dense index — загрузим при первом использовании

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.settings.dense_model_name)
            if self._dense_dim is None:
                self._dense_dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def _ensure_dense_index(self, dim: int) -> None:
        if self._dense_index is not None:
            return
        if self.dense_index_path.exists():
            index = faiss.read_index(str(self.dense_index_path))
            if not isinstance(index, faiss.IndexIDMap2):
                index = faiss.IndexIDMap2(index)
            self._dense_index = index
            self._dense_dim = index.d
            return
        base = faiss.IndexFlatIP(dim)
        self._dense_index = faiss.IndexIDMap2(base)
        self._dense_dim = dim

    def _remove_from_dense(self, ids: List[int]) -> None:
        if not ids:
            return
        if self._dense_index is None:
            if self.dense_index_path.exists():
                index = faiss.read_index(str(self.dense_index_path))
                if not isinstance(index, faiss.IndexIDMap2):
                    index = faiss.IndexIDMap2(index)
                self._dense_index = index
            else:
                return
        selector = faiss.IDSelectorArray(np.array(ids, dtype=np.int64))
        self._dense_index.remove_ids(selector)

    def _persist_dense_index(self, clear: bool = False) -> None:
        if clear:
            if self.dense_index_path.exists():
                self.dense_index_path.unlink()
            self._dense_index = None
            self._dense_dim = None
            return
        if self._dense_index is None:
            return
        faiss.write_index(self._dense_index, str(self.dense_index_path))

    def _persist_metadata(self) -> None:
        payload = {
            "dense": {"dim": self._dense_dim},
            "chunks": {str(cid): meta for cid, meta in self._metadata.items()},
        }
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _persist_bm25_tokens(self) -> None:
        payload = {
            "tokens": {str(cid): tokens for cid, tokens in self._bm25_tokens.items()},
        }
        self.bm25_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rebuild_bm25_model(self) -> None:
        if not self._bm25_tokens:
            self._bm25_model = None
            self._bm25_ids = []
            self._bm25_corpus = []
            return
        items = sorted(self._bm25_tokens.items(), key=lambda item: item[0])
        ids = [cid for cid, _ in items]
        corpus = [tokens for _, tokens in items]
        df = Counter()
        for tokens in corpus:
            unique = set(tokens)
            for token in unique:
                df[token] += 1
        filtered_corpus: List[List[str]] = []
        for tokens in corpus:
            filtered = [tok for tok in tokens if df[tok] >= self.settings.bm25_min_df]
            filtered_corpus.append(filtered or tokens)
        if filtered_corpus:
            self._bm25_model = BM25Okapi(filtered_corpus)
            self._bm25_ids = ids
            self._bm25_corpus = filtered_corpus
        else:
            self._bm25_model = None
            self._bm25_ids = []
            self._bm25_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self._token_pattern.findall(text)]

    def _record_version(self, note: str) -> None:
        session = SessionLocal()
        try:
            version = datetime.utcnow().isoformat()
            for name in ("dense", "bm25"):
                row = (
                    session.query(models.IndexVersion)
                    .filter(models.IndexVersion.name == name)
                    .one_or_none()
                )
                if row is None:
                    row = models.IndexVersion(name=name, version=version, source_hash=note, notes=note)
                    session.add(row)
                else:
                    row.version = version
                    row.source_hash = note
                    row.notes = note
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _reset_state(self) -> None:
        self._metadata = {}
        self._bm25_tokens = {}
        self._bm25_model = None
        self._bm25_ids = []
        self._bm25_corpus = []
        self._dense_index = None
        self._dense_dim = None
        if self.dense_index_path.exists():
            self.dense_index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        if self.bm25_path.exists():
            self.bm25_path.unlink()

