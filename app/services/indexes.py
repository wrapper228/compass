"""Local index abstractions for the hybrid retrieval pipeline."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import structlog

from app.services.tokenization import get_tokenizer

logger = structlog.get_logger(__name__)


@dataclass
class ChunkRecord:
    """Representation of a chunk stored in the local indices."""

    chunk_id: str
    job_id: str
    path: str
    idx: int
    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    token_count: int
    version: str

    @property
    def short_citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass
class DenseVector:
    chunk_id: str
    vector: List[float]


class DenseIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._vectors: Dict[str, DenseVector] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._vectors = {
                item["chunk_id"]: DenseVector(item["chunk_id"], item["vector"])
                for item in data
            }
            logger.info("dense_index.loaded", count=len(self._vectors))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("dense_index.load_failed", error=str(exc))
            self._vectors = {}

    def _persist(self) -> None:
        payload = [asdict(vec) for vec in self._vectors.values()]
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def reset(self) -> None:
        self._vectors.clear()
        self._persist()

    def upsert(self, vectors: Iterable[DenseVector]) -> None:
        for v in vectors:
            self._vectors[v.chunk_id] = v
        self._persist()

    def delete_missing(self, known_chunk_ids: Sequence[str]) -> None:
        known = set(known_chunk_ids)
        removed = [cid for cid in list(self._vectors) if cid not in known]
        for cid in removed:
            self._vectors.pop(cid, None)
        if removed:
            logger.info("dense_index.pruned", removed=len(removed))
            self._persist()

    def get_vector(self, chunk_id: str) -> Optional[List[float]]:
        vec = self._vectors.get(chunk_id)
        if vec is None:
            return None
        return vec.vector

    def search(self, query: List[float], top_k: int) -> List[Tuple[str, float]]:
        if not query:
            return []
        results: List[Tuple[str, float]] = []
        for vec in self._vectors.values():
            score = _cosine_similarity(query, vec.vector)
            results.append((vec.chunk_id, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


class SparseIndex:
    """Simple BM25 implementation stored locally.

    A deliberately small implementation keeps tests deterministic and avoids a
    heavyweight dependency on a search engine during CI.  The math follows the
    standard BM25 formula.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._doc_freqs: Dict[str, int] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._avg_len: float = 0.0
        self._inverted_index: Dict[str, Dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._doc_freqs = raw.get("doc_freqs", {})
            self._doc_lengths = raw.get("doc_lengths", {})
            self._avg_len = raw.get("avg_len", 0.0)
            self._inverted_index = raw.get("inverted_index", {})
            logger.info("sparse_index.loaded", documents=len(self._doc_lengths))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("sparse_index.load_failed", error=str(exc))
            self._doc_freqs = {}
            self._doc_lengths = {}
            self._avg_len = 0.0
            self._inverted_index = {}

    def _persist(self) -> None:
        payload = {
            "doc_freqs": self._doc_freqs,
            "doc_lengths": self._doc_lengths,
            "avg_len": self._avg_len,
            "inverted_index": self._inverted_index,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def reset(self) -> None:
        self._doc_freqs.clear()
        self._doc_lengths.clear()
        self._avg_len = 0.0
        self._inverted_index.clear()
        self._persist()

    def rebuild(self, records: Iterable[ChunkRecord]) -> None:
        tokenizer = get_tokenizer()
        inverted: Dict[str, Dict[str, int]] = {}
        doc_freqs: Dict[str, int] = {}
        doc_lengths: Dict[str, int] = {}
        for rec in records:
            tokens = tokenizer.encode(rec.text.lower())
            doc_lengths[rec.chunk_id] = len(tokens)
            seen_terms = set()
            counts: Dict[str, int] = {}
            for tok in tokens:
                term = str(tok)
                counts[term] = counts.get(term, 0) + 1
            for term, cnt in counts.items():
                inverted.setdefault(term, {})[rec.chunk_id] = cnt
                if term not in seen_terms:
                    doc_freqs[term] = doc_freqs.get(term, 0) + 1
                    seen_terms.add(term)
        self._doc_lengths = doc_lengths
        self._doc_freqs = doc_freqs
        self._inverted_index = inverted
        self._avg_len = (
            sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0.0
        )
        self._persist()

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not query:
            return []
        tokenizer = get_tokenizer()
        query_terms = [str(tok) for tok in tokenizer.encode(query.lower())]
        scores: Dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        total_docs = max(len(self._doc_lengths), 1)
        for term in query_terms:
            postings = self._inverted_index.get(term)
            if not postings:
                continue
            df = self._doc_freqs.get(term, 0)
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for doc_id, freq in postings.items():
                doc_len = self._doc_lengths.get(doc_id, 0)
                denom = freq + k1 * (1 - b + b * doc_len / (self._avg_len or 1.0))
                score = idf * ((freq * (k1 + 1)) / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + score
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


class IndexRepository:
    """Composite helper exposing dense + sparse search."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.dense_index = DenseIndex(base_path / "dense.json")
        self.sparse_index = SparseIndex(base_path / "sparse.json")
        self._chunk_cache_path = base_path / "chunks.json"
        self._chunk_cache: Dict[str, ChunkRecord] = {}
        self._load_chunk_cache()

    def _load_chunk_cache(self) -> None:
        if not self._chunk_cache_path.exists():
            return
        try:
            raw = json.loads(self._chunk_cache_path.read_text(encoding="utf-8"))
            self._chunk_cache = {
                item["chunk_id"]: ChunkRecord(**item) for item in raw
            }
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("index_repo.load_cache_failed", error=str(exc))
            self._chunk_cache = {}

    def _persist_chunk_cache(self) -> None:
        payload = [asdict(rec) for rec in self._chunk_cache.values()]
        self._chunk_cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def list_chunks(self) -> List[ChunkRecord]:
        return list(self._chunk_cache.values())

    def upsert_chunks(self, records: Iterable[ChunkRecord]) -> None:
        for rec in records:
            self._chunk_cache[rec.chunk_id] = rec
        self._persist_chunk_cache()

    def prune_chunks(self, valid_ids: Sequence[str]) -> None:
        valid = set(valid_ids)
        removed = [cid for cid in list(self._chunk_cache) if cid not in valid]
        for cid in removed:
            self._chunk_cache.pop(cid, None)
        if removed:
            logger.info("index_repo.pruned_chunks", removed=len(removed))
            self._persist_chunk_cache()

    def fetch_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        return self._chunk_cache.get(chunk_id)

    def fetch_vector(self, chunk_id: str) -> Optional[List[float]]:
        return self.dense_index.get_vector(chunk_id)

    def ensure_dense_vectors(self, vectors: Iterable[DenseVector]) -> None:
        self.dense_index.upsert(vectors)

    def ensure_sparse_index(self) -> None:
        self.sparse_index.rebuild(self._chunk_cache.values())

    def search(self, *, query_vector: Optional[List[float]], query_text: str, top_k_dense: int, top_k_sparse: int) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        dense_hits: List[Tuple[str, float]] = []
        if query_vector is not None:
            dense_hits = self.dense_index.search(query_vector, top_k_dense)
        sparse_hits = self.sparse_index.search(query_text, top_k_sparse)
        return dense_hits, sparse_hits


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # Pad the shortest vector with zeros so we can still compare – this can
        # happen when vectors were produced by different backends.
        if len(a) < len(b):
            a = list(a) + [0.0] * (len(b) - len(a))
        else:
            b = list(b) + [0.0] * (len(a) - len(b))
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


