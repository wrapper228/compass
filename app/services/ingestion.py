import csv as _csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.core.config import get_settings
from app.services.embeddings import embed_batch_sync
from app.services.indexes import ChunkRecord, DenseVector, IndexRepository
from app.services.tokenization import get_tokenizer


@dataclass
class RawChunk:
    path: str
    idx: int
    text: str
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    token_count: int


def process_zip(job_id: str, zip_bytes: bytes, workspace: Path) -> dict:
    base_dir = workspace / job_id
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    chunks_path = base_dir / "chunks.json"

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(raw_dir)

    raw_chunks: List[RawChunk] = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(raw_dir).as_posix()
        try:
            content = _extract_text(p)
        except Exception:
            continue
        for idx, chunk in enumerate(chunk_text(content)):
            raw_chunks.append(
                RawChunk(
                    path=rel,
                    idx=idx,
                    text=chunk["text"],
                    start_line=chunk["start_line"],
                    end_line=chunk["end_line"],
                    start_offset=chunk["start_offset"],
                    end_offset=chunk["end_offset"],
                    token_count=chunk["token_count"],
                )
            )

    deduped = deduplicate_chunks(raw_chunks)
    chunks_payload = [raw_chunk_to_payload(job_id, ch) for ch in deduped]
    chunks_path.write_text(json.dumps(chunks_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update local indexes
    settings = get_settings()
    index_repo = IndexRepository(Path(settings.retrieval_index_path))
    existing = [rec for rec in index_repo.list_chunks() if rec.job_id != job_id]
    chunk_records = [payload_to_record(item) for item in chunks_payload]
    valid_ids = [rec.chunk_id for rec in existing] + [rec.chunk_id for rec in chunk_records]
    index_repo.prune_chunks(valid_ids)
    index_repo.dense_index.delete_missing(valid_ids)
    index_repo.upsert_chunks(existing + chunk_records)
    embeddings = embed_batch_sync([rec.text for rec in chunk_records])
    index_repo.ensure_dense_vectors(
        DenseVector(chunk_id=rec.chunk_id, vector=embeddings[i])
        for i, rec in enumerate(chunk_records)
    )
    index_repo.ensure_sparse_index()

    return {
        "job_id": job_id,
        "chunks": len(chunk_records),
        "indexed_dense": len(chunk_records),
        "indexed_sparse": len(chunk_records),
    }


def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    if ext in {".csv"}:
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            reader = _csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        return "\n".join(rows)
    if ext in {".html", ".htm"}:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text("\n", strip=True)
    if ext in {".docx"}:
        doc = DocxDocument(path)
        return "\n".join([para.text for para in doc.paragraphs])
    if ext in {".pdf"}:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, min_tokens: int = 500, max_tokens: int = 1000) -> Iterable[dict]:
    tokenizer = get_tokenizer()
    lines = text.splitlines()
    chunks: List[dict] = []
    buf: List[str] = []
    buf_tokens = 0
    start_line = 1
    start_offset = 0
    current_offset = 0
    for idx, line in enumerate(lines):
        line_tokens = len(tokenizer.encode(line))
        line_with_newline = line + "\n"
        projected = buf_tokens + line_tokens
        if buf and projected > max_tokens:
            chunk_text = "\n".join(buf)
            end_line = start_line + len(buf) - 1
            chunks.append(
                {
                    "text": chunk_text,
                    "start_line": start_line,
                    "end_line": end_line,
                    "start_offset": start_offset,
                    "end_offset": current_offset,
                    "token_count": max(buf_tokens, 1),
                }
            )
            buf = []
            buf_tokens = 0
            start_line = idx + 1
            start_offset = current_offset
        if not buf:
            start_line = idx + 1
            start_offset = current_offset
        buf.append(line)
        buf_tokens += line_tokens
        current_offset += len(line_with_newline)
        if buf_tokens >= min_tokens:
            chunk_text = "\n".join(buf)
            end_line = start_line + len(buf) - 1
            chunks.append(
                {
                    "text": chunk_text,
                    "start_line": start_line,
                    "end_line": end_line,
                    "start_offset": start_offset,
                    "end_offset": current_offset,
                    "token_count": max(buf_tokens, 1),
                }
            )
            buf = []
            buf_tokens = 0
            start_line = idx + 2
            start_offset = current_offset
    if buf:
        chunk_text = "\n".join(buf)
        end_line = start_line + len(buf) - 1
        chunks.append(
            {
                "text": chunk_text,
                "start_line": start_line,
                "end_line": end_line,
                "start_offset": start_offset,
                "end_offset": current_offset,
                "token_count": max(buf_tokens, 1),
            }
        )
    return chunks


def raw_chunk_to_payload(job_id: str, chunk: RawChunk) -> dict:
    version = hashlib.sha1(f"{chunk.path}:{chunk.text}".encode("utf-8")).hexdigest()
    chunk_id = f"{job_id}:{version[:16]}:{chunk.idx}"
    return {
        "chunk_id": chunk_id,
        "job_id": job_id,
        "path": chunk.path,
        "idx": chunk.idx,
        "text": chunk.text,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "token_count": chunk.token_count,
        "version": version,
    }


def payload_to_record(payload: dict) -> ChunkRecord:
    return ChunkRecord(**payload)


def _shingles(text: str, k: int = 5) -> List[str]:
    tokens = text.split()
    if len(tokens) <= k:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + k]) for i in range(0, len(tokens) - k + 1)]


def deduplicate_chunks(items: List[RawChunk], threshold: float = 0.85) -> List[RawChunk]:
    if not items:
        return items
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    kept: List[RawChunk] = []
    idx = 0
    for it in items:
        mh = MinHash(num_perm=64)
        for sh in _shingles(it.text, 5):
            mh.update(sh.encode("utf-8", errors="ignore"))
        cands = lsh.query(mh)
        if cands:
            continue
        lsh.insert(str(idx), mh)
        kept.append(it)
        idx += 1
    return kept


