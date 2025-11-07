import io
import json
import zipfile
from pathlib import Path
from typing import List
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader
from datasketch import MinHash, MinHashLSH
import json as _json
import csv as _csv

import anyio


def process_zip(job_id: str, zip_bytes: bytes, workspace: Path) -> dict:
    base_dir = workspace / job_id
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    chunks_path = base_dir / "chunks.json"

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(raw_dir)

    texts: List[dict] = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in {".txt", ".md"}:
            rel = p.relative_to(raw_dir).as_posix()
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for idx, chunk in enumerate(chunk_paragraphs(content)):
                texts.append({
                    "path": rel,
                    "idx": idx,
                    "text": chunk,
                })
        elif ext in {".json"}:
            rel = p.relative_to(raw_dir).as_posix()
            try:
                data = _json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                content = _json.dumps(data, ensure_ascii=False, indent=2)
            except Exception:
                continue
            for idx, chunk in enumerate(chunk_paragraphs(content)):
                texts.append({"path": rel, "idx": idx, "text": chunk})
        elif ext in {".csv"}:
            rel = p.relative_to(raw_dir).as_posix()
            try:
                rows = []
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    reader = _csv.reader(f)
                    for row in reader:
                        rows.append(", ".join(row))
                content = "\n".join(rows)
            except Exception:
                continue
            for idx, chunk in enumerate(chunk_paragraphs(content)):
                texts.append({"path": rel, "idx": idx, "text": chunk})
        elif ext in {".html", ".htm"}:
            rel = p.relative_to(raw_dir).as_posix()
            try:
                html = p.read_text(encoding="utf-8", errors="ignore")
                soup = BeautifulSoup(html, "lxml")
                content = soup.get_text("\n", strip=True)
            except Exception:
                continue
            for idx, chunk in enumerate(chunk_paragraphs(content)):
                texts.append({"path": rel, "idx": idx, "text": chunk})
        elif ext in {".docx"}:
            rel = p.relative_to(raw_dir).as_posix()
            try:
                doc = DocxDocument(p)
                content = "\n".join([para.text for para in doc.paragraphs])
            except Exception:
                continue
            for idx, chunk in enumerate(chunk_paragraphs(content)):
                texts.append({"path": rel, "idx": idx, "text": chunk})
        elif ext in {".pdf"}:
            rel = p.relative_to(raw_dir).as_posix()
            try:
                reader = PdfReader(str(p))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                content = "\n".join(pages)
            except Exception:
                continue
            for idx, chunk in enumerate(chunk_paragraphs(content)):
                texts.append({"path": rel, "idx": idx, "text": chunk})

    # Дедупликация MinHash по чанкам
    texts = deduplicate_chunks(texts)

    chunks_path.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Пытаемся индексировать в векторку (если настроено)
    try:
        from app.services.retrieval import get_qdrant, ensure_collection, upsert_points
        from app.services.embeddings import get_embedding

        client = get_qdrant()
        if client is None:
            return {"job_id": job_id, "chunks": len(texts), "indexed": 0}

        # Получим размер вектора из первого эмбеддинга
        first_vec = anyio.run(get_embedding, texts[0]["text"]) if texts else []
        if not first_vec:
            return {"job_id": job_id, "chunks": len(texts), "indexed": 0}
        ensure_collection(client, vector_size=len(first_vec))

        vectors = []
        payloads = []
        for i, t in enumerate(texts):
            vec = anyio.run(get_embedding, t["text"])  # sync wrapper
            vectors.append(vec)
            payloads.append({
                "id": f"{job_id}-{i}",
                "job_id": job_id,
                "path": t["path"],
                "idx": t["idx"],
                "text": t["text"],
            })
        upsert_points(client, vectors, payloads)
        return {"job_id": job_id, "chunks": len(texts), "indexed": len(vectors)}
    except Exception:
        # Индексация необязательна для работы MVP
        return {"job_id": job_id, "chunks": len(texts), "indexed": 0}


def chunk_paragraphs(text: str, max_len: int = 1200) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for para in paras:
        if len(para) <= max_len:
            chunks.append(para)
        else:
            # Грубая разбивка длинных параграфов
            buf = []
            cur_len = 0
            for sent in para.split(". "):
                if cur_len + len(sent) + 2 > max_len:
                    if buf:
                        chunks.append(". ".join(buf).strip())
                        buf = []
                        cur_len = 0
                buf.append(sent)
                cur_len += len(sent) + 2
            if buf:
                chunks.append(". ".join(buf).strip())
    return chunks


def _shingles(text: str, k: int = 5) -> List[str]:
    tokens = text.split()
    if len(tokens) <= k:
        return [" ".join(tokens)]
    return [" ".join(tokens[i:i+k]) for i in range(0, len(tokens) - k + 1)]


def deduplicate_chunks(items: List[dict], threshold: float = 0.85) -> List[dict]:
    if not items:
        return items
    # LSH по MinHash для быстрых кандидатов
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    kept: List[dict] = []
    idx = 0
    for it in items:
        mh = MinHash(num_perm=64)
        for sh in _shingles(it["text"], 5):
            mh.update(sh.encode("utf-8", errors="ignore"))
        # Найти похожие
        cands = lsh.query(mh)
        if cands:
            # Пропускаем как дубликат
            continue
        lsh.insert(str(idx), mh)
        kept.append(it)
        idx += 1
    return kept


