from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List

import tiktoken
from datasketch import MinHash, MinHashLSH


@dataclass
class Chunk:
    text: str
    ordinal: int
    start_line: int
    end_line: int
    char_start: int
    char_end: int
    token_count: int
    sha256: str


@lru_cache(maxsize=1)
def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _build_line_index(text: str) -> List[int]:
    starts: List[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        starts.append(pos)
        pos += len(line)
    starts.append(pos)
    return starts


def _char_to_line(char_idx: int, line_starts: List[int]) -> int:
    if char_idx <= 0:
        return 1
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= char_idx:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def chunk_document(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    if not text.strip():
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    encoding = _get_encoding()
    tokens = encoding.encode(text, allowed_special=set())
    if not tokens:
        return []

    token_strings = [
        encoding.decode_single_token_bytes(tok).decode("utf-8", errors="ignore") for tok in tokens
    ]
    char_offsets = [0] * (len(tokens) + 1)
    total = 0
    for idx, token_str in enumerate(token_strings):
        char_offsets[idx] = total
        total += len(token_str)
    char_offsets[len(tokens)] = total

    line_starts = _build_line_index(text)
    step = max(1, chunk_size - overlap)

    chunks: List[Chunk] = []
    start_token = 0
    ordinal = 0
    while start_token < len(tokens):
        end_token = min(len(tokens), start_token + chunk_size)
        raw = "".join(token_strings[start_token:end_token])
        stripped = raw.strip()
        if stripped:
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            char_start = char_offsets[start_token] + leading
            char_end = char_offsets[end_token] - trailing
            if char_end <= char_start:
                char_end = char_offsets[end_token]
            start_line = _char_to_line(char_start, line_starts)
            end_line = _char_to_line(max(char_end - 1, char_start), line_starts)
            chunks.append(
                Chunk(
                    text=stripped,
                    ordinal=ordinal,
                    start_line=start_line,
                    end_line=end_line,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=end_token - start_token,
                    sha256=hashlib.sha256(stripped.encode("utf-8")).hexdigest(),
                )
            )
            ordinal += 1
        if end_token >= len(tokens):
            break
        start_token += step

    return chunks


def deduplicate_chunks(chunks: Iterable[Chunk], threshold: float = 0.85) -> List[Chunk]:
    items = list(chunks)
    if not items:
        return []
    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    kept: List[Chunk] = []
    for idx, chunk in enumerate(items):
        mh = MinHash(num_perm=64)
        tokens = chunk.text.split()
        if len(tokens) <= 5:
            shingles = [" ".join(tokens)]
        else:
            shingles = [" ".join(tokens[i : i + 5]) for i in range(0, len(tokens) - 4)]
        for sh in shingles:
            mh.update(sh.encode("utf-8", errors="ignore"))
        cands = lsh.query(mh)
        if cands:
            continue
        lsh.insert(str(idx), mh)
        kept.append(chunk)
    return kept

