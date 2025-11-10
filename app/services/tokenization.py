"""Tokenization utilities for ingestion and retrieval pipelines.

The implementation prefers ``tiktoken`` when available, but gracefully falls back
 to a lightweight whitespace/token heuristic so the service can operate in
 environments where OpenAI tooling is not installed (e.g. CI).  The API is kept
 intentionally small: callers either need a reusable tokenizer instance or a
 helper to count tokens in a piece of text.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List


@dataclass
class Tokenizer:
    """A tiny wrapper that provides ``encode`` and ``count`` helpers.

    The object tries to keep the underlying implementation opaque so we can swap
    it for ``tiktoken``/``sentencepiece`` without touching the call sites.
    """

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def count(self, text: str) -> int:
        return len(self.encode(text))


class _TiktokenTokenizer(Tokenizer):
    def __init__(self) -> None:
        import tiktoken  # type: ignore

        # ``cl100k_base`` is reasonably close to GPT style tokenisation and is
        # available without model-specific merges.
        self._enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> List[int]:  # pragma: no cover - thin wrapper
        return list(self._enc.encode(text, disallowed_special=()))


class _HeuristicTokenizer(Tokenizer):
    """Lightweight fallback that approximates token counts.

    We split on whitespace and punctuation boundaries to approximate GPT style
    tokens.  While crude, it is deterministic and keeps unit tests fast and
    hermetic.
    """

    def encode(self, text: str) -> List[int]:
        import re

        if not text:
            return []
        # ``\w`` clusters, punctuation and whitespace tokens; the goal is not to
        # be linguistically perfect but to give a reasonable upper bound on the
        # token budget for chunking.
        pattern = re.compile(r"\w+|[^\w\s]")
        return [hash(tok) & 0xFFFFFFFF for tok in pattern.findall(text)]


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    """Return a singleton tokenizer instance.

    ``tiktoken`` is optional, therefore we import it lazily and fall back to the
    heuristic implementation when the dependency is missing.
    """

    try:
        return _TiktokenTokenizer()
    except Exception:  # pragma: no cover - fallback path is deterministic
        return _HeuristicTokenizer()


def count_tokens(text: str) -> int:
    return get_tokenizer().count(text)


