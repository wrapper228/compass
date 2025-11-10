from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence

import anyio
import httpx
import structlog

from app.core.config import get_settings
from app.services.tokenization import get_tokenizer

logger = structlog.get_logger(__name__)


class EmbeddingsUnavailable(Exception):
    """Raised when no embedding backend can serve a request."""


@dataclass
class EmbeddingResult:
    vector: List[float]


class BaseEmbeddingBackend:
    """Abstract helper that exposes synchronous and async interfaces."""

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        raise EmbeddingsUnavailable

    async def aembed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return self.embed_batch(texts)


class RemoteEmbeddingBackend(BaseEmbeddingBackend):
    def __init__(self) -> None:
        self.settings = get_settings()
        if not (
            self.settings.embeddings_api_base
            and self.settings.embeddings_api_key
            and self.settings.embeddings_model
        ):
            raise EmbeddingsUnavailable("Embeddings provider is not configured")

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        # Delegate to the async path to avoid duplicating HTTP logic.
        return anyio.run(self.aembed_batch, texts)

    async def aembed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self.settings.embeddings_api_base.rstrip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {self.settings.embeddings_api_key}"}
        payload = {"input": list(texts), "model": self.settings.embeddings_model}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:  # pragma: no cover - network errors
                raise EmbeddingsUnavailable(str(exc))
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


class HashEmbeddingBackend(BaseEmbeddingBackend):
    """Deterministic lightweight embeddings used as a safety net.

    The backend hashes 3-gram tokens into a fixed 512 dimensional space.  While
    obviously not semantically rich, it allows the retrieval stack to function in
    offline developer environments and in unit tests where heavy models are
    unavailable.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.tokenizer = get_tokenizer()

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            tokens = self.tokenizer.encode(text.lower())
            vec = [0.0] * self.dim
            if not tokens:
                vectors.append(vec)
                continue
            for i in range(len(tokens) - 2):
                tri = (tokens[i], tokens[i + 1], tokens[i + 2])
                h = int(hashlib.sha1(str(tri).encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            if norm:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors


class LocalModelEmbeddingBackend(BaseEmbeddingBackend):  # pragma: no cover - optional
    def __init__(self, model_name: str | None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise EmbeddingsUnavailable(str(exc))
        if not model_name:
            model_name = "intfloat/multilingual-e5-base"
        self.model = SentenceTransformer(model_name)

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [list(vec) for vec in self.model.encode(list(texts), normalize_embeddings=True)]


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        backends: List[BaseEmbeddingBackend] = []
        # Local model first when explicitly requested.
        if settings.local_embedding_model:
            try:
                backends.append(LocalModelEmbeddingBackend(settings.local_embedding_model))
            except EmbeddingsUnavailable as exc:
                logger.warning("embeddings.local_unavailable", error=str(exc))
        # Remote backend next.
        try:
            backends.append(RemoteEmbeddingBackend())
        except EmbeddingsUnavailable:
            pass
        # Final fallback – deterministic hash embeddings.
        backends.append(HashEmbeddingBackend())
        self.backends = backends
        self._cache: dict[str, List[float]] = {}

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        last_error: Exception | None = None
        for backend in self.backends:
            try:
                vectors = backend.embed_batch(texts)
                for text, vec in zip(texts, vectors):
                    self._cache[str(text)] = vec
                return vectors
            except EmbeddingsUnavailable as exc:
                last_error = exc
                continue
        raise EmbeddingsUnavailable(str(last_error) if last_error else "no backend")

    async def aembed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        last_error: Exception | None = None
        for backend in self.backends:
            try:
                vectors = await backend.aembed_batch(texts)
                for text, vec in zip(texts, vectors):
                    self._cache[str(text)] = vec
                return vectors
            except EmbeddingsUnavailable as exc:
                last_error = exc
                continue
        raise EmbeddingsUnavailable(str(last_error) if last_error else "no backend")

    def embed(self, text: str) -> List[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self.embed_batch([text])[0]
        self._cache[text] = vec
        return vec

    async def aembed(self, text: str) -> List[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = (await self.aembed_batch([text]))[0]
        self._cache[text] = vec
        return vec


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


async def get_embedding(text: str) -> List[float]:
    service = get_embedding_service()
    return await service.aembed(text)


def get_embedding_sync(text: str) -> List[float]:
    service = get_embedding_service()
    return service.embed(text)


def embed_batch_sync(texts: Sequence[str]) -> List[List[float]]:
    service = get_embedding_service()
    return service.embed_batch(texts)


