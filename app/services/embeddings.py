from __future__ import annotations

from typing import List, Optional

import httpx

from app.core.config import get_settings


class EmbeddingsUnavailable(Exception):
    pass


async def get_embedding(text: str) -> List[float]:
    settings = get_settings()
    if not (settings.embeddings_api_base and settings.embeddings_api_key and settings.embeddings_model):
        raise EmbeddingsUnavailable("Embeddings provider is not configured")

    url = f"{settings.embeddings_api_base.rstrip('/')}/embeddings"
    headers = {"Authorization": f"Bearer {settings.embeddings_api_key}"}
    payload = {"input": text, "model": settings.embeddings_model}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["data"][0]["embedding"]


