from __future__ import annotations

from typing import List, Dict, Optional

from app.services.embeddings import get_embedding, EmbeddingsUnavailable
from app.services.retrieval import get_qdrant, search


async def retrieve_for_text(text: str, top_k: int = 6) -> List[Dict]:
    try:
        vec = await get_embedding(text)
    except EmbeddingsUnavailable:
        return []

    client = get_qdrant()
    if client is None:
        return []
    return search(client, vec, top_k=top_k)


