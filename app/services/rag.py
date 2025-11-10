from __future__ import annotations

from typing import Dict, List

from app.services.retrieval import retrieve


async def retrieve_for_text(text: str, top_k: int = 6) -> List[Dict]:
    try:
        return await retrieve(text, top_k)
    except Exception:
        return []


