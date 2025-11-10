from fastapi import APIRouter, HTTPException

from app.schemas.chat import RetrievalOptions
from app.services.rag import retrieve_for_text


router = APIRouter(prefix="/api")


@router.get("/memory/search")
async def memory_search(q: str, top_k: int = 6):
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    bundle = await retrieve_for_text(q, RetrievalOptions(top_k=top_k))
    return bundle


