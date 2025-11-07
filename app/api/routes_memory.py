from fastapi import APIRouter, HTTPException

from app.services.rag import retrieve_for_text


router = APIRouter(prefix="/api")


@router.get("/memory/search")
async def memory_search(q: str, top_k: int = 6):
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    hits = await retrieve_for_text(q, top_k=top_k)
    return hits


