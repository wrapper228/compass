from typing import List, Optional, Literal
from pydantic import BaseModel, Field


ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class RetrievalOptions(BaseModel):
    top_k: int = 6
    filters: Optional[dict] = None
    strategy: Optional[str] = "hybrid"
    max_iterations: Optional[int] = None


class ChatGenerateRequest(BaseModel):
    session_id: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    retrieval: Optional[RetrievalOptions] = None


class ChatGenerateResponse(BaseModel):
    message: ChatMessage
    usage: Optional[dict] = None
    memories_written: List[str] = []
    retrieval: Optional[dict] = None


