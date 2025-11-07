from typing import Optional

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas.chat import (
    ChatGenerateRequest,
    ChatGenerateResponse,
    ChatMessage,
)
from app.services.llm_gateway import stream_reply_stub, chat_completion
from app.services.rag import retrieve_for_text
from app.services.dialog import compose_messages
from app.services.analysis import classify_emotion
from app.services.proactivity import score_proactivity, proactive_prefix
from app.services.dialog import choose_strategy
from app.services.protocols import next_reflective_steps
from app.db.session import get_db
from app.db import models
from sqlalchemy.orm import Session
from fastapi import Depends
from app.services.memory import update_session_summary
from app.services.preferences import extract_preferences


router = APIRouter(prefix="/api")


@router.post("/chat/generate", response_model=ChatGenerateResponse)
def chat_generate(payload: ChatGenerateRequest, db: Session = Depends(get_db)) -> ChatGenerateResponse:
    last_user: Optional[ChatMessage] = next(
        (m for m in reversed(payload.messages) if m.role == "user"), None
    )
    reply_text = None
    if last_user:
        # RAG
        retrieved = []
        if payload.retrieval and payload.retrieval.top_k > 0:
            # best-effort, игнорируем ошибки
            try:
                retrieved = anyio.run(retrieve_for_text, last_user.content, payload.retrieval.top_k)
            except Exception:
                retrieved = []
        # Compose and call LLM
        full_msgs = compose_messages(last_user.content, retrieved)
        llm_resp = anyio.run(chat_completion, full_msgs, True)
        reply_text = llm_resp or f"Принято: {last_user.content[:500]}"

        # Эмоции (последние 2 реплики user)
        emotions = []
        try:
            last_user_texts = [m.content for m in payload.messages if m.role == "user"][-2:]
            for t in last_user_texts:
                emotions.append(anyio.run(classify_emotion, t))
        except Exception:
            emotions = []

        # Проактивность
        s_prefix = None
        try:
            s_obj = db.get(models.SessionModel, session.id)
            summary_txt = s_obj.summary_text if s_obj else None
            s, reasons = score_proactivity(emotions, summary_txt, db)
            s_prefix = proactive_prefix(s, reasons)
        except Exception:
            s_prefix = None
        if s_prefix:
            reply_text = f"{s_prefix}\n\n{reply_text}"

        # Пошаговый протокол для Сократического режима
        try:
            if choose_strategy(last_user.content) == "socratic":
                steps = next_reflective_steps(last_user.content)
                reply_text += "\n\nШаги для рефлексии:\n- " + "\n- ".join(steps)
        except Exception:
            pass
    else:
        reply_text = f"Принято. Это заглушка ответа. Сообщений в контексте: {len(payload.messages)}."
    # Persist session/messages (MVP)
    if payload.session_id is None:
        session = models.SessionModel()
        db.add(session)
        db.flush()  # get id
    else:
        session = db.get(models.SessionModel, int(payload.session_id))
        if session is None:
            session = models.SessionModel()
            db.add(session)
            db.flush()

    for m in payload.messages[-3:]:  # сохраним последние 3
        db.add(models.Message(session_id=session.id, role=m.role, content=m.content))

    assistant_msg = models.Message(session_id=session.id, role="assistant", content=reply_text)
    db.add(assistant_msg)
    db.commit()
    # Обновим краткую сводку сессии (best-effort)
    try:
        update_session_summary(db, session)
    except Exception:
        pass
    # Извлечём предпочтения/приоритеты (best-effort)
    try:
        extract_preferences(db, session)
    except Exception:
        pass

    return ChatGenerateResponse(
        message=ChatMessage(role="assistant", content=reply_text),
        usage={"stub": True},
        memories_written=[],
    )


@router.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        # Ждём простое текстовое сообщение пользователя
        data = await ws.receive_text()
        # Отправляем токены по мере готовности (стуб)
        async for token in stream_reply_stub(f"Принято: {data}"):
            await ws.send_text(token)
        await ws.send_text("[DONE]")
    except WebSocketDisconnect:
        return


