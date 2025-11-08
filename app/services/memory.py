from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session
import anyio
from app.db import models

from app.db import models
from app.services.llm_gateway import chat_completion


def get_last_texts(db: Session, session_id: int, limit: int = 20) -> List[str]:
    msgs = (
        db.query(models.Message)
        .filter(models.Message.session_id == session_id)
        .order_by(models.Message.id.desc())
        .limit(limit)
        .all()
    )
    return [m.content for m in reversed(msgs)]


def naive_summary(texts: List[str], max_len: int = 600) -> str:
    joined = " \n".join(texts)
    return joined[-max_len:]


def update_session_summary(db: Session, session: models.SessionModel) -> None:
    texts = get_last_texts(db, session.id, 20)
    if not texts:
        return
    try:
        # best-effort с LLM
        prompt = (
            "Суммируй диалог кратко (3-5 пунктов) с фокусом на цели/намерения/сомнения."
            " Заверши одним предложением о следующем шаге."
        )
        msgs = [
            {"role": "system", "content": "Ты краткий суммаризатор."},
            {"role": "user", "content": prompt + "\n\n" + "\n".join(texts)},
        ]
        summary = anyio.run(chat_completion, msgs, False)
    except Exception:
        summary = None
    final_summary = summary or naive_summary(texts)
    session.summary_text = final_summary
    db.add(session)
    # Сохраняем как эпизодическую память
    try:
        mem = models.Memory(
            type="episodic",
            title=f"Session {session.id} summary",
            content=final_summary,
            importance=1,
            source_ref=f"session:{session.id}",
        )
        db.add(mem)
    except Exception:
        pass
    db.commit()


def get_brief_memory_context(db: Session, limit_semantic: int = 5, limit_pref: int = 5, limit_epi: int = 3) -> list[str]:
    texts: list[str] = []
    try:
        sem = (
            db.query(models.Memory)
            .filter(models.Memory.type == "semantic")
            .order_by(models.Memory.id.desc())
            .limit(limit_semantic)
            .all()
        )
        prefs = (
            db.query(models.Memory)
            .filter(models.Memory.type == "preference")
            .order_by(models.Memory.id.desc())
            .limit(limit_pref)
            .all()
        )
        epis = (
            db.query(models.Memory)
            .filter(models.Memory.type == "episodic")
            .order_by(models.Memory.id.desc())
            .limit(limit_epi)
            .all()
        )
        for m in sem + prefs + epis:
            texts.append(m.content)
    except Exception:
        pass
    return texts


