from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

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
    session.summary_text = summary or naive_summary(texts)
    db.add(session)
    db.commit()


