from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from app.db import models
from app.services.llm_gateway import chat_completion


def extract_preferences(db: Session, session: models.SessionModel) -> int:
    texts = [m.content for m in session.messages if m.role == "user"][-10:]
    if not texts:
        return 0
    try:
        msgs = [
            {"role": "system", "content": "Выдели 3-5 устойчивых предпочтений/приоритетов пользователя списком, кратко."},
            {"role": "user", "content": "\n".join(texts)},
        ]
        resp = anyio.run(chat_completion, msgs, False)
    except Exception:
        resp = None
    if not resp:
        return 0
    count = 0
    for line in resp.splitlines():
        line = line.strip("- •\t ")
        if not line:
            continue
        mem = models.Memory(
            type="preference",
            title=line[:80],
            content=line,
            importance=1,
            source_ref=f"session:{session.id}",
        )
        db.add(mem)
        count += 1
    db.commit()
    return count


