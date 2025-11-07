from __future__ import annotations

from typing import List, Tuple

from sqlalchemy.orm import Session

from app.db import models


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def tokenize(s: str) -> set[str]:
    return set(w.lower() for w in s.split())


def detect_loops(db: Session, user_id: int | None = None, k: int = 6, thr: float = 0.6) -> Tuple[bool, List[str]]:
    # Берём последние k сессий и сравниваем их summary между собой
    q = db.query(models.SessionModel).order_by(models.SessionModel.id.desc()).limit(k)
    sessions = list(reversed(q.all()))
    reasons: List[str] = []
    if len(sessions) < 3:
        return False, reasons
    sims = []
    for i in range(len(sessions) - 1):
        a = tokenize(sessions[i].summary_text or "")
        b = tokenize(sessions[i + 1].summary_text or "")
        sims.append(jaccard(a, b))
    high = sum(1 for s in sims if s >= thr)
    if high >= 2:
        reasons.append("высокое сходство между несколькими соседними сессиями")
        return True, reasons
    return False, reasons


