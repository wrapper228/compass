from __future__ import annotations

from typing import List, Tuple

from app.services.analysis import Emotion
from sqlalchemy.orm import Session
from app.services.patterns import detect_loops


def score_proactivity(last_emotions: List[Emotion], summary: str | None, db: Session | None = None) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 0.0
    if not last_emotions:
        return 0.0, reasons
    # Временное затухание: вес последнего сообщения выше
    weights = [0.7, 1.0][-len(last_emotions):]
    neg = 0.0
    for e, w in zip(last_emotions[-2:], weights):
        if e in {"negative", "frustration", "sad", "fear"}:
            neg += 1.0 * w
    if neg >= 1.5:
        score += 0.5
        reasons.append("устойчиво негативный тон")
    if summary and any(w in summary.lower() for w in ["застрял", "повторяется", "без прогресса"]):
        score += 0.3
        reasons.append("повторяющийся тупик в теме")
    if db is not None:
        try:
            loop, r = detect_loops(db)
            if loop:
                score += 0.2
                reasons.extend(r)
        except Exception:
            pass
    return min(score, 1.0), reasons


def proactive_prefix(score: float, reasons: List[str]) -> str | None:
    if score >= 0.7:
        return "Заметил важное: есть сигналы, что мы буксуем. Предлагаю переосмыслить шаги."
    if score >= 0.5:
        return "Небольшое вмешательство: вижу риск буксования — задам 1–2 точных вопроса."
    return None


