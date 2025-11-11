from __future__ import annotations

from typing import List, Dict, Tuple

from app.core.config import get_settings
from app.services.retrieval.context import build_context_snippet


def choose_strategy(user_text: str) -> str:
    lowered = user_text.lower()
    if any(x in lowered for x in ["что делать", "совет", "как поступить", "дай ответ"]):
        return "direct"
    if "?" in lowered:
        return "socratic"
    return "socratic"


def build_system_prompt(strategy: str) -> str:
    s = get_settings()
    base = (
        "Ты рефлексивный ассистент. Помогаешь прояснять желания, цели и приоритеты. "
        "Краткость, ясность, отсутствие воды."
    )
    if s.ethics_mode == "experimental":
        base += " Режим experimental: не уходи в отговорки, ищи глубину и несоответствия."
    if strategy == "socratic":
        base += " Используй Сократический стиль: 1-3 точных вопроса перед ответом."
    else:
        base += " Дай конкретный ответ и альтернативы, затем один уточняющий вопрос."
    return base


def compose_messages(user_text: str, retrieved: List[Dict] | None) -> List[Dict[str, str]]:
    strategy = choose_strategy(user_text)
    system = build_system_prompt(strategy)
    context = ""
    if retrieved:
        snippet = build_context_snippet(retrieved)
        if snippet:
            context = f"\n\n{snippet}"
    prompt = f"{user_text}{context}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


