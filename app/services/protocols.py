from __future__ import annotations

from typing import List


def next_reflective_steps(user_text: str) -> List[str]:
    steps: List[str] = []
    steps.append("Сформулируй текущее желание/цель одной фразой (до 15 слов).")
    steps.append("Назови 1 ключевую преграду и 1 ресурс, который уже есть.")
    steps.append("Опиши следующий минимальный шаг (<= 20 минут) и время, когда его сделаешь.")
    return steps


