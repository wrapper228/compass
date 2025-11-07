from __future__ import annotations

from typing import Literal

from app.services.llm_gateway import chat_completion

Emotion = Literal["positive", "neutral", "negative", "frustration", "sad", "fear", "excited"]


async def classify_emotion(text: str) -> Emotion:
    # Лёгкий best-effort через LLM, иначе эвристика
    try:
        msgs = [
            {"role": "system", "content": "Классификатор эмоций. Ответь одним словом из: positive, neutral, negative, frustration, sad, fear, excited."},
            {"role": "user", "content": text[:1000]},
        ]
        res = await chat_completion(msgs, False)
        if isinstance(res, str):
            ans = res.strip().lower().split()[0]
            if ans in {"positive", "neutral", "negative", "frustration", "sad", "fear", "excited"}:
                return ans  # type: ignore[return-value]
    except Exception:
        pass
    low = text.lower()
    if any(w in low for w in ["злюсь", "ненавижу", "надоело", "раздражает", "бесит"]):
        return "frustration"
    if any(w in low for w in ["печально", "грусть", "плохо", "устал"]):
        return "sad"
    if any(w in low for w in ["страх", "боюсь", "тревога"]):
        return "fear"
    if any(w in low for w in ["ура", "класс", "рад", "восторг"]):
        return "excited"
    return "neutral"


