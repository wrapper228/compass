from typing import AsyncGenerator, Iterable, List, Dict, Optional

import httpx

from app.core.config import get_settings


async def stream_reply_stub(text: str) -> AsyncGenerator[str, None]:
    for token in tokenize_for_stream(text):
        yield token


def tokenize_for_stream(text: str) -> Iterable[str]:
    for word in text.split():
        yield word + " "


def select_model(is_complex: bool) -> Optional[str]:
    s = get_settings()
    if is_complex and s.llm_model_smart:
        return s.llm_model_smart
    return s.llm_model_fast or s.llm_model_smart


async def chat_completion(messages: List[Dict[str, str]], is_complex: bool = False) -> Optional[str]:
    s = get_settings()
    if not (s.llm_api_base and s.llm_api_key):
        return None
    model = select_model(is_complex)
    if not model:
        return None
    url = f"{s.llm_api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {s.llm_api_key}"}
    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content


