import pytest

from app.core.config import get_settings
from app.services.llm_gateway import chat_completion


@pytest.mark.asyncio
async def test_chat_completion_handles_non_ascii_key(monkeypatch, caplog):
    get_settings.cache_clear()
    monkeypatch.setenv("LLM_API_BASE", "http://example.com")
    monkeypatch.setenv("LLM_API_KEY", "“bad-key”")
    monkeypatch.setenv("LLM_MODEL_FAST", "gpt-test")
    monkeypatch.setenv("RETRIEVAL_ENABLED", "false")

    caplog.clear()
    result = await chat_completion([{"role": "user", "content": "привет"}])

    assert result is None
    assert any("LLM_API_KEY содержит не-ASCII" in message for message in caplog.messages)
