from fastapi.testclient import TestClient

from app.main import app


def test_chat_generate_stub():
    client = TestClient(app)
    payload = {
        "messages": [
            {"role": "user", "content": "Привет, мир"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    r = client.post("/api/chat/generate", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["message"]["role"] == "assistant"
    assert "Принято" in data["message"]["content"]

