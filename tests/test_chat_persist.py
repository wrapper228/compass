from fastapi.testclient import TestClient

from app.main import app
from app.db.session import SessionLocal
from app.db import models


def test_chat_persists_session_and_messages():
    client = TestClient(app)
    payload = {"messages": [{"role": "user", "content": "Привет"}]}
    r = client.post("/api/chat/generate", json=payload)
    assert r.status_code == 200

    db = SessionLocal()
    try:
        sessions = db.query(models.SessionModel).all()
        messages = db.query(models.Message).all()
        assert len(sessions) >= 1
        assert len(messages) >= 2  # user + assistant
    finally:
        db.close()


