from fastapi.testclient import TestClient

from app.main import app


def test_ws_chat_streams():
    client = TestClient(app)
    with client.websocket_connect("/api/ws/chat") as ws:
        ws.send_text("ping")
        # Должны получить хотя бы один токен и [DONE]
        text = ws.receive_text()
        assert isinstance(text, str)
        # Считываем остаток до DONE
        done = False
        for _ in range(20):
            msg = ws.receive_text()
            if msg == "[DONE]":
                done = True
                break
        assert done


