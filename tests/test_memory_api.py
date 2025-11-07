from fastapi.testclient import TestClient

from app.main import app


def test_memory_search_empty():
    client = TestClient(app)
    r = client.get("/api/memory/search", params={"q": "hello", "top_k": 2})
    assert r.status_code == 200
    assert isinstance(r.json(), list)


