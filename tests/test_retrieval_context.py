import numpy as np

from app.services.retrieval import context


class StubIndices:
    def encode_queries(self, texts):
        n = len(texts)
        return np.eye(n, dtype=np.float32)


class StubPipeline:
    def __init__(self):
        self.indices = StubIndices()


def test_build_context_snippet(monkeypatch):
    monkeypatch.setattr(context, "get_pipeline_instance", lambda: StubPipeline())

    retrieved = [
        {
            "text": "Первый документ о стратегии продукта. Упоминается roadmap и сроки.",
            "path": "docs/product.md",
            "start_line": 1,
            "end_line": 3,
            "score": 0.9,
        },
        {
            "text": "Второй документ о финансовых показателях и бюджете команды.",
            "path": "docs/finance.md",
            "start_line": 4,
            "end_line": 6,
            "score": 0.8,
        },
    ]

    snippet = context.build_context_snippet(retrieved)
    assert "Контекст из базы знаний" in snippet
    assert "docs/product.md" in snippet
    assert "docs/finance.md" in snippet

