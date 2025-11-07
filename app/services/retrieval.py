from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import get_settings


def get_qdrant() -> Optional[QdrantClient]:
    s = get_settings()
    if not s.qdrant_url:
        return None
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)  # type: ignore[arg-type]


def ensure_collection(client: QdrantClient, vector_size: int) -> None:
    s = get_settings()
    collections = [c.name for c in client.get_collections().collections]
    if s.qdrant_collection not in collections:
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )


def upsert_points(client: QdrantClient, vectors: List[List[float]], payloads: List[dict]) -> None:
    s = get_settings()
    points = [
        qm.PointStruct(id=payload.get("id"), vector=vectors[i], payload=payloads[i])
        for i in range(len(payloads))
    ]
    client.upsert(collection_name=s.qdrant_collection, points=points)


def search(client: QdrantClient, vector: List[float], top_k: int = 6, filters: Optional[dict] = None) -> List[dict]:
    s = get_settings()
    res = client.search(collection_name=s.qdrant_collection, query_vector=vector, limit=top_k)
    return [
        {"score": r.score, **(r.payload or {})} for r in res
    ]


