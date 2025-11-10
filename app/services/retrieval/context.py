from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

import tiktoken
from sklearn.cluster import MiniBatchKMeans

from app.core.config import get_settings
from app.services.retrieval import get_pipeline_instance

_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class ClusterBlock:
    title: str
    citations: List[str]
    score: float


def build_context_snippet(retrieved: List[dict]) -> str:
    if not retrieved:
        return ""

    settings = get_settings()
    limit = max(1, min(len(retrieved), settings.context_clusters * 3))
    top_items = retrieved[:limit]
    pipeline = get_pipeline_instance()

    texts = [item["text"] for item in top_items]
    embeddings = pipeline.indices.encode_queries(texts)

    n_clusters = min(settings.context_clusters, len(top_items))
    if n_clusters <= 0:
        return ""
    if len(top_items) == 1:
        clusters = [ClusterBlock(
            title=_summarize_text(top_items[0]["text"]),
            citations=[_format_citation(top_items[0])],
            score=top_items[0].get("score", 0.0),
        )]
    else:
        labels = _cluster_embeddings(embeddings, n_clusters)
        clusters = _build_clusters(labels, top_items)

    clusters.sort(key=lambda c: c.score, reverse=True)

    max_tokens = settings.max_context_tokens
    blocks: List[str] = []
    remaining_tokens = max_tokens

    for cluster in clusters:
        for citations_count in range(min(3, len(cluster.citations)), 0, -1):
            block = _render_block(cluster, citations_count)
            tokens = len(_encoding.encode(block))
            if tokens <= remaining_tokens:
                blocks.append(block)
                remaining_tokens -= tokens
                break

    if not blocks:
        return ""

    return "Контекст из базы знаний:\n" + "\n\n".join(blocks)


def _cluster_embeddings(embeddings, n_clusters: int) -> List[int]:
    if len(embeddings) <= n_clusters:
        return list(range(len(embeddings)))
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=max(8, n_clusters * 2),
        n_init="auto",
    )
    labels = km.fit_predict(embeddings)
    return labels.tolist()


def _build_clusters(labels: Sequence[int], items: List[dict]) -> List[ClusterBlock]:
    grouped: List[List[dict]] = [[] for _ in range(max(labels) + 1)]
    for idx, label in enumerate(labels):
        grouped[label].append(items[idx])

    clusters: List[ClusterBlock] = []
    for group in grouped:
        if not group:
            continue
        group_sorted = sorted(group, key=lambda it: it.get("score", 0.0), reverse=True)
        title = _summarize_text(" ".join(it["text"] for it in group_sorted[:2]))
        citations = [_format_citation(it) for it in group_sorted]
        clusters.append(
            ClusterBlock(
                title=title,
                citations=citations,
                score=group_sorted[0].get("score", 0.0),
            )
        )
    return clusters


def _summarize_text(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text[:200]
    summary = sentences[0]
    if len(summary) < 120 and len(sentences) > 1:
        summary = summary + " " + sentences[1]
    return summary.strip()[:300]


def _format_citation(item: dict) -> str:
    path = item.get("path", "unknown")
    start = item.get("start_line", 0)
    end = item.get("end_line", 0)
    snippet = item.get("text", "").strip().replace("\n", " ")
    if len(snippet) > 260:
        snippet = snippet[:260].rstrip() + "…"
    return f"[{path}:{start}-{end}] {snippet}"


def _render_block(cluster: ClusterBlock, citations_count: int) -> str:
    citations = "\n".join(f"- {c}" for c in cluster.citations[:citations_count])
    return f"• {cluster.title}\n{citations}"

