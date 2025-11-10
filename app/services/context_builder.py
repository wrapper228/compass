from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

import structlog

from app.core.config import get_settings
from app.services.llm_gateway import chat_completion
from app.services.tokenization import count_tokens
from app.services.retrieval_pipeline import RetrievalHit

logger = structlog.get_logger(__name__)


@dataclass
class ContextQuote:
    text: str
    citation: str


@dataclass
class ContextCluster:
    path: str
    summary: str
    quotes: List[ContextQuote]


@dataclass
class ContextBundle:
    prompt: str
    clusters: List[ContextCluster]


async def build_context(hits: Iterable[RetrievalHit]) -> ContextBundle:
    settings = get_settings()
    clusters = _cluster_hits(hits)
    context_clusters: List[ContextCluster] = []
    total_tokens = 0
    budget = settings.retrieval_max_context_tokens
    for path, path_hits in clusters.items():
        summary = await _summarise_cluster(path, path_hits)
        quotes = _build_quotes(path_hits)
        cluster_text = _format_cluster(path, summary, quotes)
        cluster_tokens = count_tokens(cluster_text)
        if total_tokens + cluster_tokens > budget and context_clusters:
            break
        total_tokens += cluster_tokens
        context_clusters.append(ContextCluster(path=path, summary=summary, quotes=quotes))
    prompt = _format_prompt(context_clusters)
    return ContextBundle(prompt=prompt, clusters=context_clusters)


def _cluster_hits(hits: Iterable[RetrievalHit]) -> Dict[str, List[RetrievalHit]]:
    clusters: Dict[str, List[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        clusters[getattr(hit, "path", "unknown")].append(hit)
    return clusters


async def _summarise_cluster(path: str, hits: List[RetrievalHit]) -> str:
    joined = " \n".join(getattr(h, "text", "") for h in hits)
    prompt = [
        {
            "role": "system",
            "content": (
                "Суммаризуй фрагменты документа коротко (2 предложения). Укажи ключевые факты."
            ),
        },
        {
            "role": "user",
            "content": f"Документ: {path}\nФрагменты:\n{joined}\n\nСуммаризуй:",
        },
    ]
    try:
        summary = await chat_completion(prompt, is_complex=False)
        if summary:
            return summary.strip()
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("context.summary_failed", error=str(exc))
    # Fallback: take first sentence of first hit
    text = getattr(hits[0], "text", "")
    sentence = text.split(".")[0]
    return sentence.strip()


def _build_quotes(hits: List[RetrievalHit]) -> List[ContextQuote]:
    quotes: List[ContextQuote] = []
    for hit in hits:
        snippet = getattr(hit, "text", "")[:400].strip()
        citation = getattr(hit, "citation", getattr(hit, "path", ""))
        quotes.append(ContextQuote(text=snippet, citation=citation))
    return quotes


def _format_cluster(path: str, summary: str, quotes: List[ContextQuote]) -> str:
    quotes_text = "\n".join(f"- {q.text} ({q.citation})" for q in quotes)
    return f"Документ: {path}\nРезюме: {summary}\nЦитаты:\n{quotes_text}"


def _format_prompt(clusters: List[ContextCluster]) -> str:
    if not clusters:
        return ""
    parts = [
        "Контекст из базы знаний. Используй факты и цитаты, указывай источник в ответе.",
    ]
    for cluster in clusters:
        quotes_text = "\n".join(f"- {quote.text} ({quote.citation})" for quote in cluster.quotes)
        parts.append(
            f"[Документ: {cluster.path}]\nРезюме: {cluster.summary}\nЦитаты:\n{quotes_text}"
        )
    return "\n\n".join(parts)


