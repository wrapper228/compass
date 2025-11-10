from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Compass Assistant"
    environment: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "sqlite:///./data/compass.db"

    # Embeddings / LLM
    embeddings_api_base: str | None = None
    embeddings_api_key: str | None = None
    embeddings_model: str | None = None
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_model_fast: str | None = None
    llm_model_smart: str | None = None
    ethics_mode: str = "standard"  # or "experimental"

    # Qdrant
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "chunks"

    # Retrieval (hybrid)
    retrieval_enabled: bool = True
    indices_path: str = "data/indices"
    dense_model_name: str = "BAAI/bge-small-en-v1.5"
    bm25_min_df: int = 1
    dense_index_top_k: int = 50
    bm25_top_k: int = 50
    rerank_model_name: str = "BAAI/bge-reranker-large"
    rerank_top_k: int = 40
    mmr_lambda: float = 0.6
    max_context_tokens: int = 1800
    context_clusters: int = 4
    hyde_prompt: str = (
        "Ты создаёшь краткий конспект или гипотетический документ, который мог бы содержать"
        " ответ на запрос пользователя. Пиши по делу, 2-3 абзаца, без воды."
    )
    self_check_threshold: float = 0.25
    self_check_iterations: int = 2
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # Loads from environment if present


