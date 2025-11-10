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
    local_embedding_model: str | None = None
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_model_fast: str | None = None
    llm_model_smart: str | None = None
    ethics_mode: str = "standard"  # or "experimental"

    # Retrieval stack configuration
    data_dir: str = "data"
    retrieval_index_path: str = "data/indexes"
    retrieval_dense_top_k: int = 12
    retrieval_sparse_top_k: int = 24
    retrieval_rrf_k: int = 60
    retrieval_mmr_lambda: float = 0.5
    retrieval_self_check_max_iterations: int = 2
    retrieval_max_context_tokens: int = 1400
    retrieval_enabled: bool = True
    cross_encoder_model: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # Loads from environment if present


