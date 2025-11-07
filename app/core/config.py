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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # Loads from environment if present


