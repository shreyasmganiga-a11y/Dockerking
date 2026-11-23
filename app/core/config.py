from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_NAME: str = "Gemini Qdrant RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Google Gemini Settings
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    EMBEDDING_MODEL: str = "text-embedding-004"
    EMBEDDING_DIMENSION: int = 768

    # Qdrant Vector Database Settings (Qdrant Cloud / Remote)
    QDRANT_URL: Optional[str] = "https://50301dc3-7940-4502-a339-e45356deead7.us-west-2-0.aws.cloud.qdrant.io:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_HOST: Optional[str] = None
    QDRANT_PORT: Optional[int] = None
    QDRANT_COLLECTION_NAME: str = "rag_documents"

    # Document Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    DEFAULT_TOP_K: int = 4


settings = Settings()
