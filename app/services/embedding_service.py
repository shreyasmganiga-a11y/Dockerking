import time
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings
from app.core.logging import logger


class EmbeddingService:
    """Service for generating dense embeddings using Google Gemini."""

    def __init__(self, api_key: str = settings.GEMINI_API_KEY, model_name: str = settings.EMBEDDING_MODEL):
        self.api_key = api_key
        self.model_name = model_name

    def _get_embeddings_client(self, model: str, api_key: str) -> GoogleGenerativeAIEmbeddings:
        clean_model = model.replace("models/", "")
        return GoogleGenerativeAIEmbeddings(
            model=clean_model,
            google_api_key=api_key or settings.GEMINI_API_KEY,
        )

    def embed_documents(self, texts: List[str], api_key: str = None) -> List[List[float]]:
        """Generates embeddings for document chunks with retry and fallback."""
        active_key = api_key or self.api_key or settings.GEMINI_API_KEY
        if not active_key:
            raise ValueError("Gemini API key is required to generate embeddings.")

        candidate_models = [self.model_name.replace("models/", "")]
        for fallback in ["text-embedding-004", "gemini-embedding-001"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model in candidate_models:
            try:
                client = self._get_embeddings_client(model, active_key)
                # Batch embed
                return client.embed_documents(texts)
            except Exception as e:
                last_error = e
                err_msg = str(e)
                if any(x in err_msg for x in ["NOT_FOUND", "404", "not supported", "deprecated"]):
                    logger.warning(f"Embedding model '{model}' unavailable, trying fallback...")
                    continue
                elif "429" in err_msg or "quota" in err_msg.lower():
                    logger.warning("Embedding rate limit reached, waiting 2s...")
                    time.sleep(2)
                    try:
                        client = self._get_embeddings_client(model, active_key)
                        return client.embed_documents(texts)
                    except Exception as retry_err:
                        last_error = retry_err
                        continue
                else:
                    raise e

        raise RuntimeError(f"All embedding models failed. Last error: {last_error}")

    def embed_query(self, query: str, api_key: str = None) -> List[float]:
        """Generates embedding for a single user query."""
        embeddings = self.embed_documents([query], api_key=api_key)
        return embeddings[0]


embedding_service = EmbeddingService()
