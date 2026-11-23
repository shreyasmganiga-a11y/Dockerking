import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import DocumentChunk, SearchResult, CollectionInfo


def _sanitize_error(error_msg: str) -> str:
    """Removes sensitive credentials from error messages before exposure or logging."""
    if settings.QDRANT_API_KEY and settings.QDRANT_API_KEY in error_msg:
        return error_msg.replace(settings.QDRANT_API_KEY, "[REDACTED]")
    return error_msg


class QdrantService:
    """Service managing interactions with Qdrant Cloud or local instances."""

    def __init__(self):
        self.client = self._initialize_client()

    def _initialize_client(self) -> QdrantClient:
        """Initializes connection to Qdrant Cloud using QDRANT_URL and QDRANT_API_KEY."""
        # 1. Qdrant Cloud / Remote URL connection (Primary)
        if settings.QDRANT_URL:
            is_cloud = "cloud.qdrant.io" in settings.QDRANT_URL or settings.QDRANT_URL.startswith("https://")
            if is_cloud and not settings.QDRANT_API_KEY:
                logger.warning(
                    "QDRANT_URL points to Qdrant Cloud, but QDRANT_API_KEY is not set. "
                    "Requests will fail until QDRANT_API_KEY is supplied."
                )

            logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
            return QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=10.0,
            )

        # 2. In-memory mode for offline unit testing
        if settings.QDRANT_HOST == ":memory:":
            logger.info("Initializing Qdrant in in-memory mode")
            return QdrantClient(location=":memory:")

        # 3. Host / Port fallback
        if settings.QDRANT_HOST:
            try:
                port = settings.QDRANT_PORT or 6333
                logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{port}")
                client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=port,
                    api_key=settings.QDRANT_API_KEY or None,
                    timeout=5.0,
                )
                client.get_collections()
                return client
            except Exception as e:
                logger.warning(
                    f"Could not connect to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT} ({_sanitize_error(str(e))}). "
                    "Falling back to local in-memory Qdrant client."
                )
                return QdrantClient(location=":memory:")

        logger.warning("Neither QDRANT_URL nor QDRANT_HOST is configured. Initializing in-memory Qdrant client.")
        return QdrantClient(location=":memory:")

    def check_health(self) -> Dict[str, Any]:
        """Returns health status of the Qdrant connection without exposing credentials."""
        is_cloud = bool(settings.QDRANT_URL and ("cloud.qdrant.io" in settings.QDRANT_URL or settings.QDRANT_URL.startswith("https://")))
        if is_cloud and not settings.QDRANT_API_KEY:
            return {
                "status": "unhealthy",
                "error": "QDRANT_API_KEY is required for Qdrant Cloud authentication",
                "mode": "cloud",
            }
        try:
            collections = self.client.get_collections()
            return {
                "status": "healthy",
                "collections_count": len(collections.collections),
                "mode": "cloud" if settings.QDRANT_URL else ("remote" if settings.QDRANT_HOST != ":memory:" else "in-memory"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": _sanitize_error(str(e)),
                "mode": "cloud" if settings.QDRANT_URL else "remote",
            }

    def ensure_collection(self, collection_name: str = None, vector_size: int = settings.EMBEDDING_DIMENSION):
        """Ensures the specified collection exists in Qdrant with Cosine distance metric without recreating."""
        col_name = collection_name or settings.QDRANT_COLLECTION_NAME
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == col_name for c in collections)
            if not exists:
                logger.info(f"Creating collection '{col_name}' with vector size {vector_size}")
                self.client.create_collection(
                    collection_name=col_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as e:
            sanitized = _sanitize_error(str(e))
            logger.error(f"Error ensuring collection '{col_name}': {sanitized}")
            raise RuntimeError(f"Error ensuring collection '{col_name}': {sanitized}")

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        collection_name: str = None,
    ) -> int:
        """Upserts document chunks and corresponding dense vectors into Qdrant."""
        col_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.ensure_collection(col_name, vector_size=len(embeddings[0]))

        points = []
        for chunk, vector in zip(chunks, embeddings):
            # Ensure point_id is a valid UUID or integer as required by Qdrant
            point_id = chunk.chunk_id
            try:
                uuid.UUID(str(point_id))
            except (ValueError, AttributeError):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk.chunk_id)))

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk.text,
                        "source_file": chunk.doc_name,
                        "page": chunk.page,
                    },
                )
            )

        self.client.upsert(
            collection_name=col_name,
            points=points,
            wait=True,
        )
        logger.info(f"Upserted {len(points)} chunks into collection '{col_name}'")
        return len(points)

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = settings.DEFAULT_TOP_K,
        collection_name: str = None,
    ) -> List[SearchResult]:
        """Searches for top-k similar chunks using cosine similarity."""
        col_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.ensure_collection(col_name, vector_size=len(query_vector))

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=col_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                )
                hits = response.points
            else:
                hits = self.client.search(
                    collection_name=col_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
        except Exception as e:
            sanitized = _sanitize_error(str(e))
            logger.error(f"Qdrant search error in '{col_name}': {sanitized}")
            raise RuntimeError(f"Qdrant search error: {sanitized}")

        results: List[SearchResult] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                SearchResult(
                    id=str(hit.id),
                    text=payload.get("text", ""),
                    source_file=payload.get("source_file", "Unknown"),
                    page=payload.get("page", 1),
                    score=float(hit.score),
                )
            )
        return results

    def list_collections(self) -> List[CollectionInfo]:
        """Returns metadata for all available collections."""
        try:
            cols = self.client.get_collections().collections
            info_list = []
            for c in cols:
                try:
                    c_info = self.client.get_collection(c.name)
                    info_list.append(
                        CollectionInfo(
                            name=c.name,
                            status=str(c_info.status),
                            vectors_count=c_info.vectors_count or 0,
                            points_count=c_info.points_count or 0,
                        )
                    )
                except Exception:
                    info_list.append(CollectionInfo(name=c.name, status="available"))
            return info_list
        except Exception as e:
            logger.error(f"Error listing collections: {_sanitize_error(str(e))}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Deletes a collection and its vector index."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {_sanitize_error(str(e))}")
            return False


qdrant_service = QdrantService()
