from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Header, HTTPException, status

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import (
    IngestResponse,
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    CollectionInfo,
    HealthResponse,
)
from app.services.document_service import document_service
from app.services.embedding_service import embedding_service
from app.services.qdrant_service import qdrant_service
from app.services.rag_service import rag_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify backend service and Qdrant Cloud connectivity."""
    qdrant_health = qdrant_service.check_health()
    endpoint = settings.QDRANT_URL or (f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}" if settings.QDRANT_HOST else "configured")
    return HealthResponse(
        status="healthy" if qdrant_health.get("status") == "healthy" else "degraded",
        version=settings.APP_VERSION,
        qdrant_status=qdrant_health.get("status", "unknown"),
        qdrant_host=endpoint,
        gemini_model=settings.GEMINI_MODEL,
    )


@router.post(
    "/documents/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Upload and ingest documents into Qdrant Cloud",
)
@router.post(
    "/index-documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF, DOCX, TXT, or MD documents"),
    collection_name: Optional[str] = Form(None, description="Optional target collection name"),
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key"),
):
    """
    Ingests uploaded documents:
    1. Extracts text and page metadata.
    2. Chunks documents using sentence-aware splitting.
    3. Generates dense embeddings using Google Gemini.
    4. Upserts vectors and chunk payloads into Qdrant Cloud.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for upload.",
        )

    api_key = x_gemini_api_key or settings.GEMINI_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gemini API key is required. Set GEMINI_API_KEY in .env or pass X-Gemini-API-Key header.",
        )

    target_collection = collection_name or settings.QDRANT_COLLECTION_NAME
    all_chunks = []
    processed_files = []

    for file in files:
        filename = file.filename or "uploaded_file"
        try:
            content = await file.read()
            if not content:
                continue

            pages = document_service.extract_text_from_file(filename, content)
            if not pages:
                logger.warning(f"File '{filename}' yielded no readable text.")
                continue

            chunks = document_service.chunk_document(filename, pages)
            all_chunks.extend(chunks)
            processed_files.append(filename)
        except Exception as e:
            logger.error(f"Error processing file '{filename}': {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process '{filename}': {str(e)}",
            )

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No extractable text found across all uploaded documents.",
        )

    # Generate Embeddings
    try:
        texts = [c.text for c in all_chunks]
        embeddings = embedding_service.embed_documents(texts, api_key=api_key)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings: {str(e)}",
        )

    # Store in Qdrant Cloud
    try:
        total_upserted = qdrant_service.upsert_chunks(
            chunks=all_chunks,
            embeddings=embeddings,
            collection_name=target_collection,
        )
    except Exception as e:
        logger.error(f"Qdrant storage error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store vectors in Qdrant: {str(e)}",
        )

    return IngestResponse(
        status="success",
        collection_name=target_collection,
        total_files=len(processed_files),
        total_chunks=total_upserted,
        filenames=processed_files,
        message=f"Successfully indexed {len(processed_files)} file(s) into {total_upserted} chunks in Qdrant Cloud.",
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat & RAG"],
    summary="Query document knowledge with grounded Gemini generation",
)
@router.post(
    "/query",
    response_model=ChatResponse,
    include_in_schema=False,
)
async def chat_with_documents(
    request: ChatRequest,
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key"),
):
    """
    RAG conversational query:
    1. Embeds question.
    2. Retrieves top-k similar chunks from Qdrant Cloud.
    3. Submits grounded context to Google Gemini.
    4. Returns grounded answer with citations and verified evidence.
    """
    api_key = x_gemini_api_key or settings.GEMINI_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gemini API key is required. Set GEMINI_API_KEY in .env or pass X-Gemini-API-Key header.",
        )

    try:
        response = rag_service.answer_query(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            model_name=request.model,
            api_key=api_key,
        )
        return response
    except Exception as e:
        logger.error(f"RAG query execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Direct semantic similarity search on Qdrant Cloud",
)
async def semantic_search(
    request: SearchRequest,
    x_gemini_api_key: Optional[str] = Header(None, alias="X-Gemini-API-Key"),
):
    """Directly queries Qdrant Cloud for semantic similarity matches without calling LLM generation."""
    api_key = x_gemini_api_key or settings.GEMINI_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Gemini API key is required to embed query.",
        )

    target_collection = request.collection_name or settings.QDRANT_COLLECTION_NAME

    try:
        query_vector = embedding_service.embed_query(request.query, api_key=api_key)
        results = qdrant_service.search_similar(
            query_vector=query_vector,
            limit=request.top_k,
            collection_name=target_collection,
        )
        return SearchResponse(
            query=request.query,
            collection_name=target_collection,
            total_results=len(results),
            results=results,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/collections",
    response_model=List[CollectionInfo],
    tags=["Collections"],
    summary="List Qdrant collections",
)
async def list_collections():
    """Lists all vector collections in the Qdrant Cloud cluster."""
    return qdrant_service.list_collections()


@router.delete(
    "/collections/{collection_name}",
    tags=["Collections"],
    summary="Delete a Qdrant collection",
)
async def delete_collection(collection_name: str):
    """Removes a collection and all stored document vectors from Qdrant Cloud."""
    success = qdrant_service.delete_collection(collection_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found or could not be deleted.",
        )
    return {"status": "success", "message": f"Collection '{collection_name}' successfully deleted."}
