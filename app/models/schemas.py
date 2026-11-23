from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a chunked slice of an ingested document."""
    chunk_id: str = Field(..., description="Unique ID for chunk")
    doc_name: str = Field(..., description="Source document file name")
    page: int = Field(default=1, description="Page number if applicable")
    text: str = Field(..., description="Text content of the chunk")


class IngestResponse(BaseModel):
    """Response returned upon document ingestion."""
    status: str = "success"
    collection_name: str
    total_files: int
    total_chunks: int
    filenames: List[str]
    message: str


class EvidenceItem(BaseModel):
    """Reference citation linking answers to source documents."""
    source_id: int = Field(..., description="Numbered reference index e.g. 1, 2")
    source_file: str = Field(..., description="Originating filename")
    page: int = Field(..., description="Page number")
    snippet: str = Field(..., description="Verbatim quote or relevant excerpt")
    score: Optional[float] = Field(default=None, description="Similarity score from Qdrant")


class ChatRequest(BaseModel):
    """Request payload for RAG question answering."""
    question: str = Field(..., min_length=1, description="User question to answer")
    collection_name: Optional[str] = Field(default=None, description="Target Qdrant collection")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Number of context chunks to retrieve")
    model: Optional[str] = Field(default=None, description="Optional override for Gemini model")


class ChatResponse(BaseModel):
    """Grounded answer with verified evidence citations."""
    answer: str = Field(..., description="Direct answer derived strictly from document context")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Source citations")
    model_used: str = Field(..., description="Gemini model that generated the answer")
    total_retrieved: int = Field(..., description="Number of context passages searched")


class SearchRequest(BaseModel):
    """Direct semantic similarity search request."""
    query: str = Field(..., min_length=1, description="Text to search for")
    collection_name: Optional[str] = Field(default=None, description="Target Qdrant collection")
    top_k: Optional[int] = Field(default=4, ge=1, le=50, description="Max results")


class SearchResult(BaseModel):
    """Individual search result item from Qdrant."""
    id: str
    text: str
    source_file: str
    page: int
    score: float


class SearchResponse(BaseModel):
    """Response containing retrieved search results."""
    query: str
    collection_name: str
    total_results: int
    results: List[SearchResult]


class CollectionInfo(BaseModel):
    """Status and statistics of a Qdrant collection."""
    name: str
    status: str
    vectors_count: Optional[int] = 0
    points_count: Optional[int] = 0


class HealthResponse(BaseModel):
    """Service health status and dependency checks."""
    status: str
    version: str
    qdrant_status: str
    qdrant_host: str
    gemini_model: str
