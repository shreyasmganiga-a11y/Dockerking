import io
import os
import uuid
from typing import List, Tuple
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.schemas import DocumentChunk
from app.core.config import settings
from app.core.logging import logger


class DocumentService:
    """Service responsible for parsing and chunking document files."""

    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "(?<=[.!?]) ", " ", ""],
        )

    def extract_text_from_file(self, filename: str, content: bytes) -> List[Tuple[str, int]]:
        """
        Extracts text content and page numbers from uploaded file bytes.
        Returns a list of (page_text, page_number) tuples.
        """
        ext = os.path.splitext(filename)[1].lower()
        pages_data: List[Tuple[str, int]] = []

        if ext == ".pdf":
            try:
                reader = pypdf.PdfReader(io.BytesIO(content))
                for idx, page in enumerate(reader.pages):
                    text = (page.extract_text() or "").strip()
                    if text:
                        pages_data.append((text, idx + 1))
            except Exception as e:
                logger.error(f"Failed to read PDF '{filename}': {e}")
                raise ValueError(f"Could not parse PDF '{filename}': {e}")

        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(io.BytesIO(content))
                paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                full_text = "\n\n".join(paragraphs).strip()
                if full_text:
                    pages_data.append((full_text, 1))
            except Exception as e:
                logger.error(f"Failed to read DOCX '{filename}': {e}")
                raise ValueError(f"Could not parse DOCX '{filename}': {e}")

        elif ext in [".txt", ".md", ".markdown"]:
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    text = content.decode(encoding).strip()
                    if text:
                        pages_data.append((text, 1))
                    break
                except UnicodeDecodeError:
                    continue
        else:
            raise ValueError(f"Unsupported document format '{ext}'. Supported: .pdf, .docx, .txt, .md")

        return pages_data

    def chunk_document(self, filename: str, pages: List[Tuple[str, int]]) -> List[DocumentChunk]:
        """Splits document pages into chunks with metadata."""
        chunks: List[DocumentChunk] = []

        for text, page_num in pages:
            split_texts = self.splitter.split_text(text)
            for split_text in split_texts:
                clean_text = split_text.strip()
                if clean_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(uuid.uuid4()),
                            doc_name=filename,
                            page=page_num,
                            text=clean_text,
                        )
                    )

        return chunks


document_service = DocumentService()
