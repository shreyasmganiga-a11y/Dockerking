import re
from typing import List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.core.logging import logger
from app.models.schemas import EvidenceItem, ChatResponse, SearchResult
from app.services.qdrant_service import qdrant_service
from app.services.embedding_service import embedding_service


STRICT_SYSTEM_PROMPT = """You are a precise, grounded, and factual Retrieval-Augmented Generation (RAG) assistant.
Answer the user's question accurately using ONLY the provided context snippets.

Strict Guidelines:
1. Rely ONLY on the facts explicitly stated in the context passages below.
2. In your answer, cite the supporting context passages using inline numerical markers like [1], [2] immediately after each claim.
3. Do NOT extrapolate, hallucinate, or bring in outside world knowledge.
4. If the context does not contain sufficient information to answer the question, state:
   "I cannot find sufficient information in the provided documents to answer this question."
5. Be direct, clear, objective, and professional.
"""


class RAGService:
    """Service orchestrating document retrieval and Gemini grounded generation."""

    def __init__(self):
        self.default_model = settings.GEMINI_MODEL

    def _format_context(self, search_results: List[SearchResult]) -> Tuple[str, List[EvidenceItem]]:
        """Formats context snippets for LLM prompt and compiles evidence items."""
        context_parts = []
        evidence_items: List[EvidenceItem] = []

        for idx, item in enumerate(search_results, start=1):
            snippet = item.text.strip()
            # Delimited context block
            block = f"--- [Snippet {idx}] Source: {item.source_file} | Page: {item.page} ---\n{snippet}"
            context_parts.append(block)

            evidence_items.append(
                EvidenceItem(
                    source_id=idx,
                    source_file=item.source_file,
                    page=item.page,
                    snippet=snippet[:250] + "..." if len(snippet) > 250 else snippet,
                    score=round(item.score, 4),
                )
            )

        return "\n\n".join(context_parts), evidence_items

    def answer_query(
        self,
        question: str,
        collection_name: str = None,
        top_k: int = None,
        model_name: str = None,
        api_key: str = None,
    ) -> ChatResponse:
        """Executes full RAG workflow: embed query -> search Qdrant -> invoke Gemini LLM."""
        active_key = api_key or settings.GEMINI_API_KEY
        if not active_key:
            raise ValueError("Gemini API key is required to query the RAG service.")

        active_top_k = top_k or settings.DEFAULT_TOP_K
        preferred_model = model_name or self.default_model

        # 1. Embed query
        query_vector = embedding_service.embed_query(question, api_key=active_key)

        # 2. Retrieve top-k similar chunks from Qdrant
        search_results = qdrant_service.search_similar(
            query_vector=query_vector,
            limit=active_top_k,
            collection_name=collection_name,
        )

        if not search_results:
            return ChatResponse(
                answer="No relevant documents found in the database. Please ingest documents first.",
                evidence=[],
                model_used=preferred_model,
                total_retrieved=0,
            )

        # 3. Format grounded prompt
        formatted_context, evidence = self._format_context(search_results)
        user_prompt = f"""Context Information:
{formatted_context}

User Question: {question}

Direct Grounded Answer:"""

        messages = [
            SystemMessage(content=STRICT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # 4. Invoke Gemini with fallback models
        candidate_models = [preferred_model.replace("models/", "")]
        for fallback in ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-3.8-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        answer_text = ""
        actual_model_used = preferred_model
        last_error = None

        for cand_model in candidate_models:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=cand_model,
                    google_api_key=active_key,
                    temperature=0.1,
                )
                response = llm.invoke(messages)
                content = response.content
                if isinstance(content, str):
                    answer_text = content
                elif isinstance(content, list):
                    answer_text = "".join(str(p.get("text", p) if isinstance(p, dict) else p) for p in content)
                else:
                    answer_text = str(content)
                actual_model_used = cand_model
                break
            except Exception as e:
                last_error = e
                err_msg = str(e)
                if any(x in err_msg for x in ["NOT_FOUND", "404", "not supported", "deprecated"]):
                    logger.warning(f"Gemini model '{cand_model}' failed with 404, attempting fallback...")
                    continue
                else:
                    raise e

        if not answer_text and last_error:
            raise RuntimeError(f"Failed to generate answer from Gemini: {last_error}")

        return ChatResponse(
            answer=answer_text.strip(),
            evidence=evidence,
            model_used=actual_model_used,
            total_retrieved=len(search_results),
        )


rag_service = RAGService()
