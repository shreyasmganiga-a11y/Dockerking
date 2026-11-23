"""
Production-Ready Document Retrieval-Augmented Generation (RAG) System
Powered by Google Gemini, FAISS, and LangChain.
"""

import os
import io
import time
from typing import List, Dict, Any, Optional

import streamlit as st
import pypdf
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# ==============================================================================
# Page Configuration & Modern Aesthetics
# ==============================================================================
st.set_page_config(
    page_title="Gemini Document RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for sleek cards, citations, and modern badge chips
st.markdown(
    """
    <style>
    /* Main container tweaks */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1000px;
    }
    
    /* Header styling */
    .rag-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .rag-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .rag-header p {
        color: #71717a;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    
    /* Citation container */
    .citation-card {
        background-color: rgba(240, 242, 246, 0.6);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    @media (prefers-color-scheme: dark) {
        .citation-card {
            background-color: rgba(30, 35, 45, 0.7);
            border-color: rgba(255, 255, 255, 0.12);
        }
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-doc {
        background-color: rgba(59, 130, 246, 0.15);
        color: #2563eb;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .badge-page {
        background-color: rgba(16, 185, 129, 0.15);
        color: #059669;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .citation-snippet {
        font-style: italic;
        color: #4b5563;
        line-height: 1.45;
        white-space: pre-wrap;
    }
    @media (prefers-color-scheme: dark) {
        .citation-snippet {
            color: #d1d5db;
        }
        .badge-doc {
            color: #60a5fa;
            background-color: rgba(59, 130, 246, 0.25);
        }
        .badge-page {
            color: #34d399;
            background-color: rgba(16, 185, 129, 0.25);
        }
    }
    
    /* Status banner */
    .doc-pill {
        display: inline-block;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 12px;
        background: rgba(128, 128, 128, 0.15);
        margin: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_files_info" not in st.session_state:
    st.session_state.indexed_files_info = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "active_embedding_model" not in st.session_state:
    st.session_state.active_embedding_model = None


# ==============================================================================
# Helper Functions: PDF Parsing, Chunking & Indexing
# ==============================================================================
def parse_uploaded_pdfs(uploaded_files: List[Any]) -> List[Document]:
    """
    Extracts text and page metadata directly from uploaded PDF streams using pypdf.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        
    Returns:
        List of LangChain Document objects with source and page metadata.
    """
    documents: List[Document] = []
    
    for file in uploaded_files:
        try:
            pdf_bytes = io.BytesIO(file.read())
            # Reset file pointer for any future operations
            file.seek(0)
            
            reader = pypdf.PdfReader(pdf_bytes)
            num_pages = len(reader.pages)
            
            for page_idx in range(num_pages):
                page = reader.pages[page_idx]
                page_text = page.extract_text() or ""
                
                # Clean up whitespace
                page_text = page_text.strip()
                if page_text:
                    doc = Document(
                        page_content=page_text,
                        metadata={
                            "source": file.name,
                            "page": page_idx + 1,
                            "total_pages": num_pages,
                        },
                    )
                    documents.append(doc)
        except Exception as e:
            st.error(f"Error reading {file.name}: {str(e)}")
            
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Chunk size: 1000, Overlap: 200.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def build_faiss_vector_store(
    chunks: List[Document], api_key: str, preferred_model: str = "gemini-embedding-001"
) -> Optional[FAISS]:
    """
    Generates embeddings via Google Generative AI Embeddings and builds an in-memory FAISS index.
    Attempts preferred_model first; if a 404 NOT_FOUND occurs (e.g. retired model),
    automatically attempts fallback models like gemini-embedding-001 or text-embedding-004.
    
    Args:
        chunks: List of chunked Document objects.
        api_key: User's Google Gemini API Key.
        preferred_model: Embedding model name to try first.
        
    Returns:
        FAISS vector store instance or None on error.
    """
    # Candidate models in order of priority (stripped of any redundant 'models/' prefix)
    candidate_models = [preferred_model.replace("models/", "")]
    for fallback in ["gemini-embedding-001", "text-embedding-004"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)
            
    last_error = None
    for model_name in candidate_models:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=api_key,
            )
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.active_embedding_model = model_name
            return vector_store
        except Exception as e:
            last_error = e
            err_msg = str(e)
            # If 404 / NOT_FOUND, try next candidate
            if "NOT_FOUND" in err_msg or "404" in err_msg or "not supported" in err_msg:
                continue
            else:
                st.error(f"Vector Store indexing error: {err_msg}")
                return None
                
    st.error(f"Vector Store indexing error: {str(last_error)}")
    return None



# ==============================================================================
# Helper Functions: RAG Workflow & Prompt Grounding
# ==============================================================================
STRICT_SYSTEM_PROMPT = """You are a precise, grounded, and factual AI assistant.
Answer the user's question accurately using ONLY the provided context. If the answer cannot be determined from the context, state that clearly.

Strict Guidelines:
1. Rely ONLY on the facts explicitly mentioned in the context below.
2. Do NOT extrapolate, hallucinate, speculate, or introduce external knowledge.
3. Be direct and concise. Do NOT start answers with filler phrases like "Based on the provided documents," "According to the context," or "From the documents." Answer the question directly (e.g., "The student's name is Shreyas M.").
4. If the context does not contain sufficient information to answer the question with certainty, state: "I cannot find sufficient information in the provided documents to answer this question."
5. Maintain a clear, objective, and professional tone.
"""


def extract_text_from_chunk(chunk: Any) -> str:
    """
    Safely extracts clean string text from an LLM stream chunk.
    Handles strings, LangChain AIMessageChunk, and Google GenAI 2.0+ structured content blocks
    (e.g., [{'type': 'text', 'text': '...'}, {'type': 'text', 'extras': {...}}]).
    """
    if chunk is None:
        return ""
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part and isinstance(part["text"], str):
                    parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(str(part.text))
        return "".join(parts)
    elif isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        return ""
    return ""


def format_context_for_prompt(retrieved_docs: List[Document]) -> str:
    """Formats retrieved document chunks with clear source delimiters."""
    context_parts = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        source_name = doc.metadata.get("source", "Unknown Document")
        page_num = doc.metadata.get("page", "N/A")
        part = f"--- [Snippet {idx}] Source: {source_name} | Page: {page_num} ---\n{doc.page_content.strip()}"
        context_parts.append(part)
    return "\n\n".join(context_parts)


@st.cache_data(show_spinner=False, ttl=300)
def fetch_available_gemini_models(api_key_str: str) -> List[str]:
    """
    Queries Google GenAI API to discover supported models for generateContent.
    Falls back to a curated robust list if discovery fails or key is missing.
    """
    fallback_list = [
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]
    if not api_key_str:
        return fallback_list
    try:
        from google import genai
        client = genai.Client(api_key=api_key_str)
        found = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", []) or []
            name = (m.name or "").replace("models/", "")
            if "gemini" in name.lower() and ("generateContent" in actions or not actions):
                # Filter out embeddings, realtime, tts, or image-generation only models
                if not any(x in name.lower() for x in ["vision", "realtime", "imagen", "embedding", "thinking", "tts"]):
                    found.append(name)
        # Prioritize modern 3.x models, then flash models
        def sort_priority(model_name: str):
            is_3 = "3." in model_name or "3-" in model_name
            is_flash = "flash" in model_name
            return (not is_3, not is_flash, model_name)

        sorted_models = sorted(found, key=sort_priority)
        return sorted_models if sorted_models else fallback_list
    except Exception:
        return fallback_list


def stream_gemini_with_fallback(
    messages: List[Any],
    preferred_model: str,
    api_key: str,
):
    """
    Streams response from Gemini LLM. If preferred_model raises 404/NOT_FOUND,
    automatically falls back to next candidate model (e.g. gemini-3.8-flash, gemini-3.1-pro-preview).
    Returns:
        tuple: (generator, actual_model_used)
    """
    candidate_models = [preferred_model.replace("models/", "")]
    for fallback in [
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)
            
    last_exception = None
    for model_name in candidate_models:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.1,  # Low temperature for strict factual accuracy
                streaming=True,
            )
            stream = llm.stream(messages)
            # Fetch first chunk to verify model availability
            first_chunk = next(stream)
            
            def _gen(f_chunk, s_stream):
                first_text = extract_text_from_chunk(f_chunk)
                if first_text:
                    yield first_text
                for chunk in s_stream:
                    text = extract_text_from_chunk(chunk)
                    if text:
                        yield text
                    
            return _gen(first_chunk, stream), model_name
        except Exception as e:
            last_exception = e
            err_msg = str(e)
            if any(term in err_msg for term in ["NOT_FOUND", "404", "not supported", "no longer available", "deprecated"]):
                continue
            else:
                raise e

    raise last_exception


def answer_rag_query(
    query: str,
    vector_store: FAISS,
    api_key: str,
    model_name: str,
    top_k: int = 4,
):
    """
    Performs similarity search and calls Gemini LLM with grounded prompt.
    Returns:
        tuple: (response_stream_or_text, retrieved_docs, model_used)
    """
    # 1. Similarity Search
    retrieved_docs = vector_store.similarity_search(query, k=top_k)
    
    if not retrieved_docs:
        return None, [], model_name
    
    # 2. Build Grounded Context
    formatted_context = format_context_for_prompt(retrieved_docs)
    
    user_prompt_content = f"""Context:
{formatted_context}

Question:
{query}

Answer:"""

    messages = [
        SystemMessage(content=STRICT_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt_content),
    ]
    
    # 3. Stream with resilient fallback
    response_stream, actual_model = stream_gemini_with_fallback(
        messages=messages,
        preferred_model=model_name,
        api_key=api_key,
    )
    return response_stream, retrieved_docs, actual_model



# ==============================================================================
# Sidebar Setup & Controls
# ==============================================================================
with st.sidebar:
    st.title("⚙️ RAG Configuration")
    
    # 1. API Key Configuration
    st.subheader("1. Google Gemini API Key")
    env_api_key = os.environ.get("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Enter Gemini API Key",
        value=env_api_key,
        type="password",
        help="Get your API key at https://aistudio.google.com/app/apikey",
        placeholder="AIzaSy...",
    )
    
    if api_key:
        st.success("API Key provided", icon="✅")
    else:
        st.warning("Please supply a valid Gemini API key to proceed.", icon="🔑")
    
    # 2. Model Selection
    st.subheader("2. Model Selection")
    available_models = fetch_available_gemini_models(api_key.strip()) if api_key else [
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-1.5-flash",
    ]
    selected_model = st.selectbox(
        "Gemini LLM Variant",
        options=available_models,
        index=0,
        help="Select the Gemini model variant for answering queries. Dynamically queried when API key is provided.",
    )
    
    selected_embedding_model = st.selectbox(
        "Embedding Model",
        options=["gemini-embedding-001", "text-embedding-004"],
        index=0,
        help="Google embedding model. 'gemini-embedding-001' is active and recommended. Automatic fallback will engage if a model is deprecated/unavailable in your API tier.",
    )
    
    st.markdown("---")
    
    # 3. Document Uploader
    st.subheader("3. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to index for retrieval.",
    )
    
    # Process & Index Button
    process_btn = st.button("⚡ Process & Index Documents", use_container_width=True, type="primary")
    
    if process_btn:
        if not api_key:
            st.error("Cannot index documents without an API key. Please input your Gemini API Key above.")
        elif not uploaded_files:
            st.warning("Please select at least one PDF file before processing.")
        else:
            with st.spinner("Extracting text from PDF(s)..."):
                raw_docs = parse_uploaded_pdfs(uploaded_files)
                
            if not raw_docs:
                st.error("No readable text found in the uploaded PDF(s). Please verify the files are not scanned images.")
            else:
                progress_bar = st.progress(30, text="Splitting text into semantic chunks...")
                chunks = chunk_documents(raw_docs)
                time.sleep(0.3)
                
                progress_bar.progress(60, text=f"Generating embeddings for {len(chunks)} chunks using {selected_embedding_model}...")
                vector_store = build_faiss_vector_store(chunks, api_key, preferred_model=selected_embedding_model)
                
                if vector_store:
                    progress_bar.progress(100, text="Indexing complete!")
                    st.session_state.vector_store = vector_store
                    st.session_state.total_chunks = len(chunks)
                    st.session_state.indexed_files_info = [
                        {"name": f.name, "size_kb": round(f.size / 1024, 1)} for f in uploaded_files
                    ]
                    active_emb = st.session_state.get("active_embedding_model", selected_embedding_model)
                    st.success(f"Successfully indexed {len(chunks)} chunks across {len(uploaded_files)} PDF(s) using `{active_emb}`!")
                else:
                    progress_bar.empty()
    
    # Show active index status
    if st.session_state.vector_store is not None:
        st.markdown("---")
        st.markdown("### 📊 Active Vector Index")
        st.markdown(f"**Total Indexed Chunks:** `{st.session_state.total_chunks}`")
        st.markdown(f"**Embedding Model:** `{st.session_state.get('active_embedding_model', selected_embedding_model)}`")
        st.markdown("**Indexed Files:**")
        for f in st.session_state.indexed_files_info:
            st.markdown(f"- 📄 `{f['name']}` ({f['size_kb']} KB)")
        
        # Reset / Clear Index Button
        if st.button("🗑️ Clear Vector Index & History", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.indexed_files_info = []
            st.session_state.total_chunks = 0
            st.session_state.messages = []
            st.session_state.active_embedding_model = None
            st.rerun()


# ==============================================================================
# Main Chat & Interactive Interface
# ==============================================================================
st.markdown(
    """
    <div class="rag-header">
        <h1>📚 Document RAG Assistant</h1>
        <p>Ground-truth Q&A with Google Gemini embeddings and <code>FAISS</code> vector retrieval.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Initial info Banner if index not ready
if st.session_state.vector_store is None:
    st.info(
        "👋 **Welcome! To get started:**\n"
        "1. Enter your **Google Gemini API Key** in the sidebar.\n"
        "2. Upload one or more **PDF documents**.\n"
        "3. Click **'⚡ Process & Index Documents'** to generate vector embeddings.\n"
        "4. Ask questions grounded strictly in your document evidence.",
        icon="ℹ️",
    )


# Function to render verified source evidence expander
def render_evidence_citations(sources: List[Dict[str, Any]]):
    """Renders formatted citation evidence cards inside a styled expander."""
    if not sources:
        return
    with st.expander("🔍 Verified Source Evidence & Citations", expanded=False):
        st.caption(f"Retrieved {len(sources)} most relevant chunks from the FAISS vector index:")
        for idx, src in enumerate(sources, start=1):
            doc_name = src.get("source", "Unknown Document")
            page_num = src.get("page", "N/A")
            content = src.get("content", "").strip()
            
            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-header">
                        <span class="badge-doc">📄 {doc_name}</span>
                        <span class="badge-page">🔖 Page {page_num}</span>
                    </div>
                    <div class="citation-snippet">"{content}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Render existing conversation messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    sources = message.get("sources", [])
    
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and sources:
            render_evidence_citations(sources)


# Handle user chat input
if prompt := st.chat_input("Ask a question about your indexed documents..."):
    # Guard Checks
    if not api_key:
        st.error("Please provide your Google Gemini API Key in the sidebar before asking questions.")
    elif st.session_state.vector_store is None:
        st.warning("No documents are currently indexed! Please upload and process PDF documents first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Process Assistant Response with RAG Retrieval
        with st.chat_message("assistant"):
            with st.spinner("Searching document vector store & formulating answer..."):
                try:
                    response_stream, retrieved_docs, model_used = answer_rag_query(
                        query=prompt,
                        vector_store=st.session_state.vector_store,
                        api_key=api_key,
                        model_name=selected_model,
                        top_k=4,
                    )
                    
                    # Prepare sources list for citation rendering and session state
                    formatted_sources = [
                        {
                            "source": doc.metadata.get("source", "Unknown Document"),
                            "page": doc.metadata.get("page", "N/A"),
                            "content": doc.page_content[:600] + ("..." if len(doc.page_content) > 600 else ""),
                        }
                        for doc in retrieved_docs
                    ]
                    
                    # Stream the output directly into the chat container
                    full_response = ""
                    placeholder = st.empty()
                    for chunk in response_stream:
                        chunk_text = extract_text_from_chunk(chunk)
                        full_response += chunk_text
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    
                    # Render the citations immediately beneath the answer
                    render_evidence_citations(formatted_sources)
                    
                    # Save assistant message to conversational history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": formatted_sources,
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred while generating response: {str(e)}")
