# 🚀 Gemini Qdrant Document RAG Service (Qdrant Cloud)

Production-ready, service-oriented Retrieval-Augmented Generation (RAG) backend built with **FastAPI**, **Google Gemini**, and **Qdrant Cloud Vector Database**.

---

## 🏗️ Architecture & Cloud Migration

The project communicates directly with **Qdrant Cloud** over HTTPS:

```text
GitHub → GitHub Actions → AWS ECR → AWS EC2 → FastAPI Docker Container → Qdrant Cloud (HTTPS)
```

- **Qdrant Cloud Cluster Endpoint**:
  `https://50301dc3-7940-4502-a339-e45356deead7.us-west-2-0.aws.cloud.qdrant.io:6333`
- **Vector Metric**: Cosine Similarity (Embedding Dimension: 768).
- **Embeddings**: Google Gemini `text-embedding-004` (or `gemini-embedding-001`).
- **Generation**: Google Gemini `gemini-2.5-flash` (with automated fallbacks).

---

## 📁 Project Structure

```text
RAG/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app, lifespan, CORS, error handling
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # Pydantic Settings reading .env (QDRANT_URL, QDRANT_API_KEY)
│   │   └── logging.py              # Structured application logger (sanitizes secrets)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic request & response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_service.py     # File extraction (PDF, DOCX, TXT) and sentence chunking
│   │   ├── embedding_service.py    # Google Gemini dense embeddings with retries
│   │   ├── qdrant_service.py       # Qdrant Cloud client, collections, vector upsert & search
│   │   └── rag_service.py          # Grounded Gemini generation with verified citations
│   └── api/
│       ├── __init__.py
│       └── routes.py               # REST API endpoints
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD: GitHub Actions → AWS ECR → EC2 deployment
├── Dockerfile                      # Container build definition (credential-free)
├── docker-compose.yml              # FastAPI container only (no local Qdrant container)
├── .dockerignore                   # Explicitly ignores venv/, .env, __pycache__, etc.
├── .gitignore                      # Prevents committing .env, keys, and temporary files
├── .env                            # Active environment variables (git-ignored)
├── .env.example                    # Template environment variables (safe placeholders)
├── requirements.txt                # Production dependencies (including qdrant-client)
└── README.md                       # Documentation & API reference
```

---

## 🔐 Environment & Secret Configuration

Create a local `.env` file (copied from `.env.example`):

```bash
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=text-embedding-004
EMBEDDING_DIMENSION=768

# Qdrant Cloud Vector Database
QDRANT_URL=https://50301dc3-7940-4502-a339-e45356deead7.us-west-2-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_new_qdrant_cloud_api_key_here
QDRANT_COLLECTION_NAME=rag_documents

# Chunking & Retrieval Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
DEFAULT_TOP_K=4
```

> **Important**: Never commit `.env` or hardcode `QDRANT_API_KEY` into source code. `.env` is excluded in `.gitignore` and `.dockerignore`.

---

## ⚡ Running with Docker Compose

With the migration to Qdrant Cloud, Docker Compose now manages **only the FastAPI backend container**:

```bash
docker compose up --build -d
```

- **FastAPI API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 💻 Running Locally (Python venv)

```powershell
# 1. Activate your virtual environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 🚀 GitHub Actions CI/CD Deployment

The deployment pipeline is configured in `.github/workflows/deploy.yml`.

### Required GitHub Repository Secrets

Configure the following secrets under **Repository Settings → Secrets and variables → Actions**:

| Secret Name | Description |
|---|---|
| `QDRANT_URL` | `https://50301dc3-7940-4502-a339-e45356deead7.us-west-2-0.aws.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | Your newly generated Qdrant Cloud API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `AWS_ACCESS_KEY_ID` | AWS IAM Access Key for ECR push |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM Secret Access Key for ECR push |
| `EC2_HOST` | Public IP or DNS of the AWS EC2 instance |
| `EC2_USER` | EC2 SSH username (e.g. `ubuntu` or `ec2-user`) |
| `EC2_SSH_KEY` | Private SSH key (PEM) to connect to EC2 |

### Security Measures:
1. **No Build-Time Secrets**: Credentials are never passed as Docker build arguments or stored in image layers.
2. **Secure Runtime Injection**: The workflow creates a `.env` file directly on EC2 with `chmod 600` permissions.
3. **No Secret Leaks**: Workflow logs and application logs do not print or expose API keys.
