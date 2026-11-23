from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.logging import logger
from app.api.routes import router as api_router
from app.services.qdrant_service import qdrant_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to initialize connections and collections."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    try:
        is_cloud = bool(settings.QDRANT_URL and ("cloud.qdrant.io" in settings.QDRANT_URL or settings.QDRANT_URL.startswith("https://")))
        if is_cloud and not settings.QDRANT_API_KEY:
            logger.warning(
                "QDRANT_API_KEY is not configured. Qdrant Cloud collection initialization deferred "
                "until valid credentials are provided."
            )
        else:
            qdrant_service.ensure_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vector_size=settings.EMBEDDING_DIMENSION,
            )
            logger.info(f"Default Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' verified.")
    except Exception as e:
        logger.warning(f"Initial Qdrant collection verification warning: {e}")
    
    yield
    
    logger.info("Shutting down RAG Service.")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Production-grade FastAPI service implementing Retrieval-Augmented Generation (RAG) "
        "powered by Google Gemini models and Qdrant Cloud Vector Database."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", include_in_schema=False)
async def root_health():
    """Top-level health endpoint."""
    qdrant_health = qdrant_service.check_health()
    return {
        "status": "healthy" if qdrant_health.get("status") == "healthy" else "degraded",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "qdrant": qdrant_health,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
