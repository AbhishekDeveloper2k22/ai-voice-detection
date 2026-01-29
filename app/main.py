"""
AI Voice Detection API - Main Application
Detects whether a voice sample is AI-generated or Human
Supports: Tamil, English, Hindi, Malayalam, Telugu
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from app.routes import voice_detection_router
from app.config import get_settings
from app import __version__


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"🚀 AI Voice Detection API v{__version__} starting up...")
    settings = get_settings()
    logger.info(f"📌 Supported languages: {', '.join(settings.supported_languages)}")
    logger.info(f"🔐 API authentication enabled")
    
    yield
    
    # Shutdown
    logger.info("👋 AI Voice Detection API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="AI Voice Detection API",
    description="""
## AI-Generated Voice Detection System

This API analyzes voice samples to determine if they are **AI-generated** or spoken by a **human**.

### Supported Languages
- 🇮🇳 **Tamil**
- 🇬🇧 **English**
- 🇮🇳 **Hindi**
- 🇮🇳 **Malayalam**
- 🇮🇳 **Telugu**

### Features
- Base64-encoded MP3 audio input
- Real-time voice analysis
- Confidence scoring (0.0 - 1.0)
- Detailed explanations for decisions

### Authentication
All requests require an API key in the `x-api-key` header.
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    """
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(
    voice_detection_router,
    prefix="/api",
    tags=["Voice Detection"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    settings = get_settings()
    return {
        "name": "AI Voice Detection API",
        "version": __version__,
        "status": "running",
        "supported_languages": settings.supported_languages,
        "docs": "/docs",
        "endpoints": {
            "health": "/api/health",
            "voice_detection": "/api/voice-detection"
        }
    }


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
