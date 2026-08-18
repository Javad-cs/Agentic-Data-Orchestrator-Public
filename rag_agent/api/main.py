import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

from src.config.models import SystemConfig
from src.agents import create_fast_lane

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown:
    - Load config
    - Initialize Fast Lane
    - Cleanup on shutdown
    """
    # Startup
    logger.info("Starting RAG API...")
    
    try:
        # Load config
        config = SystemConfig()
        app.state.config = config
        
        # Initialize Fast Lane
        logger.info("Initializing Fast Lane...")
        fast_lane = await create_fast_lane(config)
        app.state.fast_lane = fast_lane
        
        logger.info(" RAG API ready")
        
        yield
        
    finally:
        # Shutdown
        logger.info("Shutting down RAG API...")
        
        if hasattr(app.state, 'fast_lane'):
            await app.state.fast_lane.close()
        
        logger.info(" RAG API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="Enterprise RAG system with Fast/Slow Lane architecture",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes (imported after app creation to avoid circular import)
app.include_router(router)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "fast_lane_initialized": hasattr(app.state, 'fast_lane')
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "RAG Agent API",
        "version": "0.1.0",
        "endpoints": {
            "query": "/query (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }