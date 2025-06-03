from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.db import Base, engine
from app.core.healthcheck import check_all_services
from app.core.mcp_process_manager import start_process_monitor
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check service dependencies
check_all_services()

# Create database tables
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
except Exception as e:
    logger.error(f"Error creating database tables: {str(e)}")

app = FastAPI(
    title="LLM Enterprise Platform",
    description="Enterprise-grade LLM platform with FastAPI, AutoGen, and MCP",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    logger.info("Starting MCP process monitor...")
    asyncio.create_task(start_process_monitor())
    logger.info("MCP process monitor started")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0"
    }

# Import and include routers
from app.api.v1 import api_router
app.include_router(api_router, prefix="/api/v1") 