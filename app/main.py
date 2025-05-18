from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="LLM Enterprise Platform",
    description="Enterprise-grade LLM platform with FastAPI, AutoGen, and MCP",
    version="0.1.0"
)

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
# from app.api.v1 import router as api_v1_router
# app.include_router(api_v1_router, prefix="/api/v1") 