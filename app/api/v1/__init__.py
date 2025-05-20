from fastapi import APIRouter
from app.api.v1.endpoints import document
from app.api.routes import llm
from app.api.v1.endpoints import settings
from app.api.v1.endpoints import mcp_tools

api_router = APIRouter()

api_router.include_router(
    document.router,
    prefix="/documents",
    tags=["documents"]
)

api_router.include_router(
    llm.router,
    prefix="",
    tags=["llm"]
)

api_router.include_router(
    settings.router,
    prefix="/settings",
    tags=["settings"]
)

api_router.include_router(
    mcp_tools.router,
    prefix="/mcp/tools",
    tags=["mcp"]
) 