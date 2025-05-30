from fastapi import APIRouter
from app.api.v1.endpoints import document, document_upload_fix
from app.api.routes import llm
from app.api.v1.endpoints import settings
from app.api.v1.endpoints import mcp_tools
from app.api.v1.endpoints import mcp_servers
from app.api.v1.endpoints import langchain
from app.api.v1.endpoints import langgraph_agents

api_router = APIRouter()

api_router.include_router(
    document.router,
    prefix="/documents",
    tags=["documents"]
)

api_router.include_router(
    document_upload_fix.router,
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

api_router.include_router(
    mcp_servers.router,
    prefix="/mcp/servers",
    tags=["mcp"]
)

api_router.include_router(
    langchain.router,
    prefix="/langchain",
    tags=["langchain"]
)

api_router.include_router(
    langgraph_agents.router,
    prefix="/langgraph",
    tags=["langgraph"]
) 