from fastapi import APIRouter
from app.api.v1.endpoints import mcp_tools, mcp_agent

api_router = APIRouter()

api_router.include_router(mcp_tools.router, prefix="/mcp/tools", tags=["mcp-tools"])
api_router.include_router(mcp_agent.router, prefix="/mcp/agent", tags=["mcp-agent"]) 