"""
Automation API endpoints integration
Routes automation endpoints into the main FastAPI application
"""
from fastapi import APIRouter
from app.automation.api.automation_endpoints import router as automation_router
from app.automation.api.apinode_mcp_endpoints import router as apinode_mcp_router

router = APIRouter()

# Include automation endpoints with prefix
router.include_router(
    automation_router,
    prefix="/automation",
    tags=["AI Automation"],
)

# Include APINode MCP endpoints
router.include_router(
    apinode_mcp_router,
    prefix="/automation",
    tags=["APINode MCP Tools"],
)