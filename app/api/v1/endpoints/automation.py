"""
Automation API endpoints integration
Routes automation endpoints into the main FastAPI application
"""
from fastapi import APIRouter
from app.automation.api.automation_endpoints import router as automation_router

router = APIRouter()

# Include automation endpoints with prefix
router.include_router(
    automation_router,
    prefix="/automation",
    tags=["AI Automation"],
)