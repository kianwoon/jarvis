from fastapi import APIRouter
from app.api.v1.endpoints import document
from app.api.routes import llm

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