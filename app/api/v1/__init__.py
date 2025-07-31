from fastapi import APIRouter
from app.api.v1.endpoints import document, document_upload_fix, document_multi, document_multi_progress, document_preview, document_classify, document_intelligent, document_progress
from app.api.routes import llm
from app.api.v1.endpoints import settings
from app.api.v1.endpoints import mcp_tools
from app.api.v1.endpoints import mcp_servers
from app.api.v1.endpoints import system_tools
from app.api.v1.endpoints import langchain
from app.api.v1.endpoints import intelligent_chat
from app.api.v1.endpoints import langgraph_agents
from app.api.v1.endpoints import collections
from app.api.v1.endpoints import model_presets
from app.api.v1.endpoints import oauth
from app.api.v1.endpoints import oauth_flow
from app.api.v1.endpoints import agent_templates
from app.api.v1.endpoints import agent_recommendations
from app.api.v1.endpoints import automation
from app.api.v1.endpoints import temp_documents
from app.api.v1.endpoints import knowledge_graph
from app.api.v1.endpoints import knowledge_graph_schema
from app.api.v1.endpoints import prompt_management
import logging

logger = logging.getLogger(__name__)
logger.info("Importing collections router...")

api_router = APIRouter()

api_router.include_router(
    document.router,
    prefix="/documents",
    tags=["documents"]
)

api_router.include_router(
    document_progress.router,
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
    system_tools.router,
    prefix="/mcp",
    tags=["system-tools"]
)

api_router.include_router(
    langchain.router,
    prefix="/langchain",
    tags=["langchain"]
)


api_router.include_router(
    intelligent_chat.router,
    prefix="/chat",
    tags=["intelligent-chat"]
)

api_router.include_router(
    langgraph_agents.router,
    prefix="/langgraph",
    tags=["langgraph"]
)

api_router.include_router(
    document_multi.router,
    prefix="/documents",
    tags=["documents"]
)

api_router.include_router(
    document_multi_progress.router,
    prefix="/documents",
    tags=["documents"]
)

api_router.include_router(
    document_preview.router,
    prefix="/documents",
    tags=["documents"]
)

api_router.include_router(
    document_classify.router,
    prefix="/document",
    tags=["document-classification"]
)

api_router.include_router(
    document_intelligent.router,
    prefix="/documents",
    tags=["intelligent-documents"]
)

logger.info("Including collections router...")
api_router.include_router(
    collections.router,
    prefix="/collections",
    tags=["collections"]
)
logger.info("Collections router included successfully")

api_router.include_router(
    model_presets.router,
    prefix="/model-presets",
    tags=["model-presets"]
)

api_router.include_router(
    oauth.router,
    prefix="/oauth",
    tags=["oauth"]
)

api_router.include_router(
    oauth_flow.router,
    prefix="/oauth/flow",
    tags=["oauth-flow"]
)


api_router.include_router(
    agent_templates.router,
    prefix="/agent-templates",
    tags=["agent-templates"]
)


api_router.include_router(
    agent_recommendations.router,
    prefix="/agent-recommendations",
    tags=["agent-recommendations"]
)



api_router.include_router(
    automation.router,
    prefix="",
    tags=["automation"]
)

api_router.include_router(
    temp_documents.router,
    prefix="/temp-documents",
    tags=["temp-documents"]
)

api_router.include_router(
    knowledge_graph.router,
    prefix="/knowledge-graph",
    tags=["knowledge-graph"]
)

api_router.include_router(
    knowledge_graph_schema.router,
    prefix="/knowledge-graph",
    tags=["knowledge-graph-schema"]
)

api_router.include_router(
    prompt_management.router,
    prefix="/prompts",
    tags=["prompts"]
)
