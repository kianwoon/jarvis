"""
RAG MCP Tool Registration
========================

Registers the RAG search service as an MCP tool for agent consumption.
"""

import logging
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, MCPTool
from app.core.mcp_tools_cache import reload_enabled_mcp_tools

logger = logging.getLogger(__name__)

def register_rag_mcp_tool():
    """Register RAG search as an MCP tool"""
    
    db = SessionLocal()
    try:
        # Check if tool already exists
        existing_tool = db.query(MCPTool).filter(MCPTool.name == "knowledge_search").first()
        
        tool_definition = {
            "name": "knowledge_search",
            "description": "Intelligent document retrieval for in-house info using vector and keyword search with automatic collection selection",
            "endpoint": "internal://rag_mcp_service",  # Internal service endpoint
            "method": "POST",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question"
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Specific collections to search. If not provided, auto-detects best collections"
                    },
                    "max_documents": {
                        "type": "integer",
                        "description": "Maximum number of documents to return",
                        "default": 8,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "include_content": {
                        "type": "boolean",
                        "description": "Whether to include full document content or just metadata",
                        "default": True
                    }
                },
                "required": ["query"]
            },
            "is_active": True,
            "is_manual": True  # Mark as manually registered system tool
        }
        
        if existing_tool:
            # Update existing tool
            for key, value in tool_definition.items():
                if key != "name":  # Don't update the name
                    setattr(existing_tool, key, value)
            logger.info("Updated existing RAG MCP tool")
        else:
            # Create new tool
            new_tool = MCPTool(**tool_definition)
            db.add(new_tool)
            logger.info("Registered new RAG MCP tool")
        
        db.commit()
        
        # Reload cache to make tool available immediately
        reload_enabled_mcp_tools()
        logger.info("RAG MCP tool registration completed successfully")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to register RAG MCP tool: {e}")
        raise
    finally:
        db.close()

def unregister_rag_mcp_tool():
    """Unregister RAG search MCP tool"""
    
    db = SessionLocal()
    try:
        tool = db.query(MCPTool).filter(MCPTool.name == "knowledge_search").first()
        if tool:
            db.delete(tool)
            db.commit()
            reload_enabled_mcp_tools()
            logger.info("RAG MCP tool unregistered successfully")
        else:
            logger.warning("RAG MCP tool not found for unregistration")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to unregister RAG MCP tool: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    register_rag_mcp_tool()