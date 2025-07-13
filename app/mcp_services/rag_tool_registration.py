"""
RAG MCP Tool Registration
========================

Registers the RAG search service as an MCP tool for agent consumption.
No more hardcoding - uses configurable settings.
"""

import logging
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, MCPTool
from app.core.mcp_tools_cache import reload_enabled_mcp_tools
from app.core.rag_tool_config import get_rag_tool_config

logger = logging.getLogger(__name__)

def register_rag_mcp_tool():
    """Register RAG search as an MCP tool using configurable settings"""
    
    # Get configuration (no more hardcoding!)
    config = get_rag_tool_config()
    
    # Get available collections for the description
    from app.core.collection_registry_cache import get_all_collections
    collections_list = []
    try:
        collections = get_all_collections()
        if collections:
            collections_list = [f"{c.get('collection_name', '')}" for c in collections]
    except Exception as e:
        logger.warning(f"Could not fetch collections for tool description: {e}")
    
    # Build collections description
    collections_desc = "Optional: Specific collections to search. If not provided, auto-detects best collections."
    if collections_list:
        collections_desc += f" Available collections: {', '.join(collections_list)}"
    
    db = SessionLocal()
    try:
        # Check if tool already exists (using configurable name)
        tool_name = config.get_tool_name()
        existing_tool = db.query(MCPTool).filter(MCPTool.name == tool_name).first()
        
        # Get tool definition from config (no hardcoding!)
        tool_definition = config.get_tool_definition(collections_desc)
        
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
    """Unregister RAG search MCP tool using configurable settings"""
    
    # Get configuration (no more hardcoding!)
    config = get_rag_tool_config()
    tool_name = config.get_tool_name()
    
    db = SessionLocal()
    try:
        tool = db.query(MCPTool).filter(MCPTool.name == tool_name).first()
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