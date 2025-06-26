"""
System Tools Management API
===========================

Manages internal system tools like RAG search, document processing, etc.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, MCPTool
from app.core.mcp_tools_cache import reload_enabled_mcp_tools
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SystemToolResponse(BaseModel):
    id: int
    name: str
    description: str
    endpoint: str
    method: str
    parameters: Dict[str, Any]
    is_active: bool
    is_internal: bool
    tool_type: str
    version: str

@router.get("/system-tools", response_model=List[SystemToolResponse])
async def get_system_tools(db: Session = Depends(get_db)):
    """Get all system tools"""
    try:
        # Get tools that are internal services or system tools
        system_tools = db.query(MCPTool).filter(
            MCPTool.endpoint.like('internal://%') |
            (MCPTool.is_manual == True)
        ).all()
        
        result = []
        for tool in system_tools:
            result.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description or "System tool",
                "endpoint": tool.endpoint,
                "method": tool.method,
                "parameters": tool.parameters or {},
                "is_active": tool.is_active,
                "is_internal": tool.endpoint.startswith('internal://') if tool.endpoint else False,
                "tool_type": "internal" if tool.endpoint and tool.endpoint.startswith('internal://') else "system",
                "version": "1.0.0"  # Can be enhanced to track versions
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get system tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system-tools/reinstall")
async def reinstall_system_tools(db: Session = Depends(get_db)):
    """Reinstall/re-register all system tools"""
    try:
        logger.info("Reinstalling system tools...")
        
        # Re-register RAG tool
        from app.mcp_services.rag_tool_registration import register_rag_mcp_tool
        register_rag_mcp_tool()
        
        # Add other system tools here as they're created
        # Example: register_document_processor_tool()
        # Example: register_collection_manager_tool()
        
        # Reload cache
        reload_enabled_mcp_tools()
        
        logger.info("System tools reinstalled successfully")
        return {"success": True, "message": "System tools reinstalled successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reinstall system tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reinstall system tools: {str(e)}")

@router.put("/system-tools/{tool_id}/status")
async def toggle_system_tool_status(tool_id: int, db: Session = Depends(get_db)):
    """Toggle system tool active status"""
    try:
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="System tool not found")
        
        # Check if it's actually a system tool
        if not (tool.endpoint and tool.endpoint.startswith('internal://')) and not tool.is_manual:
            raise HTTPException(status_code=400, detail="Not a system tool")
        
        # Toggle status
        tool.is_active = not tool.is_active
        db.commit()
        
        # Reload cache
        reload_enabled_mcp_tools()
        
        return {
            "success": True, 
            "message": f"Tool {tool.name} {'enabled' if tool.is_active else 'disabled'}",
            "is_active": tool.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to toggle system tool status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-tools/{tool_id}")
async def get_system_tool_details(tool_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific system tool"""
    try:
        tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
        if not tool:
            raise HTTPException(status_code=404, detail="System tool not found")
        
        return {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "endpoint": tool.endpoint,
            "method": tool.method,
            "parameters": tool.parameters,
            "headers": tool.headers,
            "is_active": tool.is_active,
            "is_manual": tool.is_manual,
            "created_at": tool.created_at,
            "updated_at": tool.updated_at,
            "server_id": tool.server_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system tool details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system-tools/cache/reload")
async def reload_system_tools_cache():
    """Reload MCP tools cache without reinstalling"""
    try:
        logger.info("Reloading MCP tools cache...")
        reload_enabled_mcp_tools()
        logger.info("MCP tools cache reloaded successfully")
        return {"success": True, "message": "MCP tools cache reloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to reload MCP tools cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload cache: {str(e)}")

@router.post("/system-tools/validate")
async def validate_system_tools():
    """Validate that all system tools are properly registered and functional"""
    try:
        results = {}
        
        # Test RAG tool
        try:
            from app.mcp_services.rag_mcp_service import execute_rag_search
            import asyncio
            
            # Quick test search
            test_result = await execute_rag_search(
                query="test validation query",
                max_documents=1,
                include_content=False
            )
            
            results["rag_search"] = {
                "status": "healthy" if test_result.get("jsonrpc") == "2.0" else "error",
                "response_time": test_result.get("result", {}).get("execution_time_ms", 0),
                "details": "RAG search responding normally"
            }
            
        except Exception as e:
            results["rag_search"] = {
                "status": "error",
                "details": f"RAG search validation failed: {str(e)}"
            }
        
        # Add validation for other system tools here
        
        overall_status = "healthy" if all(
            result["status"] == "healthy" for result in results.values()
        ) else "degraded"
        
        return {
            "overall_status": overall_status,
            "tools": results,
            "timestamp": "2024-01-01T00:00:00Z"  # Will be enhanced with actual timestamp
        }
        
    except Exception as e:
        logger.error(f"System tools validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))