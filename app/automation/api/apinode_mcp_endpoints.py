"""
APINode MCP Tool Execution Endpoints
Handles execution of APINode tools called by agents through MCP interface
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.automation.integrations.apinode_mcp_bridge import apinode_mcp_bridge
from app.core.langfuse_integration import get_tracer

logger = logging.getLogger(__name__)
router = APIRouter()

class APINodeToolExecutionRequest(BaseModel):
    """Request model for APINode tool execution"""
    tool_name: str
    parameters: Dict[str, Any]
    workflow_id: str
    execution_id: str
    
class APINodeToolExecutionResponse(BaseModel):
    """Response model for APINode tool execution"""
    success: bool
    result: Any = None
    error: str = None
    metadata: Dict[str, Any] = {}

@router.post("/internal/workflow/apinode/execute", response_model=APINodeToolExecutionResponse)
async def execute_apinode_tool(request: APINodeToolExecutionRequest):
    """
    Execute an APINode tool call from an agent
    
    This endpoint is called by the MCP tool execution system when an agent
    calls an APINode tool that was registered for a workflow.
    """
    tracer = get_tracer()
    
    try:
        logger.info(f"[APINODE MCP EXECUTION] Executing tool: {request.tool_name}")
        logger.info(f"[APINODE MCP EXECUTION] Parameters: {request.parameters}")
        
        # Execute the APINode tool
        result = await apinode_mcp_bridge.execute_apinode_tool(
            tool_name=request.tool_name,
            parameters=request.parameters,
            workflow_id=request.workflow_id,
            execution_id=request.execution_id
        )
        
        logger.info(f"[APINODE MCP EXECUTION] Tool execution completed: {result.get('success', False)}")
        
        return APINodeToolExecutionResponse(
            success=result.get("success", False),
            result=result.get("result"),
            error=result.get("error"),
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"[APINODE MCP EXECUTION] Tool execution failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"APINode tool execution failed: {str(e)}"
        )

@router.get("/internal/workflow/apinode/tools/{workflow_id}")
async def get_workflow_apinode_tools(workflow_id: str):
    """
    Get all APINode tools registered for a specific workflow
    
    This endpoint can be used for debugging and monitoring
    """
    try:
        tools = apinode_mcp_bridge.get_workflow_tools(workflow_id)
        return {
            "workflow_id": workflow_id,
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"[APINODE MCP] Failed to get workflow tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workflow tools: {str(e)}"
        )

@router.get("/internal/workflow/apinode/tools")
async def get_all_registered_apinode_tools():
    """
    Get all registered APINode tools across all workflows
    
    This endpoint can be used for debugging and monitoring
    """
    try:
        registered_tools = apinode_mcp_bridge.get_all_registered_tools()
        return {
            "registered_tools": list(registered_tools),
            "count": len(registered_tools)
        }
    except Exception as e:
        logger.error(f"[APINODE MCP] Failed to get all registered tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get all registered tools: {str(e)}"
        )

@router.delete("/internal/workflow/apinode/cleanup")
async def cleanup_all_apinode_tools():
    """
    Clean up all registered APINode tools
    
    This endpoint can be used for debugging and reset operations
    """
    try:
        apinode_mcp_bridge.cleanup_all_workflow_tools()
        return {"message": "All APINode tools cleaned up successfully"}
    except Exception as e:
        logger.error(f"[APINODE MCP] Failed to cleanup tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup tools: {str(e)}"
        )