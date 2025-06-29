"""
API Endpoints for AI Automation with Langflow Integration
Following existing FastAPI patterns from the codebase
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import uuid
import asyncio
from datetime import datetime

from app.automation.core.automation_cache import (
    get_automation_workflows, 
    get_workflow_by_id, 
    invalidate_workflow_cache,
    cache_workflow_execution,
    get_workflow_execution
)
from app.automation.integrations.postgres_bridge import postgres_bridge
from app.automation.core.automation_executor import AutomationExecutor
from app.automation.api.node_schema_endpoint import router as node_schema_router

logger = logging.getLogger(__name__)
router = APIRouter()

# Include node schema endpoints
router.include_router(node_schema_router, prefix="/schema", tags=["Node Schema"])

# Pydantic models
class WorkflowCreate(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    langflow_config: Dict[str, Any] = Field(..., description="Langflow configuration JSON")
    trigger_config: Optional[Dict[str, Any]] = Field(None, description="Trigger configuration")
    is_active: bool = Field(True, description="Whether workflow is active")
    created_by: Optional[str] = Field("user", description="Creator identifier")

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    langflow_config: Optional[Dict[str, Any]] = None
    trigger_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class WorkflowExecuteRequest(BaseModel):
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data for workflow")
    execution_mode: str = Field("sync", description="Execution mode: sync, async, or stream")
    message: Optional[str] = Field(None, description="Message to trigger workflow (for agent-based workflows)")

class WorkflowExecuteStreamRequest(BaseModel):
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data for workflow")
    message: Optional[str] = Field(None, description="Message to trigger workflow")

class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    langflow_config: Dict[str, Any]
    trigger_config: Optional[Dict[str, Any]]
    is_active: bool
    created_by: Optional[str] = "system"
    created_at: Optional[str]
    updated_at: Optional[str]

class ExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: int
    status: str
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    execution_log: Optional[List[Dict[str, Any]]]
    error_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]

@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows():
    """List all automation workflows"""
    try:
        workflows_dict = get_automation_workflows()
        workflows_list = []
        
        for workflow_id, workflow_data in workflows_dict.items():
            workflows_list.append(WorkflowResponse(**workflow_data))
        
        logger.info(f"[AUTOMATION API] Listed {len(workflows_list)} workflows")
        return workflows_list
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(workflow: WorkflowCreate):
    """Create new automation workflow"""
    try:
        # Create workflow in database
        workflow_id = postgres_bridge.create_workflow(workflow.dict())
        
        if not workflow_id:
            raise HTTPException(status_code=500, detail="Failed to create workflow")
        
        # Invalidate cache to refresh
        invalidate_workflow_cache()
        
        # Get created workflow
        created_workflow = postgres_bridge.get_workflow(workflow_id)
        if not created_workflow:
            raise HTTPException(status_code=500, detail="Failed to retrieve created workflow")
        
        logger.info(f"[AUTOMATION API] Created workflow: {workflow_id}")
        return WorkflowResponse(**created_workflow)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error creating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: int):
    """Get specific workflow by ID"""
    try:
        workflow = get_workflow_by_id(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowResponse(**workflow)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error getting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(workflow_id: int, updates: WorkflowUpdate):
    """Update automation workflow"""
    try:
        # Update in database
        success = postgres_bridge.update_workflow(workflow_id, updates.dict(exclude_unset=True))
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or update failed")
        
        # Invalidate cache
        invalidate_workflow_cache()
        
        # Get updated workflow
        updated_workflow = postgres_bridge.get_workflow(workflow_id)
        if not updated_workflow:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated workflow")
        
        logger.info(f"[AUTOMATION API] Updated workflow: {workflow_id}")
        return WorkflowResponse(**updated_workflow)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error updating workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: int):
    """Delete automation workflow"""
    try:
        success = postgres_bridge.delete_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Invalidate cache
        invalidate_workflow_cache()
        
        logger.info(f"[AUTOMATION API] Deleted workflow: {workflow_id}")
        return {"message": "Workflow deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error deleting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/execute")
async def execute_workflow(
    workflow_id: int, 
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks
):
    """Execute automation workflow"""
    try:
        # Get workflow
        workflow = get_workflow_by_id(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if not workflow.get("is_active", False):
            raise HTTPException(status_code=400, detail="Workflow is not active")
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Create execution record
        execution_data = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "status": "running",
            "input_data": request.input_data,
            "execution_log": []
        }
        
        db_execution_id = postgres_bridge.create_execution(execution_data)
        if not db_execution_id:
            raise HTTPException(status_code=500, detail="Failed to create execution record")
        
        # Cache execution for real-time tracking
        cache_workflow_execution(execution_id, execution_data)
        
        if request.execution_mode == "async":
            # Execute in background
            background_tasks.add_task(
                execute_workflow_background,
                workflow_id,
                execution_id,
                workflow.get("langflow_config", {}),
                request.input_data
            )
            
            return {
                "execution_id": execution_id,
                "status": "running",
                "message": "Workflow execution started in background"
            }
        else:
            # Execute synchronously
            try:
                executor = AutomationExecutor()
                result = await executor.execute_workflow(
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    langflow_config=workflow.get("langflow_config", {}),
                    input_data=request.input_data,
                    message=request.message
                )
                
                return {
                    "execution_id": execution_id,
                    "status": "completed",
                    "result": result
                }
            except Exception as e:
                # Update execution with error
                postgres_bridge.update_execution(execution_id, {
                    "status": "failed",
                    "error_message": str(e)
                })
                
                logger.error(f"[AUTOMATION API] Workflow execution failed: {e}")
                raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error executing workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/execute/stream")
async def execute_workflow_stream(
    workflow_id: int, 
    request: WorkflowExecuteStreamRequest
):
    """Execute workflow with real-time streaming updates"""
    from fastapi.responses import StreamingResponse
    import json
    
    try:
        # Get workflow
        workflow = get_workflow_by_id(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if not workflow.get("is_active", False):
            raise HTTPException(status_code=400, detail="Workflow is not active")
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # Create execution record
        execution_data = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "status": "running",
            "input_data": request.input_data,
            "message": request.message,
            "execution_log": []
        }
        
        db_execution_id = postgres_bridge.create_execution(execution_data)
        if not db_execution_id:
            raise HTTPException(status_code=500, detail="Failed to create execution record")
        
        # Cache execution for real-time tracking
        cache_workflow_execution(execution_id, execution_data)
        
        async def stream_workflow_execution():
            try:
                executor = AutomationExecutor()
                
                async for update in executor.execute_workflow_stream(
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    langflow_config=workflow.get("langflow_config", {}),
                    input_data=request.input_data,
                    message=request.message
                ):
                    yield f"data: {json.dumps(update)}\n\n"
                    
            except Exception as e:
                logger.error(f"[AUTOMATION API] Streaming execution failed: {e}")
                error_update = {
                    "type": "workflow_error",
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_update)}\n\n"
        
        return StreamingResponse(
            stream_workflow_execution(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error starting streaming execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(execution_id: str):
    """Get execution status and results"""
    try:
        # Try cache first for real-time data
        execution = get_workflow_execution(execution_id)
        
        # Fallback to database
        if not execution:
            execution = postgres_bridge.get_execution(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return ExecutionResponse(**execution)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error getting execution {execution_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions")
async def list_all_executions(limit: int = 50):
    """List recent executions across all workflows"""
    try:
        from app.core.db import SessionLocal, AutomationExecution
        db = SessionLocal()
        try:
            executions = db.query(AutomationExecution).order_by(
                AutomationExecution.started_at.desc()
            ).limit(limit).all()
            
            result = []
            for execution in executions:
                result.append({
                    "id": execution.id,
                    "workflow_id": execution.workflow_id,
                    "execution_id": execution.execution_id,
                    "status": execution.status,
                    "started_at": execution.started_at.isoformat() if execution.started_at else None,
                    "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                    "error_message": execution.error_message
                })
            
            return {
                "executions": result,
                "total": len(result)
            }
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error listing all executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workflows/{workflow_id}/executions")
async def list_workflow_executions(workflow_id: int, limit: int = 50):
    """List recent executions for a workflow"""
    try:
        executions = postgres_bridge.get_workflow_executions(workflow_id, limit)
        return {
            "workflow_id": workflow_id,
            "executions": executions,
            "total": len(executions)
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error listing executions for workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/integrations/tools")
async def list_available_tools():
    """List available MCP tools for workflow design"""
    try:
        from app.automation.integrations.mcp_bridge import mcp_bridge
        tools = mcp_bridge.get_available_tools()
        return {
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/integrations/agents")
async def list_available_agents():
    """List available agents for workflow design"""
    try:
        from app.automation.integrations.agent_bridge import agent_bridge
        agents = agent_bridge.get_available_agents()
        return {
            "agents": agents,
            "count": len(agents)
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_automation_status():
    """Get automation system status"""
    try:
        from app.automation.core.automation_cache import get_cache_stats
        from app.automation.integrations.redis_bridge import workflow_redis
        
        cache_stats = get_cache_stats()
        redis_status = workflow_redis.get_connection_status()
        
        return {
            "status": "healthy",
            "cache": cache_stats,
            "redis": redis_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error getting status: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/cache/reload")
async def reload_cache():
    """Reload automation cache"""
    try:
        from app.automation.core.automation_cache import reload_automation_workflows
        workflows = reload_automation_workflows()
        return {
            "message": "Cache reloaded successfully",
            "workflows_cached": len(workflows),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error reloading cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Langflow Integration Endpoints

@router.post("/integrations/tools/execute")
async def execute_mcp_tool_for_langflow(request: Dict[str, Any]):
    """Execute MCP tool for Langflow custom nodes"""
    try:
        from app.automation.integrations.mcp_bridge import mcp_bridge
        
        tool_name = request.get("tool_name")
        parameters = request.get("parameters", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")
        
        # Execute tool via MCP bridge
        result = mcp_bridge.execute_tool_sync(tool_name, parameters)
        
        return {
            "success": result.get("success", False),
            "result": result.get("result"),
            "error": result.get("error"),
            "execution_time": 0  # TODO: Add timing
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error executing MCP tool {request.get('tool_name')}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/integrations/agents/execute") 
async def execute_agent_for_langflow(request: Dict[str, Any]):
    """Execute agent for Langflow custom nodes"""
    try:
        from app.automation.integrations.agent_bridge import agent_bridge
        
        agent_name = request.get("agent_name")
        query = request.get("query")
        context = request.get("context")
        
        if not agent_name or not query:
            raise HTTPException(status_code=400, detail="agent_name and query are required")
        
        # Execute agent via agent bridge
        result = agent_bridge.execute_agent_sync(agent_name, query, context)
        
        return result
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error executing agent {request.get('agent_name')}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/integrations/redis/execute")
async def execute_redis_for_langflow(request: Dict[str, Any]):
    """Execute Redis operation for Langflow custom nodes"""
    try:
        from app.automation.integrations.redis_bridge import workflow_redis
        
        operation = request.get("operation")
        key = request.get("key")
        value = request.get("value")
        ttl = request.get("ttl", 3600)
        
        if not operation or not key:
            raise HTTPException(status_code=400, detail="operation and key are required")
        
        result = None
        
        if operation == "get":
            result = workflow_redis.get(key)
        elif operation == "set":
            if value is None:
                raise HTTPException(status_code=400, detail="value is required for set operation")
            workflow_redis.set(key, value, expire=ttl)
            result = "OK"
        elif operation == "delete":
            result = workflow_redis.delete(key)
        elif operation == "exists":
            result = workflow_redis.exists(key)
        elif operation == "keys":
            result = workflow_redis.keys(key)  # key is pattern for keys operation
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported operation: {operation}")
        
        return {
            "success": True,
            "result": result,
            "operation": operation,
            "key": key
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error executing Redis operation {request.get('operation')}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_workflow_background(
    workflow_id: int,
    execution_id: str,
    langflow_config: Dict[str, Any],
    input_data: Optional[Dict[str, Any]]
):
    """Background task for async workflow execution"""
    try:
        logger.info(f"[AUTOMATION API] Starting background execution: {execution_id}")
        
        executor = AutomationExecutor()
        result = await executor.execute_workflow(
            workflow_id=workflow_id,
            execution_id=execution_id,
            langflow_config=langflow_config,
            input_data=input_data
        )
        
        # Update execution with results
        postgres_bridge.update_execution(execution_id, {
            "status": "completed",
            "output_data": result
        })
        
        logger.info(f"[AUTOMATION API] Background execution completed: {execution_id}")
        
    except Exception as e:
        logger.error(f"[AUTOMATION API] Background execution failed: {e}")
        
        # Update execution with error
        postgres_bridge.update_execution(execution_id, {
            "status": "failed",
            "error_message": str(e)
        })