"""
API Endpoints for AI Automation with CORRECTED Langfuse Integration
Following EXACT standard chat mode Langfuse pattern from langchain.py
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
from app.automation.api.cache_endpoint import router as cache_router
from app.automation.core.resource_monitor import get_resource_monitor
from app.core.unified_mcp_service import get_mcp_pool_stats
from app.langchain.dynamic_agent_system import agent_instance_pool
from app.core.redis_client import get_redis_pool_info

logger = logging.getLogger(__name__)
router = APIRouter()

# Include node schema endpoints
router.include_router(node_schema_router, prefix="/schema", tags=["Node Schema"])

# Include cache management endpoints
router.include_router(cache_router, prefix="", tags=["Cache Management"])

# Include external trigger endpoints
from app.automation.api.trigger_endpoints import router as trigger_router
router.include_router(trigger_router, prefix="/external", tags=["External Triggers"])

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
        invalidate_workflow_cache(workflow_id)
        
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
        # Validate workflow data before update
        update_data = updates.dict(exclude_unset=True)
        if 'langflow_config' in update_data:
            langflow_config = update_data['langflow_config']
            if langflow_config and isinstance(langflow_config, dict):
                nodes = langflow_config.get('nodes', [])
                if not nodes:
                    logger.error(f"[AUTOMATION API] Rejecting workflow update: No nodes in langflow_config")
                    raise HTTPException(
                        status_code=400, 
                        detail="Invalid workflow configuration: No nodes found. This would make the workflow non-executable."
                    )
        
        # Update in database
        success = postgres_bridge.update_workflow(workflow_id, update_data)
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found or update failed")
        
        # Invalidate cache for this specific workflow
        invalidate_workflow_cache(workflow_id)
        
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
        
        # Invalidate cache for this specific workflow
        invalidate_workflow_cache(workflow_id)
        
        logger.info(f"[AUTOMATION API] Deleted workflow: {workflow_id}")
        return {"message": "Workflow deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error deleting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions")
async def list_executions(limit: int = 10):
    """List recent workflow executions"""
    try:
        # Get recent executions from database (includes automatic cleanup)
        executions = postgres_bridge.get_recent_executions(limit)
        
        logger.info(f"[AUTOMATION API] Listed {len(executions)} executions")
        return {
            "executions": executions,
            "total": len(executions)
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error listing executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/executions/cleanup")
async def cleanup_stale_executions():
    """Manually cleanup stale running executions"""
    try:
        cleaned_count = postgres_bridge.cleanup_stale_executions(timeout_minutes=30)
        logger.info(f"[AUTOMATION API] Manually cleaned up {cleaned_count} stale executions")
        return {
            "message": f"Cleaned up {cleaned_count} stale executions",
            "cleaned_count": cleaned_count
        }
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error cleaning up executions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/workflows/{workflow_id}/execute/stream")
async def execute_workflow_stream(
    workflow_id: int, 
    request: WorkflowExecuteStreamRequest
):
    """Execute workflow with CORRECTED Langfuse tracing - Following EXACT standard chat mode pattern"""
    from fastapi.responses import StreamingResponse
    import json
    
    # DEBUG LOGGING for workflow execution
    logger.info(f"[WORKFLOW EXECUTION DEBUG] === EXECUTING WORKFLOW ID {workflow_id} ===")
    logger.info(f"[WORKFLOW EXECUTION DEBUG] Request data: {request.dict()}")
    logger.info(f"[WORKFLOW EXECUTION DEBUG] Input data: {request.input_data}")
    logger.info(f"[WORKFLOW EXECUTION DEBUG] Message: {request.message}")
    
    # EXACT SAME LANGFUSE PATTERN AS STANDARD CHAT MODE
    from app.core.langfuse_integration import get_tracer
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Temporarily get model from LLM settings, will update after workflow is loaded
    from app.core.llm_settings_cache import get_llm_settings
    llm_settings = get_llm_settings()
    model_name = llm_settings.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    logger.debug(f"Automation endpoint - tracer enabled: {tracer.is_enabled()}")
    
    try:
        # Get workflow
        logger.info(f"[WORKFLOW EXECUTION DEBUG] Fetching workflow {workflow_id} from cache/database...")
        workflow = get_workflow_by_id(workflow_id)
        if not workflow:
            logger.error(f"[WORKFLOW EXECUTION DEBUG] Workflow {workflow_id} not found!")
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        logger.info(f"[WORKFLOW EXECUTION DEBUG] Workflow {workflow_id} found. Name: {workflow.get('name', 'Unknown')}")
        logger.info(f"[WORKFLOW EXECUTION DEBUG] Workflow is_active: {workflow.get('is_active', False)}")
        
        if not workflow.get("is_active", False):
            logger.error(f"[WORKFLOW EXECUTION DEBUG] Workflow {workflow_id} is not active!")
            raise HTTPException(status_code=400, detail="Workflow is not active")
        
        # Extract models from agent nodes in the workflow
        workflow_models = []
        
        # Use langflow_config as that's what the automation executor uses
        langflow_config = workflow.get("langflow_config", {})
        nodes = langflow_config.get("nodes", [])
        
        logger.info(f"[AUTOMATION API] Workflow keys: {list(workflow.keys())}")
        logger.info(f"[AUTOMATION API] Langflow config found: {'langflow_config' in workflow}")
        logger.info(f"[AUTOMATION API] Workflow has {len(nodes)} nodes")
        
        for node in nodes:
            node_type = node.get("type")
            logger.info(f"[AUTOMATION API] Node type: {node_type}, id: {node.get('id')}")
            
            if node_type in ["AgentNode", "agentnode"]:
                # Use the EXACT same extraction logic as agent_workflow_executor
                node_data = node.get("data", {})
                logger.info(f"[AUTOMATION API] AgentNode data keys: {list(node_data.keys())}")
                
                # The model is stored in node_data.node.model based on the data structure
                node_info = node_data.get("node", {})
                logger.info(f"[AUTOMATION API] node_info keys: {list(node_info.keys())}")
                
                # Extract model from the correct path
                node_model = node_info.get("model")
                
                # Fallback to other possible locations
                agent_config = node_data.get("agent_config", {})
                if not node_model:
                    node_model = agent_config.get("model")
                
                if not node_model:
                    node_model = node_data.get("model")
                
                logger.info(f"[AUTOMATION API] AgentNode model: {node_model}")
                logger.info(f"[AUTOMATION API] node_info.get('model'): {node_info.get('model')}")
                logger.info(f"[AUTOMATION API] agent_config.get('model'): {agent_config.get('model')}")
                logger.info(f"[AUTOMATION API] node_data.get('model'): {node_data.get('model')}")
                
                if node_model:
                    workflow_models.append(node_model)
        
        # Update model name if models found in workflow  
        if workflow_models:
            model_name = workflow_models[0]
            logger.info(f"[AUTOMATION API] Using model from workflow config: {model_name}, all models: {workflow_models}")
        else:
            logger.info(f"[AUTOMATION API] No models in workflow, using LLM settings: {model_name}")
            # If no models found in workflow, use the same pattern as RAG endpoint
            from app.core.llm_settings_cache import get_main_llm_full_config
            main_llm_config = get_main_llm_full_config(llm_settings)
            model_name = main_llm_config.get("model", "unknown")
            logger.info(f"[AUTOMATION API] Updated to use main LLM config model: {model_name}")
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        
        # EXACT SAME TRACE CREATION AS STANDARD CHAT MODE
        if tracer.is_enabled():
            trace = tracer.create_trace(
                name="automation-workflow",  # Similar to "rag-workflow"
                input=request.message or str(request.input_data),
                metadata={
                    "endpoint": "/api/v1/automation/workflows/execute/stream",
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "has_message": request.message is not None,
                    "has_input_data": request.input_data is not None,
                    "model": model_name,
                    "models_in_workflow": workflow_models if workflow_models else []
                }
            )
            logger.debug(f"Automation trace created: {trace is not None}")
            
            # EXACT SAME EXECUTION SPAN AS STANDARD CHAT MODE
            automation_span = None
            if trace:
                automation_span = tracer.create_span(
                    trace,
                    name="automation-execution",  # Similar to "rag-execution"
                    metadata={
                        "operation": "automation_workflow_execution",
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "model": model_name
                    }
                )
                logger.debug(f"Automation span created: {automation_span is not None}")
            
            # REMOVED: automation-generation should NOT exist
            # Generations are only for actual LLM calls, not workflow coordination
            # The DynamicMultiAgentSystem will create proper generations for each agent's LLM calls
        
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
            """Stream response following EXACT standard chat mode pattern"""
            try:
                chunk_count = 0
                collected_output = ""
                final_answer = ""
                
                logger.info(f"[WORKFLOW EXECUTION DEBUG] Creating AutomationExecutor for workflow {workflow_id}")
                executor = AutomationExecutor()
                
                logger.info(f"[WORKFLOW EXECUTION DEBUG] About to call execute_workflow_stream")
                logger.info(f"[WORKFLOW EXECUTION DEBUG] Execution ID: {execution_id}")
                logger.info(f"[WORKFLOW EXECUTION DEBUG] Langflow config nodes count: {len(workflow.get('langflow_config', {}).get('nodes', []))}")
                logger.debug(f"Automation API - About to call execute_workflow_stream with trace")
                
                # EXACT SAME PATTERN: Pass trace to service layer (like rag_answer)
                async for update in executor.execute_workflow_stream(
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    langflow_config=workflow.get("langflow_config", {}),
                    input_data=request.input_data,
                    message=request.message,
                    trace=trace  # PASS TRACE EXACTLY LIKE STANDARD CHAT MODE
                ):
                    chunk_count += 1
                    event_type = update.get("type", "unknown")
                    
                    # Stream events in real-time (like standard chat)
                    yield f"data: {json.dumps(update)}\n\n"
                    
                    # Collect final answer for trace completion
                    if event_type == "workflow_result":
                        final_answer = update.get("response", "")
                        collected_output += json.dumps(update)
                    elif event_type == "workflow_complete":
                        if not final_answer:
                            final_answer = "Automation workflow completed"
                
                logger.debug(f"Automation API - Processed {chunk_count} events")
                
                # TRACE COMPLETION - No generation to end since we removed automation-generation
                if tracer.is_enabled():
                    try:
                        workflow_output = final_answer if final_answer else "Automation workflow completed"
                        
                        # Update trace with final result
                        if trace:
                            trace.update(
                                output=workflow_output,
                                metadata={
                                    "success": True,
                                    "source": "automation_workflow",
                                    "response_length": len(final_answer) if final_answer else len(collected_output),
                                    "streaming": True,
                                    "workflow_id": workflow_id,
                                    "execution_id": execution_id,
                                    "model": model_name,  # Include the model from workflow config
                                    "models_used": workflow_models if workflow_models else [model_name]  # All models in workflow
                                }
                            )
                            logger.debug(f"Automation trace updated")
                        
                        # Flush traces (copy from langchain.py)
                        tracer.flush()
                        logger.debug(f"Automation traces flushed")
                    except Exception as e:
                        logger.warning(f"Failed to update Langfuse trace/generation: {e}")
                        
            except Exception as e:
                logger.error(f"[AUTOMATION API] Streaming execution failed: {e}")
                
                # ERROR HANDLING - No generation to end since we removed automation-generation  
                if tracer.is_enabled():
                    try:
                        error_output = f"Error: {str(e)}"
                        
                        # Update trace with error (exact pattern from langchain.py)
                        if trace:
                            trace.update(
                                output=f"Error: {str(e)}",
                                metadata={
                                    "success": False,
                                    "error": str(e),
                                    "workflow_id": workflow_id,
                                    "execution_id": execution_id,
                                    "model": model_name,
                                    "models_used": workflow_models if workflow_models else [model_name]
                                }
                            )
                        
                        tracer.flush()
                    except:
                        pass  # Don't fail the request if tracing fails
                
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


# ===== RESOURCE MONITORING ENDPOINTS =====

@router.get("/monitor/resources")
async def get_resource_stats():
    """
    Get comprehensive resource usage statistics for automation workflows
    """
    try:
        resource_monitor = get_resource_monitor()
        
        # Get all resource statistics
        workflow_stats = resource_monitor.get_workflow_stats()
        mcp_stats = get_mcp_pool_stats()
        agent_pool_stats = agent_instance_pool.get_pool_stats()
        redis_stats = get_redis_pool_info()
        
        return {
            "workflow_resources": workflow_stats,
            "mcp_subprocess_pool": mcp_stats,
            "agent_instance_pool": agent_pool_stats,
            "redis_connection_pool": redis_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error getting resource stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/resources/workflow/{workflow_id}/{execution_id}")
async def get_workflow_resource_stats(workflow_id: int, execution_id: str):
    """
    Get resource usage statistics for a specific workflow execution
    """
    try:
        resource_monitor = get_resource_monitor()
        stats = resource_monitor.get_workflow_stats(workflow_id, execution_id)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error getting workflow resource stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitor/health")
async def get_system_health():
    """
    Get overall system health status for automation workflows
    """
    try:
        resource_monitor = get_resource_monitor()
        
        # Get resource statistics
        stats = resource_monitor.get_workflow_stats()
        mcp_stats = get_mcp_pool_stats()
        agent_stats = agent_instance_pool.get_pool_stats()
        redis_stats = get_redis_pool_info()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        # Check resource utilization
        if stats.get("system_utilization", {}).get("memory_utilization", 0) > 0.8:
            health_status = "warning"
            issues.append("High memory utilization")
        
        if stats.get("system_utilization", {}).get("subprocess_utilization", 0) > 0.8:
            health_status = "warning"  
            issues.append("High subprocess utilization")
        
        # Check agent pool utilization
        if agent_stats.get("pool_utilization", 0) > 0.9:
            health_status = "warning"
            issues.append("Agent pool near capacity")
        
        # Check for recent alerts
        recent_alerts = stats.get("recent_alerts", [])
        if len(recent_alerts) > 5:
            health_status = "critical"
            issues.append(f"Multiple resource alerts: {len(recent_alerts)}")
        
        # Check Redis connection
        if redis_stats.get("status") != "active":
            health_status = "critical"
            issues.append("Redis connection pool unavailable")
        
        return {
            "status": health_status,
            "issues": issues,
            "active_workflows": stats.get("active_workflows", 0),
            "system_utilization": stats.get("system_utilization", {}),
            "pool_utilizations": {
                "agent_pool": agent_stats.get("pool_utilization", 0),
                "mcp_pool": mcp_stats.get("subprocess_pool", {}).get("pool_utilization", 0)
            },
            "recent_alerts_count": len(recent_alerts),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitor/cleanup")
async def force_resource_cleanup():
    """
    Force cleanup of automation resources (admin endpoint)
    """
    try:
        logger.info("[AUTOMATION API] Force resource cleanup requested")
        
        # Clean up MCP subprocesses
        from app.core.unified_mcp_service import cleanup_mcp_subprocesses
        await cleanup_mcp_subprocesses()
        
        # Clean up agent instance pool
        await agent_instance_pool.cleanup_all()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("[AUTOMATION API] Force resource cleanup completed")
        
        return {
            "message": "Resource cleanup completed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[AUTOMATION API] Error in force cleanup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/parallel-animation/{node_id}")
async def test_parallel_animation(node_id: str):
    """
    Test endpoint to trigger ParallelNode animation events for debugging
    """
    from fastapi.responses import StreamingResponse
    import json
    import asyncio
    
    async def test_animation_stream():
        """Send test animation events"""
        try:
            logger.info(f"[TEST ANIMATION] Starting test for ParallelNode {node_id}")
            
            # Send node start event
            start_event = {
                "type": "node_start",
                "node_id": node_id,
                "node_type": "ParallelNode",
                "workflow_id": 999,
                "execution_id": "test-animation",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"[TEST ANIMATION] Sending start event: {start_event}")
            yield f"data: {json.dumps(start_event)}\n\n"
            
            # Wait 3 seconds to simulate processing
            await asyncio.sleep(3)
            
            # Send node complete event  
            complete_event = {
                "type": "node_complete",
                "node_id": node_id,
                "node_type": "ParallelNode", 
                "output": "Test animation completed",
                "workflow_id": 999,
                "execution_id": "test-animation",
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"[TEST ANIMATION] Sending complete event: {complete_event}")
            yield f"data: {json.dumps(complete_event)}\n\n"
            
            logger.info(f"[TEST ANIMATION] Test completed for ParallelNode {node_id}")
            
        except Exception as e:
            logger.error(f"[TEST ANIMATION] Error: {e}")
            error_event = {
                "type": "node_error",
                "node_id": node_id,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps(error_event)}\n\n"
    
    return StreamingResponse(
        test_animation_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )