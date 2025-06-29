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

@router.post("/workflows/{workflow_id}/execute/stream")
async def execute_workflow_stream(
    workflow_id: int, 
    request: WorkflowExecuteStreamRequest
):
    """Execute workflow with CORRECTED Langfuse tracing - Following EXACT standard chat mode pattern"""
    from fastapi.responses import StreamingResponse
    import json
    
    # EXACT SAME LANGFUSE PATTERN AS STANDARD CHAT MODE
    from app.core.langfuse_integration import get_tracer
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing (copy from langchain.py)
    from app.core.llm_settings_cache import get_llm_settings
    llm_settings = get_llm_settings()
    model_name = llm_settings.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    logger.debug(f"Automation endpoint - tracer enabled: {tracer.is_enabled()}")
    
    try:
        # Get workflow
        workflow = get_workflow_by_id(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if not workflow.get("is_active", False):
            raise HTTPException(status_code=400, detail="Workflow is not active")
        
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
                    "model": model_name
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
            
            # EXACT SAME GENERATION CREATION AS STANDARD CHAT MODE
            if automation_span:
                generation = tracer.create_generation_with_usage(
                    trace=trace,
                    name="automation-generation",  # Similar to "rag-generation"
                    model=model_name,
                    input_text=request.message or str(request.input_data),
                    parent_span=automation_span,
                    metadata={
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "model": model_name,
                        "endpoint": "automation"
                    }
                )
                logger.debug(f"Automation generation created: {generation is not None}")
        
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
                
                executor = AutomationExecutor()
                
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
                    yield f"data: {json.dumps(update)}\\n\\n"
                    
                    # Collect final answer for trace completion
                    if event_type == "workflow_result":
                        final_answer = update.get("response", "")
                        collected_output += json.dumps(update)
                    elif event_type == "workflow_complete":
                        if not final_answer:
                            final_answer = "Automation workflow completed"
                
                logger.debug(f"Automation API - Processed {chunk_count} events")
                
                # EXACT SAME TRACE COMPLETION AS STANDARD CHAT MODE
                if tracer.is_enabled():
                    try:
                        generation_output = final_answer if final_answer else "Automation workflow completed"
                        
                        # Estimate token usage for cost tracking (copy from langchain.py)
                        usage = tracer.estimate_token_usage(
                            request.message or str(request.input_data), 
                            generation_output
                        )
                        
                        # End the generation with results (exact pattern from langchain.py)
                        if generation:
                            generation.end(
                                output=generation_output,
                                usage_details=usage,
                                metadata={
                                    "response_length": len(final_answer) if final_answer else len(collected_output),
                                    "source": "automation_workflow",
                                    "streaming": True,
                                    "has_final_answer": bool(final_answer),
                                    "workflow_id": workflow_id,
                                    "execution_id": execution_id,
                                    "model": model_name,
                                    "input_length": len(request.message or str(request.input_data)),
                                    "output_length": len(generation_output),
                                    "estimated_tokens": usage
                                }
                            )
                            logger.debug(f"Automation generation completed")
                        
                        # Update trace with final result (exact pattern from langchain.py)
                        if trace:
                            trace.update(
                                output=generation_output,
                                metadata={
                                    "success": True,
                                    "source": "automation_workflow",
                                    "response_length": len(final_answer) if final_answer else len(collected_output),
                                    "streaming": True,
                                    "workflow_id": workflow_id,
                                    "execution_id": execution_id
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
                
                # EXACT SAME ERROR HANDLING AS STANDARD CHAT MODE
                if tracer.is_enabled():
                    try:
                        # Estimate usage even for errors (copy from langchain.py)
                        error_output = f"Error: {str(e)}"
                        usage = tracer.estimate_token_usage(
                            request.message or str(request.input_data), 
                            error_output
                        )
                        
                        # End generation with error (exact pattern from langchain.py)
                        if generation:
                            generation.end(
                                output=error_output,
                                usage_details=usage,
                                metadata={
                                    "success": False,
                                    "error": str(e),
                                    "workflow_id": workflow_id,
                                    "execution_id": execution_id,
                                    "model": model_name,
                                    "estimated_tokens": usage
                                }
                            )
                        
                        # Update trace with error (exact pattern from langchain.py)
                        if trace:
                            trace.update(
                                output=f"Error: {str(e)}",
                                metadata={
                                    "success": False,
                                    "error": str(e),
                                    "workflow_id": workflow_id,
                                    "execution_id": execution_id
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
                yield f"data: {json.dumps(error_update)}\\n\\n"
        
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