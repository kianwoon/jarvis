"""
AI Automation Endpoint - Following Standard Chat Mode Langfuse Pattern
Exactly matches the tracing pattern used in /api/v1/endpoints/langchain.py
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime

from app.core.langfuse_integration import get_tracer
from app.automation.core.agent_workflow_executor_standard_pattern import AgentWorkflowExecutor
from app.automation.integrations.postgres_bridge import postgres_bridge

logger = logging.getLogger(__name__)
router = APIRouter()

class AutomationRequest(BaseModel):
    workflow_id: int
    workflow_config: Dict[str, Any]
    input_data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    execution_id: Optional[str] = None

@router.post("/execute")
async def execute_automation_workflow(request: AutomationRequest):
    """
    Execute automation workflow following EXACT standard chat mode Langfuse pattern
    Matches the pattern from /api/v1/endpoints/langchain.py rag_endpoint
    """
    
    # EXACT SAME PATTERN AS STANDARD CHAT MODE
    # Initialize Langfuse tracing (copy from langchain.py)
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing (copy from langchain.py)
    from app.core.llm_settings_cache import get_llm_settings
    llm_settings = get_llm_settings()
    model_name = llm_settings.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    # EXACT SAME TRACE CREATION AS STANDARD CHAT MODE
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="automation-workflow",  # Similar to "rag-workflow"
            input=request.message or str(request.input_data),
            metadata={
                "endpoint": "/api/v1/automation/execute",
                "workflow_id": request.workflow_id,
                "execution_id": request.execution_id,
                "has_message": request.message is not None,
                "has_input_data": request.input_data is not None,
                "model": model_name
            }
        )
        
        # EXACT SAME EXECUTION SPAN AS STANDARD CHAT MODE
        automation_span = None
        if trace:
            automation_span = tracer.create_span(
                trace,
                name="automation-execution",  # Similar to "rag-execution"
                metadata={
                    "operation": "automation_workflow_execution",
                    "workflow_id": request.workflow_id,
                    "execution_id": request.execution_id,
                    "model": model_name
                }
            )
        
        # EXACT SAME GENERATION CREATION AS STANDARD CHAT MODE
        if automation_span:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="automation-generation",  # Similar to "rag-generation"
                model=model_name,
                input_text=request.message or str(request.input_data),
                parent_span=automation_span,
                metadata={
                    "workflow_id": request.workflow_id,
                    "execution_id": request.execution_id,
                    "model": model_name,
                    "endpoint": "automation"
                }
            )
    
    # Generate execution ID if not provided
    execution_id = request.execution_id or f"auto-{int(datetime.utcnow().timestamp())}"
    
    async def stream():
        """Stream response following EXACT standard chat mode pattern"""
        try:
            chunk_count = 0
            collected_output = ""
            final_answer = ""
            workflow_result_data = None
            
            # Initialize workflow executor
            executor = AgentWorkflowExecutor()
            
            print(f"[DEBUG] API endpoint - About to call execute_agent_workflow with trace")
            
            # EXACT SAME PATTERN: Pass trace to service layer (like rag_answer)
            async for event in executor.execute_agent_workflow(
                workflow_id=request.workflow_id,
                execution_id=execution_id,
                workflow_config=request.workflow_config,
                input_data=request.input_data,
                message=request.message,
                trace=trace  # PASS TRACE EXACTLY LIKE STANDARD CHAT MODE
            ):
                chunk_count += 1
                event_type = event.get("type", "unknown")
                
                # Stream events in real-time (like standard chat)
                yield json.dumps(event) + "\\n"
                
                # Collect final answer and result data for trace completion and database storage
                if event_type == "workflow_result":
                    final_answer = event.get("response", "")
                    workflow_result_data = event  # Store complete result data
                    collected_output += json.dumps(event)
                elif event_type == "workflow_complete":
                    if not final_answer:
                        final_answer = "Automation workflow completed"
            
            print(f"[DEBUG] API endpoint - Processed {chunk_count} events")
            
            # Save workflow result to database
            if workflow_result_data:
                try:
                    # Prepare output data for database storage
                    output_data = {
                        "final_response": workflow_result_data.get("response", ""),
                        "agent_outputs": workflow_result_data.get("agent_outputs", {}),
                        "output_config": workflow_result_data.get("output_config"),
                        "execution_timestamp": datetime.now(datetime.UTC).isoformat(),
                        "workflow_id": workflow_result_data.get("workflow_id"),
                        "execution_id": workflow_result_data.get("execution_id")
                    }
                    
                    # Update execution record with output data
                    update_success = postgres_bridge.update_execution(execution_id, {
                        "output_data": output_data,
                        "status": "completed"
                    })
                    
                    if update_success:
                        logger.info(f"[DATABASE] Successfully saved workflow result for execution {execution_id}")
                        print(f"[DEBUG] Database update successful - output_data saved with response length: {len(final_answer)}")
                    else:
                        logger.error(f"[DATABASE] Failed to save workflow result for execution {execution_id}")
                        print(f"[DEBUG] Database update failed for execution {execution_id}")
                        
                except Exception as db_error:
                    logger.error(f"[DATABASE] Error saving workflow result: {db_error}")
                    print(f"[DEBUG] Database save error: {db_error}")
            else:
                logger.warning(f"[DATABASE] No workflow result data to save for execution {execution_id}")
                print(f"[DEBUG] No workflow_result_data captured for execution {execution_id}")
            
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
                                "workflow_id": request.workflow_id,
                                "execution_id": execution_id,
                                "model": model_name,
                                "input_length": len(request.message or str(request.input_data)),
                                "output_length": len(generation_output),
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with final result (exact pattern from langchain.py)
                    if trace:
                        trace.update(
                            output=generation_output,
                            metadata={
                                "success": True,
                                "source": "automation_workflow",
                                "response_length": len(final_answer) if final_answer else len(collected_output),
                                "streaming": True,
                                "workflow_id": request.workflow_id,
                                "execution_id": execution_id
                            }
                        )
                    
                    # Flush traces (copy from langchain.py)
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
        except Exception as e:
            logger.error(f"Automation workflow error: {e}")
            
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
                                "workflow_id": request.workflow_id,
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
                                "workflow_id": request.workflow_id,
                                "execution_id": execution_id
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            
            # Always send final answer even on errors (copy from langchain.py)
            yield json.dumps({
                "type": "workflow_error",
                "error": str(e),
                "workflow_id": request.workflow_id,
                "execution_id": execution_id
            }) + "\\n"
    
    return StreamingResponse(stream(), media_type="application/json")