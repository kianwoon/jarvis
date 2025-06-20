from fastapi import APIRouter
from pydantic import BaseModel
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier, QueryType
from app.core.langfuse_integration import get_tracer
from fastapi.responses import StreamingResponse
import json as json_module  # Import with alias to avoid scope issues
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize enhanced query classifier
query_classifier = EnhancedQueryClassifier()

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # Add this for backward compatibility
    use_langgraph: bool = True
    collections: Optional[List[str]] = None  # Specific collections to search
    collection_strategy: str = "auto"  # "auto", "specific", or "all"
    skip_classification: bool = False  # Allow bypassing classification if needed

class RAGResponse(BaseModel):
    answer: str
    source: str
    conversation_id: Optional[str] = None

@router.post("/rag")
async def rag_endpoint(request: RAGRequest):
    # Initialize Langfuse tracing
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    from app.core.llm_settings_cache import get_llm_settings
    llm_settings = get_llm_settings()
    model_name = llm_settings.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="rag-workflow",
            input=request.question,
            metadata={
                "endpoint": "/api/v1/langchain/rag",
                "conversation_id": request.conversation_id,
                "thinking": request.thinking,
                "collections": request.collections,
                "collection_strategy": request.collection_strategy,
                "use_langgraph": request.use_langgraph,
                "model": model_name
            }
        )
        
        # Create RAG execution span for proper hierarchy
        rag_span = None
        if trace:
            rag_span = tracer.create_span(
                trace,
                name="rag-execution",
                metadata={
                    "operation": "rag_search_and_generation",
                    "thinking": request.thinking,
                    "collections": request.collections,
                    "collection_strategy": request.collection_strategy,
                    "use_langgraph": request.use_langgraph,
                    "model": model_name
                }
            )
        
        # Create generation within the RAG span for detailed observability
        if rag_span:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="rag-generation",
                model=model_name,
                input_text=request.question,
                parent_span=rag_span,
                metadata={
                    "thinking": request.thinking,
                    "collections": request.collections,
                    "collection_strategy": request.collection_strategy,
                    "use_langgraph": request.use_langgraph,
                    "model": model_name,
                    "endpoint": "rag"
                }
            )
    
    # Use session_id as conversation_id if conversation_id is not provided
    conversation_id = request.conversation_id or request.session_id
    
    # Classify the query unless explicitly skipped
    routing = None
    if not request.skip_classification:
        routing = await query_classifier.get_routing_recommendation(request.question)
        logger.info(f"Query routing: {routing['primary_type']} (confidence: {routing['confidence']:.2f})")
        
        # Handle based on classification (using enhanced classifier types)
        primary_type = routing['primary_type']
        is_hybrid = routing.get('is_hybrid', False)
        
        # Check for hybrid queries that need multiple handlers
        if is_hybrid:
            # Handle hybrid queries (TOOL_RAG, TOOL_LLM, RAG_LLM, TOOL_RAG_LLM)
            return handle_hybrid_query(request, routing)
        elif primary_type == QueryType.TOOL.value and routing['confidence'] > 0.7:
            # Let main rag_answer handle tool queries to maintain streaming integrity
            # handle_tool_query causes streaming issues due to duplicate tool execution
            pass  # Fall through to main stream() function
        elif primary_type == QueryType.LLM.value and routing['confidence'] > 0.8:
            # Route to direct LLM only if high confidence
            return handle_direct_llm_query(request, routing)
        elif primary_type == QueryType.MULTI_AGENT.value and routing['confidence'] > 0.6:
            # Route to multi-agent system
            return handle_multi_agent_query(request, routing)
        # Fall through to RAG for RAG_SEARCH or low confidence cases
    
    async def stream():
        try:
            chunk_count = 0
            # Add classification metadata to first chunk if available
            if not request.skip_classification and routing:
                classification_chunk = json_module.dumps({
                    "type": "classification",
                    "routing": routing,
                    "handler": "rag"
                }) + "\n"
                yield classification_chunk
                
            print(f"[DEBUG] API endpoint - About to call rag_answer with stream=True")
            
            # Map enhanced classifier types to simple optimization types
            simple_query_type = None
            if routing:
                enhanced_type = routing['primary_type']
                type_mapping = {
                    "rag": "RAG",
                    "tool": "TOOLS", 
                    "llm": "LLM",
                    "code": "LLM",
                    "multi_agent": "RAG",
                    "tool_rag": "TOOLS",
                    "tool_llm": "TOOLS", 
                    "rag_llm": "RAG",
                    "tool_rag_llm": "TOOLS"
                }
                simple_query_type = type_mapping.get(enhanced_type, "LLM")
                print(f"[DEBUG] API endpoint - Mapped '{enhanced_type}' to '{simple_query_type}'")
            
            from app.langchain.service import rag_answer
            rag_stream = await rag_answer(
                request.question, 
                thinking=request.thinking, 
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=request.use_langgraph,
                collections=request.collections,
                collection_strategy=request.collection_strategy,
                query_type=simple_query_type,
                trace=rag_span or trace
            )
            print(f"[DEBUG] API endpoint - Got rag_stream: {type(rag_stream)}")
            
            # rag_answer returns async generator
            async for chunk in rag_stream:
                try:
                    if chunk:  # Only yield non-empty chunks
                        chunk_count += 1
                        yield chunk
                        # Add small delay to ensure streaming
                        import asyncio
                        await asyncio.sleep(0)
                except GeneratorExit:
                    # Client disconnected, log and stop gracefully
                    print(f"[INFO] Client disconnected after {chunk_count} chunks")
                    break
                except Exception as chunk_error:
                    print(f"[ERROR] Error yielding chunk {chunk_count}: {chunk_error}")
                    # Try to yield error message, but don't fail the whole stream
                    try:
                        yield json_module.dumps({"error": f"Chunk error: {str(chunk_error)}"}) + "\n"
                    except:
                        break  # If we can't even send error, client is likely gone
        except (ConnectionError, BrokenPipeError) as conn_error:
            print(f"[INFO] Connection lost during RAG processing: {conn_error}")
            # Client disconnected, don't try to send anything
        except Exception as e:
            print(f"[ERROR] RAG endpoint error: {e}")
            import traceback
            traceback.print_exc()
            try:
                yield json_module.dumps({"error": f"RAG processing failed: {str(e)}"}) + "\n"
            except:
                # If we can't send the error, client is likely disconnected
                print(f"[ERROR] Could not send error response, client likely disconnected")
    
    print(f"[DEBUG] API endpoint - Returning StreamingResponse")
    
    # Create a wrapper to capture the output for Langfuse tracing
    async def stream_with_tracing():
        collected_output = ""
        final_answer = ""
        source_info = ""
        
        try:
            async for chunk in stream():
                # Collect the streamed data
                if chunk and chunk.strip():
                    collected_output += chunk
                    
                    # Try to extract the final answer from the response chunks
                    if '"answer"' in chunk:
                        try:
                            chunk_data = json_module.loads(chunk.strip())
                            if "answer" in chunk_data:
                                final_answer = chunk_data["answer"]
                                # Clean up thinking tags from the answer for cleaner output
                                import re
                                if final_answer:
                                    final_answer = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL).strip()
                            if "source" in chunk_data:
                                source_info = chunk_data["source"]
                        except:
                            pass  # Continue streaming even if parsing fails
                
                yield chunk
            
            # Update Langfuse generation and trace with the final output
            if tracer.is_enabled():
                try:
                    # Ensure we have meaningful output for the generation
                    generation_output = final_answer if final_answer else "Response generated successfully"
                    
                    # Estimate token usage for cost tracking
                    usage = tracer.estimate_token_usage(request.question, generation_output)
                    
                    # End the generation with results including usage
                    if generation:
                        generation.end(
                            output=generation_output,
                            usage_details=usage,
                            metadata={
                                "response_length": len(final_answer) if final_answer else len(collected_output),
                                "source": source_info or "rag-chat",
                                "streaming": True,
                                "has_answer": bool(final_answer),
                                "conversation_id": conversation_id,
                                "output_captured": bool(final_answer),
                                "model": model_name,
                                "input_length": len(request.question),
                                "output_length": len(generation_output),
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with final result
                    if trace:
                        trace.update(
                            output=final_answer if final_answer else collected_output[:500],
                            metadata={
                                "success": True,
                                "source": source_info,
                                "response_length": len(final_answer) if final_answer else len(collected_output),
                                "streaming": True,
                                "conversation_id": conversation_id
                            }
                        )
                    
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
            # Note: workflow_span is only for multi-agent mode, not standard RAG
                    
        except Exception as e:
            # Update generation and trace with error
            if tracer.is_enabled():
                try:
                    # Estimate usage even for errors
                    error_output = f"Error: {str(e)}"
                    usage = tracer.estimate_token_usage(request.question, error_output)
                    
                    # End generation with error
                    if generation:
                        generation.end(
                            output=error_output,
                            usage_details=usage,
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": conversation_id,
                                "model": model_name,
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with error
                    if trace:
                        trace.update(
                            output=f"Error: {str(e)}",
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": conversation_id
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            raise
    
    return StreamingResponse(stream_with_tracing(), media_type="application/json")

class MultiAgentRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    selected_agents: Optional[List[str]] = None  # Specific agents to use
    max_iterations: int = 10
    conversation_history: Optional[List[Dict]] = None  # Previous messages for context

class MultiAgentResponse(BaseModel):
    final_answer: str
    agent_responses: List[Dict[str, str]]
    conversation_id: str

class LargeGenerationRequest(BaseModel):
    task_description: str
    target_count: int = 100
    chunk_size: Optional[int] = None  # Auto-calculated if not provided
    conversation_id: Optional[str] = None
    use_redis: bool = True  # Use Redis for state management
    conversation_history: Optional[List[Dict]] = None

class GenerationProgressRequest(BaseModel):
    session_id: str

@router.post("/multi-agent")
async def multi_agent_endpoint(request: MultiAgentRequest):
    """Process a question using the multi-agent system"""
    # Initialize Langfuse tracing
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    from app.core.llm_settings_cache import get_llm_settings
    llm_settings = get_llm_settings()
    model_name = llm_settings.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="multi-agent-workflow",
            input=request.question,
            metadata={
                "endpoint": "/api/v1/langchain/multi-agent",
                "conversation_id": request.conversation_id,
                "selected_agents": request.selected_agents,
                "max_iterations": request.max_iterations,
                "model": model_name
            }
        )
        
        # Note: Individual agent generations will be created by each agent
        # No single top-level generation needed for multi-agent mode
    
    # Create multi-agent workflow span for proper hierarchy
    workflow_span = None
    if trace:
        try:
            if tracer.is_enabled():
                workflow_span = tracer.create_multi_agent_workflow_span(
                    trace, "multi-agent", request.selected_agents or []
                )
        except Exception as e:
            print(f"[DEBUG] Failed to create multi-agent workflow span: {e}")
    
    system = MultiAgentSystem(conversation_id=request.conversation_id, trace=workflow_span if workflow_span else trace)
    
    async def stream_events_with_tracing():
        collected_output = ""
        final_response = ""
        agent_outputs = []
        
        try:
            async for event in system.stream_events(
                request.question,
                selected_agents=request.selected_agents,
                max_iterations=request.max_iterations,
                conversation_history=request.conversation_history
            ):
                # Events from multi-agent system should always be dicts
                if not isinstance(event, dict):
                    print(f"[ERROR] Received non-dict event: {type(event)} - {str(event)[:200]}...")
                    continue
                
                # Collect event data for Langfuse
                collected_output += json_module.dumps(event) + "\n"
                
                # Log key events
                event_type = event.get('type', 'unknown')
                if event_type == 'agent_complete':
                    agent = event.get('agent', 'unknown')
                    content = event.get('content', '')
                    content_length = len(content)
                    print(f"[INFO] Agent {agent} completed ({content_length} chars)")
                    agent_outputs.append({"agent": agent, "content": content[:500]})
                elif event_type == 'final_response':
                    final_response = event.get('response', '')
                    response_length = len(final_response)
                    print(f"[INFO] Multi-agent processing complete ({response_length} chars)")
                
                # Convert to JSON string with newline, same as standard chat
                yield json_module.dumps(event, ensure_ascii=False) + "\n"
            
            # Update Langfuse generation and trace with the final output
            if tracer.is_enabled():
                try:
                    generation_output = final_response if final_response else "Multi-agent processing completed"
                    
                    # Estimate token usage for cost tracking
                    usage = tracer.estimate_token_usage(request.question, generation_output)
                    
                    # End the generation with results including usage
                    if generation:
                        generation.end(
                            output=generation_output,
                            usage_details=usage,
                            metadata={
                                "response_length": len(final_response) if final_response else len(collected_output),
                                "source": "multi-agent",
                                "streaming": True,
                                "has_final_response": bool(final_response),
                                "conversation_id": request.conversation_id,
                                "agent_count": len(agent_outputs),
                                "model": model_name,
                                "input_length": len(request.question),
                                "output_length": len(generation_output),
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with final result
                    if trace:
                        trace.update(
                            output=generation_output,
                            metadata={
                                "success": True,
                                "source": "multi-agent",
                                "agent_outputs": agent_outputs,
                                "response_length": len(final_response) if final_response else len(collected_output),
                                "streaming": True,
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
            # End multi-agent workflow span with success
            if workflow_span:
                try:
                    tracer.end_span_with_result(
                        workflow_span,
                        {
                            "agent_count": len(agent_outputs),
                            "successful_agents": len([a for a in agent_outputs if a.get("content")]),
                            "final_response_length": len(final_response) if final_response else 0,
                            "total_output_length": len(collected_output)
                        },
                        success=bool(final_response)
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to end multi-agent workflow span: {e}")
                    
        except Exception as e:
            # Update generation and trace with error
            if tracer.is_enabled():
                try:
                    # Estimate usage even for errors
                    error_output = f"Error: {str(e)}"
                    usage = tracer.estimate_token_usage(request.question, error_output)
                    
                    # Individual agent generations handle their own errors
                    # No top-level generation to end in multi-agent mode
                    
                    # Update trace with error
                    if trace:
                        trace.update(
                            output=f"Error: {str(e)}",
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            
            # End multi-agent workflow span with error
            if workflow_span:
                try:
                    tracer.end_span_with_result(workflow_span, None, False, str(e))
                except Exception as span_error:
                    print(f"[WARNING] Failed to end multi-agent workflow span with error: {span_error}")
            
            raise
    
    return StreamingResponse(
        stream_events_with_tracing(),
        media_type="application/json"
    )

@router.post("/large-generation")
async def large_generation_endpoint(request: LargeGenerationRequest):
    """
    Handle large generation tasks that transcend context limits
    using intelligent chunking and Redis-based state management
    """
    # Initialize Langfuse tracing
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    from app.core.llm_settings_cache import get_llm_settings
    llm_settings = get_llm_settings()
    model_name = llm_settings.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="large-generation-workflow",
            input=request.task_description,
            metadata={
                "endpoint": "/api/v1/langchain/large-generation",
                "conversation_id": request.conversation_id,
                "target_count": request.target_count,
                "chunk_size": request.chunk_size,
                "use_redis": request.use_redis,
                "model": model_name
            }
        )
        
        # Create generation within the trace for detailed observability
        if trace:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="large-generation-generation",
                model=model_name,
                input_text=request.task_description,
                metadata={
                    "target_count": request.target_count,
                    "chunk_size": request.chunk_size,
                    "use_redis": request.use_redis,
                    "model": model_name,
                    "endpoint": "large-generation"
                }
            )
    
    system = MultiAgentSystem(conversation_id=request.conversation_id, trace=trace)
    
    async def stream_events_with_tracing():
        collected_output = ""
        final_result = ""
        chunks_processed = 0
        
        try:
            # Choose Redis or in-memory continuity manager
            if request.use_redis:
                # Use Redis-based approach for production
                async for event in system.stream_large_generation_events(
                    query=request.task_description,
                    target_count=request.target_count,
                    chunk_size=request.chunk_size,
                    conversation_history=request.conversation_history
                ):
                    # Collect event data for Langfuse
                    collected_output += json_module.dumps(event) + "\n"
                    
                    # Track chunk processing
                    if event.get('type') == 'chunk_complete':
                        chunks_processed += 1
                    elif event.get('type') == 'generation_complete':
                        final_result = event.get('final_result', '')
                    
                    yield json_module.dumps(event, ensure_ascii=False) + "\n"
            else:
                # Use in-memory approach for testing
                async for event in system.stream_large_generation_events(
                    query=request.task_description,
                    target_count=request.target_count,
                    chunk_size=request.chunk_size,
                    conversation_history=request.conversation_history
                ):
                    # Collect event data for Langfuse
                    collected_output += json_module.dumps(event) + "\n"
                    
                    # Track chunk processing
                    if event.get('type') == 'chunk_complete':
                        chunks_processed += 1
                    elif event.get('type') == 'generation_complete':
                        final_result = event.get('final_result', '')
                    
                    yield json_module.dumps(event, ensure_ascii=False) + "\n"
            
            # Update Langfuse generation and trace with the final output
            if tracer.is_enabled():
                try:
                    generation_output = final_result if final_result else f"Large generation completed - {chunks_processed} chunks processed"
                    
                    # Estimate token usage for cost tracking (large generation can be expensive)
                    usage = tracer.estimate_token_usage(request.task_description, generation_output)
                    
                    # End the generation with results including usage
                    if generation:
                        generation.end(
                            output=generation_output,
                            usage_details=usage,
                            metadata={
                                "response_length": len(final_result) if final_result else len(collected_output),
                                "source": "large-generation",
                                "streaming": True,
                                "chunks_processed": chunks_processed,
                                "target_count": request.target_count,
                                "conversation_id": request.conversation_id,
                                "use_redis": request.use_redis,
                                "model": model_name,
                                "input_length": len(request.task_description),
                                "output_length": len(generation_output),
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with final result
                    if trace:
                        trace.update(
                            output=generation_output,
                            metadata={
                                "success": True,
                                "source": "large-generation",
                                "chunks_processed": chunks_processed,
                                "response_length": len(final_result) if final_result else len(collected_output),
                                "streaming": True,
                                "conversation_id": request.conversation_id
                            }
                        )
                    
                    tracer.flush()
                except Exception as e:
                    print(f"[WARNING] Failed to update Langfuse trace/generation: {e}")
            
            # End multi-agent workflow span with success
            if workflow_span:
                try:
                    tracer.end_span_with_result(
                        workflow_span,
                        {
                            "agent_count": len(agent_outputs),
                            "successful_agents": len([a for a in agent_outputs if a.get("content")]),
                            "final_response_length": len(final_response) if final_response else 0,
                            "total_output_length": len(collected_output)
                        },
                        success=bool(final_response)
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to end multi-agent workflow span: {e}")
                    
        except Exception as e:
            # Update generation and trace with error
            if tracer.is_enabled():
                try:
                    # Estimate usage even for errors
                    error_output = f"Error: {str(e)}"
                    usage = tracer.estimate_token_usage(request.task_description, error_output)
                    
                    # End generation with error
                    if generation:
                        generation.end(
                            output=error_output,
                            usage_details=usage,
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": request.conversation_id,
                                "chunks_processed": chunks_processed,
                                "model": model_name,
                                "estimated_tokens": usage
                            }
                        )
                    
                    # Update trace with error
                    if trace:
                        trace.update(
                            output=f"Error: {str(e)}",
                            metadata={
                                "success": False,
                                "error": str(e),
                                "conversation_id": request.conversation_id,
                                "chunks_processed": chunks_processed
                            }
                        )
                    
                    tracer.flush()
                except:
                    pass  # Don't fail the request if tracing fails
            
            error_event = {
                "type": "large_generation_error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield json_module.dumps(error_event, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_events_with_tracing(),
        media_type="application/json"
    )

@router.get("/large-generation/progress/{session_id}")
async def get_generation_progress(session_id: str):
    """Get progress for an ongoing large generation task"""
    try:
        # Try Redis-based progress first
        from app.agents.redis_continuation_manager import RedisContinuityManager
        
        redis_manager = RedisContinuityManager(session_id=session_id)
        progress = await redis_manager.get_progress_status()
        
        return {
            "session_id": session_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "available": False,
            "timestamp": datetime.now().isoformat()
        }

@router.post("/large-generation/resume")
async def resume_generation(request: dict):
    """Resume a large generation task from a specific point"""
    session_id = request.get("session_id")
    from_chunk = request.get("from_chunk", 0)
    
    if not session_id:
        return {"error": "session_id is required"}
    
    try:
        system = MultiAgentSystem()
        
        async def stream_resume_events():
            async for event in system.resume_chunked_generation(
                session_id=session_id,
                from_chunk=from_chunk
            ):
                yield json_module.dumps(event, ensure_ascii=False) + "\n"
        
        return StreamingResponse(
            stream_resume_events(),
            media_type="application/json"
        )
        
    except Exception as e:
        error_event = {
            "type": "resume_error",
            "error": str(e),
            "session_id": session_id
        }
        return StreamingResponse(
            iter([json_module.dumps(error_event, ensure_ascii=False) + "\n"]),
            media_type="application/json"
        )

@router.delete("/large-generation/cleanup/{session_id}")
async def cleanup_generation_session(session_id: str):
    """Clean up Redis keys and resources for a completed generation session"""
    try:
        from app.agents.redis_continuation_manager import RedisContinuityManager
        
        redis_manager = RedisContinuityManager(session_id=session_id)
        await redis_manager.cleanup_session()
        
        return {
            "session_id": session_id,
            "status": "cleaned_up",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "status": "cleanup_failed",
            "timestamp": datetime.now().isoformat()
        }

def handle_tool_query(request: RAGRequest, routing: Dict):
    """Handle queries that require tool usage"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "tool_handler",
                "message": "This query requires external tools."
            }) + "\n"
            
            # Tool execution will be handled by rag_answer internally
            
            yield json_module.dumps({
                "type": "status",
                "message": f"🔧 Tool execution required. Using dynamic tool selection..."
            }) + "\n"
            
            # Let rag_answer handle tool execution internally to avoid duplication
            # This prevents the enhanced question issue that triggers large generation detection
            
            try:
                # Use original question and let rag_answer handle tool execution
                # The service module already has logic to execute tools when query_type = "TOOLS"
                from app.langchain.service import rag_answer
                for chunk in rag_answer(
                    request.question,
                    thinking=request.thinking,
                    stream=True,
                    conversation_id=conversation_id,
                    use_langgraph=False,
                    collections=request.collections,
                    collection_strategy=request.collection_strategy
                ):
                    yield chunk
                    
            except Exception as e:
                # Fallback to regular RAG if tool execution fails
                yield json_module.dumps({
                    "type": "tool_error",
                    "message": f"Tool execution failed: {str(e)}. Falling back to regular response."
                }) + "\n"
                
                from app.langchain.service import rag_answer
                
                for chunk in rag_answer(
                    request.question,
                    thinking=request.thinking,
                    stream=True,
                    conversation_id=conversation_id,
                    use_langgraph=False,
                    collections=request.collections,
                    collection_strategy=request.collection_strategy
                ):
                    yield chunk
                
        except Exception as e:
            logger.error(f"Tool handler error: {e}")
            yield json_module.dumps({"error": f"Tool handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_direct_llm_query(request: RAGRequest, routing: Dict):
    """Handle queries that can be answered directly by LLM without RAG"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "direct_llm",
                "message": "This is a general query. Using direct LLM response."
            }) + "\n"
            
            # Use the service module but skip RAG retrieval
            from app.langchain.service import get_llm_response_direct
            
            # Stream direct LLM response
            chunk_count = 0
            for chunk in get_llm_response_direct(
                request.question,
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id
            ):
                if chunk:
                    chunk_count += 1
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Direct LLM handler error: {e}")
            yield json_module.dumps({"error": f"Direct LLM handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_multi_agent_query(request: RAGRequest, routing: Dict):
    """Handle complex queries that need multiple agents"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "multi_agent",
                "message": "This is a complex query. Using multi-agent system."
            }) + "\n"
            
            # Get suggested agents
            suggested_agents = routing.get("routing", {}).get("suggested_agents", [])
            
            # Create multi-agent request
            multi_request = MultiAgentRequest(
                question=request.question,
                conversation_id=conversation_id,
                selected_agents=suggested_agents if suggested_agents else None,
                max_iterations=10
            )
            
            # Use existing multi-agent endpoint logic
            system = MultiAgentSystem(conversation_id=conversation_id)
            
            # Stream responses
            import asyncio
            
            async def agent_stream():
                async for event in system.stream_events(
                    request.question,
                    selected_agents=suggested_agents,
                    max_iterations=10
                ):
                    yield json_module.dumps(event, ensure_ascii=False) + "\n"
            
            async def async_wrapper():
                chunks = []
                async for chunk in agent_stream():
                    chunks.append(chunk)
                return chunks
            
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                chunks = loop.run_until_complete(async_wrapper())
                for chunk in chunks:
                    yield chunk
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Multi-agent handler error: {e}")
            yield json_module.dumps({"error": f"Multi-agent handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_hybrid_query(request: RAGRequest, routing: Dict):
    """Handle hybrid queries that require multiple components (Tools + RAG + LLM)"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "hybrid",
                "message": f"This is a hybrid query requiring: {routing['primary_type']}"
            }) + "\n"
            
            # Extract component types
            primary_type = routing['primary_type']
            components = []
            
            # Determine which components are needed
            if primary_type == QueryType.TOOL_RAG.value:
                components = ["tool", "rag"]
            elif primary_type == QueryType.TOOL_LLM.value:
                components = ["tool", "llm"]
            elif primary_type == QueryType.RAG_LLM.value:
                components = ["rag", "llm"]
            elif primary_type == QueryType.TOOL_RAG_LLM.value:
                components = ["tool", "rag", "llm"]
            
            # Collect results from each component
            results = {}
            
            # 1. Tool component - Get real-time data
            if "tool" in components:
                yield json_module.dumps({
                    "type": "status",
                    "message": "🔧 Searching for real-time information..."
                }) + "\n"
                
                # Get MCP tools
                from app.core.mcp_tools_cache import get_enabled_mcp_tools
                available_tools = get_enabled_mcp_tools()
                
                # TODO: Execute appropriate tools based on query
                # For now, we'll indicate tool execution
                tool_results = {
                    "status": "Tool search would be executed here",
                    "suggested_tools": routing.get("routing", {}).get("suggested_tools", [])
                }
                results["tool"] = tool_results
                
                yield json_module.dumps({
                    "type": "tool_result",
                    "data": tool_results
                }) + "\n"
            
            # 2. RAG component - Search knowledge base
            if "rag" in components:
                yield json_module.dumps({
                    "type": "status",
                    "message": "📚 Searching knowledge base..."
                }) + "\n"
                
                # Get RAG results using the service
                from app.langchain.service import handle_rag_query
                
                try:
                    # Get RAG results
                    context, _ = handle_rag_query(
                        question=request.question,
                        thinking=request.thinking,
                        collections=request.collections or ["documents"],
                        collection_strategy=request.collection_strategy or "auto"
                    )
                    
                    results = {
                        "documents": context if isinstance(context, list) else [],
                        "context": context if isinstance(context, str) else ""
                    }
                    
                    # Format context
                    rag_context = "\n\n".join([
                        f"Source {i+1}:\n{result['text'][:500]}..."
                        for i, result in enumerate(results[:3])
                    ]) if results else None
                    
                except Exception as e:
                    logger.warning(f"RAG retrieval error: {e}")
                    rag_context = None
                
                results["rag"] = rag_context
                
                if rag_context:
                    yield json_module.dumps({
                        "type": "rag_result",
                        "data": {
                            "context_found": True,
                            "num_sources": len(rag_context.split('\n\n')) if isinstance(rag_context, str) else 0
                        }
                    }) + "\n"
            
            # 3. Synthesize with LLM
            yield json_module.dumps({
                "type": "status",
                "message": "🤖 Synthesizing comprehensive answer..."
            }) + "\n"
            
            # Build enhanced prompt with all component results
            enhanced_prompt = build_hybrid_prompt(request.question, results, components)
            
            # Stream the final synthesized answer using rag_answer
            from app.langchain.service import rag_answer
            
            # Use rag_answer with the enhanced prompt as a custom context
            for chunk in rag_answer(
                enhanced_prompt,
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=False,  # Direct LLM for synthesis
                collections=request.collections,
                collection_strategy="none"  # We already have context
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Hybrid handler error: {e}")
            yield json_module.dumps({"error": f"Hybrid handling failed: {str(e)}"}) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

def build_hybrid_prompt(question: str, results: Dict, components: List[str]) -> str:
    """Build an enhanced prompt that combines results from all components"""
    prompt_parts = [f"User Question: {question}\n"]
    
    if "tool" in components and "tool" in results:
        tool_data = results["tool"]
        prompt_parts.append("\n## Real-time Information:")
        if tool_data.get("suggested_tools"):
            prompt_parts.append(f"Suggested tools: {', '.join(tool_data['suggested_tools'])}")
        prompt_parts.append("(Tool execution results would appear here)")
    
    if "rag" in components and "rag" in results:
        rag_data = results["rag"]
        if rag_data:
            prompt_parts.append("\n## Knowledge Base Context:")
            prompt_parts.append(str(rag_data))
    
    prompt_parts.append("\n## Task:")
    prompt_parts.append("Based on the above information, provide a comprehensive answer that:")
    prompt_parts.append("1. Incorporates relevant real-time data (if available)")
    prompt_parts.append("2. Uses knowledge from the knowledge base (if available)")
    prompt_parts.append("3. Synthesizes all information into a coherent response")
    prompt_parts.append("4. Clearly indicates which information comes from which source")
    
    return "\n".join(prompt_parts) 