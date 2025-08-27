from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.core.db import get_db
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier, QueryType
from app.core.langfuse_integration import get_tracer
from app.core.query_classifier_settings_cache import get_query_classifier_settings
from app.core.temp_document_manager import TempDocumentManager
from app.langchain.hybrid_rag_orchestrator import HybridRAGOrchestrator
from app.core.simple_conversation_manager import conversation_manager
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
from app.core.redis_client import get_redis_client
from fastapi.responses import StreamingResponse
import json as json_module  # Import with alias to avoid scope issues
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize enhanced query classifier
query_classifier = EnhancedQueryClassifier()

# Initialize temp document manager
temp_doc_manager = TempDocumentManager()

# Initialize hybrid RAG orchestrator
hybrid_rag_orchestrator = HybridRAGOrchestrator()

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # Add this for backward compatibility
    use_langgraph: bool = True
    selected_agent: Optional[str] = None  # For single agent mode (@agent feature)
    collections: Optional[List[str]] = None  # Specific collections to search
    collection_strategy: str = "auto"  # "auto", "specific", or "all"
    skip_classification: bool = False  # Allow bypassing classification if needed
    include_temp_docs: Optional[bool] = None  # Include temporary documents in search
    active_temp_doc_ids: Optional[List[str]] = None  # Specific temp doc IDs to include
    # Hybrid RAG mode parameters
    use_hybrid_rag: bool = False  # Enable hybrid RAG mode
    hybrid_strategy: str = "temp_priority"  # "temp_priority", "parallel_fusion", "persistent_only"
    fallback_to_persistent: bool = True  # Fallback to persistent RAG if no temp results
    temp_results_weight: float = 0.7  # Weight for temp results in fusion (0.0-1.0)
    # Radiating mode parameters
    use_radiating: bool = False  # Enable radiating coverage mode
    radiating_config: Optional[Dict[str, Any]] = None  # Radiating settings (depth, strategy, filters, etc.)

class RAGResponse(BaseModel):
    answer: str
    source: str
    conversation_id: Optional[str] = None

@router.post("/rag")
async def rag_endpoint(request: RAGRequest):
    """SIMPLE RAG: Query -> Get Documents -> LLM -> Response. NO complex processing."""
    
    conversation_id = request.conversation_id or request.session_id
    
    # Check for single agent mode (@agent feature)
    if request.selected_agent:
        logger.info(f"🎯 SINGLE AGENT MODE: {request.selected_agent} - Bypassing classification")
        return handle_single_agent_query(request, request.selected_agent, None, None)
    
    # Check for radiating mode
    if request.use_radiating:
        logger.info(f"🌟 RADIATING MODE ACTIVATED - Bypassing classification")
        return handle_radiating_query(request, None, None)
    
    async def stream():
        try:
            # SIMPLE RAG: Just save user message and get documents
            if conversation_id:
                await conversation_manager.add_message(conversation_id, "user", request.question)
            
            # STEP 1: Get documents directly - NO batching, NO extraction, NO verification
            yield json_module.dumps({"type": "status", "message": "Searching documents..."}) + "\n"
            
            from app.langchain.service import call_mcp_tool
            rag_result = call_mcp_tool(
                tool_name="rag_knowledge_search",
                parameters={
                    "query": request.question,
                    "include_content": True,
                    "max_documents": 5,
                    "collections": request.collections
                }
            )
            
            documents = []
            if isinstance(rag_result, dict) and 'result' in rag_result:
                result_data = rag_result['result']
                if 'documents' in result_data:
                    documents = result_data['documents']
            
            if not documents:
                yield json_module.dumps({"answer": "I couldn't find relevant information.", "source": "RAG"}) + "\n"
                return
                
            # STEP 2: Build simple context - NO complex templates, NO formatting from DB
            context = "\n\n".join([
                f"Document: {doc.get('source', 'Unknown')}\n{doc.get('content', '')}" 
                for doc in documents
            ])
            
            # STEP 3: Simple prompt - NO complex synthesis, NO conversation history
            prompt = f"Based on this context, answer the question:\n\nContext:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"
            
            # STEP 4: Get LLM response directly - NO rag_answer, NO complex routing
            yield json_module.dumps({"type": "status", "message": "Generating response..."}) + "\n"
            
            from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
            llm_settings = get_llm_settings()
            llm_config = get_main_llm_full_config(llm_settings)
            
            from app.llm.ollama import OllamaLLM
            llm = OllamaLLM(
                base_url=llm_config.get("base_url", "http://host.docker.internal:11434"),
                model=llm_config.get("model", "llama3.1:8b")
            )
            
            # STEP 5: Stream response directly
            complete_answer = ""
            async for response in llm.generate_stream(prompt):
                if response.text:
                    complete_answer += response.text
                    yield json_module.dumps({"token": response.text}) + "\n"
            
            # STEP 6: Save response and send completion - SIMPLE
            if conversation_id and complete_answer:
                await conversation_manager.add_message(conversation_id, "assistant", complete_answer)
            
            # Format documents for UI
            formatted_docs = [{
                "content": doc.get('content', ''),
                "source": doc.get('source', 'Unknown'),
                "relevance_score": doc.get('score', 0.0)
            } for doc in documents]
            
            yield json_module.dumps({
                "answer": complete_answer,
                "source": "RAG",
                "conversation_id": conversation_id,
                "documents": formatted_docs
            }) + "\n"
        
        except Exception as e:
            logger.error(f"Simple RAG error: {e}")
            yield json_module.dumps({"error": f"RAG failed: {str(e)}"}) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

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
    """Process a question using the LangGraph multi-agent system"""
    
    # Initialize Langfuse tracing
    tracer = get_tracer()
    trace = None
    generation = None
    
    # Get model name from LLM settings for proper tracing
    llm_settings = get_llm_settings()
    main_llm_config = get_main_llm_full_config(llm_settings)
    model_name = main_llm_config.get("model", "unknown")
    
    if not tracer._initialized:
        tracer.initialize()
    
    if tracer.is_enabled():
        trace = tracer.create_trace(
            name="langgraph-multi-agent-workflow",
            input=request.question,
            metadata={
                "endpoint": "/api/v1/langchain/multi-agent",
                "conversation_id": request.conversation_id,
                "selected_agents": request.selected_agents,
                "max_iterations": request.max_iterations,
                "model": model_name,
                "system_type": "langgraph"
            }
        )
        
        # Create generation for LangGraph multi-agent workflow
        if trace:
            generation = tracer.create_generation_with_usage(
                trace=trace,
                name="langgraph-multi-agent-generation",
                model=model_name,
                input_text=request.question,
                metadata={
                    "selected_agents": request.selected_agents,
                    "max_iterations": request.max_iterations,
                    "system_type": "langgraph",
                    "model": model_name,
                    "endpoint": "multi-agent"
                }
            )
    
    # Create multi-agent workflow span for proper hierarchy
    workflow_span = None
    if trace:
        try:
            if tracer.is_enabled():
                workflow_span = tracer.create_multi_agent_workflow_span(
                    trace, "langgraph-multi-agent", request.selected_agents or []
                )
        except Exception as e:
            pass
    
    # Use LangGraph multi-agent system with streaming callback
    from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
    
    async def stream_events_with_tracing():
        collected_output = ""
        final_response = ""
        agent_outputs = []  # Initialize to track agent outputs
        
        try:
            # Use streaming multi-agent function like working RAG pattern
            from app.langchain.langgraph_multi_agent_system import langgraph_multi_agent_answer
            
            # Get streaming generator (same pattern as rag_answer)
            multi_agent_stream = await langgraph_multi_agent_answer(
                question=request.question,
                conversation_id=request.conversation_id,
                stream=True,
                trace=workflow_span or trace
            )
            
            # Stream chunks exactly like RAG endpoint does
            chunk_count = 0
            async for chunk in multi_agent_stream:
                try:
                    if chunk:  # Only yield non-empty chunks
                        chunk_count += 1
                        yield chunk
                        # Collect agent outputs from chunks
                        try:
                            if isinstance(chunk, str):
                                chunk_data = json_module.loads(chunk.strip())
                                if chunk_data.get("type") == "agent_response":
                                    agent_outputs.append({
                                        "agent": chunk_data.get("agent", "unknown"),
                                        "content": chunk_data.get("response", "")
                                    })
                        except:
                            pass  # Ignore parsing errors
                        # Add small delay to ensure streaming
                        import asyncio
                        await asyncio.sleep(0.01)
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
                        
            collected_output = f"Multi-agent streaming completed with {chunk_count} chunks"
            
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
                            usage=usage,
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
    llm_settings = get_llm_settings()
    main_llm_config = get_main_llm_full_config(llm_settings)
    model_name = main_llm_config.get("model", "unknown")
    
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
                            usage=usage,
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
            
            # Large generation doesn't use workflow spans - removed incorrect code
                    
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
                            usage=usage,
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

def _detect_comprehensive_query(message: str) -> bool:
    """
    Detect queries that require comprehensive data coverage rather than cached context.
    
    These queries need fresh Milvus retrieval to ensure complete results rather than
    potentially incomplete cached conversation context.
    
    Args:
        message: The user's query message
        
    Returns:
        bool: True if this is a comprehensive query requiring fresh data retrieval
    """
    import re
    
    # Patterns that indicate comprehensive queries requiring fresh data
    comprehensive_patterns = [
        # "find all X from all Y" patterns
        r'(?:find|get|show|list)\s+(?:all|everything|every)\s+\w+\s+from\s+(?:all|every)',
        
        # "all/complete/full" modifiers for broad coverage
        r'(?:all|complete|full|comprehensive)\s+(?:list|overview|summary)',
        
        # "find/show all/every X" patterns
        r'(?:find|show)\s+(?:all|every|each)\s+\w+(?:\s+and\s+\w+)*',
        
        # "from all sources" patterns
        r'from\s+(?:all|every)\s+source',
        
        # Explicit comprehensive search terms
        r'(?:complete|comprehensive|full)\s+(?:retrieval|search|query)',
        
        # "everything about" patterns
        r'(?:everything|anything)\s+about',
        
        # "any/all instances of" patterns
        r'(?:any|all)\s+instances?\s+of',
        
        # "total/entire collection" patterns
        r'(?:total|entire)\s+(?:collection|dataset|repository)',
        
        # "show everything from" patterns
        r'show\s+everything\s+from',
        
        # "all X in the system/database" patterns
        r'all\s+\w+\s+in\s+the\s+(?:system|database|platform|application)'
    ]
    
    # Check if message matches any comprehensive query pattern
    message_lower = message.lower()
    for pattern in comprehensive_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False

def _log_routing_decision(message: str, is_comprehensive: bool, is_current_data: bool, conversation_id: str, handler_name: str):
    """
    Log comprehensive routing decision details for debugging.
    
    Args:
        message: The user's query message
        is_comprehensive: Whether this was detected as a comprehensive query
        is_current_data: Whether this was detected as a current data query  
        conversation_id: The conversation ID if available
        handler_name: Name of the handler (DIRECT, HYBRID, RADIATING)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Truncate message for logging to avoid spam
    truncated_message = message[:100] + "..." if len(message) > 100 else message
    
    logger.info(f"[{handler_name} ROUTING] Query: '{truncated_message}'")
    logger.info(f"[{handler_name} ROUTING] Comprehensive query: {is_comprehensive}")
    logger.info(f"[{handler_name} ROUTING] Current data query: {is_current_data}")
    logger.info(f"[{handler_name} ROUTING] Has conversation ID: {bool(conversation_id)}")
    
    if is_comprehensive:
        logger.info(f"[{handler_name} ROUTING] → BYPASSING conversation context for fresh data retrieval")
    elif is_current_data:
        logger.info(f"[{handler_name} ROUTING] → BYPASSING conversation context to avoid contamination")
    elif conversation_id:
        logger.info(f"[{handler_name} ROUTING] → USING conversation context")
    else:
        logger.info(f"[{handler_name} ROUTING] → NO CONTEXT (no conversation ID)")

async def decide_tools_with_llm(query: str, available_tools: Dict, agent_name: str, llm_config: Dict) -> List[tuple]:
    """
    Use LLM to intelligently decide which tools to use based on the query.
    Returns a list of (tool_name, parameters) tuples.
    """
    from app.llm.ollama import OllamaLLM
    from app.llm.base import LLMConfig
    import os
    import re
    
    try:
        # Build tool descriptions for the prompt
        tool_descriptions = []
        for tool_name, tool_info in available_tools.items():
            if isinstance(tool_info, dict):
                description = tool_info.get('description', f'Tool: {tool_name}')
                # Get detailed description from manifest if available
                manifest = tool_info.get('manifest', {})
                if manifest and 'tools' in manifest:
                    for tool_def in manifest['tools']:
                        if tool_def.get('name') == tool_name:
                            description = tool_def.get('description', description)
                            break
                
                # Get input schema for parameters
                input_schema = {}
                if manifest and 'tools' in manifest:
                    for tool_def in manifest['tools']:
                        if tool_def.get('name') == tool_name:
                            input_schema = tool_def.get('inputSchema', {}).get('properties', {})
                            break
                
                tool_descriptions.append(f"- {tool_name}: {description}")
                if input_schema:
                    params_desc = ", ".join([f"{k}: {v.get('description', v.get('type', 'any'))}" for k, v in input_schema.items()])
                    tool_descriptions.append(f"  Parameters: {params_desc}")
        
        # Create the tool decision prompt
        tool_decision_prompt = f"""You are {agent_name}, analyzing a user query to determine which tools (if any) would be helpful.

User Query: {query}

Available Tools:
{chr(10).join(tool_descriptions)}

Instructions:
1. Analyze the user's query carefully
2. Determine which tools (if any) would help answer the query
3. For each tool you want to use, output a function call in this exact format:
   call_tool_TOOLNAME(param1="value1", param2="value2")
4. If no tools are needed, simply write "No tools needed"
5. Extract parameters intelligently from the user's query
6. Multiple tools can be called if needed

Examples:
- For "search for information about Python": call_tool_google_search(query="Python programming language", num_results=5)
- For "what time is it?": call_tool_datetime()
- For "weather in New York": call_tool_weather(location="New York")

Your response (only function calls or "No tools needed"):"""

        # Initialize LLM for tool decision
        decision_llm_config = LLMConfig(
            model_name=llm_config["model"],
            temperature=0.3,  # Lower temperature for more deterministic tool selection
            top_p=0.9,
            max_tokens=500  # Tool decisions should be concise
        )
        
        # Get model server URL from settings
        llm_settings = get_llm_settings()
        model_server = os.environ.get("OLLAMA_BASE_URL") or llm_settings.get('model_server', '').strip()
        if not model_server:
            from app.core.config import get_settings
            settings = get_settings()
            model_server = settings.OLLAMA_BASE_URL
        
        llm = OllamaLLM(decision_llm_config, base_url=model_server)
        
        # Get LLM decision
        response = await llm.generate(tool_decision_prompt)
        logger.info(f"🤖 LLM tool decision response: {response}")
        
        # Extract text from response object
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse the response for function calls
        tools_to_execute = []
        
        if "no tools needed" in response_text.lower():
            logger.info("🔍 LLM decided no tools are needed")
            return []
        
        # Pattern to match function calls: call_tool_NAME(params)
        function_pattern = r'call_tool_(\w+)\s*\(([^)]*)\)'
        matches = re.finditer(function_pattern, response_text, re.IGNORECASE)
        
        for match in matches:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters
            parameters = {}
            if params_str.strip():
                # Pattern to match parameter assignments
                param_pattern = r'(\w+)\s*=\s*["\']([^"\']*)["\']'
                param_matches = re.finditer(param_pattern, params_str)
                
                for param_match in param_matches:
                    param_name = param_match.group(1)
                    param_value = param_match.group(2)
                    
                    # Convert to appropriate type
                    if param_name == 'num_results' and param_value.isdigit():
                        parameters[param_name] = int(param_value)
                    elif param_value.lower() in ['true', 'false']:
                        parameters[param_name] = param_value.lower() == 'true'
                    else:
                        parameters[param_name] = param_value
            
            # Verify tool exists in available tools
            if tool_name in available_tools:
                tools_to_execute.append((tool_name, parameters))
                logger.info(f"📌 LLM selected tool: {tool_name} with params: {parameters}")
            else:
                logger.warning(f"⚠️ LLM selected unavailable tool: {tool_name}")
        
        return tools_to_execute
        
    except Exception as e:
        logger.error(f"Error in LLM tool decision: {e}")
        return []

def handle_single_agent_query(request: RAGRequest, agent_name: str, trace=None, rag_span=None):
    """Handle single agent queries (@agent feature)"""
    from fastapi.responses import StreamingResponse
    from app.core.langgraph_agents_cache import get_agent_by_name
    from app.llm.ollama import OllamaLLM
    from app.llm.base import LLMConfig
    from app.core.mcp_tools_cache import get_enabled_mcp_tools
    from app.langchain.service import call_mcp_tool
    from app.core.config import get_settings
    import os
    import json
    
    conversation_id = request.conversation_id or request.session_id
    
    async def stream():
        try:
            # Get agent configuration
            agent_data = get_agent_by_name(agent_name)
            
            if not agent_data:
                # Fallback configuration if agent not found
                llm_settings = get_llm_settings()
                mode_config = get_main_llm_full_config(llm_settings)
                system_prompt = f"You are {agent_name}, a specialized AI assistant. Please respond directly to the user's question using your expertise."
            else:
                # Use agent's specific configuration
                agent_system_prompt = agent_data.get("system_prompt", "")
                agent_config = agent_data.get("config", {})
                
                # Determine model and parameters based on agent config
                if agent_config.get('use_main_llm'):
                    main_llm_config = get_main_llm_full_config()
                    model_name = main_llm_config.get('model', 'qwen3:30b-a3b-instruct-2507-q4_K_M')
                    actual_max_tokens = agent_config.get('max_tokens', main_llm_config.get('max_tokens', 2000))
                    actual_temperature = agent_config.get('temperature', main_llm_config.get('temperature', 0.7))
                    top_p = agent_config.get('top_p', main_llm_config.get('top_p', 0.9))
                else:
                    # Agent-specific model or fallback
                    model_name = agent_config.get("model")
                    if not model_name:
                        main_llm_config = get_main_llm_full_config()
                        model_name = main_llm_config.get('model', 'qwen3:30b-a3b-instruct-2507-q4_K_M')
                    
                    actual_max_tokens = agent_config.get('max_tokens', 2000)
                    actual_temperature = agent_config.get('temperature', 0.7)
                    top_p = agent_config.get('top_p', 0.9)
                
                mode_config = {
                    "model": model_name,
                    "temperature": float(actual_temperature),
                    "top_p": float(top_p),
                    "max_tokens": int(actual_max_tokens)
                }
                
                # Construct system prompt - simplified to avoid infinite compliance loops
                system_prompt = f"""You are {agent_name}. {agent_system_prompt}

Important: Respond directly to the user's question. Do not include any meta-commentary, instructions, or explanations of your process."""
            
            # Get conversation history if available
            conversation_history = ""
            if conversation_id:
                from app.core.simple_conversation_manager import conversation_manager
                history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                if history:
                    conversation_history = conversation_manager.format_history_for_prompt(history, "")
            
            # Check if agent has tools and execute them if needed
            tool_results_context = ""
            tools_used = []
            
            if agent_data and agent_data.get("tools"):
                agent_tools = agent_data.get("tools", [])
                logger.info(f"🔧 Agent {agent_name} has tools: {agent_tools}")
                
                # Get all enabled MCP tools
                enabled_tools = get_enabled_mcp_tools()
                
                # Filter to only agent's assigned tools
                available_tools = {
                    tool_name: tool_info 
                    for tool_name, tool_info in enabled_tools.items() 
                    if tool_name in agent_tools
                }
                
                if available_tools:
                    logger.info(f"📋 Available tools for {agent_name}: {list(available_tools.keys())}")
                    
                    # Use LLM to intelligently decide which tools to use
                    tools_to_execute = await decide_tools_with_llm(
                        request.question, 
                        available_tools, 
                        agent_name,
                        mode_config
                    )
                    
                    # Execute selected tools
                    if tools_to_execute:
                        tool_results = []
                        for tool_name, params in tools_to_execute:
                            try:
                                logger.info(f"🚀 Executing {tool_name} with params: {params}")
                                result = call_mcp_tool(tool_name, params, trace)
                                
                                tools_used.append(tool_name)
                                tool_results.append({
                                    "tool": tool_name,
                                    "result": result,
                                    "success": True
                                })
                                logger.info(f"✅ Tool {tool_name} executed successfully")
                                
                            except Exception as tool_error:
                                logger.error(f"❌ Error executing {tool_name}: {tool_error}")
                                tool_results.append({
                                    "tool": tool_name,
                                    "error": str(tool_error),
                                    "success": False
                                })
                        
                        # Format tool results for inclusion in prompt with enhanced source attribution
                        if tool_results:
                            successful_results = [r for r in tool_results if r.get("success")]
                            if successful_results:
                                tool_results_context = "\n\n📊 TOOL EXECUTION RESULTS\n"
                                tool_results_context += "="*60 + "\n"
                                tool_results_context += "⚠️ IMPORTANT: Each result below comes from a different tool/source.\n"
                                tool_results_context += "Do NOT mix information between different sources.\n"
                                tool_results_context += "="*60 + "\n\n"
                                
                                for idx, result in enumerate(successful_results, 1):
                                    tool_name = result['tool']
                                    tool_result = result['result']
                                    
                                    # Enhanced formatting for search tools
                                    if 'search' in tool_name.lower():
                                        tool_results_context += f"\n🔍 SOURCE #{idx}: {tool_name.upper()}\n"
                                        tool_results_context += "-"*40 + "\n"
                                        tool_results_context += f"ATTRIBUTION: Information below is ONLY from {tool_name}\n"
                                        tool_results_context += "-"*40 + "\n"
                                        tool_results_context += json.dumps(tool_result, indent=2) + "\n"
                                        tool_results_context += "-"*40 + "\n"
                                        tool_results_context += f"END OF {tool_name.upper()} RESULTS\n"
                                        tool_results_context += "="*60 + "\n"
                                    else:
                                        # Standard tool formatting
                                        tool_results_context += f"\n📊 TOOL #{idx}: {tool_name.upper()}\n"
                                        tool_results_context += "-"*40 + "\n"
                                        tool_results_context += json.dumps(tool_result, indent=2) + "\n"
                                        tool_results_context += "-"*40 + "\n"
                                
                                tool_results_context += "\n⚠️ REMINDER: Each tool result above is from a DIFFERENT source.\n"
                                tool_results_context += "Attribute information correctly to its specific source.\n"
                                tool_results_context += "NEVER mix facts from different tools/sources.\n"
            
            # Prepare the full prompt with tool results
            # Add additional reminder to not expose instructions
            full_prompt = f"""{system_prompt}

{tool_results_context}{conversation_history}

User: {request.question}

FINAL REMINDER - OUTPUT ONLY THE REQUESTED CONTENT:
- If asked for a LinkedIn post, output ONLY the post text
- If asked for analysis, output ONLY the analysis
- DO NOT output section headers, templates, or instructions
- DO NOT explain your process or methodology unless specifically asked
- START your response with the actual content, not meta-commentary"""
            
            # Initialize LLM
            llm_config = LLMConfig(
                model_name=mode_config["model"],
                temperature=float(mode_config["temperature"]),
                top_p=float(mode_config["top_p"]),
                max_tokens=int(mode_config["max_tokens"])
            )
            
            # Get model server URL
            llm_settings = get_llm_settings()
            model_server = os.environ.get("OLLAMA_BASE_URL") or llm_settings.get('model_server', '').strip() or "http://ollama:11434"
            llm = OllamaLLM(llm_config, base_url=model_server)
            
            # Send initial event
            yield json.dumps({
                "type": "chat_start",
                "conversation_id": conversation_id,
                "model": mode_config["model"],
                "thinking": request.thinking,
                "selected_agent": agent_name,
                "confidence": 1.0,
                "classification": "direct_agent"
            }) + "\n"
            
            # Send tool execution status if tools were used
            if tools_used:
                yield json.dumps({
                    "type": "status",
                    "message": f"🔧 Executed tools: {', '.join(tools_used)}"
                }) + "\n"
            
            yield json.dumps({
                "type": "status",
                "message": f"🤖 {agent_name} is responding..."
            }) + "\n"
            
            # Stream agent response
            final_response = ""
            async for response_chunk in llm.generate_stream(full_prompt):
                final_response += response_chunk.text
                
                if response_chunk.text.strip():
                    yield json.dumps({
                        "token": response_chunk.text
                    }) + "\n"
            
            # Save to conversation history
            if conversation_id:
                await conversation_manager.add_message(conversation_id, "user", request.question)
                await conversation_manager.add_message(conversation_id, "assistant", final_response)
            
            # Send completion event
            yield json.dumps({
                "answer": final_response,
                "source": f"direct_agent_{agent_name}",
                "conversation_id": conversation_id,
                "bypass_classification": True,
                "selected_agent": agent_name
            }) + "\n"
            
            # Update tracing if enabled
            if trace and rag_span:
                try:
                    rag_span.update(
                        output=final_response,
                        metadata={
                            "selected_agent": agent_name,
                            "model": mode_config["model"],
                            "bypass_classification": True,
                            "tools_used": tools_used,
                            "tool_execution": len(tools_used) > 0
                        }
                    )
                    rag_span.end()
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Single agent query error: {e}")
            yield json.dumps({
                "answer": f"Error in agent {agent_name}: {str(e)}",
                "source": "ERROR",
                "conversation_id": conversation_id
            }) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

def handle_radiating_query(request: RAGRequest, trace=None, rag_span=None):
    """Handle radiating coverage queries"""
    from fastapi.responses import StreamingResponse
    from app.langchain.radiating_agent_system import RadiatingAgent
    from app.core.simple_conversation_manager import conversation_manager
    import json
    from datetime import datetime
    
    conversation_id = request.conversation_id or request.session_id
    
    async def stream():
        try:
            # Store user message in conversation history
            if conversation_id:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.question
                )
            
            # Get conversation history if available
            conversation_history = ""
            
            # Check if this is a comprehensive query requiring fresh data retrieval
            is_comprehensive_query = _detect_comprehensive_query(request.question)
            logger.info(f"[RADIATING HANDLER] Is comprehensive query: {is_comprehensive_query}")
            
            # Log routing decision for debugging
            _log_routing_decision(
                message=request.question,
                is_comprehensive=is_comprehensive_query,
                is_current_data=False,  # Radiating handler doesn't check for current data queries
                conversation_id=conversation_id,
                handler_name="RADIATING"
            )
            
            if conversation_id and not is_comprehensive_query:
                history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                if history:
                    conversation_history = conversation_manager.format_history_for_prompt(history, request.question)
                    logger.info(f"[RADIATING HANDLER] Enhanced question with conversation context")
            elif is_comprehensive_query:
                logger.info(f"[RADIATING HANDLER] Skipping conversation history for comprehensive query to ensure complete data retrieval")
            
            # Extract radiating configuration
            radiating_config = request.radiating_config or {}
            max_depth = radiating_config.get('max_depth', 3)
            strategy = radiating_config.get('strategy', 'hybrid')
            filters = radiating_config.get('filters', {})
            include_coverage = radiating_config.get('include_coverage_data', True)
            
            # Initialize the RadiatingAgent with config if provided
            radiating_agent = RadiatingAgent(trace=trace, radiating_config=radiating_config)
            
            # Send initial event
            yield json.dumps({
                "type": "chat_start",
                "conversation_id": conversation_id,
                "mode": "radiating",
                "confidence": 1.0,
                "classification": "radiating_coverage",
                "config": {
                    "max_depth": max_depth,
                    "strategy": strategy,
                    "filters": filters
                }
            }) + "\n"
            
            # Send status
            yield json.dumps({
                "type": "status",
                "message": f"🌟 Initiating radiating coverage exploration (depth: {max_depth}, strategy: {strategy})..."
            }) + "\n"
            
            # Build context for radiating processing
            context = {
                'max_depth': max_depth,
                'strategy': strategy,
                'filters': filters,
                'include_coverage': include_coverage,
                'conversation_history': conversation_history,
                'conversation_id': conversation_id
            }
            
            # Process with radiating coverage
            final_response = ""
            entities_found = []
            relationships_found = []
            coverage_data = None
            
            async for chunk in radiating_agent.process_with_radiation(
                query=request.question,
                context=context,
                stream=True
            ):
                # Handle different chunk types
                chunk_type = chunk.get('type')
                
                if chunk_type == 'entity':
                    # Entity discovered
                    entities_found.append(chunk.get('entity'))
                    yield json.dumps({
                        "type": "entity_discovered",
                        "entity": chunk.get('entity'),
                        "confidence": chunk.get('confidence', 0.8)
                    }) + "\n"
                    
                elif chunk_type == 'relationship':
                    # Relationship found
                    relationships_found.append(chunk.get('relationship'))
                    yield json.dumps({
                        "type": "relationship_found",
                        "relationship": chunk.get('relationship')
                    }) + "\n"
                    
                elif chunk_type == 'traversal_progress':
                    # Traversal progress update
                    yield json.dumps({
                        "type": "traversal_progress",
                        "current_depth": chunk.get('current_depth'),
                        "entities_explored": chunk.get('entities_explored'),
                        "total_entities": chunk.get('total_entities')
                    }) + "\n"
                    
                elif chunk_type in ['text', 'content', 'response']:
                    # Text response chunk - handle both 'text' and 'content' types
                    text = chunk.get('text', '') or chunk.get('content', '')
                    final_response += text
                    if text.strip():
                        yield json.dumps({
                            "token": text
                        }) + "\n"
                        
                elif chunk_type == 'coverage':
                    # Coverage data
                    coverage_data = chunk.get('coverage')
                    
                elif chunk_type == 'status':
                    # Status update
                    yield json.dumps({
                        "type": "status",
                        "message": chunk.get('message', '')
                    }) + "\n"
                    
                elif chunk_type == 'metadata':
                    # Metadata from radiating system
                    entities_count = chunk.get('entities_discovered', 0)
                    relationships_count = chunk.get('relationships_found', 0)
                    if entities_count > 0:
                        entities_found.extend([{}] * entities_count)  # Placeholder for count
                    if relationships_count > 0:
                        relationships_found.extend([{}] * relationships_count)  # Placeholder for count
                    coverage_data = {
                        'entities_discovered': entities_count,
                        'relationships_found': relationships_count,
                        'processing_time_ms': chunk.get('processing_time_ms'),
                        'coverage_depth': chunk.get('coverage_depth', 0)
                    }
            
            # Save to conversation history
            if conversation_id and final_response:
                await conversation_manager.add_message(conversation_id, "assistant", final_response)
            
            # Send completion event with full results
            completion_event = {
                "answer": final_response,
                "source": "radiating_coverage",
                "conversation_id": conversation_id,
                "bypass_classification": True,
                "mode": "radiating",
                "entities_found": len(entities_found),
                "relationships_found": len(relationships_found)
            }
            
            # Add coverage data if available
            if coverage_data:
                completion_event["coverage"] = coverage_data
            
            yield json.dumps(completion_event) + "\n"
            
            # Update tracing if enabled
            if trace and rag_span:
                try:
                    rag_span.update(
                        output=final_response,
                        metadata={
                            "mode": "radiating",
                            "max_depth": max_depth,
                            "strategy": strategy,
                            "entities_found": len(entities_found),
                            "relationships_found": len(relationships_found),
                            "bypass_classification": True
                        }
                    )
                    rag_span.end()
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Radiating query error: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "message": f"Error in radiating coverage: {str(e)}",
                "source": "ERROR",
                "conversation_id": conversation_id,
                "timestamp": datetime.now().isoformat()
            }) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

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
            
    # Create a wrapper to capture the output for Langfuse tracing in direct tool execution
    async def stream_with_tracing():
        collected_output = ""
        final_answer = ""
        source_info = ""
        
        try:
            async for chunk in stream():
                # Collect the streamed data
                if chunk and chunk.strip():
                    collected_output += chunk
                    # logger.info(f"[TRACE DEBUG] Direct tool chunk received: {chunk[:200]}...")  # Commented out to reduce log noise
                    
                    # Try to extract the final answer from the response chunks
                    if '"answer"' in chunk:
                        try:
                            chunk_data = json_module.loads(chunk.strip())
                            if "answer" in chunk_data:
                                final_answer = chunk_data["answer"]
                                # logger.info(f"[TRACE DEBUG] Direct tool extracted final answer: {final_answer[:100]}...")  # Commented out
                                if final_answer:
                                    pass  # logger.info(f"[TRACE DEBUG] Direct tool final answer: {final_answer[:100]}...")  # Commented out
                            if "source" in chunk_data:
                                source_info = chunk_data["source"]
                        except Exception as e:
                            pass  # Continue streaming even if parsing fails
                
                yield chunk
            
            
        except Exception as e:
            logger.error(f"Direct tool streaming error: {e}")
            yield json_module.dumps({"error": f"Direct tool streaming failed: {str(e)}"}) + "\n"
    
    return StreamingResponse(stream_with_tracing(), media_type="application/json")

def extract_primary_subject(query: str) -> str:
    """
    Extract the primary subject/entity from a user query for relevance filtering.
    This helps prevent mixing information from different entities in search results.
    """
    import re
    
    # Remove common question words and phrases
    query_lower = query.lower()
    question_patterns = [
        r'^(what|when|where|who|why|how|is|are|does|do|can|could|would|will|should)\s+',
        r'^(tell me about|explain|describe|find|search for|look up|get me|show me)\s+',
        r'^(information about|info about|details about|data about)\s+',
    ]
    
    cleaned_query = query
    for pattern in question_patterns:
        cleaned_query = re.sub(pattern, '', query_lower, flags=re.IGNORECASE)
    
    # Extract product names, entities, or main topics
    # Look for quoted terms first (highest priority)
    quoted_match = re.search(r'"([^"]+)"', query)
    if quoted_match:
        return quoted_match.group(1)
    
    # Look for proper nouns (capitalized words)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    if proper_nouns:
        # Return the longest proper noun phrase
        return max(proper_nouns, key=len)
    
    # Look for product tiers or specific identifiers
    tier_patterns = [
        r'\b(pro|plus|premium|enterprise|team|free|basic|standard|advanced)\b',
        r'\b([a-zA-Z]+\s+(?:pro|plus|premium|enterprise|team))\b',
    ]
    for pattern in tier_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(0)
    
    # Fallback: Return the cleaned query focusing on key nouns
    # Remove articles and common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as'}
    words = cleaned_query.split()
    key_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if key_words:
        # Return first 3 key words as the primary subject
        return ' '.join(key_words[:3])
    
    # Last resort: return original query
    return query

def handle_direct_tool_query(request: RAGRequest, routing: Dict, trace=None):
    """Handle confident tool classifications with direct execution"""
    # logger.info(f"[DIRECT HANDLER DEBUG] Function called! routing: {routing}")  # Commented out to avoid logging entire routing object
    logger.info(f"[DIRECT HANDLER] handle_direct_tool_query called for tool query")
    conversation_id = request.conversation_id or request.session_id
    
    async def stream():
        logger.info(f"[DIRECT HANDLER] stream() function started")
        # Initialize enhanced_question early to avoid UnboundLocalError
        enhanced_question = request.question
        
        try:
            # Store user message in conversation history
            if conversation_id:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.question
                )
            
            # Initialize enhanced_question 
            enhanced_question = request.question
            logger.info(f"[DIRECT HANDLER] About to send classification info")
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "direct_tool_execution",
                "message": f"Directly executing tool with {routing['confidence']:.2f} confidence"
            }) + "\n"
            
            # Get the suggested tool from classification
            # logger.info(f"[DIRECT HANDLER] Full routing structure: {routing}")
            suggested_tools = routing.get('suggested_tools', [])
            # Also check nested structure
            if not suggested_tools and 'routing' in routing:
                suggested_tools = routing['routing'].get('suggested_tools', [])
            logger.info(f"[DIRECT HANDLER] Suggested tools: {suggested_tools}")
            
            # Skip conversation history for real-time/current data queries to avoid contamination
            tools_for_current_data = ['get_datetime', 'get_current_time', 'current_time', 'datetime', 'now']  
            is_current_data_query = any(tool in suggested_tools for tool in tools_for_current_data)
            
            # Check if this is a comprehensive query requiring fresh data retrieval
            is_comprehensive_query = _detect_comprehensive_query(request.question)
            
            logger.info(f"[DIRECT HANDLER] Tools for current data: {tools_for_current_data}")
            logger.info(f"[DIRECT HANDLER] Is current data query: {is_current_data_query}")
            logger.info(f"[DIRECT HANDLER] Is comprehensive query: {is_comprehensive_query}")
            
            # Log routing decision for debugging
            _log_routing_decision(
                message=request.question,
                is_comprehensive=is_comprehensive_query,
                is_current_data=is_current_data_query,
                conversation_id=conversation_id,
                handler_name="DIRECT"
            )
            
            # Retrieve conversation history and enhance question with context
            # Skip context for current data queries AND comprehensive queries to ensure fresh/complete results
            if conversation_id and not is_current_data_query and not is_comprehensive_query:
                # Get last 3 exchanges (6 messages: 3 user + 3 assistant)
                history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                if history:
                    # Format history for LLM context
                    enhanced_question = conversation_manager.format_history_for_prompt(history, request.question)
                    logger.info(f"[DIRECT HANDLER] Retrieved {len(history)} messages for conversation {conversation_id}")
                    logger.info(f"[DIRECT HANDLER] Enhanced question with conversation context")
            elif is_current_data_query:
                logger.info(f"[DIRECT HANDLER] Skipping conversation history for current data query to avoid contamination")
                # enhanced_question already set to request.question above
            elif is_comprehensive_query:
                logger.info(f"[DIRECT HANDLER] Skipping conversation history for comprehensive query to ensure complete data retrieval")
                # enhanced_question already set to request.question above
            else:
                logger.info(f"[DIRECT HANDLER] No conversation ID, using original question")
                # enhanced_question already set to request.question above
            if not suggested_tools:
                # Fallback to rag_answer if no tools suggested  
                logger.info(f"[DIRECT HANDLER] No suggested tools found, falling back to regular rag_answer flow")
                yield json_module.dumps({
                    "type": "status", 
                    "message": "No specific tool suggested, falling back to planning..."
                }) + "\n"
                
                from app.langchain.service import rag_answer
                rag_stream = await rag_answer(
                    enhanced_question,  # Use enhanced_question which includes conversation history
                    thinking=request.thinking,
                    stream=True,
                    conversation_id=conversation_id,
                    use_langgraph=request.use_langgraph,
                    collections=request.collections,
                    collection_strategy=request.collection_strategy,
                    query_type="TOOLS",
                    trace=trace
                )
                async for chunk in rag_stream:
                    yield chunk
                return
            
            # Execute tools directly without redundant classification
            logger.info(f"[DIRECT HANDLER] Executing tools directly: {suggested_tools}")
            
            yield json_module.dumps({
                "type": "status", 
                "message": f"Executing {suggested_tools[0]} directly..."
            }) + "\n"
            
            # Import tool execution function
            from app.langchain.service import call_mcp_tool
            
            tool_results = []
            for tool_name in suggested_tools[:1]:  # Execute primary tool only
                try:
                    logger.info(f"[DIRECT HANDLER] Executing tool: {tool_name}")
                    
                    # Get tool schema to determine if parameters are needed
                    from app.core.mcp_tools_cache import get_enabled_mcp_tools
                    available_tools = get_enabled_mcp_tools()
                    tool_info = available_tools.get(tool_name, {})
                    tool_params_schema = tool_info.get('parameters', {})
                    
                    # Check if tool requires parameters based on schema
                    required_params = []
                    if isinstance(tool_params_schema, dict):
                        required_params = tool_params_schema.get('required', [])
                    
                    parameters = {}
                    if not required_params:
                        # Tool needs no parameters - execute immediately
                        logger.info(f"[DIRECT HANDLER] Tool {tool_name} requires no parameters")
                    else:
                        # Tool needs parameters - determine them intelligently
                        yield json_module.dumps({
                            "type": "status", 
                            "message": f"Planning parameters for {tool_name}..."
                        }) + "\n"
                        
                        # Use simple parameter mapping for common patterns
                        if "query" in required_params:
                            parameters["query"] = request.question
                        
                        # For complex tools with multiple required params, use intelligent planning
                        if len(required_params) > 1 or ("query" not in required_params and required_params):
                            from app.langchain.intelligent_tool_planner import get_tool_planner
                            planner = get_tool_planner()
                            
                            execution_plan = await planner.plan_tool_execution(
                                task=request.question,
                                context={"user_query": request.question},
                                mode="standard"
                            )
                            
                            # Find the matching tool in the plan
                            tool_plan = None
                            for planned_tool in execution_plan.tools:
                                if planned_tool.tool_name == tool_name:
                                    tool_plan = planned_tool
                                    break
                            
                            if tool_plan:
                                parameters = tool_plan.parameters
                    
                    # Send progress update before execution
                    yield json_module.dumps({
                        "type": "status", 
                        "message": f"Executing {tool_name}..."
                    }) + "\n"
                    
                    logger.info(f"[DIRECT HANDLER] Using planned parameters for {tool_name}: {parameters}")
                    
                    result = call_mcp_tool(
                        tool_name=tool_name,
                        parameters=parameters,
                        trace=trace
                    )
                    
                    tool_results.append({
                        "tool": tool_name,
                        "success": True,
                        "result": result
                    })
                    
                    yield json_module.dumps({
                        "type": "tool_execution",
                        "tool": tool_name,
                        "success": True,
                        "result": result
                    }) + "\n"
                    
                except Exception as e:
                    logger.error(f"[DIRECT HANDLER] Tool {tool_name} failed: {e}")
                    tool_results.append({
                        "tool": tool_name,
                        "success": False,
                        "error": str(e)
                    })
            
            # Generate final response with tool results
            logger.info(f"[DIRECT HANDLER] Processing {len(tool_results)} tool results")
            logger.info(f"[DEBUG TOOL_RESULTS] Tool results structure: {tool_results}")
            logger.info(f"[DEBUG TOOL_RESULTS] Success check: tool_results={bool(tool_results)}, any_success={any(r.get('success') for r in tool_results) if tool_results else False}")
            if tool_results and any(r.get('success') for r in tool_results):
                try:
                    # Send progress update before synthesis
                    yield json_module.dumps({
                        "type": "status", 
                        "message": "Generating response from tool results..."
                    }) + "\n"
                    
                    # Build context from tool results and extract documents
                    tool_context = ""
                    extracted_documents = []
                    
                    for tr in tool_results:
                        if tr.get('success'):
                            # Extract readable text from tool result
                            tool_name = tr['tool']
                            raw_result = tr['result']
                            
                            # Format tool result with enhanced source attribution
                            if tool_name == 'google_search' or 'search' in tool_name.lower():
                                # Extract primary subject for better focus
                                primary_subject = extract_primary_subject(request.question)
                                
                                # Special handling for search results with source attribution
                                tool_context += f"\n{'='*60}\n"
                                tool_context += f"🔍 SEARCH RESULTS FROM: {tool_name.upper()}\n"
                                tool_context += f"Query: {request.question}\n"
                                tool_context += f"PRIMARY SUBJECT: {primary_subject}\n"
                                tool_context += f"{'='*60}\n"
                                tool_context += f"⚠️ CRITICAL RELEVANCE FILTERING REQUIREMENTS:\n"
                                tool_context += f"• PRIMARY FOCUS: Extract ONLY information about '{primary_subject}'\n"
                                tool_context += f"• Each result below may discuss MULTIPLE entities/topics\n"
                                tool_context += f"• DO NOT assume all information in a result relates to '{primary_subject}'\n"
                                tool_context += f"• If a result mentions multiple products/entities, carefully distinguish between them\n"
                                tool_context += f"• Before using any fact, verify: 'Is this specifically about {primary_subject}?'\n"
                                tool_context += f"• When in doubt, OMIT information rather than risk mixing entities\n"
                                tool_context += f"{'='*60}\n\n"
                                
                                if isinstance(raw_result, dict) and 'content' in raw_result:
                                    # Parse search results with source attribution
                                    search_items = []
                                    for item in raw_result['content']:
                                        if isinstance(item, dict) and item.get('type') == 'text':
                                            text = item.get('text', '')
                                            # Try to extract URL/source from the text if available
                                            lines = text.split('\n')
                                            result_num = len(search_items) + 1
                                            
                                            # Format each search result with clear numbering and separation
                                            formatted_result = f"📌 RESULT #{result_num}:\n"
                                            formatted_result += f"⚠️ RELEVANCE FILTER: Only use information about '{primary_subject}'\n"
                                            formatted_result += f"{'-'*40}\n"
                                            formatted_result += text
                                            formatted_result += f"\n{'-'*40}\n"
                                            formatted_result += f"✓ VERIFICATION: Does the above specifically relate to '{primary_subject}'?\n"
                                            formatted_result += f"✓ ACTION: Extract ONLY '{primary_subject}' information, ignore other entities\n"
                                            formatted_result += f"{'-'*40}\n"
                                            search_items.append(formatted_result)
                                    
                                    if search_items:
                                        tool_context += "\n".join(search_items)
                                        tool_context += f"\n{'='*60}\n"
                                        tool_context += f"⚠️ END OF {tool_name.upper()} RESULTS\n"
                                        tool_context += f"FINAL REMINDER: Focus ONLY on '{primary_subject}'\n"
                                        tool_context += f"DO NOT mix information about different entities/products\n"
                                        tool_context += f"{'='*60}\n\n"
                                    else:
                                        tool_context += f"No readable results found from {tool_name}\n\n"
                                else:
                                    # Fallback for other formats
                                    readable_result = str(raw_result)
                                    tool_context += f"Raw search results:\n{readable_result}\n"
                                    tool_context += f"\n{'='*60}\n\n"
                                
                            else:
                                # Standard formatting for non-search tools
                                if isinstance(raw_result, dict) and 'content' in raw_result:
                                    # Extract text from content array
                                    content_parts = []
                                    for item in raw_result['content']:
                                        if isinstance(item, dict) and item.get('type') == 'text':
                                            content_parts.append(item.get('text', ''))
                                    readable_result = ' '.join(content_parts) if content_parts else str(raw_result)
                                else:
                                    readable_result = str(raw_result)
                                
                                # Format with clear tool attribution
                                tool_context += f"\n{'='*60}\n"
                                tool_context += f"📊 TOOL: {tool_name.upper()}\n"
                                tool_context += f"{'='*60}\n"
                                tool_context += f"{readable_result}\n"
                                tool_context += f"{'='*60}\n\n"
                            
                            logger.info(f"[DIRECT HANDLER] Formatted {tool_name} result with enhanced attribution")
                            
                            # Extract documents from RAG search results
                            # Import is_rag_tool function for flexible tool name matching
                            from app.langchain.service import is_rag_tool
                            if is_rag_tool(tr['tool']) and isinstance(tr.get('result'), dict):
                                # Parse JSON-RPC response
                                result_data = tr['result']
                                if 'result' in result_data and isinstance(result_data['result'], dict):
                                    rag_result = result_data['result']
                                    if 'documents' in rag_result:
                                        extracted_documents.extend(rag_result['documents'])
                    
                    logger.info(f"[DIRECT HANDLER] Tool context built: {tool_context[:200]}...")
                    logger.info(f"[DIRECT HANDLER] Extracted {len(extracted_documents)} documents from tool results")
                    
                    # CRITICAL FIX: When tool results exist, format conversation history as secondary context
                    # Tool results are authoritative and take precedence over conversation history
                    logger.info("[TOOL RESULTS WIN] Tool results exist - formatting conversation history as secondary context")
                    
                    # Get conversation history if available but label it clearly
                    labeled_conversation_history = ""
                    if conversation_id:
                        raw_history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                        if raw_history:
                            # Format history with clear labeling
                            history_text = conversation_manager.format_history_for_prompt(raw_history, "")
                            labeled_conversation_history = f"""
📝 PREVIOUS CONVERSATION (may contain outdated information):
{history_text}

⚠️ NOTE: The above conversation history is provided for context only. 
If there are any conflicts between the conversation history and the current search results,
ALWAYS trust the search results as they contain the most up-to-date information."""
                            logger.info(f"[TOOL RESULTS WIN] Included {len(raw_history)} messages as labeled secondary context")
                    
                    # Use unified synthesis with proper message format
                    from app.langchain.service import build_messages_for_synthesis
                    
                    logger.info(f"[DEBUG CALL] About to call build_messages_for_synthesis with original question and labeled history")
                    
                    try:
                        messages, source_label, full_context, system_prompt = build_messages_for_synthesis(
                            question=request.question,  # Use original question
                            query_type="TOOLS",
                            tool_context=tool_context,
                            conversation_history=labeled_conversation_history,  # Pass labeled history
                            thinking=request.thinking if hasattr(request, 'thinking') else False
                        )
                        logger.info(f"[DEBUG CALL] build_messages_for_synthesis returned successfully, messages count: {len(messages)}")
                    except Exception as e:
                        logger.error(f"[DEBUG CALL] build_messages_for_synthesis failed with error: {e}")
                        raise
                    
                    # For backward compatibility, create a single prompt
                    synthesis_prompt = "\n\n".join([msg["content"] for msg in messages])
                    
                    logger.info(f"[DIRECT HANDLER] About to stream synthesis response")
                    
                    # Get LLM settings first
                    from app.llm.ollama import OllamaLLM
                    from app.llm.base import LLMConfig
                    
                    llm_settings = get_llm_settings()
                    
                    # Apply dynamic detection and model fallback for main LLM configuration
                    main_llm_base = llm_settings.get('main_llm', {})
                    model_name = main_llm_base.get('model', '')
                    
                    # Disable automatic fallback to investigate the real issue
                    model_fallback_applied = False
                    original_model = model_name
                    
                    # DISABLED FALLBACK - Let's find the real issue
                    # if 'qwen3:30b-a3b-instruct-2507-q4_K_M' in model_name:
                    #     fallback_model = 'qwen3:30b-a3b-q4_K_M'
                    #     logger.warning(f"[MAIN LLM FALLBACK] Detected problematic model {model_name}, switching to {fallback_model}")
                    #     # ... fallback logic disabled
                    
                    if True:  # Always use original model now
                        # Try to get cached behavior or use model name heuristics for other models
                        detected_mode = None
                        try:
                            from app.llm.response_analyzer import response_analyzer
                            cached_profile = response_analyzer._get_cached_behavior(model_name)
                            if cached_profile and cached_profile.confidence > 0.7:
                                detected_mode = 'thinking' if cached_profile.is_thinking_model else 'non-thinking'
                                logger.info(f"[MAIN LLM DETECTION] Using cached behavior for {model_name}: {detected_mode} (confidence: {cached_profile.confidence:.2f})")
                            else:
                                # Use model name heuristics for instruct models
                                if 'instruct' in model_name.lower() and '2507' in model_name:
                                    detected_mode = 'non-thinking'
                                    logger.info(f"[MAIN LLM DETECTION] Using heuristic detection for {model_name}: {detected_mode}")
                        except Exception as e:
                            logger.warning(f"[MAIN LLM DETECTION] Failed to apply dynamic detection: {e}")
                        
                        main_llm_config = get_main_llm_full_config(llm_settings, override_mode=detected_mode)
                    
                    # Log effective configuration
                    logger.info(f"[MAIN LLM CONFIG] Original model: {original_model}")
                    logger.info(f"[MAIN LLM CONFIG] Final model: {main_llm_config.get('model')}")
                    logger.info(f"[MAIN LLM CONFIG] Model fallback applied: {model_fallback_applied}")
                    logger.info(f"[MAIN LLM CONFIG] Effective mode: {main_llm_config.get('effective_mode', 'unknown')}")
                    logger.info(f"[MAIN LLM CONFIG] Mode overridden: {main_llm_config.get('mode_overridden', False)}")
                    logger.info(f"[MAIN LLM CONFIG] Temperature: {main_llm_config.get('temperature', 'not_set')}")
                    logger.info(f"[MAIN LLM CONFIG] Top-p: {main_llm_config.get('top_p', 'not_set')}")
                    logger.info(f"[MAIN LLM CONFIG] Max tokens: {main_llm_config.get('max_tokens', 'not_set')}")
                    
                    # Create synthesis generation span for Langfuse tracing
                    synthesis_generation_span = None
                    if trace:
                        try:
                            from app.core.langfuse_integration import get_tracer
                            tracer = get_tracer()
                            if tracer.is_enabled():
                                synthesis_generation_span = tracer.create_llm_generation_span(
                                    trace,
                                    model=main_llm_config.get('model'),
                                    prompt=messages,  # Pass messages instead of single prompt
                                    operation="direct_tool_synthesis"
                                )
                                logger.info(f"[DIRECT HANDLER] Created synthesis generation span")
                        except Exception as e:
                            logger.warning(f"Failed to create synthesis generation span: {e}")
                    
                    # Configure LLM for streaming using refactored config
                    max_tokens_from_config = main_llm_config.get('max_tokens', 4000)
                    llm_config = LLMConfig(
                        model_name=main_llm_config.get('model'),
                        temperature=float(main_llm_config.get('temperature', 0.8)),
                        max_tokens=int(max_tokens_from_config),
                        top_p=float(main_llm_config.get('top_p', 0.9))
                    )
                    
                    # Get model server from settings - no hardcoded fallback
                    model_server = main_llm_config.get('model_server', '').strip()
                    
                    if not model_server:
                        # Use environment variable as last resort
                        import os
                        model_server = os.environ.get("OLLAMA_BASE_URL", "")
                    
                    if not model_server:
                        logger.error("No model server configured in settings or environment")
                        raise ValueError("Model server must be configured in LLM settings")
                    
                    llm = OllamaLLM(llm_config, base_url=model_server)
                    
                    # Stream synthesis response token by token using chat with messages
                    final_response = ""
                    # Log system prompt being used
                    logger.info(f"[DIRECT HANDLER] Using system prompt: {system_prompt[:100]}...")
                    logger.info(f"[DIRECT HANDLER] Full system prompt length: {len(system_prompt)}")
                    logger.info(f"[DIRECT HANDLER] Messages array length: {len(messages)}")
                    for i, msg in enumerate(messages):
                        logger.info(f"[DIRECT HANDLER] Message {i} - Role: {msg.get('role')}, Content length: {len(msg.get('content', ''))}")
                        if msg.get('role') == 'system':
                            logger.info(f"[DIRECT HANDLER] System message preview: {msg.get('content', '')[:200]}...")
                    
                    # Use chat_stream with messages instead of generate_stream
                    # Track response for dynamic behavior detection
                    response_tokens = []
                    first_analysis_done = False
                    chunk_count = 0
                    
                    
                    async for response_chunk in llm.chat_stream(messages):
                        chunk_count += 1
                        chunk_text = response_chunk.text
                        final_response += chunk_text
                        response_tokens.append(chunk_text)
                        
                        # Log first few chunks for debugging
                        if chunk_count <= 5:
                            pass
                        
                        # Log every 100 chunks to track progress
                        if chunk_count % 100 == 0:
                            pass
                        
                        # Analyze first 10 tokens for thinking behavior detection
                        if not first_analysis_done and len(response_tokens) >= 10:
                            first_analysis_done = True
                            try:
                                from app.llm.response_analyzer import detect_model_thinking_behavior
                                sample_text = ''.join(response_tokens[:20])  # First 20 tokens
                                model_name = main_llm_config.get('model', 'unknown')
                                is_thinking, confidence = detect_model_thinking_behavior(sample_text, model_name)
                                
                                logger.info(f"[DYNAMIC DETECTION] Model: {model_name}")
                                logger.info(f"[DYNAMIC DETECTION] Detected thinking behavior: {is_thinking} (confidence: {confidence:.2f})")
                                logger.info(f"[DYNAMIC DETECTION] Sample analyzed: {sample_text[:100]}...")
                                
                                # Log if detection differs from configured mode
                                configured_mode = main_llm_config.get('mode', 'thinking')
                                expected_thinking = configured_mode == 'thinking'
                                if is_thinking != expected_thinking and confidence > 0.7:
                                    logger.warning(f"[DYNAMIC DETECTION] Mode mismatch! Configured: {configured_mode}, Detected: {'thinking' if is_thinking else 'non-thinking'}")
                                    
                            except Exception as e:
                                logger.warning(f"[DYNAMIC DETECTION] Analysis failed: {e}")
                        
                        if response_chunk.text.strip():
                            yield json_module.dumps({
                                "token": response_chunk.text
                            }) + "\n"
                    
                    logger.info(f"[DIRECT HANDLER] Synthesis completed, response length: {len(final_response)}")
                    
                    # End synthesis generation span with output
                    if synthesis_generation_span:
                        try:
                            tracer = get_tracer()
                            usage = tracer.estimate_token_usage(synthesis_prompt, final_response)
                            synthesis_generation_span.end(
                                output=final_response,
                                usage=usage,
                                metadata={
                                    "response_length": len(final_response),
                                    "operation": "direct_tool_synthesis",
                                    "tool_results_count": len(tool_results),
                                    "optimization": "direct_execution"
                                }
                            )
                            logger.info(f"[DIRECT HANDLER] Synthesis generation span ended with output")
                        except Exception as e:
                            logger.warning(f"Failed to end synthesis generation span: {e}")
                    
                    # Process final response based on detected model behavior
                    processed_response = final_response
                    try:
                        from app.llm.response_analyzer import detect_model_thinking_behavior
                        model_name = main_llm_config.get('model', 'unknown')
                        is_thinking, confidence = detect_model_thinking_behavior(final_response, model_name)
                        
                        if is_thinking and confidence > 0.8:
                            # Remove thinking tags from response
                            import re
                            processed_response = re.sub(r'<think>.*?</think>', '', final_response, flags=re.DOTALL | re.IGNORECASE)
                            processed_response = processed_response.strip()
                            
                            logger.info(f"[RESPONSE PROCESSING] Removed thinking tags from response")
                            logger.info(f"[RESPONSE PROCESSING] Original length: {len(final_response)}, Processed length: {len(processed_response)}")
                        
                    except Exception as e:
                        logger.warning(f"[RESPONSE PROCESSING] Failed to process thinking tags: {e}")
                        processed_response = final_response
                    
                    # Send final answer with documents
                    response_data = {
                        "answer": processed_response,
                        "source": "DIRECT_TOOL_EXECUTION", 
                        "context": tool_context,
                        "query_type": "TOOLS",
                        "metadata": {
                            "tools_executed": [tr['tool'] for tr in tool_results if tr.get('success')],
                            "direct_execution": True,
                            "optimization": "eliminated_redundant_classification"
                        }
                    }
                    
                    # Add documents if any were extracted
                    if extracted_documents:
                        response_data["documents"] = extracted_documents
                        response_data["metadata"]["documents_found"] = len(extracted_documents)
                    
                    yield json_module.dumps(response_data) + "\n"
                    
                    logger.info(f"[DIRECT HANDLER] Final answer sent to frontend")
                    
                    # Save assistant response to conversation history (use processed response)
                    if conversation_id and processed_response:
                        await conversation_manager.add_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=processed_response,
                            metadata={"source": "DIRECT_TOOL_EXECUTION", "tools_executed": [tr['tool'] for tr in tool_results if tr.get('success')]}
                        )
                    
                except Exception as synthesis_error:
                    logger.error(f"[DIRECT HANDLER] Synthesis failed: {synthesis_error}")
                    yield json_module.dumps({
                        "type": "error",
                        "message": f"Response synthesis failed: {str(synthesis_error)}"
                    }) + "\n"
            else:
                # Fallback if tools failed
                logger.warning(f"[DIRECT HANDLER] No successful tool results")
                yield json_module.dumps({
                    "type": "error",
                    "message": "Tool execution failed, please try again"
                }) + "\n"
                    
        except Exception as e:
            yield json_module.dumps({
                "type": "handler_error",
                "error": str(e),
                "message": "Direct tool execution failed, falling back to planning..."
            }) + "\n"
            
            # Fallback to regular rag_answer
            from app.langchain.service import rag_answer
            rag_stream = await rag_answer(
                enhanced_question,
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=request.use_langgraph,
                collections=request.collections,
                collection_strategy=request.collection_strategy,
                query_type="TOOLS",
                trace=trace
            )
            async for chunk in rag_stream:
                yield chunk
                    
        except Exception as e:
            yield json_module.dumps({
                "type": "handler_error",
                "error": str(e),
                "message": "Direct tool handler failed completely"
            }) + "\n"
    
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
    
    async def stream():
        try:
            # Send classification info
            yield json_module.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "langgraph_multi_agent",
                "message": "This is a complex query. Using LangGraph multi-agent system."
            }) + "\n"
            
            # Use LangGraph multi-agent system
            from app.langchain.langgraph_multi_agent_system import langgraph_multi_agent_answer
            
            yield json_module.dumps({
                "type": "status",
                "message": "🤖 Initializing LangGraph multi-agent workflow..."
            }) + "\n"
            
            # Process with LangGraph system
            result = await langgraph_multi_agent_answer(
                question=request.question,
                conversation_id=conversation_id
            )
            
            # Stream the results
            yield json_module.dumps({
                "type": "agent_selection",
                "agents_selected": result.get("agents_used", []),
                "execution_pattern": result.get("execution_pattern", ""),
                "selection_reasoning": result.get("agent_selection_reasoning", ""),
                "confidence_score": result.get("confidence_score", 0.0)
            }) + "\n"
            
            yield json_module.dumps({
                "type": "status",
                "message": f"🔄 Executing {result.get('execution_pattern', 'sequential')} collaboration pattern..."
            }) + "\n"
            
            # Stream final response
            final_answer = result.get("answer", "")
            
            # Stream token by token for consistency with other handlers
            import asyncio
            for i in range(0, len(final_answer), 10):  # Stream in chunks of 10 characters
                chunk = final_answer[i:i+10]
                yield json_module.dumps({
                    "token": chunk
                }) + "\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
            # Send final answer
            yield json_module.dumps({
                "answer": final_answer,
                "source": "LANGGRAPH_MULTI_AGENT",
                "agents_used": result.get("agents_used", []),
                "execution_pattern": result.get("execution_pattern", ""),
                "confidence_score": result.get("confidence_score", 0.0),
                "conversation_id": conversation_id,
                "metadata": result.get("metadata", {})
            }) + "\n"
                
        except Exception as e:
            logger.error(f"LangGraph multi-agent handler error: {e}")
            yield json_module.dumps({"error": f"LangGraph multi-agent handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_hybrid_query(request: RAGRequest, routing: Dict):
    """Handle hybrid queries that require multiple components (Tools + RAG + LLM)"""
    conversation_id = request.conversation_id or request.session_id
    
    async def stream():
        try:
            # Store user message in conversation history
            if conversation_id:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=request.question
                )
            
            # Retrieve conversation history and enhance question with context
            enhanced_question = request.question
            
            # Check if this is a comprehensive query requiring fresh data retrieval
            is_comprehensive_query = _detect_comprehensive_query(request.question)
            logger.info(f"[HYBRID HANDLER] Is comprehensive query: {is_comprehensive_query}")
            
            # Log routing decision for debugging
            _log_routing_decision(
                message=request.question,
                is_comprehensive=is_comprehensive_query,
                is_current_data=False,  # Hybrid handler doesn't check for current data queries
                conversation_id=conversation_id,
                handler_name="HYBRID"
            )
            
            if conversation_id and not is_comprehensive_query:
                # Get last 3 exchanges (6 messages: 3 user + 3 assistant)
                history = await conversation_manager.get_conversation_history(conversation_id, limit=6)
                if history:
                    # Format history for LLM context
                    enhanced_question = conversation_manager.format_history_for_prompt(history, request.question)
                    logger.info(f"[HYBRID HANDLER] Enhanced question with conversation context")
            elif is_comprehensive_query:
                logger.info(f"[HYBRID HANDLER] Skipping conversation history for comprehensive query to ensure complete data retrieval")
            
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
            enhanced_prompt = build_hybrid_prompt(enhanced_question, results, components)
            
            # Stream the final synthesized answer using rag_answer
            from app.langchain.service import rag_answer
            
            # Create hybrid context with the results
            hybrid_synthesis_context = {
                "sources": [],
                "strategy": "hybrid_synthesis",
                "pre_formatted_context": enhanced_prompt,
                "component_results": results
            }
            
            # Use rag_answer with the enhanced question and pass context via hybrid_context
            final_answer = ""
            async for chunk in rag_answer(
                enhanced_question,  # Pass enhanced_question to maintain conversation context
                thinking=request.thinking,
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=False,  # Direct LLM for synthesis
                collections=request.collections,
                collection_strategy="none",  # We already have context
                query_type="SYNTHESIS",  # Use SYNTHESIS mode to skip additional searches
                hybrid_context=hybrid_synthesis_context
            ):
                # Extract token from chunk for accumulation
                if isinstance(chunk, str):
                    try:
                        data = json_module.loads(chunk.strip())
                        if data.get("token"):
                            final_answer += data["token"]
                        elif data.get("answer"):
                            final_answer = data["answer"]
                    except:
                        pass
                yield chunk
            
            # Save assistant response to conversation history
            if conversation_id and final_answer:
                await conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_answer,
                    metadata={"source": "hybrid_handler", "components": components}
                )
                
        except Exception as e:
            logger.error(f"Hybrid handler error: {e}")
            yield json_module.dumps({"error": f"Hybrid handling failed: {str(e)}"}) + "\n"
    
    return StreamingResponse(stream(), media_type="application/json")

def build_hybrid_prompt(question: str, results: Dict, components: List[str]) -> str:
    """Build an enhanced prompt that combines results from all components"""
    # Get prompt templates from database
    from app.core.prompt_settings_cache import get_synthesis_prompts, get_system_behaviors
    synthesis_templates = get_synthesis_prompts()
    behaviors = get_system_behaviors()
    
    # Use template for hybrid prompt building
    hybrid_template = synthesis_templates.get('hybrid_synthesis', 
        "User Question: {question}\n{context}\n\nProvide a comprehensive answer.")
    
    # Build context sections
    context_parts = []
    
    if "tool" in components and "tool" in results:
        tool_data = results["tool"]
        context_parts.append("\n## Real-time Information:")
        if tool_data.get("suggested_tools"):
            context_parts.append(f"Suggested tools: {', '.join(tool_data['suggested_tools'])}")
        context_parts.append("(Tool execution results would appear here)")
    
    if "rag" in components and "rag" in results:
        rag_data = results["rag"]
        if rag_data:
            context_parts.append("\n## Knowledge Base Context:")
            context_parts.append(str(rag_data))
    
    # Add synthesis instructions if configured
    if behaviors.get('include_synthesis_instructions', True):
        context_parts.append("\n## Task:")
        context_parts.append("Based on the above information, provide a comprehensive answer that:")
        context_parts.append("1. Incorporates relevant real-time data (if available)")
        context_parts.append("2. Uses knowledge from the knowledge base (if available)")
        context_parts.append("3. Synthesizes all information into a coherent response")
        if behaviors.get('include_sources', True):
            context_parts.append("4. Clearly indicates which information comes from which source")
    
    context = "\n".join(context_parts)
    
    # Apply template
    if "{question}" in hybrid_template and "{context}" in hybrid_template:
        return hybrid_template.format(question=question, context=context)
    else:
        # Fallback for simple concatenation
        return f"User Question: {question}\n{context}"

# ===== TEMPORARY DOCUMENT ENDPOINTS =====

@router.post("/upload-document")
async def upload_temp_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(...),
    ttl_hours: int = Form(2),
    auto_include: bool = Form(True)
):
    """Upload a temporary document for conversation-scoped indexing."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process document
        result = await temp_doc_manager.upload_and_process_document(
            file_content=file_content,
            filename=file.filename,
            conversation_id=conversation_id,
            ttl_hours=ttl_hours,
            auto_include=auto_include
        )
        
        if result['success']:
            return {
                "success": True,
                "temp_doc_id": result['temp_doc_id'],
                "message": f"Document '{file.filename}' uploaded and indexed successfully",
                "metadata": result['metadata']
            }
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

@router.get("/temp-documents/{conversation_id}")
async def get_temp_documents(conversation_id: str):
    """Get all temporary documents for a conversation."""
    try:
        documents = await temp_doc_manager.get_conversation_documents(conversation_id)
        return {
            "conversation_id": conversation_id,
            "documents": documents,
            "total_count": len(documents),
            "active_count": len([doc for doc in documents if doc.get('is_included', False)])
        }
    except Exception as e:
        logger.error(f"Failed to get temp documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/temp-documents/{temp_doc_id}")
async def delete_temp_document(temp_doc_id: str):
    """Delete a temporary document."""
    try:
        success = await temp_doc_manager.delete_document(temp_doc_id)
        return {"success": success}
    except Exception as e:
        logger.error(f"Failed to delete temp document: {str(e)}")
        return {"success": True}  # Idempotent - return success even if not found

@router.put("/temp-documents/{temp_doc_id}/preferences")
async def update_temp_document_preferences(
    temp_doc_id: str,
    preferences: Dict[str, Any]
):
    """Update preferences for a temporary document."""
    try:
        success = await temp_doc_manager.update_document_preferences(temp_doc_id, preferences)
        if success:
            return {"success": True, "message": "Preferences updated"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Failed to update preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def _get_notebook_id_from_conversation(db: Session, conversation_id: str) -> Optional[str]:
    """Get notebook_id associated with a conversation_id from database."""
    try:
        query = text("""
            SELECT notebook_id 
            FROM notebook_conversations 
            WHERE conversation_id = :conversation_id
            LIMIT 1
        """)
        result = db.execute(query, {"conversation_id": conversation_id})
        row = result.fetchone()
        return row.notebook_id if row else None
    except Exception as e:
        logger.warning(f"Failed to get notebook_id for conversation {conversation_id}: {e}")
        return None

async def _clear_notebook_caches(notebook_id: str, conversation_id: str):
    """Clear notebook-related Redis caches when conversation is deleted."""
    try:
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis client not available for cache cleanup")
            return
        
        # Cache patterns to clear based on notebook_id
        cache_patterns = [
            f"notebook_content_count:{notebook_id}:*",
            f"task_plan_{notebook_id}_*",
            f"llm_settings_notebook_{notebook_id}",
            f"notebook_rag_cache:{notebook_id}:*",
            f"notebook_sources:{notebook_id}:*"
        ]
        
        cleared_keys = 0
        for pattern in cache_patterns:
            try:
                # Use SCAN to find keys matching pattern
                keys = []
                cursor = 0
                while True:
                    cursor, batch = redis_client.scan(cursor=cursor, match=pattern, count=100)
                    keys.extend(batch)
                    if cursor == 0:
                        break
                
                # Delete found keys
                if keys:
                    redis_client.delete(*keys)
                    cleared_keys += len(keys)
                    logger.debug(f"Cleared {len(keys)} keys matching pattern: {pattern}")
                    
            except Exception as e:
                logger.warning(f"Failed to clear cache pattern {pattern}: {e}")
        
        if cleared_keys > 0:
            logger.info(f"Successfully cleared {cleared_keys} notebook cache entries for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Failed to clear notebook caches for conversation {conversation_id}: {e}")
        # Don't raise - cache cleanup failure shouldn't break conversation deletion

@router.delete("/conversation/{conversation_id}")
async def clear_conversation_history(conversation_id: str, db: Session = Depends(get_db)):
    """Clear conversation history from Redis for a specific conversation ID and cleanup associated caches."""
    try:
        # First clear the conversation history
        await conversation_manager.clear_conversation(conversation_id)
        
        # Get notebook_id associated with this conversation for cache cleanup
        notebook_id = await _get_notebook_id_from_conversation(db, conversation_id)
        
        if notebook_id:
            # Clear notebook-related Redis caches
            await _clear_notebook_caches(notebook_id, conversation_id)
            logger.info(f"Cleared notebook caches for notebook {notebook_id} after conversation {conversation_id} deletion")
        
        return {"success": True, "message": f"Conversation {conversation_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear conversation {conversation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}") 