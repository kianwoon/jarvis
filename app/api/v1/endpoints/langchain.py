from fastapi import APIRouter
from pydantic import BaseModel
from app.langchain.service import rag_answer
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier, QueryType
from fastapi.responses import StreamingResponse
import json
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
def rag_endpoint(request: RAGRequest):
    # Use session_id as conversation_id if conversation_id is not provided
    conversation_id = request.conversation_id or request.session_id
    
    # Classify the query unless explicitly skipped
    routing = None
    if not request.skip_classification:
        routing = query_classifier.get_routing_recommendation(request.question)
        logger.info(f"Query routing: {routing['primary_type']} (confidence: {routing['confidence']:.2f})")
        
        # Handle based on classification (using enhanced classifier types)
        primary_type = routing['primary_type']
        is_hybrid = routing.get('is_hybrid', False)
        
        # Check for hybrid queries that need multiple handlers
        if is_hybrid:
            # Handle hybrid queries (TOOL_RAG, TOOL_LLM, RAG_LLM, TOOL_RAG_LLM)
            return handle_hybrid_query(request, routing)
        elif primary_type == QueryType.TOOL.value and routing['confidence'] > 0.7:
            # Route to tool handler
            return handle_tool_query(request, routing)
        elif primary_type == QueryType.LLM.value and routing['confidence'] > 0.8:
            # Route to direct LLM only if high confidence
            return handle_direct_llm_query(request, routing)
        elif primary_type == QueryType.MULTI_AGENT.value and routing['confidence'] > 0.6:
            # Route to multi-agent system
            return handle_multi_agent_query(request, routing)
        # Fall through to RAG for RAG_SEARCH or low confidence cases
    
    def stream():
        try:
            chunk_count = 0
            # Add classification metadata to first chunk if available
            if not request.skip_classification and routing:
                classification_chunk = json.dumps({
                    "type": "classification",
                    "routing": routing,
                    "handler": "rag"
                }) + "\n"
                yield classification_chunk
                
            for chunk in rag_answer(
                request.question, 
                thinking=request.thinking, 
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=request.use_langgraph,
                collections=request.collections,
                collection_strategy=request.collection_strategy
            ):
                try:
                    if chunk:  # Only yield non-empty chunks
                        chunk_count += 1
                        yield chunk
                except GeneratorExit:
                    # Client disconnected, log and stop gracefully
                    print(f"[INFO] Client disconnected after {chunk_count} chunks")
                    break
                except Exception as chunk_error:
                    print(f"[ERROR] Error yielding chunk {chunk_count}: {chunk_error}")
                    # Try to yield error message, but don't fail the whole stream
                    try:
                        yield json.dumps({"error": f"Chunk error: {str(chunk_error)}"}) + "\n"
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
                yield json.dumps({"error": f"RAG processing failed: {str(e)}"}) + "\n"
            except:
                # If we can't send the error, client is likely disconnected
                print(f"[ERROR] Could not send error response, client likely disconnected")
    
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
    """Process a question using the multi-agent system"""
    system = MultiAgentSystem(conversation_id=request.conversation_id)
    
    async def stream_events():
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
            
            # Log key events
            event_type = event.get('type', 'unknown')
            if event_type == 'agent_complete':
                agent = event.get('agent', 'unknown')
                content_length = len(event.get('content', ''))
                print(f"[INFO] Agent {agent} completed ({content_length} chars)")
            elif event_type == 'final_response':
                response_length = len(event.get('response', ''))
                print(f"[INFO] Multi-agent processing complete ({response_length} chars)")
            
            # Convert to JSON string with newline, same as standard chat
            yield json.dumps(event, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_events(),
        media_type="application/json"
    )

@router.post("/large-generation")
async def large_generation_endpoint(request: LargeGenerationRequest):
    """
    Handle large generation tasks that transcend context limits
    using intelligent chunking and Redis-based state management
    """
    system = MultiAgentSystem(conversation_id=request.conversation_id)
    
    async def stream_events():
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
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            else:
                # Use in-memory approach for testing
                async for event in system.stream_large_generation_events(
                    query=request.task_description,
                    target_count=request.target_count,
                    chunk_size=request.chunk_size,
                    conversation_history=request.conversation_history
                ):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    
        except Exception as e:
            error_event = {
                "type": "large_generation_error",
                "error": str(e),
                "timestamp": json.dumps(datetime.now().isoformat())
            }
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_events(),
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
                yield json.dumps(event, ensure_ascii=False) + "\n"
        
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
            iter([json.dumps(error_event, ensure_ascii=False) + "\n"]),
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
            yield json.dumps({
                "type": "classification",
                "routing": routing,
                "handler": "tool_handler",
                "message": "This query requires external tools."
            }) + "\n"
            
            # Get suggested tools
            suggested_tools = routing.get("routing", {}).get("suggested_tools", [])
            
            # For now, provide a response indicating tool functionality
            yield json.dumps({
                "type": "status",
                "message": f"🔧 Tool search functionality would execute here. Suggested tools: {', '.join(suggested_tools) if suggested_tools else 'web_search'}"
            }) + "\n"
            
            # Since we don't have actual tool execution yet, 
            # fall back to regular LLM response with a note about tools
            enhanced_question = f"""
{request.question}

Note: This query would benefit from real-time tool execution (web search, APIs, etc).
Currently providing a response based on training data. 
In a full implementation, this would include live web search results.
"""
            
            # Use rag_answer to provide a response
            from app.langchain.service import rag_answer
            
            for chunk in rag_answer(
                enhanced_question,
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
            yield json.dumps({"error": f"Tool handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_direct_llm_query(request: RAGRequest, routing: Dict):
    """Handle queries that can be answered directly by LLM without RAG"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json.dumps({
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
            yield json.dumps({"error": f"Direct LLM handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_multi_agent_query(request: RAGRequest, routing: Dict):
    """Handle complex queries that need multiple agents"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json.dumps({
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
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            
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
            yield json.dumps({"error": f"Multi-agent handling failed: {str(e)}"}) + "\n"
            
    return StreamingResponse(stream(), media_type="application/json")

def handle_hybrid_query(request: RAGRequest, routing: Dict):
    """Handle hybrid queries that require multiple components (Tools + RAG + LLM)"""
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            # Send classification info
            yield json.dumps({
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
                yield json.dumps({
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
                
                yield json.dumps({
                    "type": "tool_result",
                    "data": tool_results
                }) + "\n"
            
            # 2. RAG component - Search knowledge base
            if "rag" in components:
                yield json.dumps({
                    "type": "status",
                    "message": "📚 Searching knowledge base..."
                }) + "\n"
                
                # Get RAG results using the service
                from app.langchain.service import hybrid_retrieval_with_rerank
                from app.rag.qdrant_retriever import QdrantRetriever
                
                try:
                    # Initialize retriever
                    retriever = QdrantRetriever()
                    
                    # Get RAG results
                    results = hybrid_retrieval_with_rerank(
                        retriever,
                        request.question,
                        collection_name="documents",  # Default collection
                        k=5
                    )
                    
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
                    yield json.dumps({
                        "type": "rag_result",
                        "data": {
                            "context_found": True,
                            "num_sources": len(rag_context.split('\n\n')) if isinstance(rag_context, str) else 0
                        }
                    }) + "\n"
            
            # 3. Synthesize with LLM
            yield json.dumps({
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
            yield json.dumps({"error": f"Hybrid handling failed: {str(e)}"}) + "\n"
    
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