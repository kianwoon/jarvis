from fastapi import APIRouter
from pydantic import BaseModel
from app.langchain.service import rag_answer
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from fastapi.responses import StreamingResponse
import json
from typing import Optional, List, Dict
from datetime import datetime

router = APIRouter()

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None  # Add this for backward compatibility
    use_langgraph: bool = True

class RAGResponse(BaseModel):
    answer: str
    source: str
    conversation_id: Optional[str] = None

@router.post("/rag")
def rag_endpoint(request: RAGRequest):
    # Use session_id as conversation_id if conversation_id is not provided
    conversation_id = request.conversation_id or request.session_id
    
    def stream():
        try:
            chunk_count = 0
            for chunk in rag_answer(
                request.question, 
                thinking=request.thinking, 
                stream=True,
                conversation_id=conversation_id,
                use_langgraph=request.use_langgraph
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