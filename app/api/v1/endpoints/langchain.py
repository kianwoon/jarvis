from fastapi import APIRouter
from pydantic import BaseModel
from app.langchain.service import rag_answer
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from fastapi.responses import StreamingResponse
import json
from typing import Optional, List, Dict

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
        for chunk in rag_answer(
            request.question, 
            thinking=request.thinking, 
            stream=True,
            conversation_id=conversation_id,
            use_langgraph=request.use_langgraph
        ):
            yield chunk
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
            
            # Convert to JSON string with newline, same as standard chat
            yield json.dumps(event, ensure_ascii=False) + "\n"
    
    return StreamingResponse(
        stream_events(),
        media_type="application/json"
    ) 