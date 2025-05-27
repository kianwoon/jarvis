from fastapi import APIRouter
from pydantic import BaseModel
from app.langchain.service import rag_answer
from fastapi.responses import StreamingResponse
import json
from typing import Optional

router = APIRouter()

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False
    conversation_id: Optional[str] = None
    use_langgraph: bool = True

class RAGResponse(BaseModel):
    answer: str
    source: str
    conversation_id: Optional[str] = None

@router.post("/rag")
def rag_endpoint(request: RAGRequest):
    def stream():
        for chunk in rag_answer(
            request.question, 
            thinking=request.thinking, 
            stream=True,
            conversation_id=request.conversation_id,
            use_langgraph=request.use_langgraph
        ):
            yield chunk
    return StreamingResponse(stream(), media_type="application/json") 