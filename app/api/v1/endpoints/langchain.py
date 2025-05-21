from fastapi import APIRouter
from pydantic import BaseModel
from app.langchain.service import rag_answer
from fastapi.responses import StreamingResponse
import json

router = APIRouter()

class RAGRequest(BaseModel):
    question: str
    thinking: bool = False

class RAGResponse(BaseModel):
    answer: str
    source: str

@router.post("/rag")
def rag_endpoint(request: RAGRequest):
    def stream():
        for chunk in rag_answer(request.question, thinking=request.thinking, stream=True):
            yield chunk
    return StreamingResponse(stream(), media_type="application/json") 