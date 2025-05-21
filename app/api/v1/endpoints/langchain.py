from fastapi import APIRouter
from pydantic import BaseModel
from app.langchain.service import rag_answer

router = APIRouter()

class RAGRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    answer: str

@router.post("/rag", response_model=RAGResponse)
def rag_endpoint(request: RAGRequest):
    answer = rag_answer(request.question)
    return RAGResponse(answer=answer["answer"]) 