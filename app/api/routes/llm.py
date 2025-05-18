from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.llm import QwenInference, LLMConfig
import asyncio

router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for the LLM")
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 100

class GenerateResponse(BaseModel):
    text: str
    metadata: dict

@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        config = LLMConfig(
            model_name="Qwen/Qwen-7B-Chat",  # Optionally, get from env/config
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        inference = QwenInference(config)
        response = await inference.generate(request.prompt)
        return GenerateResponse(text=response.text, metadata=response.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 