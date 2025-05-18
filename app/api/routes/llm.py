from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import asyncio
from fastapi.responses import StreamingResponse

router = APIRouter()

# Load model ONCE at startup
config = LLMConfig(
    model_name="qwen:7b",  # Ollama model name
    temperature=0.7,
    top_p=1.0,
    max_tokens=100
)
inference = OllamaLLM(config)

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
        # Update config params for this request
        inference.config.temperature = request.temperature
        inference.config.top_p = request.top_p
        inference.config.max_tokens = request.max_tokens
        response = await inference.generate(request.prompt)
        return GenerateResponse(text=response.text, metadata=response.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    async def event_stream():
        async for chunk in inference.generate_stream(request.prompt):
            yield f"data: {chunk.text}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream") 