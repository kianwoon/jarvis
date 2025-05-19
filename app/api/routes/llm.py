from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import asyncio
from fastapi.responses import StreamingResponse
import os

router = APIRouter()

# Load model ONCE at startup
config = LLMConfig(
    model_name="qwen3:30b-a3b",  # Ollama model name for Qwen3-30B-A3B (MoE)
    temperature=0.7,
    top_p=1.0,
    max_tokens=2048
)
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
print("OLLAMA_BASE_URL at runtime:", ollama_base_url)
inference = OllamaLLM(config, base_url=ollama_base_url)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for the LLM")
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 100

class GenerateResponse(BaseModel):
    text: str
    metadata: dict

@router.get("/current_model")
async def get_current_model():
    """Return the current model being used by the LLM API."""
    return {
        "model_name": inference.model_name,
        "display_name": "Qwen3-30B-A3B (MoE)"
    }

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
        try:
            # Update config params for this request
            inference.config.temperature = request.temperature
            inference.config.top_p = request.top_p
            inference.config.max_tokens = request.max_tokens
            
            async for chunk in inference.generate_stream(request.prompt):
                if chunk and chunk.text:
                    yield f"data: {chunk.text}\n\n"
        except Exception as e:
            print(f"Streaming error: {str(e)}")  # Log the error
            yield f"data: [ERROR] {str(e)}\n\n"
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    ) 