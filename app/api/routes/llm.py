from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.core.llm_settings_cache import get_llm_settings
import asyncio
from fastapi.responses import StreamingResponse
import os
import httpx
import re

router = APIRouter()

ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")

# Helper to create a new inference object with the latest settings
def get_inference(thinking: bool = False):
    settings = get_llm_settings()
    required_fields = ["model", "max_tokens"]
    missing = [f for f in required_fields if f not in settings or settings[f] is None]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing required LLM config fields: {', '.join(missing)}")
    mode = settings["thinking_mode"] if thinking else settings["non_thinking_mode"]
    for param in ["temperature", "top_p", "top_k", "min_p"]:
        if param not in mode:
            raise HTTPException(status_code=500, detail=f"Missing '{param}' in {'thinking_mode' if thinking else 'non_thinking_mode'}")
    config = LLMConfig(
        model_name=settings["model"],
        temperature=float(mode["temperature"]),
        top_p=float(mode["top_p"]),
        max_tokens=int(settings["max_tokens"])
    )
    return OllamaLLM(config, base_url=ollama_base_url)

@router.get("/ollama/models")
async def list_ollama_models():
    """List available Ollama models from the Ollama server."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ollama_base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            # The models are in data['models'], each with a 'name' field
            models = [m['name'] for m in data.get('models', [])]
            return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch models from Ollama: {str(e)}")

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for the LLM")
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096

class GenerateResponse(BaseModel):
    text: str
    reasoning: str | None
    answer: str
    metadata: dict

@router.get("/current_model")
async def get_current_model():
    inference = get_inference()
    return {
        "model_name": inference.model_name,
        "display_name": "Qwen3-30B-A3B (MoE)"
    }

@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        inference = get_inference()
        # Update config params for this request
        inference.config.temperature = request.temperature
        inference.config.top_p = request.top_p
        inference.config.max_tokens = request.max_tokens
        response = await inference.generate(request.prompt)
        text = response.text or ""
        reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return {
            "text": text,
            "reasoning": reasoning[0] if reasoning else None,
            "answer": answer,
            "metadata": response.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_stream")
async def generate_stream(request: GenerateRequest):
    inference = get_inference()
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