from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.core.llm_settings_cache import get_llm_settings
from app.core.langfuse_integration import trace_llm_call, get_tracer
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
    model_name = inference.model_name
    
    # Try to get model details from Ollama
    display_name = model_name  # Default to model name
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{ollama_base_url}/api/show",
                json={"name": model_name}
            )
            if resp.status_code == 200:
                data = resp.json()
                # Extract display name from model details if available
                if "details" in data and "family" in data["details"]:
                    family = data["details"]["family"]
                    # Format the display name based on model info
                    if "parameter_size" in data["details"]:
                        param_size = data["details"]["parameter_size"]
                        display_name = f"{family.title()} {param_size}"
                    else:
                        display_name = family.title()
                elif "modelfile" in data:
                    # Try to extract from modelfile
                    display_name = model_name.replace(":", " ").replace("-", " ").title()
                
                # Special handling for deepseek models
                if "deepseek" in model_name.lower():
                    # Parse deepseek-r1:8b format
                    parts = model_name.split(":")
                    if len(parts) == 2:
                        model_type = parts[0].replace("-", " ").title()
                        size = parts[1].upper()
                        display_name = f"{model_type} {size}"
                    else:
                        display_name = model_name.replace("-", " ").title()
    except Exception as e:
        print(f"Failed to get model details from Ollama: {e}")
        # Fallback parsing for common patterns
        if ":" in model_name:
            parts = model_name.split(":")
            if len(parts) == 2:
                base = parts[0].replace("-", " ").title()
                size = parts[1].upper()
                display_name = f"{base} {size}"
    
    return {
        "model_name": model_name,
        "display_name": display_name
    }

@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    tracer = get_tracer()
    trace = None
    generation = None
    
    try:
        inference = get_inference()
        # Update config params for this request
        inference.config.temperature = request.temperature
        inference.config.top_p = request.top_p
        inference.config.max_tokens = request.max_tokens
        
        # Create Langfuse trace and generation
        # Initialize tracer if not already done
        if not tracer._initialized:
            tracer.initialize()
            
        if tracer.is_enabled():
            trace = tracer.create_trace(
                name="llm-generate",
                input=request.prompt,
                metadata={
                    "endpoint": "/api/v1/generate",
                    "model": inference.config.model_name
                }
            )
            
            if trace:
                generation = tracer.create_generation(
                    trace=trace,
                    name="ollama-generation",
                    model=inference.config.model_name,
                    input=request.prompt,
                    metadata={
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "max_tokens": request.max_tokens
                    }
                )
        
        response = await inference.generate(request.prompt)
        text = response.text or ""
        reasoning = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)
        answer = text.strip()
        
        result = {
            "text": text,
            "reasoning": reasoning[0] if reasoning else None,
            "answer": answer,
            "metadata": response.metadata
        }
        
        # Update generation with results
        if generation:
            generation.end(
                output=text,
                metadata={
                    "response_length": len(text),
                    "has_reasoning": bool(reasoning),
                    "model_metadata": response.metadata,
                    "input_length": len(request.prompt),
                    "output_length": len(text)
                }
            )
        
        # Update trace with final result
        if trace:
            trace.update(
                output=result,
                metadata={
                    "success": True,
                    "response_type": "reasoning" if reasoning else "direct"
                }
            )
        
        return result
        
    except Exception as e:
        # Log error in generation and trace
        if generation:
            generation.end(
                output=f"Error: {str(e)}",
                metadata={"error": str(e), "level": "ERROR"}
            )
        
        if trace:
            trace.update(
                output=f"Error: {str(e)}",
                metadata={"error": str(e), "success": False}
            )
        
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Flush traces
        if tracer.is_enabled():
            tracer.flush()

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