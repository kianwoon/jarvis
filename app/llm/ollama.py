import httpx
from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from typing import AsyncGenerator
import json
from app.core.llm_settings_cache import get_llm_settings

class OllamaLLM(BaseLLM):
    def __init__(self, config: LLMConfig, base_url: str = "http://localhost:11434"):
        super().__init__(config)
        self.base_url = base_url
        self.model_name = config.model_name

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Get context length from settings or use model-specific defaults
        llm_settings = get_llm_settings()
        context_length = llm_settings.get("context_length", 128000)  # Default to 128k for DeepSeek
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,  # Important: disable streaming for single response
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
                "num_ctx": context_length,  # Set context window size
            }
        }
        async with httpx.AsyncClient(timeout=30.0) as client:  # 30 second timeout
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            
            # With stream=False, Ollama returns a single JSON response
            try:
                data = response.json()
                return LLMResponse(
                    text=data.get("response", ""),
                    metadata={"model": self.model_name, "done": data.get("done", True)}
                )
            except json.JSONDecodeError:
                # Fallback: try parsing line by line if single JSON fails
                lines = response.text.strip().splitlines()
                full_text = ""
                for line in lines:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                full_text += data["response"]
                        except json.JSONDecodeError:
                            continue
                
                return LLMResponse(
                    text=full_text,
                    metadata={"model": self.model_name}
                )

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        # Get context length from settings or use model-specific defaults
        llm_settings = get_llm_settings()
        context_length = llm_settings.get("context_length", 128000)  # Default to 128k for DeepSeek
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
                "num_ctx": context_length,  # Set context window size
            },
            "stream": True
        }
        
        print(f"[DEBUG] Ollama payload - model: {self.model_name}, num_predict: {self.config.max_tokens}, num_ctx: {context_length}")
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data_json = json.loads(line)
                        if "response" in data_json:
                            yield LLMResponse(
                                text=data_json["response"],
                                metadata={"model": self.model_name, "streaming": True}
                            )
                    except Exception:
                        continue

    async def embed(self, text: str):
        raise NotImplementedError("Embedding is not supported for OllamaLLM.")

class JarvisLLM:
    def __init__(self, mode=None, max_tokens=None, base_url: str = "http://localhost:11434"):
        # Default to 'non-thinking' if mode is not specified
        self.mode = mode if mode in ("thinking", "non-thinking") else "non-thinking"
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.llm = self._build_llm(self.mode, self.max_tokens)

    def _build_llm(self, mode, max_tokens):
        settings = get_llm_settings()
        if "model" not in settings or "max_tokens" not in settings:
            raise RuntimeError("Missing required LLM config fields: model, max_tokens")
        mode_settings = settings["thinking_mode"] if mode == "thinking" else settings["non_thinking_mode"]
        for param in ["temperature", "top_p"]:
            if param not in mode_settings:
                raise RuntimeError(f"Missing '{param}' in {'thinking_mode' if mode == 'thinking' else 'non_thinking_mode'}")
        # Handle both string and int values for max_tokens
        if max_tokens is not None:
            max_tokens_value = int(max_tokens)
        else:
            max_tokens_raw = settings["max_tokens"]
            try:
                max_tokens_value = int(max_tokens_raw)
            except (ValueError, TypeError):
                print(f"[WARNING] Invalid max_tokens in settings: {max_tokens_raw}, using 16384")
                max_tokens_value = 16384
                
        config = LLMConfig(
            model_name=settings["model"],
            temperature=float(mode_settings["temperature"]),
            top_p=float(mode_settings["top_p"]),
            max_tokens=max_tokens_value
        )
        return OllamaLLM(config, base_url=self.base_url)

    def set_mode(self, mode=None, max_tokens=None):
        self.mode = mode if mode in ("thinking", "non-thinking") else "non-thinking"
        self.max_tokens = max_tokens
        self.llm = self._build_llm(self.mode, self.max_tokens)

    async def invoke(self, prompt: str) -> str:
        response = await self.llm.generate(prompt)
        return response.text 