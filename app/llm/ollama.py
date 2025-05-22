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
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            }
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            # Patch: Only parse the first JSON object in the response
            lines = response.text.strip().splitlines()
            data = json.loads(lines[0])
            return LLMResponse(
                text=data["response"],
                metadata={"model": self.model_name}
            )

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
            "stream": True
        }
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
        config = LLMConfig(
            model_name=settings["model"],
            temperature=float(mode_settings["temperature"]),
            top_p=float(mode_settings["top_p"]),
            max_tokens=int(max_tokens) if max_tokens is not None else int(settings["max_tokens"])
        )
        return OllamaLLM(config, base_url=self.base_url)

    def set_mode(self, mode=None, max_tokens=None):
        self.mode = mode if mode in ("thinking", "non-thinking") else "non-thinking"
        self.max_tokens = max_tokens
        self.llm = self._build_llm(self.mode, self.max_tokens)

    async def invoke(self, prompt: str) -> str:
        response = await self.llm.generate(prompt)
        return response.text 