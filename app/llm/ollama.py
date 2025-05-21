import httpx
from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from typing import AsyncGenerator
import json

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