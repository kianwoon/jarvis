from typing import Any, Dict, List, Optional, AsyncGenerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.llm.base import BaseLLM, LLMConfig, LLMResponse
from app.core.config import get_settings

class QwenLLM(BaseLLM):
    """Qwen LLM implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.settings = get_settings()
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            trust_remote_code=True
        )
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from Qwen."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            **kwargs
        )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return LLMResponse(
            text=response_text,
            metadata={"model": self.config.model_name}
        )
    
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Generate streaming text from Qwen."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            streamer=streamer,
            **kwargs
        )
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the output
        for text in streamer:
            yield LLMResponse(
                text=text,
                metadata={"model": self.config.model_name}
            )
    
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings using Qwen's embedding model."""
        # TODO: Implement embedding generation
        # This will require a separate embedding model
        raise NotImplementedError("Embedding generation not implemented yet") 