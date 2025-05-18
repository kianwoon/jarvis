from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncGenerator
from pydantic import BaseModel

class LLMResponse(BaseModel):
    """Standardized LLM response format."""
    text: str
    metadata: Dict[str, Any] = {}

class LLMConfig(BaseModel):
    """Base configuration for LLM models."""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[LLMResponse, None]:
        """Generate streaming text from the LLM."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for the given text."""
        pass 