from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class Document(BaseModel):
    """Document model for RAG system."""
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    """Search result model."""
    document: Document
    score: float
    metadata: Dict[str, Any] = {}

class BaseRetriever(ABC):
    """Abstract base class for document retrievers."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """Search for relevant documents."""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the retriever."""
        pass

class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""
    
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        pass 