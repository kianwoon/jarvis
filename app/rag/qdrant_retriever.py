from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.rag.base import BaseRetriever, Document, SearchResult
from app.core.config import get_settings

class QdrantRetriever(BaseRetriever):
    """Qdrant-based document retriever."""
    
    def __init__(self, collection_name: str):
        self.settings = get_settings()
        self.collection_name = collection_name
        self.client = QdrantClient(
            host=self.settings.QDRANT_HOST,
            port=self.settings.QDRANT_PORT
        )
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists with proper configuration."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Default embedding size
                    distance=models.Distance.COSINE
                )
            )
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Qdrant."""
        points = []
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                raise ValueError(f"Document {i} has no embedding")
            
            points.append(models.PointStruct(
                id=i,
                vector=doc.embedding,
                payload={
                    "content": doc.content,
                    "metadata": doc.metadata
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """Search for relevant documents in Qdrant."""
        # Note: query embedding should be generated before calling this method
        query_embedding = kwargs.get("query_embedding")
        if query_embedding is None:
            raise ValueError("query_embedding is required")
        
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return [
            SearchResult(
                document=Document(
                    content=result.payload["content"],
                    metadata=result.payload["metadata"]
                ),
                score=result.score,
                metadata={"id": result.id}
            )
            for result in search_results
        ]
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from Qdrant."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=document_ids
            )
        ) 