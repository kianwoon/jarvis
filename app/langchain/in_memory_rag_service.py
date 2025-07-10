"""
In-memory RAG service for fast temporary document processing.
Uses FAISS for efficient vector operations without persistent storage.
"""

import asyncio
import uuid
import numpy as np
import logging
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core.in_memory_rag_settings import (
    get_in_memory_rag_settings, 
    get_vector_store_config,
    get_embedding_config,
    get_chunking_config,
    VectorStoreType,
    EmbeddingModelType
)

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    chunk_id: str = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            self.chunk_id = str(uuid.uuid4())

@dataclass
class QueryResult:
    """Result from querying the in-memory RAG service."""
    chunk: DocumentChunk
    score: float
    rank: int

@dataclass
class RAGResponse:
    """Complete response from RAG query."""
    query: str
    results: List[QueryResult]
    total_chunks: int
    processing_time_ms: float
    source_info: Dict[str, Any]

class EmbeddingServiceInterface(ABC):
    """Abstract interface for embedding services."""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        pass

class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""
    
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors with metadata to the store."""
        pass
    
    @abstractmethod
    async def search(self, query_vector: List[float], top_k: int, threshold: float) -> List[Tuple[int, float]]:
        """Search for similar vectors and return indices with scores."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all vectors from the store."""
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """Get the number of vectors in the store."""
        pass

class FAISSVectorStore(VectorStoreInterface):
    """FAISS-based vector store for fast similarity search."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.metadata_store: List[Dict[str, Any]] = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        try:
            import faiss
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"Initialized FAISS index with dimension {self.dimension}")
        except ImportError:
            logger.error("FAISS not available, falling back to simple vector store")
            raise ImportError("FAISS is required for FAISSVectorStore")
    
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to FAISS index."""
        try:
            if not vectors:
                return True
            
            # Convert to numpy array and normalize for cosine similarity
            vector_array = np.array(vectors, dtype=np.float32)
            
            # Normalize vectors for cosine similarity
            norms = np.linalg.norm(vector_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            vector_array = vector_array / norms
            
            # Add to FAISS index
            self.index.add(vector_array)
            
            # Store metadata
            self.metadata_store.extend(metadata)
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS index: {e}")
            return False
    
    async def search(self, query_vector: List[float], top_k: int, threshold: float) -> List[Tuple[int, float]]:
        """Search FAISS index for similar vectors."""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query vector
            query_array = np.array([query_vector], dtype=np.float32)
            norm = np.linalg.norm(query_array)
            if norm > 0:
                query_array = query_array / norm
            
            # Search FAISS index
            scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            # Filter by threshold and return results
            results = []
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if score >= threshold and idx != -1:  # -1 indicates no match found
                    results.append((int(idx), float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            return []
    
    async def clear(self) -> None:
        """Clear FAISS index."""
        try:
            self.index.reset()
            self.metadata_store.clear()
            logger.info("Cleared FAISS index")
        except Exception as e:
            logger.error(f"Failed to clear FAISS index: {e}")
    
    def get_vector_count(self) -> int:
        """Get number of vectors in FAISS index."""
        return self.index.ntotal if self.index else 0

class SimpleVectorStore(VectorStoreInterface):
    """Simple in-memory vector store using numpy for basic similarity search."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: List[List[float]] = []
        self.metadata_store: List[Dict[str, Any]] = []
    
    async def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to simple store."""
        try:
            self.vectors.extend(vectors)
            self.metadata_store.extend(metadata)
            logger.info(f"Added {len(vectors)} vectors to simple vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to add vectors to simple store: {e}")
            return False
    
    async def search(self, query_vector: List[float], top_k: int, threshold: float) -> List[Tuple[int, float]]:
        """Search using cosine similarity."""
        try:
            if not self.vectors:
                return []
            
            # Convert to numpy arrays
            query_array = np.array(query_vector)
            vector_array = np.array(self.vectors)
            
            # Normalize for cosine similarity
            query_norm = np.linalg.norm(query_array)
            vector_norms = np.linalg.norm(vector_array, axis=1)
            
            if query_norm == 0:
                return []
            
            # Calculate cosine similarities
            similarities = np.dot(vector_array, query_array) / (vector_norms * query_norm)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by threshold
            results = []
            for i, idx in enumerate(top_indices):
                score = similarities[idx]
                if score >= threshold:
                    results.append((int(idx), float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search simple vector store: {e}")
            return []
    
    async def clear(self) -> None:
        """Clear simple vector store."""
        self.vectors.clear()
        self.metadata_store.clear()
        logger.info("Cleared simple vector store")
    
    def get_vector_count(self) -> int:
        """Get number of vectors."""
        return len(self.vectors)

class HuggingFaceEmbeddingService(EmbeddingServiceInterface):
    """HuggingFace-based embedding service."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize HuggingFace model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Initialized HuggingFace model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not available")
            raise ImportError("sentence-transformers is required for HuggingFaceEmbeddingService")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query."""
        try:
            embedding = self.model.encode([query], normalize_embeddings=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

class MockEmbeddingService(EmbeddingServiceInterface):
    """Mock embedding service for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.info(f"Initialized mock embedding service with dimension {dimension}")
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings."""
        import hashlib
        embeddings = []
        for text in texts:
            # Create deterministic but varied embeddings based on text hash
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Convert hash to embedding
            embedding = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                value = int.from_bytes(chunk + b'\x00' * (4 - len(chunk)), 'big')
                embedding.append(float(value) / (2**32))
            
            # Pad or trim to desired dimension
            while len(embedding) < self.dimension:
                embedding.extend(embedding[:min(len(embedding), self.dimension - len(embedding))])
            embedding = embedding[:self.dimension]
            
            # Normalize
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate mock query embedding."""
        embeddings = await self.embed_texts([query])
        return embeddings[0] if embeddings else []

class QwenHTTPEmbeddingService(EmbeddingServiceInterface):
    """HTTP-based embedding service using Qwen embedder."""
    
    def __init__(self, endpoint: str = "http://qwen-embedder:8050/embed"):
        self.endpoint = endpoint
        self.dimension = 512  # Qwen embedding dimension
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Qwen HTTP service."""
        try:
            embeddings = []
            for text in texts:
                # Normalize text for consistent embeddings
                normalized_text = text.lower().strip()
                payload = {"texts": [normalized_text]}
                
                response = requests.post(self.endpoint, json=payload, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                embeddings.append(result["embeddings"][0])
            
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings via Qwen HTTP: {e}")
            # Fallback to mock embeddings if HTTP fails
            mock_service = MockEmbeddingService(self.dimension)
            return await mock_service.embed_texts(texts)
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate query embedding using Qwen HTTP service."""
        embeddings = await self.embed_texts([query])
        return embeddings[0] if embeddings else []

class InMemoryRAGService:
    """High-performance in-memory RAG service for temporary documents."""
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.config = get_in_memory_rag_settings()
        self.chunks: List[DocumentChunk] = []
        self.vector_store: VectorStoreInterface = None
        self.embedding_service: EmbeddingServiceInterface = None
        self.created_at = datetime.now()
        self.last_access = datetime.now()
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize vector store and embedding service."""
        # Initialize vector store
        vector_config = get_vector_store_config()
        
        if self.config.vector_store_type == VectorStoreType.FAISS:
            try:
                self.vector_store = FAISSVectorStore(self.config.embedding_dimension)
            except ImportError:
                logger.warning("FAISS not available, falling back to simple vector store")
                self.vector_store = SimpleVectorStore(self.config.embedding_dimension)
        else:
            self.vector_store = SimpleVectorStore(self.config.embedding_dimension)
        
        # Initialize embedding service
        embedding_config = get_embedding_config()
        
        if self.config.embedding_model_type == EmbeddingModelType.HUGGINGFACE:
            try:
                self.embedding_service = HuggingFaceEmbeddingService(
                    self.config.embedding_model_name,
                    device=embedding_config.get('device', 'cpu')
                )
            except ImportError:
                logger.warning("HuggingFace not available, using Qwen HTTP embeddings")
                self.embedding_service = QwenHTTPEmbeddingService()
        elif self.config.embedding_model_type == EmbeddingModelType.QWEN_ENDPOINT:
            self.embedding_service = QwenHTTPEmbeddingService()
        else:
            # Use Qwen HTTP service as primary, fallback to mock
            logger.info("Using Qwen HTTP embedding service")
            self.embedding_service = QwenHTTPEmbeddingService()
        
        logger.info(f"Initialized in-memory RAG service for conversation {self.conversation_id}")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the in-memory RAG service."""
        try:
            start_time = datetime.now()
            
            # Convert documents to chunks
            new_chunks = []
            texts_to_embed = []
            
            for doc in documents:
                # Split document into chunks if needed
                content = doc.get('content', '')
                if len(content) > self.config.chunk_size:
                    chunks = self._chunk_text(content, doc.get('metadata', {}))
                else:
                    chunks = [DocumentChunk(
                        content=content,
                        metadata=doc.get('metadata', {})
                    )]
                
                new_chunks.extend(chunks)
                texts_to_embed.extend([chunk.content for chunk in chunks])
            
            # Generate embeddings
            embeddings = await self.embedding_service.embed_texts(texts_to_embed)
            
            if len(embeddings) != len(new_chunks):
                logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(new_chunks)}")
                return False
            
            # Add embeddings to chunks
            for chunk, embedding in zip(new_chunks, embeddings):
                chunk.embedding = embedding
            
            # Add to vector store
            metadata_list = [chunk.metadata for chunk in new_chunks]
            success = await self.vector_store.add_vectors(embeddings, metadata_list)
            
            if success:
                self.chunks.extend(new_chunks)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(f"Added {len(new_chunks)} chunks in {processing_time:.2f}ms")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    async def query(self, query: str, top_k: Optional[int] = None) -> RAGResponse:
        """Query the in-memory RAG service."""
        try:
            start_time = datetime.now()
            self.last_access = datetime.now()
            
            if top_k is None:
                top_k = self.config.max_results_per_query
            
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_query(query)
            if not query_embedding:
                return RAGResponse(
                    query=query,
                    results=[],
                    total_chunks=len(self.chunks),
                    processing_time_ms=0,
                    source_info={'error': 'Failed to generate query embedding'}
                )
            
            # Search vector store
            search_results = await self.vector_store.search(
                query_embedding, 
                top_k, 
                self.config.similarity_threshold
            )
            
            # Convert to QueryResults
            results = []
            for rank, (chunk_idx, score) in enumerate(search_results):
                if chunk_idx < len(self.chunks):
                    results.append(QueryResult(
                        chunk=self.chunks[chunk_idx],
                        score=score,
                        rank=rank + 1
                    ))
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return RAGResponse(
                query=query,
                results=results,
                total_chunks=len(self.chunks),
                processing_time_ms=processing_time,
                source_info={
                    'conversation_id': self.conversation_id,
                    'vector_store_type': self.config.vector_store_type.value,
                    'embedding_model': self.config.embedding_model_name,
                    'similarity_threshold': self.config.similarity_threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to query in-memory RAG: {e}")
            return RAGResponse(
                query=query,
                results=[],
                total_chunks=len(self.chunks),
                processing_time_ms=0,
                source_info={'error': str(e)}
            )
    
    def _chunk_text(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into chunks."""
        try:
            chunks = []
            chunk_config = get_chunking_config()
            
            # Simple text splitter implementation
            chunk_size = chunk_config['chunk_size']
            chunk_overlap = chunk_config['chunk_overlap']
            
            start = 0
            chunk_index = 0
            
            while start < len(text):
                # Define chunk end
                end = start + chunk_size
                if end > len(text):
                    end = len(text)
                
                # Try to end at sentence boundary
                if end < len(text):
                    for sep in chunk_config['separators']:
                        sep_pos = text.rfind(sep, start, end)
                        if sep_pos != -1:
                            end = sep_pos + len(sep)
                            break
                
                # Create chunk
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        'chunk_index': chunk_index,
                        'start_char': start,
                        'end_char': end,
                        'chunk_length': len(chunk_text)
                    })
                    
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        metadata=chunk_metadata
                    ))
                    chunk_index += 1
                
                # Move to next chunk with overlap
                start = max(start + 1, end - chunk_overlap)
                if start >= end:
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            return []
    
    async def clear(self) -> None:
        """Clear all documents from the service."""
        await self.vector_store.clear()
        self.chunks.clear()
        logger.info(f"Cleared in-memory RAG service for conversation {self.conversation_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            'conversation_id': self.conversation_id,
            'document_count': len(self.chunks),
            'vector_count': self.vector_store.get_vector_count(),
            'created_at': self.created_at.isoformat(),
            'last_access': self.last_access.isoformat(),
            'config': {
                'vector_store_type': self.config.vector_store_type.value,
                'embedding_model': self.config.embedding_model_name,
                'similarity_threshold': self.config.similarity_threshold,
                'max_results': self.config.max_results_per_query
            }
        }
    
    def is_expired(self) -> bool:
        """Check if the service has expired based on TTL."""
        if not hasattr(self, 'ttl_hours'):
            self.ttl_hours = self.config.default_ttl_hours
        
        expiry_time = self.created_at + timedelta(hours=self.ttl_hours)
        return datetime.now() > expiry_time

# Global registry for conversation-based RAG services
_rag_services: Dict[str, InMemoryRAGService] = {}

async def get_in_memory_rag_service(conversation_id: str) -> InMemoryRAGService:
    """Get or create an in-memory RAG service for a conversation."""
    global _rag_services
    
    if conversation_id not in _rag_services:
        _rag_services[conversation_id] = InMemoryRAGService(conversation_id)
        logger.info(f"Created new in-memory RAG service for conversation {conversation_id}")
    
    return _rag_services[conversation_id]

async def cleanup_expired_services() -> int:
    """Clean up expired RAG services."""
    global _rag_services
    
    expired_services = []
    for conv_id, service in _rag_services.items():
        if service.is_expired():
            expired_services.append(conv_id)
    
    for conv_id in expired_services:
        await _rag_services[conv_id].clear()
        del _rag_services[conv_id]
        logger.info(f"Cleaned up expired RAG service for conversation {conv_id}")
    
    return len(expired_services)