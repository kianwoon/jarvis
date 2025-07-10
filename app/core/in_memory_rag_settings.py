"""
Configuration settings for in-memory RAG operations.
Provides centralized configuration for temporary document processing and vector operations.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VectorStoreType(Enum):
    """Supported vector store types for in-memory operations."""
    FAISS = "faiss"
    CHROMADB = "chromadb"
    SIMPLE = "simple"

class EmbeddingModelType(Enum):
    """Supported embedding model types."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    QWEN_ENDPOINT = "qwen_endpoint"
    MOCK = "mock"

@dataclass
class InMemoryRAGConfig:
    """Configuration for in-memory RAG operations."""
    
    # Vector Store Configuration
    vector_store_type: VectorStoreType = VectorStoreType.FAISS
    similarity_threshold: float = 0.5
    max_results_per_query: int = 5
    
    # Embedding Configuration
    embedding_model_type: EmbeddingModelType = EmbeddingModelType.HUGGINGFACE
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents_per_conversation: int = 10
    
    # Performance Settings
    enable_parallel_processing: bool = True
    batch_size: int = 10
    cache_embeddings: bool = True
    
    # Hybrid RAG Settings
    temp_doc_priority_weight: float = 0.8
    persistent_rag_weight: float = 0.2
    min_temp_doc_score: float = 0.3
    fallback_to_persistent: bool = True
    
    # Session Management
    default_ttl_hours: int = 2
    auto_cleanup_interval_minutes: int = 30
    max_session_memory_mb: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if self.temp_doc_priority_weight + self.persistent_rag_weight != 1.0:
            logger.warning(
                f"Priority weights don't sum to 1.0: temp={self.temp_doc_priority_weight}, "
                f"persistent={self.persistent_rag_weight}"
            )
        
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        
        if self.max_documents_per_conversation <= 0:
            raise ValueError("max_documents_per_conversation must be positive")

class InMemoryRAGSettings:
    """Singleton class for managing in-memory RAG settings."""
    
    _instance: Optional['InMemoryRAGSettings'] = None
    _config: Optional[InMemoryRAGConfig] = None
    
    def __new__(cls) -> 'InMemoryRAGSettings':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = self._load_config()
    
    def _load_config(self) -> InMemoryRAGConfig:
        """Load configuration from environment variables and defaults."""
        try:
            # Vector Store Configuration
            vector_store_type = VectorStoreType(
                os.getenv("IN_MEMORY_VECTOR_STORE", VectorStoreType.FAISS.value)
            )
            similarity_threshold = float(
                os.getenv("IN_MEMORY_SIMILARITY_THRESHOLD", "0.5")
            )
            max_results = int(
                os.getenv("IN_MEMORY_MAX_RESULTS", "5")
            )
            
            # Embedding Configuration
            embedding_type = EmbeddingModelType(
                os.getenv("IN_MEMORY_EMBEDDING_TYPE", EmbeddingModelType.QWEN_ENDPOINT.value)
            )
            embedding_model = os.getenv(
                "IN_MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            )
            embedding_dim = int(
                os.getenv("IN_MEMORY_EMBEDDING_DIMENSION", "512")
            )
            
            # Document Processing
            chunk_size = int(os.getenv("IN_MEMORY_CHUNK_SIZE", "1000"))
            chunk_overlap = int(os.getenv("IN_MEMORY_CHUNK_OVERLAP", "200"))
            max_docs = int(os.getenv("IN_MEMORY_MAX_DOCS_PER_CONV", "10"))
            
            # Performance Settings
            enable_parallel = os.getenv("IN_MEMORY_PARALLEL_PROCESSING", "true").lower() == "true"
            batch_size = int(os.getenv("IN_MEMORY_BATCH_SIZE", "10"))
            cache_embeddings = os.getenv("IN_MEMORY_CACHE_EMBEDDINGS", "true").lower() == "true"
            
            # Hybrid RAG Settings
            temp_weight = float(os.getenv("IN_MEMORY_TEMP_PRIORITY_WEIGHT", "0.8"))
            persistent_weight = float(os.getenv("IN_MEMORY_PERSISTENT_WEIGHT", "0.2"))
            min_temp_score = float(os.getenv("IN_MEMORY_MIN_TEMP_SCORE", "0.3"))
            fallback = os.getenv("IN_MEMORY_FALLBACK_TO_PERSISTENT", "true").lower() == "true"
            
            # Session Management
            ttl_hours = int(os.getenv("IN_MEMORY_DEFAULT_TTL_HOURS", "2"))
            cleanup_interval = int(os.getenv("IN_MEMORY_CLEANUP_INTERVAL_MINUTES", "30"))
            max_memory = int(os.getenv("IN_MEMORY_MAX_SESSION_MEMORY_MB", "100"))
            
            config = InMemoryRAGConfig(
                vector_store_type=vector_store_type,
                similarity_threshold=similarity_threshold,
                max_results_per_query=max_results,
                embedding_model_type=embedding_type,
                embedding_model_name=embedding_model,
                embedding_dimension=embedding_dim,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_documents_per_conversation=max_docs,
                enable_parallel_processing=enable_parallel,
                batch_size=batch_size,
                cache_embeddings=cache_embeddings,
                temp_doc_priority_weight=temp_weight,
                persistent_rag_weight=persistent_weight,
                min_temp_doc_score=min_temp_score,
                fallback_to_persistent=fallback,
                default_ttl_hours=ttl_hours,
                auto_cleanup_interval_minutes=cleanup_interval,
                max_session_memory_mb=max_memory
            )
            
            logger.info(f"Loaded in-memory RAG configuration: {vector_store_type.value} vector store")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load in-memory RAG configuration: {e}")
            logger.info("Using default configuration")
            return InMemoryRAGConfig()
    
    def get_config(self) -> InMemoryRAGConfig:
        """Get the current configuration."""
        return self._config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values."""
        if self._config is None:
            self._config = InMemoryRAGConfig()
        
        # Create new config with updated values
        config_dict = self._config.__dict__.copy()
        config_dict.update(kwargs)
        
        self._config = InMemoryRAGConfig(**config_dict)
        logger.info("Updated in-memory RAG configuration")
    
    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = self._load_config()
        logger.info("Reset in-memory RAG configuration to defaults")

# Global instance
_settings_instance = None

def get_in_memory_rag_settings() -> InMemoryRAGConfig:
    """Get the current in-memory RAG configuration."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = InMemoryRAGSettings()
    return _settings_instance.get_config()

def update_in_memory_rag_settings(**kwargs) -> None:
    """Update in-memory RAG configuration."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = InMemoryRAGSettings()
    _settings_instance.update_config(**kwargs)

def get_vector_store_config() -> Dict[str, Any]:
    """Get vector store specific configuration."""
    config = get_in_memory_rag_settings()
    
    base_config = {
        'similarity_threshold': config.similarity_threshold,
        'max_results': config.max_results_per_query,
        'embedding_dimension': config.embedding_dimension
    }
    
    if config.vector_store_type == VectorStoreType.FAISS:
        base_config.update({
            'index_type': 'IndexFlatIP',  # Inner product for cosine similarity
            'enable_gpu': False,  # Keep on CPU for simplicity
            'nprobe': 10
        })
    elif config.vector_store_type == VectorStoreType.CHROMADB:
        base_config.update({
            'collection_name_prefix': 'temp_rag_',
            'distance_function': 'cosine',
            'persist_directory': None  # In-memory only
        })
    
    return base_config

def get_embedding_config() -> Dict[str, Any]:
    """Get embedding model specific configuration."""
    config = get_in_memory_rag_settings()
    
    base_config = {
        'model_type': config.embedding_model_type.value,
        'model_name': config.embedding_model_name,
        'dimension': config.embedding_dimension,
        'batch_size': config.batch_size,
        'cache_embeddings': config.cache_embeddings
    }
    
    if config.embedding_model_type == EmbeddingModelType.HUGGINGFACE:
        base_config.update({
            'device': 'cpu',  # Force CPU to avoid CUDA issues
            'normalize_embeddings': True
        })
    elif config.embedding_model_type == EmbeddingModelType.OPENAI:
        base_config.update({
            'api_key': os.getenv('OPENAI_API_KEY'),
            'model': 'text-embedding-ada-002'
        })
    elif config.embedding_model_type == EmbeddingModelType.QWEN_ENDPOINT:
        base_config.update({
            'endpoint_url': os.getenv('QWEN_EMBEDDER_URL', 'http://qwen-embedder:8050/embed')
        })
    
    return base_config

def get_chunking_config() -> Dict[str, Any]:
    """Get document chunking configuration."""
    config = get_in_memory_rag_settings()
    
    return {
        'chunk_size': config.chunk_size,
        'chunk_overlap': config.chunk_overlap,
        'separators': ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        'length_function': len,
        'is_separator_regex': False
    }