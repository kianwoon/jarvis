"""
Standalone RAG Fallback Implementation
=====================================

Simple RAG implementation that can be used directly by MCP tools without
circular dependencies. Uses the existing vector database and embedding
infrastructure but avoids the complex langchain.service dependencies.
"""

import logging
import time
import json as json_module
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def simple_rag_search(
    query: str,
    collections: Optional[List[str]] = None,
    max_documents: int = 5,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Simple standalone RAG search implementation
    
    This is a lightweight fallback that can be used when the main RAG service
    has circular import issues. It directly uses the vector database without
    going through the langchain service layer.
    """
    start_time = time.time()
    
    try:
        logger.info(f"[RAG FALLBACK] Starting simple RAG search for query: {query[:100]}...")
        
        # Handle edge cases
        if not query or not isinstance(query, str):
            return {
                "success": False,
                "error": "Invalid query provided",
                "query": str(query),
                "collections_searched": [],
                "total_documents_found": 0,
                "documents_returned": 0,
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "documents": [],
                "search_metadata": {"fallback": True}
            }
        
        # Get collections to search
        if not collections:
            collections = get_default_collections()
        
        logger.info(f"[RAG FALLBACK] Searching collections: {collections}")
        
        # Get vector database settings
        from app.core.vector_db_settings_cache import get_vector_db_settings
        vector_settings = get_vector_db_settings()
        
        # Get embedding settings
        from app.core.embedding_settings_cache import get_embedding_settings
        embedding_settings = get_embedding_settings()
        
        # Initialize embeddings
        embedder = get_simple_embedder(embedding_settings)
        
        # Generate query embedding
        query_embedding = embedder.encode([query])[0]
        
        # Search each collection
        all_documents = []
        for collection_name in collections:
            try:
                docs = search_collection(
                    collection_name,
                    query,
                    query_embedding,
                    vector_settings,
                    max_documents
                )
                all_documents.extend(docs)
                logger.info(f"[RAG FALLBACK] Found {len(docs)} documents in {collection_name}")
            except Exception as e:
                logger.warning(f"[RAG FALLBACK] Error searching collection {collection_name}: {e}")
                continue
        
        # Sort by score and limit results
        all_documents = sorted(all_documents, key=lambda x: x.get('score', 0), reverse=True)
        limited_documents = all_documents[:max_documents]
        
        # Format documents
        formatted_documents = []
        for doc in limited_documents:
            formatted_doc = {
                'title': doc.get('title', 'Unknown'),
                'score': doc.get('score', 0.0),
                'metadata': doc.get('metadata', {})
            }
            
            if include_content:
                formatted_doc['content'] = doc.get('content', '')
            else:
                formatted_doc['content_preview'] = (doc.get('content', '') or '')[:200] + '...'
            
            formatted_documents.append(formatted_doc)
        
        execution_time = int((time.time() - start_time) * 1000)
        
        result = {
            "success": True,
            "query": query,
            "collections_searched": collections,
            "total_documents_found": len(all_documents),
            "documents_returned": len(formatted_documents),
            "execution_time_ms": execution_time,
            "documents": formatted_documents,
            "search_metadata": {
                "fallback": True,
                "search_strategy": "simple_vector_search",
                "embedding_model": embedding_settings.get('model_name', 'unknown'),
                "cache_hit": False
            }
        }
        
        logger.info(f"[RAG FALLBACK] Completed search in {execution_time}ms, found {len(formatted_documents)} documents")
        return result
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"[RAG FALLBACK] Search failed: {str(e)}")
        
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "collections_searched": collections or [],
            "total_documents_found": 0,
            "documents_returned": 0,
            "execution_time_ms": execution_time,
            "documents": [],
            "search_metadata": {"fallback": True, "error": True}
        }

def get_default_collections() -> List[str]:
    """Get default collections to search"""
    try:
        from app.core.collection_registry_cache import get_all_collections
        collections = get_all_collections()
        if collections and isinstance(collections, list):
            collection_names = [c.get('collection_name') for c in collections if c.get('collection_name')]
            if collection_names:
                return collection_names[:3]  # Limit to first 3 collections for fallback
        return ["default_knowledge"]
    except Exception as e:
        logger.warning(f"[RAG FALLBACK] Could not get collections: {e}")
        return ["default_knowledge"]

def get_simple_embedder(embedding_settings: Dict[str, Any]):
    """Get a simple embedder instance"""
    try:
        model_name = embedding_settings.get('model_name', 'all-MiniLM-L6-v2')
        
        # Try sentence transformers first (most common)
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except Exception:
            pass
        
        # Try HuggingFace embeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception:
            pass
        
        # Fallback to a basic implementation
        logger.warning("[RAG FALLBACK] Using mock embedder - no proper embedding available")
        return MockEmbedder()
        
    except Exception as e:
        logger.error(f"[RAG FALLBACK] Failed to initialize embedder: {e}")
        return MockEmbedder()

def search_collection(
    collection_name: str,
    query: str,
    query_embedding: List[float],
    vector_settings: Dict[str, Any],
    max_docs: int
) -> List[Dict[str, Any]]:
    """Search a specific collection"""
    try:
        # Try Milvus first
        docs = search_milvus_collection(collection_name, query_embedding, vector_settings, max_docs)
        if docs:
            return docs
    except Exception as e:
        logger.debug(f"[RAG FALLBACK] Milvus search failed: {e}")
    
    try:
        # Try Qdrant as fallback
        docs = search_qdrant_collection(collection_name, query_embedding, vector_settings, max_docs)
        if docs:
            return docs
    except Exception as e:
        logger.debug(f"[RAG FALLBACK] Qdrant search failed: {e}")
    
    logger.warning(f"[RAG FALLBACK] No vector database available for {collection_name}")
    return []

def search_milvus_collection(
    collection_name: str,
    query_embedding: List[float],
    vector_settings: Dict[str, Any],
    max_docs: int
) -> List[Dict[str, Any]]:
    """Search Milvus collection"""
    try:
        from pymilvus import Collection, connections
        
        # Connect to Milvus
        milvus_config = vector_settings.get('milvus', {})
        host = milvus_config.get('host', 'localhost')
        port = milvus_config.get('port', 19530)
        
        connections.connect(host=host, port=port)
        
        # Get collection
        collection = Collection(collection_name)
        collection.load()
        
        # Search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=max_docs,
            output_fields=["title", "content", "metadata"]
        )
        
        # Format results
        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    'title': hit.get('title', 'Unknown'),
                    'content': hit.get('content', ''),
                    'score': float(hit.score),
                    'metadata': hit.get('metadata', {})
                })
        
        return documents
        
    except Exception as e:
        logger.debug(f"[RAG FALLBACK] Milvus search error: {e}")
        raise

def search_qdrant_collection(
    collection_name: str,
    query_embedding: List[float],
    vector_settings: Dict[str, Any],
    max_docs: int
) -> List[Dict[str, Any]]:
    """Search Qdrant collection"""
    try:
        from qdrant_client import QdrantClient
        
        # Connect to Qdrant
        qdrant_config = vector_settings.get('qdrant', {})
        host = qdrant_config.get('host', 'localhost')
        port = qdrant_config.get('port', 6333)
        
        client = QdrantClient(host=host, port=port)
        
        # Search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=max_docs
        )
        
        # Format results
        documents = []
        for result in results:
            payload = result.payload or {}
            documents.append({
                'title': payload.get('title', 'Unknown'),
                'content': payload.get('content', ''),
                'score': float(result.score),
                'metadata': payload.get('metadata', {})
            })
        
        return documents
        
    except Exception as e:
        logger.debug(f"[RAG FALLBACK] Qdrant search error: {e}")
        raise

class MockEmbedder:
    """Mock embedder for when no real embedder is available"""
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings"""
        import hashlib
        import struct
        
        embeddings = []
        for text in texts:
            # Create a deterministic "embedding" from text hash
            text_hash = hashlib.md5(text.encode()).digest()
            # Convert to 384 floats (typical embedding size)
            embedding = []
            for i in range(0, len(text_hash), 4):
                chunk = text_hash[i:i+4]
                if len(chunk) == 4:
                    val = struct.unpack('f', chunk)[0]
                    embedding.append(val)
            
            # Pad to 384 dimensions
            while len(embedding) < 384:
                embedding.append(0.0)
                
            embeddings.append(embedding[:384])
        
        return embeddings