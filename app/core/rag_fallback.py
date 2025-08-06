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
import requests
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
        
        # Get collections to search with smart matching
        collections = smart_collection_matching(collections)
        
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

def smart_collection_matching(requested_collections: List[str]) -> List[str]:
    """
    Smart collection name matching to handle agent inference errors
    
    This fixes cases where agents guess collection names like 'partnerships' 
    when the actual collection is 'partnership'.
    """
    if not requested_collections:
        return get_default_collections()
    
    try:
        from app.core.collection_registry_cache import get_all_collections
        all_collections_data = get_all_collections()
        if not all_collections_data:
            return get_default_collections()
        
        available_collections = [c.get('collection_name') for c in all_collections_data if c.get('collection_name')]
        matched_collections = []
        
        for requested in requested_collections:
            if not requested:
                continue
                
            # Exact match first
            if requested in available_collections:
                matched_collections.append(requested)
                continue
            
            # Fuzzy matching for common variations
            requested_lower = requested.lower().strip()
            best_match = None
            
            for available in available_collections:
                available_lower = available.lower().strip()
                
                # Handle plural/singular variations
                if (requested_lower == available_lower + 's' or  # partnerships -> partnership
                    requested_lower + 's' == available_lower or   # partnership -> partnerships
                    requested_lower == available_lower):           # exact case-insensitive
                    best_match = available
                    break
                
                # Handle partial matches (be careful not to be too broad)
                if (len(requested_lower) > 4 and len(available_lower) > 4 and
                    (requested_lower in available_lower or available_lower in requested_lower)):
                    if not best_match or len(available) < len(best_match):  # prefer shorter match
                        best_match = available
            
            if best_match:
                matched_collections.append(best_match)
                logger.info(f"[RAG FALLBACK] Smart matched '{requested}' -> '{best_match}'")
            else:
                logger.warning(f"[RAG FALLBACK] No match found for collection '{requested}'")
        
        # If we found matches, use them. Otherwise fall back to searching all collections
        if matched_collections:
            return matched_collections
        else:
            logger.info(f"[RAG FALLBACK] No collections matched, falling back to default search")
            return get_default_collections()
            
    except Exception as e:
        logger.warning(f"[RAG FALLBACK] Smart matching failed: {e}")
        return requested_collections  # Return original if matching fails

def get_simple_embedder(embedding_settings: Dict[str, Any]):
    """Get a simple embedder instance"""
    try:
        # Get the actual embedding model from settings
        embedding_model = embedding_settings.get('embedding_model', 'all-MiniLM-L6-v2')
        embedding_endpoint = embedding_settings.get('embedding_endpoint', '')
        
        logger.info(f"[RAG FALLBACK] Using embedding model: {embedding_model}, endpoint: {embedding_endpoint}")
        
        # If we have a remote endpoint, use HTTP embedder
        if embedding_endpoint and embedding_endpoint.startswith('http'):
            # Replace Docker hostname with localhost for local access
            if 'qwen-embedder:8050' in embedding_endpoint:
                embedding_endpoint = embedding_endpoint.replace('qwen-embedder:8050', 'localhost:8050')
            return HTTPEmbedder(embedding_endpoint)
        
        # Try sentence transformers with the configured model
        try:
            from sentence_transformers import SentenceTransformer
            # Handle model name (strip quotes if present)
            model_name = embedding_model.strip('"').strip("'")
            return SentenceTransformer(model_name)
        except Exception as e:
            logger.debug(f"[RAG FALLBACK] SentenceTransformer failed: {e}")
        
        # Try HuggingFace embeddings
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            model_name = embedding_model.strip('"').strip("'")
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            logger.debug(f"[RAG FALLBACK] HuggingFaceEmbeddings failed: {e}")
        
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
        
        # Get Milvus config from the correct location
        milvus_config = {}
        if 'databases' in vector_settings:
            for db in vector_settings['databases']:
                if db.get('id') == 'milvus' and db.get('enabled'):
                    milvus_config = db.get('config', {})
                    break
        
        # Extract connection parameters
        uri = milvus_config.get('MILVUS_URI') or milvus_config.get('uri')
        token = milvus_config.get('MILVUS_TOKEN') or milvus_config.get('token')
        
        if not uri:
            logger.warning("[RAG FALLBACK] No Milvus URI found in configuration")
            raise Exception("No Milvus URI configured")
        
        # Connect to Milvus with proper authentication
        if token:
            connections.connect(uri=uri, token=token)
        else:
            connections.connect(uri=uri)
        
        # Get collection
        collection = Collection(collection_name)
        collection.load()
        
        # Search with correct field name
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="vector",  # Use correct field name
            param=search_params,
            limit=max_docs,
            output_fields=["content", "source"]  # Use available fields
        )
        
        # Format results
        documents = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                # Access entity fields correctly
                documents.append({
                    'title': entity.get('source') or 'Unknown',  # Use source as title
                    'content': entity.get('content') or '',
                    'score': float(hit.distance),  # Use distance for score
                    'metadata': {
                        'source': entity.get('source') or '',
                        'collection': collection_name
                    }
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

class HTTPEmbedder:
    """HTTP-based embedder for remote embedding services"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via HTTP endpoint"""
        try:
            response = requests.post(
                self.endpoint,
                json={"texts": texts},
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            if 'embeddings' in result:
                return result['embeddings']
            elif 'data' in result:
                return [item['embedding'] for item in result['data']]
            else:
                logger.error(f"[HTTPEmbedder] Unexpected response format: {result}")
                return [[0.0] * 2560 for _ in texts]  # Return zero vectors as fallback
                
        except Exception as e:
            logger.error(f"[HTTPEmbedder] Failed to get embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 2560 for _ in texts]