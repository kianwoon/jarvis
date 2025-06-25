"""
Temporary document indexer service using LlamaIndex for in-memory document indexing.
Provides session-based document indexing that exists only during conversation sessions.
"""

import asyncio
import pickle
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode

from app.core.redis_base import RedisCache
import redis
from app.core.embedding_settings_cache import get_embedding_settings
from app.document_handlers.base import ExtractedChunk
from app.document_handlers.base import DocumentHandler
from app.utils.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)

# Global mock embedding class for pickle compatibility
class MockEmbedding(BaseEmbedding):
    """Simple mock embedding for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.embed_batch_size = 10
        self.model_name = "mock_embedding"
        # Initialize Pydantic fields to avoid pickle issues
        self.__pydantic_fields_set__ = set()
        self.__pydantic_extra__ = {}
        
    def _get_query_embedding(self, query: str) -> list[float]:
        # Create a simple hash-based embedding
        import hashlib
        hash_obj = hashlib.md5(query.encode())
        hash_bytes = hash_obj.digest()
        # Convert to numbers and normalize
        embedding = [float(b) / 255.0 for b in hash_bytes]
        # Pad to standard embedding size (384 dimensions)
        while len(embedding) < 384:
            embedding.extend(embedding[:min(len(embedding), 384 - len(embedding))])
        return embedding[:384]
    
    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)
    
    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_query_embedding(text)
    
    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._get_text_embedding(text) for text in texts]
    
    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)
    
    def get_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)
    
    def get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)
    
    def __getstate__(self):
        """Custom pickle state for serialization."""
        return {
            'embed_batch_size': self.embed_batch_size,
            'model_name': self.model_name,
            '__pydantic_fields_set__': getattr(self, '__pydantic_fields_set__', set()),
            '__pydantic_extra__': getattr(self, '__pydantic_extra__', {})
        }
    
    def __setstate__(self, state):
        """Custom pickle state restoration."""
        self.embed_batch_size = state.get('embed_batch_size', 10)
        self.model_name = state.get('model_name', 'mock_embedding')
        self.__pydantic_fields_set__ = state.get('__pydantic_fields_set__', set())
        self.__pydantic_extra__ = state.get('__pydantic_extra__', {})

class TempDocumentIndexer:
    """
    Manages temporary document indexing using LlamaIndex for in-memory vector stores.
    Documents are indexed per conversation session and automatically cleaned up.
    """
    
    def __init__(self):
        self.redis_cache = RedisCache(key_prefix="temp_doc_")
        self.metadata_extractor = MetadataExtractor()
        self._binary_redis_client = None
        
    async def create_temp_index(
        self, 
        document_chunks: List[ExtractedChunk],
        temp_doc_id: str,
        conversation_id: str,
        filename: str,
        ttl_hours: int = 2
    ) -> Dict[str, Any]:
        """
        Create a temporary LlamaIndex vector store from extracted document chunks.
        
        Args:
            document_chunks: List of extracted chunks from document handlers
            temp_doc_id: Unique identifier for this temporary document
            conversation_id: Associated conversation ID
            filename: Original filename
            ttl_hours: Time-to-live in hours (default: 2)
            
        Returns:
            Dictionary containing temp document metadata and collection info
        """
        try:
            logger.info(f"Creating temporary index for {filename} (doc_id: {temp_doc_id})")
            
            # Convert ExtractedChunks to LlamaIndex Documents
            documents = []
            for i, chunk in enumerate(document_chunks):
                # Create LlamaIndex Document with metadata
                doc = Document(
                    text=chunk.content,
                    metadata={
                        **chunk.metadata,
                        'temp_doc_id': temp_doc_id,
                        'conversation_id': conversation_id,
                        'filename': filename,
                        'chunk_index': i,
                        'quality_score': chunk.quality_score,
                        'created_at': datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            # Create in-memory vector store
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Get embedding model from existing settings
            embedding_model = await self._get_embedding_model()
            
            # Create the index
            temp_index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=embedding_model,
                show_progress=True
            )
            
            # Store index in Redis with TTL using binary-safe client
            collection_name = f"temp_llamaindex_{temp_doc_id}"
            ttl_seconds = ttl_hours * 3600
            
            # Always store JSON fallback data first
            index_data = {
                'documents': [{'text': doc.text, 'metadata': doc.metadata} for doc in documents],
                'embedding_model_type': 'MockEmbedding',
                'created_at': datetime.now().isoformat()
            }
            logger.info(f"[STORAGE DEBUG] Storing JSON fallback data with {len(index_data['documents'])} documents")
            self.redis_cache.set(
                f"llamaindex_json:{collection_name}",  # Use different key for JSON
                index_data,
                expire=ttl_seconds
            )
            
            # Also try to store binary pickle (but don't fail if it doesn't work)
            try:
                serialized_index = pickle.dumps(temp_index)
                success = self._store_binary_data(
                    f"llamaindex_binary:{collection_name}",  # Use different key for binary
                    serialized_index,
                    ttl_seconds
                )
                if success:
                    logger.info(f"[STORAGE DEBUG] Binary pickle data stored successfully")
                else:
                    logger.warning(f"[STORAGE DEBUG] Binary pickle storage failed, using JSON fallback")
            except Exception as pickle_error:
                logger.warning(f"[STORAGE DEBUG] Pickle serialization failed: {str(pickle_error)}, using JSON fallback")
            
            # Store document metadata
            temp_doc_metadata = {
                'temp_doc_id': temp_doc_id,
                'conversation_id': conversation_id,
                'filename': filename,
                'collection_name': collection_name,
                'upload_timestamp': datetime.now().isoformat(),
                'expiry_timestamp': (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
                'is_included': True,  # Default to included
                'status': 'ready',
                'metadata': {
                    'file_size': sum(len(chunk.content.encode('utf-8')) for chunk in document_chunks),
                    'chunk_count': len(document_chunks),
                    'avg_quality_score': sum(chunk.quality_score for chunk in document_chunks) / len(document_chunks)
                }
            }
            
            # Store metadata with TTL
            self.redis_cache.set(
                f"metadata:{temp_doc_id}",
                temp_doc_metadata,
                expire=ttl_seconds
            )
            
            # Add to conversation's document list
            await self._add_doc_to_conversation(conversation_id, temp_doc_id, ttl_seconds)
            
            logger.info(f"Successfully created temporary index for {filename} with {len(documents)} chunks")
            return temp_doc_metadata
            
        except Exception as e:
            logger.error(f"Failed to create temporary index for {filename}: {str(e)}")
            raise
    
    def get_temp_index(self, temp_doc_id: str) -> Optional[VectorStoreIndex]:
        """
        Retrieve a temporary LlamaIndex from Redis.
        
        Args:
            temp_doc_id: Temporary document ID
            
        Returns:
            LlamaIndex VectorStoreIndex or None if not found/expired
        """
        try:
            collection_name = f"temp_llamaindex_{temp_doc_id}"
            
            # Try to get binary data first
            serialized_index = self._get_binary_data(f"llamaindex_binary:{collection_name}")
            
            if serialized_index:
                try:
                    temp_index = pickle.loads(serialized_index)
                    logger.info(f"[INDEX DEBUG] Successfully loaded binary pickle data")
                    return temp_index
                except Exception as unpickle_error:
                    logger.warning(f"Failed to unpickle index: {str(unpickle_error)}")
                    # Fall back to JSON data and recreate index
                    logger.info(f"[INDEX DEBUG] Attempting JSON fallback for {collection_name}")
                    index_data = self.redis_cache.get(f"llamaindex_json:{collection_name}")
                    logger.info(f"[INDEX DEBUG] JSON data found: {index_data is not None}")
                    if index_data:
                        logger.info(f"[INDEX DEBUG] JSON data type: {type(index_data)}")
                        if isinstance(index_data, dict):
                            logger.info(f"[INDEX DEBUG] JSON data keys: {list(index_data.keys())}")
                            recreated_index = self._recreate_index_from_data_sync(index_data)
                            logger.info(f"[INDEX DEBUG] Recreated index: {recreated_index is not None}")
                            return recreated_index
            else:
                # No binary data, try JSON fallback directly
                logger.info(f"[INDEX DEBUG] No binary data found, trying JSON fallback for {collection_name}")
                index_data = self.redis_cache.get(f"llamaindex_json:{collection_name}")
                logger.info(f"[INDEX DEBUG] JSON data found: {index_data is not None}")
                if index_data and isinstance(index_data, dict):
                    logger.info(f"[INDEX DEBUG] JSON data keys: {list(index_data.keys())}")
                    recreated_index = self._recreate_index_from_data_sync(index_data)
                    logger.info(f"[INDEX DEBUG] Recreated index: {recreated_index is not None}")
                    return recreated_index
            
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve temporary index {temp_doc_id}: {str(e)}")
            return None
    
    async def query_temp_document(
        self, 
        temp_doc_id: str, 
        query: str, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query a temporary document index.
        
        Args:
            temp_doc_id: Temporary document ID
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            Query results with sources and metadata
        """
        try:
            logger.info(f"[QUERY DEBUG] Querying temp doc {temp_doc_id} with query: {query}")
            temp_index = self.get_temp_index(temp_doc_id)
            logger.info(f"[QUERY DEBUG] Retrieved index: {temp_index is not None}")
            
            if not temp_index:
                logger.warning(f"[QUERY DEBUG] No index found for {temp_doc_id}")
                return {'error': 'Temporary document not found or expired'}
            
            logger.info(f"[QUERY DEBUG] Creating query engine...")
            # Create query engine with no LLM (we just want retrieval, not generation)
            from llama_index.core.retrievers import VectorIndexRetriever
            
            # Use retriever instead of query engine to avoid LLM requirement
            retriever = VectorIndexRetriever(
                index=temp_index,
                similarity_top_k=top_k,
            )
            
            # Execute retrieval (no LLM needed)
            logger.info(f"[QUERY DEBUG] Executing retrieval on index...")
            nodes = retriever.retrieve(query)
            logger.info(f"[QUERY DEBUG] Retrieval executed, found {len(nodes)} nodes")
            
            # Extract source information
            sources = []
            logger.info(f"[QUERY DEBUG] Processing {len(nodes)} retrieved nodes")
            for i, node in enumerate(nodes):
                logger.info(f"[QUERY DEBUG] Source {i}: text length {len(node.text)}, score {getattr(node, 'score', 'N/A')}")
                sources.append({
                    'content': node.text,
                    'metadata': node.metadata,
                    'score': node.score if hasattr(node, 'score') else 1.0
                })
            
            result = {
                'response': f"Found {len(sources)} relevant chunks from {temp_doc_id}",
                'sources': sources,
                'temp_doc_id': temp_doc_id,
                'query': query
            }
            logger.info(f"[QUERY DEBUG] Returning {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Failed to query temporary document {temp_doc_id}: {str(e)}")
            return {'error': f'Query failed: {str(e)}'}
    
    async def get_conversation_temp_docs(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get all temporary documents for a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of temporary document metadata
        """
        try:
            doc_ids = self.redis_cache.get(f"conv_temp_docs:{conversation_id}", [])
            temp_docs = []
            
            for temp_doc_id in doc_ids:
                metadata = self.redis_cache.get(f"metadata:{temp_doc_id}")
                if metadata:
                    temp_docs.append(metadata)
            
            return temp_docs
        except Exception as e:
            logger.error(f"Failed to get temp docs for conversation {conversation_id}: {str(e)}")
            return []
    
    async def update_document_inclusion(
        self, 
        temp_doc_id: str, 
        is_included: bool
    ) -> bool:
        """
        Update whether a temporary document should be included in chat context.
        
        Args:
            temp_doc_id: Temporary document ID
            is_included: Whether to include in chat
            
        Returns:
            Success status
        """
        try:
            metadata = self.redis_cache.get(f"metadata:{temp_doc_id}")
            if metadata:
                metadata['is_included'] = is_included
                
                # Get original TTL and update
                self.redis_cache.set(f"metadata:{temp_doc_id}", metadata)
                
                logger.info(f"Updated inclusion status for {temp_doc_id}: {is_included}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to update inclusion for {temp_doc_id}: {str(e)}")
            return False
    
    async def delete_temp_document(self, temp_doc_id: str) -> bool:
        """
        Manually delete a temporary document.
        
        Args:
            temp_doc_id: Temporary document ID
            
        Returns:
            Success status
        """
        try:
            # Get metadata to find conversation
            metadata = self.redis_cache.get(f"metadata:{temp_doc_id}")
            if metadata:
                conversation_id = metadata['conversation_id']
                collection_name = metadata.get('collection_name', f"temp_llamaindex_{temp_doc_id}")
                
                # Remove from conversation list
                existing_docs = self.redis_cache.get(f"conv_temp_docs:{conversation_id}", [])
                if temp_doc_id in existing_docs:
                    existing_docs.remove(temp_doc_id)
                    self.redis_cache.set(f"conv_temp_docs:{conversation_id}", existing_docs)
                
                # Delete index and metadata (both binary and JSON)
                self.redis_cache.delete(f"llamaindex_temp:{collection_name}")
                # Also try to delete binary version
                client = self._get_binary_redis_client()
                if client:
                    try:
                        client.delete(f"temp_doc_llamaindex_temp:{collection_name}")
                    except:
                        pass
                self.redis_cache.delete(f"metadata:{temp_doc_id}")
                
                logger.info(f"Successfully deleted temporary document {temp_doc_id}")
                return True
            else:
                # Document doesn't exist, but we should clean up any orphaned references
                logger.warning(f"Document {temp_doc_id} not found in metadata, cleaning up any references")
                
                # Try to clean up from all possible conversation lists
                # This is a bit brute force but ensures cleanup
                conv_keys = []
                try:
                    # Get all conversation keys (this is Redis-specific)
                    import fnmatch
                    client = self.redis_cache._get_client()
                    if client:
                        all_keys = client.keys("conv_temp_docs:*")
                        for key in all_keys:
                            conv_key = key.replace("temp_doc_", "")
                            existing_docs = self.redis_cache.get(conv_key, [])
                            if temp_doc_id in existing_docs:
                                existing_docs.remove(temp_doc_id)
                                self.redis_cache.set(conv_key, existing_docs)
                                logger.info(f"Cleaned up {temp_doc_id} from {conv_key}")
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup attempt failed: {cleanup_error}")
                
                # Clean up any potential orphaned keys
                self.redis_cache.delete(f"llamaindex_temp:temp_llamaindex_{temp_doc_id}")
                # Also clean up binary keys
                client = self._get_binary_redis_client()
                if client:
                    try:
                        client.delete(f"temp_doc_llamaindex_temp:temp_llamaindex_{temp_doc_id}")
                    except:
                        pass
                self.redis_cache.delete(f"metadata:{temp_doc_id}")
                self.redis_cache.delete(f"status:{temp_doc_id}")
                
                return True  # Return True since we've cleaned up what we can
            
        except Exception as e:
            logger.error(f"Failed to delete temporary document {temp_doc_id}: {str(e)}")
            # Even if there's an error, try basic cleanup
            try:
                self.redis_cache.delete(f"metadata:{temp_doc_id}")
                self.redis_cache.delete(f"status:{temp_doc_id}")
                self.redis_cache.delete(f"llamaindex_temp:temp_llamaindex_{temp_doc_id}")
                # Clean up binary keys too
                client = self._get_binary_redis_client()
                if client:
                    try:
                        client.delete(f"temp_doc_llamaindex_temp:temp_llamaindex_{temp_doc_id}")
                    except:
                        pass
            except:
                pass
            return False
    
    async def cleanup_expired_documents(self) -> int:
        """
        Clean up expired temporary documents.
        
        Returns:
            Number of documents cleaned up
        """
        try:
            cleaned_count = 0
            
            # Find all temporary document metadata
            pattern = "temp_doc_metadata:*"
            async for key in self.redis_cache.scan_iter(match=pattern):
                metadata = await self.redis_cache.get(key)
                if metadata:
                    expiry_time = datetime.fromisoformat(metadata['expiry_timestamp'])
                    if datetime.now() > expiry_time:
                        temp_doc_id = metadata['temp_doc_id']
                        if await self.delete_temp_document(temp_doc_id):
                            cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired temporary documents")
            
            return cleaned_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired documents: {str(e)}")
            return 0
    
    async def _get_embedding_model(self) -> BaseEmbedding:
        """Get embedding model from existing settings."""
        try:
            # Try to use existing embedding endpoint if available
            embedding_settings = get_embedding_settings()
            embedding_endpoint = embedding_settings.get('embedding_endpoint', '')
            
            if embedding_endpoint and 'qwen-embedder' in embedding_endpoint:
                # Use the existing qwen-embedder service
                logger.info("Using existing qwen-embedder service")
                # For now, create a simple wrapper (this would need custom implementation)
                # Fall through to use a local model instead
                pass
            
            # Try to use a simple embedding model that doesn't require external services
            try:
                # Use sentence-transformers if available
                from sentence_transformers import SentenceTransformer
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                
                # Use a lightweight model that can run locally
                model = HuggingFaceEmbedding(
                    model_name="all-MiniLM-L6-v2",  # Small, fast model
                    device="cpu"  # Force CPU to avoid CUDA issues
                )
                logger.info("Using local sentence-transformers model")
                return model
                
            except ImportError:
                logger.warning("sentence-transformers not available, trying OpenAI")
                
                # Try OpenAI if API key is available
                import os
                if os.getenv('OPENAI_API_KEY'):
                    from llama_index.embeddings.openai import OpenAIEmbedding
                    logger.info("Using OpenAI embeddings")
                    return OpenAIEmbedding(model="text-embedding-ada-002")
                else:
                    logger.warning("No OpenAI API key found")
            
            # Last resort: create a simple mock embedding for testing
            logger.warning("Creating mock embedding model for testing")
            return self._create_mock_embedding()
            
        except Exception as e:
            logger.error(f"Failed to load any embedding model: {str(e)}")
            # Create a mock embedding for basic functionality
            return self._create_mock_embedding()
    
    def _create_mock_embedding(self):
        """Create a simple mock embedding for testing purposes."""
        logger.info("Created mock embedding model for testing")
        return MockEmbedding()
    
    def _get_binary_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client that can handle binary data."""
        if self._binary_redis_client is None:
            try:
                import os
                self._binary_redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "redis"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    password=os.getenv("REDIS_PASSWORD", None),
                    decode_responses=False,  # Keep binary data as is
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                self._binary_redis_client.ping()
                return self._binary_redis_client
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.error(f"Binary Redis connection failed: {e}")
                self._binary_redis_client = None
        return self._binary_redis_client
    
    def _store_binary_data(self, key: str, data: bytes, ttl_seconds: int) -> bool:
        """Store binary data in Redis."""
        try:
            client = self._get_binary_redis_client()
            if not client:
                return False
            
            full_key = f"temp_doc_{key}"
            return bool(client.set(full_key, data, ex=ttl_seconds))
        except Exception as e:
            logger.error(f"Failed to store binary data: {str(e)}")
            return False
    
    def _get_binary_data(self, key: str) -> Optional[bytes]:
        """Get binary data from Redis."""
        try:
            client = self._get_binary_redis_client()
            if not client:
                return None
            
            full_key = f"temp_doc_{key}"
            return client.get(full_key)
        except Exception as e:
            logger.error(f"Failed to get binary data: {str(e)}")
            return None
    
    async def _recreate_index_from_data(self, index_data: Dict[str, Any]) -> Optional[VectorStoreIndex]:
        """Recreate index from stored document data."""
        try:
            if 'documents' not in index_data:
                return None
            
            # Recreate documents
            documents = []
            for doc_data in index_data['documents']:
                doc = Document(
                    text=doc_data['text'],
                    metadata=doc_data['metadata']
                )
                documents.append(doc)
            
            # Create new index
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            embedding_model = await self._get_embedding_model()
            
            temp_index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=embedding_model,
                show_progress=False
            )
            
            return temp_index
        except Exception as e:
            logger.error(f"Failed to recreate index from data: {str(e)}")
            return None
    
    def _recreate_index_from_data_sync(self, index_data: Dict[str, Any]) -> Optional[VectorStoreIndex]:
        """Recreate index from stored document data (synchronous version)."""
        try:
            logger.info(f"[RECREATE DEBUG] Starting index recreation")
            
            if 'documents' not in index_data:
                logger.error(f"[RECREATE DEBUG] No 'documents' key in index_data")
                return None
            
            logger.info(f"[RECREATE DEBUG] Found {len(index_data['documents'])} documents in data")
            
            # Recreate documents
            documents = []
            for i, doc_data in enumerate(index_data['documents']):
                logger.info(f"[RECREATE DEBUG] Processing document {i}: text length {len(doc_data.get('text', ''))}")
                doc = Document(
                    text=doc_data['text'],
                    metadata=doc_data['metadata']
                )
                documents.append(doc)
            
            logger.info(f"[RECREATE DEBUG] Created {len(documents)} Document objects")
            
            # Create new index
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            embedding_model = self._create_mock_embedding()  # Use sync method
            
            logger.info(f"[RECREATE DEBUG] Creating VectorStoreIndex...")
            temp_index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=storage_context,
                embed_model=embedding_model,
                show_progress=False
            )
            
            logger.info(f"[RECREATE DEBUG] Index created successfully!")
            return temp_index
        except Exception as e:
            logger.error(f"Failed to recreate index from data (sync): {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    async def _add_doc_to_conversation(
        self, 
        conversation_id: str, 
        temp_doc_id: str, 
        ttl_seconds: int
    ) -> None:
        """Add temporary document ID to conversation's document set."""
        # Note: These are sync operations as RedisCache is sync
        # For now, store as a simple list in a key
        existing_docs = self.redis_cache.get(f"conv_temp_docs:{conversation_id}", [])
        if temp_doc_id not in existing_docs:
            existing_docs.append(temp_doc_id)
        self.redis_cache.set(f"conv_temp_docs:{conversation_id}", existing_docs, expire=ttl_seconds)