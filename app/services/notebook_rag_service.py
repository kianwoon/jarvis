"""
Notebook RAG service for querying documents within notebooks using Milvus vector store.
Integrates with existing vector database and embedding infrastructure.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.notebook_models import (
    NotebookRAGResponse, NotebookRAGSource
)
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.notebook_llm_settings_cache import get_notebook_llm_settings
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import connections, Collection, utility

logger = logging.getLogger(__name__)

class NotebookRAGService:
    """
    Service for RAG queries against notebook documents using existing vector infrastructure.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._vector_settings = None
        self._embedding_settings = None
        self._embedding_function = None
        self._milvus_client = None
    
    def _get_vector_settings(self) -> Dict[str, Any]:
        """Get vector database settings with caching."""
        if not self._vector_settings:
            self._vector_settings = get_vector_db_settings()
        return self._vector_settings
    
    def _get_embedding_settings(self) -> Dict[str, Any]:
        """Get embedding settings with caching."""
        if not self._embedding_settings:
            self._embedding_settings = get_embedding_settings()
        return self._embedding_settings
    
    def _get_embedding_function(self):
        """Get embedding function instance with caching."""
        if not self._embedding_function:
            try:
                self.logger.debug("[DEBUG] Starting embedding function initialization")
                embedding_config = self._get_embedding_settings()
                self.logger.debug(f"[DEBUG] Embedding config: {embedding_config}")
                
                # Check for HTTP embedding endpoint (match document.py pattern)
                embedding_endpoint = embedding_config.get('embedding_endpoint')
                embedding_model = embedding_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                
                if embedding_endpoint:
                    # Use HTTP embedding function (same as document.py)
                    self.logger.debug(f"[DEBUG] Creating HTTPEmbeddingFunction with endpoint: {embedding_endpoint}")
                    try:
                        self._embedding_function = HTTPEmbeddingFunction(endpoint=embedding_endpoint)
                        self.logger.info(f"Initialized HTTP embedding function with endpoint: {embedding_endpoint}")
                    except Exception as http_err:
                        self.logger.error(f"[DEBUG] HTTPEmbeddingFunction creation failed: {str(http_err)}")
                        self.logger.error(f"[DEBUG] HTTPEmbeddingFunction traceback: {traceback.format_exc()}")
                        raise
                else:
                    # Use HuggingFace embeddings as fallback
                    self.logger.debug(f"[DEBUG] Creating HuggingFaceEmbeddings with model: {embedding_model}")
                    self._embedding_function = HuggingFaceEmbeddings(
                        model_name=embedding_model,
                        model_kwargs={'device': 'cpu'}
                    )
                    self.logger.info(f"Initialized HuggingFace embedding function with model: {embedding_model}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize embedding function: {str(e)}")
                self.logger.error(f"[DEBUG] Embedding function init traceback: {traceback.format_exc()}")
                # Fallback to default - but this should match Milvus dimensions
                raise Exception(f"Cannot initialize embedding function without proper configuration. Error: {str(e)}")
        
        return self._embedding_function
    
    def _get_milvus_connection_args(self) -> Dict[str, Any]:
        """Get Milvus connection arguments using active vector database configuration."""
        try:
            # Get active vector database configuration (same as DocumentAdminService)
            from app.services.vector_db_service import get_active_vector_db
            
            active_db = get_active_vector_db()
            if not active_db or active_db.get('id') != 'milvus':
                raise Exception("Milvus is not the active vector database")
            
            milvus_config = active_db.get('config', {})
            uri = milvus_config.get('MILVUS_URI')
            token = milvus_config.get('MILVUS_TOKEN', '')
            
            if not uri:
                raise Exception("Milvus URI not configured")
            
            self.logger.info(f"Using Milvus URI: {uri}")
            
            # Return connection args in format expected by pymilvus
            return {
                "uri": uri,
                "token": token,
                "alias": "notebook_rag"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get Milvus connection args: {str(e)}")
            # Fallback to legacy format for backwards compatibility
            vector_config = self._get_vector_settings()
            return {
                "host": vector_config.get('host', 'localhost'),
                "port": vector_config.get('port', 19530),
                "user": vector_config.get('user', ''),
                "password": vector_config.get('password', ''),
                "alias": "notebook_rag_fallback"
            }
    
    async def query_notebook(
        self,
        db: Session,
        notebook_id: str,
        query: str,
        top_k: int = 5,
        include_metadata: bool = True,
        collection_filter: Optional[List[str]] = None
    ) -> NotebookRAGResponse:
        """
        Query notebook documents using RAG.
        
        Args:
            db: Database session
            notebook_id: Notebook ID to query
            query: Search query
            top_k: Number of results to return
            include_metadata: Whether to include metadata
            collection_filter: Optional filter for specific collections
            
        Returns:
            RAG query results
        """
        try:
            self.logger.info(f"Querying notebook {notebook_id} with query: {query[:50]}...")
            
            # Get notebook documents and their collections
            collections_info = await self._get_notebook_collections(db, notebook_id, collection_filter)
            
            if not collections_info:
                self.logger.warning(f"No collections found for notebook {notebook_id}")
                return NotebookRAGResponse(
                    notebook_id=notebook_id,
                    query=query,
                    sources=[],
                    total_sources=0,
                    queried_documents=0,
                    collections_searched=[]
                )
            
            # Query each collection and aggregate results
            all_sources = []
            collections_searched = set()
            
            self.logger.debug("[DEBUG] Getting embedding function for query")
            embedding_function = self._get_embedding_function()
            self.logger.debug(f"[DEBUG] Embedding function type: {type(embedding_function)}")
            
            self.logger.debug("[DEBUG] Getting Milvus connection args")
            connection_args = self._get_milvus_connection_args()
            self.logger.debug(f"[DEBUG] Connection args: {connection_args}")
            
            for collection_name, document_ids in collections_info.items():
                try:
                    self.logger.debug(f"[DEBUG] Processing collection {collection_name} for documents: {document_ids}")
                    
                    # Connect to Milvus using direct pymilvus client
                    self.logger.debug(f"[DEBUG] Connecting to Milvus for collection {collection_name}")
                    try:
                        connections.connect(
                            alias=connection_args['alias'],
                            uri=connection_args.get('uri'),
                            token=connection_args.get('token', ''),
                            host=connection_args.get('host'),
                            port=connection_args.get('port')
                        )
                        
                        # Create Collection instance
                        collection = Collection(collection_name, using=connection_args['alias'])
                        collection.load()  # Ensure collection is loaded
                        self.logger.debug(f"[DEBUG] Successfully connected to collection {collection_name}")
                    except Exception as milvus_err:
                        self.logger.error(f"[DEBUG] Failed to connect to Milvus collection: {str(milvus_err)}")
                        self.logger.error(f"[DEBUG] Milvus connection traceback: {traceback.format_exc()}")
                        raise
                    
                    # Get query embedding
                    self.logger.debug(f"[DEBUG] Generating embedding for query: {query[:100]}...")
                    try:
                        if hasattr(embedding_function, 'embed_query'):
                            query_embedding = embedding_function.embed_query(query)
                        elif hasattr(embedding_function, 'encode'):
                            query_embedding = embedding_function.encode([query])[0].tolist()
                        else:
                            raise Exception(f"Unsupported embedding function type: {type(embedding_function)}")
                        
                        if not isinstance(query_embedding, list):
                            query_embedding = query_embedding.tolist()
                        
                        self.logger.debug(f"[DEBUG] Generated embedding of dimension: {len(query_embedding)}")
                    except Exception as embed_err:
                        self.logger.error(f"[DEBUG] Failed to generate embedding: {str(embed_err)}")
                        self.logger.error(f"[DEBUG] Embedding error traceback: {traceback.format_exc()}")
                        raise
                    
                    # Build filter expression for document and memory IDs
                    doc_filter = None
                    if document_ids:
                        # Create filter to include both document chunks and memories from this notebook
                        # Document IDs in Milvus are chunked: "doc_id_pX_cY" format - use prefix matching
                        # Memory IDs are exact UUIDs - use exact matching
                        conditions = []
                        
                        for doc_id in document_ids:
                            # Check if this looks like a UUID (memory ID) - exact match
                            if len(doc_id) == 36 and doc_id.count('-') == 4:
                                conditions.append(f"doc_id == '{doc_id}'")
                            else:
                                # Document ID - use prefix matching for chunks
                                conditions.append(f"doc_id like '{doc_id}%'")
                        
                        doc_filter = " or ".join(conditions)
                    
                    # Setup search parameters
                    search_params = {
                        "metric_type": "COSINE",
                        "params": {"nprobe": 10}
                    }
                    
                    # Get configurable retrieval limits for comprehensive coverage
                    try:
                        notebook_settings = get_notebook_llm_settings()
                        max_retrieval_chunks = notebook_settings.get('notebook_llm', {}).get('max_retrieval_chunks', 200)
                        retrieval_multiplier = notebook_settings.get('notebook_llm', {}).get('retrieval_multiplier', 3)
                        # Document-aware retrieval settings
                        include_neighboring_chunks = notebook_settings.get('notebook_llm', {}).get('include_neighboring_chunks', True)
                        neighbor_chunk_radius = notebook_settings.get('notebook_llm', {}).get('neighbor_chunk_radius', 2)
                        document_completeness_threshold = notebook_settings.get('notebook_llm', {}).get('document_completeness_threshold', 0.8)
                        enable_document_aware_retrieval = notebook_settings.get('notebook_llm', {}).get('enable_document_aware_retrieval', True)
                    except Exception as e:
                        self.logger.warning(f"Could not load notebook settings, using defaults: {e}")
                        max_retrieval_chunks = 200  # Default for comprehensive retrieval
                        retrieval_multiplier = 3    # Default multiplier
                        # Document-aware retrieval defaults
                        include_neighboring_chunks = True
                        neighbor_chunk_radius = 2
                        document_completeness_threshold = 0.8
                        enable_document_aware_retrieval = True
                    
                    limit = min(top_k * retrieval_multiplier, max_retrieval_chunks)
                    
                    self.logger.debug(f"[DEBUG] About to call collection.search with limit: {limit}")
                    if doc_filter:
                        self.logger.debug(f"[DEBUG] Using document filter: {doc_filter}")
                    
                    try:
                        # Use direct pymilvus Collection.search() method
                        search_results = collection.search(
                            data=[query_embedding],
                            anns_field="vector",
                            param=search_params,
                            limit=limit,
                            expr=doc_filter,
                            output_fields=["content", "doc_id", "source", "page", "doc_type", "uploaded_at", "section", "author", "hash"]
                        )
                        
                        # Convert initial results to our format
                        initial_results = []
                        for hits in search_results:
                            for hit in hits:
                                # Extract fields from Milvus result
                                content = hit.entity.get('content', '')
                                metadata = {
                                    'doc_id': hit.entity.get('doc_id', ''),
                                    'source': hit.entity.get('source', ''),
                                    'page': hit.entity.get('page', 0),
                                    'doc_type': hit.entity.get('doc_type', ''),
                                    'uploaded_at': hit.entity.get('uploaded_at', ''),
                                    'section': hit.entity.get('section', ''),
                                    'author': hit.entity.get('author', ''),
                                    'hash': hit.entity.get('hash', '')
                                }
                                
                                # Create a document-like object that matches our processing logic
                                doc = {
                                    'page_content': content,
                                    'metadata': metadata
                                }
                                
                                initial_results.append((doc, hit.score))
                        
                        # Apply document-aware retrieval logic
                        if enable_document_aware_retrieval and initial_results:
                            results = await self._apply_document_aware_retrieval(
                                initial_results, 
                                collection,
                                query_embedding,
                                search_params,
                                doc_filter,
                                include_neighboring_chunks,
                                neighbor_chunk_radius,
                                document_completeness_threshold,
                                max_retrieval_chunks
                            )
                        else:
                            results = initial_results
                        
                        self.logger.debug(f"[DEBUG] Successfully executed direct search, got {len(results)} results")
                    except Exception as search_err:
                        self.logger.error(f"[DEBUG] Direct Milvus search failed: {str(search_err)}")
                        self.logger.error(f"[DEBUG] Search error type: {type(search_err).__name__}")
                        self.logger.error(f"[DEBUG] Search traceback: {traceback.format_exc()}")
                        raise
                    
                    collections_searched.add(collection_name)
                    self.logger.debug(f"Retrieved {len(results)} results from collection {collection_name}")
                    
                    # Debug: Log first result structure if available
                    if results and len(results) > 0:
                        self.logger.debug(f"[DEBUG] Got {len(results)} results from Milvus")
                        try:
                            first_doc, first_score = results[0]
                            self.logger.debug(f"[DEBUG] First result type: {type(first_doc)}")
                            self.logger.debug(f"[DEBUG] First result score: {first_score}")
                            self.logger.debug(f"[DEBUG] First result attributes: {dir(first_doc)}")
                            if hasattr(first_doc, '__dict__'):
                                self.logger.debug(f"[DEBUG] First result __dict__: {first_doc.__dict__}")
                            elif isinstance(first_doc, dict):
                                self.logger.debug(f"[DEBUG] First result keys: {list(first_doc.keys())}")
                                self.logger.debug(f"[DEBUG] First result content: {str(first_doc)[:200]}...")
                        except Exception as debug_err:
                            self.logger.error(f"[DEBUG] Error inspecting first result: {str(debug_err)}")
                            self.logger.error(f"[DEBUG] Debug error traceback: {traceback.format_exc()}")
                    
                    # Convert results to our format
                    self.logger.debug(f"[DEBUG] Processing {len(results)} results for conversion")
                    for i, (doc, score) in enumerate(results):
                        try:
                            self.logger.debug(f"[DEBUG] Processing result {i+1}/{len(results)}")
                            self.logger.debug(f"[DEBUG] Doc type: {type(doc)}, Score: {score}")
                            
                            # Extract document metadata with robust handling
                            doc_metadata = {}
                            if hasattr(doc, 'metadata') and doc.metadata:
                                doc_metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
                                self.logger.debug(f"[DEBUG] Got metadata from doc.metadata: {doc_metadata}")
                            elif isinstance(doc, dict) and 'metadata' in doc:
                                doc_metadata = doc['metadata'] if isinstance(doc['metadata'], dict) else {}
                                self.logger.debug(f"[DEBUG] Got metadata from doc['metadata']: {doc_metadata}")
                            else:
                                self.logger.debug(f"[DEBUG] No metadata found for document")
                            
                            document_id = doc_metadata.get('doc_id', 'unknown')
                            self.logger.debug(f"[DEBUG] Extracted document_id: {document_id}")
                            
                            # Fix document ID filtering - use prefix matching for chunked IDs
                            # Chunked IDs like "33997c75bf33_p1_c6" should match base ID "33997c75bf33"
                            should_include = True
                            if document_ids:
                                should_include = False
                                for base_doc_id in document_ids:
                                    if document_id.startswith(base_doc_id):
                                        should_include = True
                                        break
                                
                                if not should_include:
                                    self.logger.debug(f"[DEBUG] Skipping document {document_id} - not in notebook document list {document_ids}")
                                    continue
                                else:
                                    self.logger.debug(f"[DEBUG] Including document {document_id} - matches notebook documents")
                            
                            # Get document name from cached results with improved memory handling
                            cached_names = getattr(self, '_current_document_names', {})
                            document_info = cached_names.get(document_id, {})
                            
                            # Enhanced name resolution for different ID formats
                            if not document_info:
                                # Check if this looks like a UUID (memory ID) - try exact match first
                                if len(document_id) == 36 and document_id.count('-') == 4:
                                    # This is likely a memory UUID, check all cached names for exact match
                                    for cached_id, cached_info in cached_names.items():
                                        if cached_id == document_id and cached_info.get('type') == 'memory':
                                            document_info = cached_info
                                            self.logger.debug(f"[MEMORY_RESOLUTION] Found exact memory match: {document_id}")
                                            break
                                # Try prefix matching for document chunks 
                                elif '_' in document_id:
                                    # For document chunks like '33997c75bf33_p0_c0', try base ID '33997c75bf33'  
                                    base_id = document_id.split('_')[0]
                                    document_info = cached_names.get(base_id, {})
                                    if document_info:
                                        self.logger.debug(f"[DOCUMENT_RESOLUTION] Found chunk base match: {base_id} for {document_id}")
                            
                            document_name = document_info.get('name')
                            source_type = document_info.get('type', 'document')
                            
                            # CRITICAL FIX: If memory source has no name, provide fallback to prevent filtering
                            if source_type == 'memory' and not document_name:
                                self.logger.warning(f"[MEMORY_FIX] Memory {document_id} has no name in cached results")
                                self.logger.warning(f"[MEMORY_FIX] Available cached names: {list(cached_names.keys())}")
                                document_name = f"Memory ({document_id[:8]}...)"
                            elif source_type == 'memory' and document_name:
                                self.logger.debug(f"[MEMORY_SUCCESS] Memory {document_id} resolved to name: '{document_name}'")
                            
                            # Add logging for source type detection to help debug issues
                            self.logger.debug(f"[SOURCE_TYPE_DEBUG] Document {document_id}: type='{source_type}', name='{document_name}'")
                            
                            
                            # Robust content extraction - handle different document formats
                            content = None
                            self.logger.debug(f"[DEBUG] Extracting content from document {document_id}")
                            
                            try:
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content
                                    self.logger.debug(f"[DEBUG] Got content from doc.page_content: {len(str(content))} chars")
                                elif hasattr(doc, 'content'):
                                    content = doc.content
                                    self.logger.debug(f"[DEBUG] Got content from doc.content: {len(str(content))} chars")
                                elif hasattr(doc, 'text'):
                                    content = doc.text
                                    self.logger.debug(f"[DEBUG] Got content from doc.text: {len(str(content))} chars")
                                elif isinstance(doc, dict):
                                    # Handle dict-like document objects
                                    self.logger.debug(f"[DEBUG] Doc is dict, available keys: {list(doc.keys())}")
                                    content = doc.get('page_content') or doc.get('content') or doc.get('text', '')
                                    if not content and 'text' in doc:
                                        # Additional check for KeyError debugging
                                        try:
                                            content = doc['text']
                                            self.logger.debug(f"[DEBUG] Successfully accessed doc['text']: {len(str(content))} chars")
                                        except KeyError as ke:
                                            self.logger.error(f"[DEBUG] KeyError accessing doc['text']: {str(ke)}")
                                            self.logger.error(f"[DEBUG] Doc keys at time of KeyError: {list(doc.keys())}")
                                            self.logger.error(f"[DEBUG] Full doc content: {str(doc)}")
                                            raise
                                    self.logger.debug(f"[DEBUG] Got content from dict: {len(str(content))} chars")
                                else:
                                    # Fallback - convert to string
                                    content = str(doc)
                                    self.logger.debug(f"[DEBUG] Got content from str(doc): {len(str(content))} chars")
                                
                            except Exception as content_err:
                                self.logger.error(f"[DEBUG] Error extracting content from document {document_id}: {str(content_err)}")
                                self.logger.error(f"[DEBUG] Content extraction error type: {type(content_err).__name__}")
                                if isinstance(content_err, KeyError):
                                    self.logger.error(f"[DEBUG] KeyError during content extraction - missing key: {str(content_err)}")
                                self.logger.error(f"[DEBUG] Content extraction traceback: {traceback.format_exc()}")
                                raise
                            
                            if not content:
                                self.logger.warning(f"[DEBUG] Empty content for document {document_id}, skipping")
                                continue
                            
                            # DEBUG: Log source creation details
                            self.logger.debug(f"[SOURCE_CREATION] Creating NotebookRAGSource:")
                            self.logger.debug(f"[SOURCE_CREATION]   document_id: {document_id}")
                            self.logger.debug(f"[SOURCE_CREATION]   document_name: '{document_name}'") 
                            self.logger.debug(f"[SOURCE_CREATION]   source_type: '{source_type}'")
                            self.logger.debug(f"[SOURCE_CREATION]   score: {score}")
                            self.logger.debug(f"[SOURCE_CREATION]   collection: {collection_name}")
                            self.logger.debug(f"[SOURCE_CREATION]   content_length: {len(content)}")
                            
                            source = NotebookRAGSource(
                                content=content,
                                metadata=doc_metadata if include_metadata else {},
                                score=float(score),
                                document_id=document_id,
                                document_name=document_name,
                                collection=collection_name,
                                source_type=source_type
                            )
                        except Exception as doc_error:
                            self.logger.error(f"[DEBUG] Error processing document result: {str(doc_error)}")
                            self.logger.error(f"[DEBUG] Document processing error type: {type(doc_error).__name__}")
                            if isinstance(doc_error, KeyError):
                                self.logger.error(f"[DEBUG] KeyError during document processing - missing key: {str(doc_error)}")
                            self.logger.error(f"[DEBUG] Problematic doc type: {type(doc)}")
                            self.logger.error(f"[DEBUG] Problematic doc content (first 200 chars): {str(doc)[:200]}")
                            self.logger.error(f"[DEBUG] Document processing traceback: {traceback.format_exc()}")
                            continue
                        
                        all_sources.append(source)
                        
                        # DEBUG: Confirm source was added to the list
                        self.logger.debug(f"[SOURCE_ADDED] Successfully added source to all_sources list")
                        self.logger.debug(f"[SOURCE_ADDED] Current all_sources count: {len(all_sources)}")
                        self.logger.debug(f"[SOURCE_ADDED] Last added source type: '{source.source_type}', name: '{source.document_name}'")
                        
                        # Use configurable multiplier for source collection
                        try:
                            notebook_settings = get_notebook_llm_settings()
                            collection_multiplier = notebook_settings.get('notebook_llm', {}).get('collection_multiplier', 4)
                        except Exception:
                            collection_multiplier = 4  # Default for comprehensive collection
                        
                        if len(all_sources) >= top_k * collection_multiplier:
                            break
                    
                    self.logger.debug(f"Found {len(results)} results from collection {collection_name}")
                    
                    # Clean up connection for this collection
                    try:
                        connections.disconnect(alias=connection_args['alias'])
                    except:
                        pass  # Ignore cleanup errors
                    
                except Exception as e:
                    self.logger.error(f"[DEBUG] Error querying collection {collection_name}: {str(e)}")
                    self.logger.error(f"[DEBUG] Collection query error type: {type(e).__name__}")
                    if isinstance(e, KeyError):
                        self.logger.error(f"[DEBUG] KeyError during collection query - missing key: {str(e)}")
                    self.logger.error(f"[DEBUG] Full error traceback for collection {collection_name}: {traceback.format_exc()}")
                    continue
            
            # DEBUG: Log all sources before sorting and filtering
            self.logger.debug(f"[FINAL_SOURCES_DEBUG] Total sources before filtering: {len(all_sources)}")
            for idx, src in enumerate(all_sources):
                self.logger.debug(f"[FINAL_SOURCES_DEBUG] Source {idx+1}: type='{src.source_type}', name='{src.document_name}', score={src.score}")
            
            # Sort all sources by score (higher is better for COSINE similarity)
            all_sources.sort(key=lambda x: x.score, reverse=True)
            
            # Take top_k results
            top_sources = all_sources[:top_k]
            
            # DEBUG: Log final sources being returned
            self.logger.debug(f"[FINAL_RESPONSE_DEBUG] Final top {top_k} sources being returned:")
            for idx, src in enumerate(top_sources):
                self.logger.debug(f"[FINAL_RESPONSE_DEBUG] Source {idx+1}: type='{src.source_type}', name='{src.document_name}', score={src.score}")
            
            self.logger.info(f"Successfully queried notebook {notebook_id}, found {len(top_sources)} sources")
            
            return NotebookRAGResponse(
                notebook_id=notebook_id,
                query=query,
                sources=top_sources,
                total_sources=len(all_sources),
                queried_documents=len(set(source.document_id for source in all_sources)),
                collections_searched=list(collections_searched)
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error querying notebook: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            if isinstance(e, KeyError):
                self.logger.error(f"KeyError details - missing key: {str(e)}")
                import traceback
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return empty result instead of raising
            return NotebookRAGResponse(
                notebook_id=notebook_id,
                query=query,
                sources=[],
                total_sources=0,
                queried_documents=0,
                collections_searched=[]
            )
    
    async def _apply_document_aware_retrieval(
        self,
        initial_results: List[tuple],
        collection,
        query_embedding: List[float],
        search_params: Dict[str, Any],
        doc_filter: Optional[str],
        include_neighboring_chunks: bool,
        neighbor_chunk_radius: int,
        document_completeness_threshold: float,
        max_retrieval_chunks: int
    ) -> List[tuple]:
        """
        Apply document-aware retrieval logic to initial results with memory-first prioritization.
        
        This method implements NotebookLM-style comprehensive information coverage by:
        1. Separating memory sources from document sources (memory-first priority)
        2. Grouping chunks by document and calculating document-level scores
        3. Retrieving ALL memory chunks for completeness (memories are user-curated)
        4. Retrieving neighboring chunks when a chunk matches
        5. Including complete documents when they exceed the completeness threshold
        6. Boosting memory chunk scores for final ranking
        
        Args:
            initial_results: Initial search results as (doc, score) tuples
            collection: Milvus collection instance
            query_embedding: Query embedding vector
            search_params: Milvus search parameters
            doc_filter: Document filter expression
            include_neighboring_chunks: Whether to include neighboring chunks
            neighbor_chunk_radius: Number of chunks to include before and after matches
            document_completeness_threshold: Score threshold for including complete documents
            max_retrieval_chunks: Maximum number of chunks to retrieve
            
        Returns:
            Enhanced results with document-aware retrieval and memory-first prioritization applied
        """
        try:
            from collections import defaultdict
            import re
            
            if not initial_results:
                return initial_results
            
            self.logger.debug(f"[DOCUMENT_AWARE] Starting document-aware retrieval with memory-first prioritization, {len(initial_results)} initial results")
            
            # Load memory prioritization settings
            try:
                notebook_settings = get_notebook_llm_settings()
                prioritize_memory_sources = notebook_settings.get('notebook_llm', {}).get('prioritize_memory_sources', True)
                memory_score_boost = notebook_settings.get('notebook_llm', {}).get('memory_score_boost', 0.2)
                include_all_memory_chunks = notebook_settings.get('notebook_llm', {}).get('include_all_memory_chunks', True)
            except Exception as e:
                self.logger.warning(f"Could not load memory prioritization settings, using defaults: {e}")
                prioritize_memory_sources = True
                memory_score_boost = 0.2
                include_all_memory_chunks = True
            
            self.logger.debug(f"[MEMORY_PRIORITY] Settings: prioritize={prioritize_memory_sources}, boost={memory_score_boost}, include_all={include_all_memory_chunks}")
            
            # Step 1: Separate memory sources from document sources
            memory_chunks = []
            document_chunks = []
            
            for doc, score in initial_results:
                if self._is_memory_source(doc):
                    memory_chunks.append((doc, score))
                    self.logger.debug(f"[MEMORY_PRIORITY] Identified memory chunk: {doc.get('metadata', {}).get('doc_id', 'unknown')}")
                else:
                    document_chunks.append((doc, score))
            
            self.logger.debug(f"[MEMORY_PRIORITY] Separated {len(memory_chunks)} memory chunks and {len(document_chunks)} document chunks")
            
            enhanced_results = []
            processed_chunk_ids = set()
            
            # Step 2: Process memory sources first (if prioritization enabled)
            if prioritize_memory_sources and memory_chunks:
                self.logger.debug(f"[MEMORY_PRIORITY] Processing memory chunks first")
                
                # Group memory chunks by document ID
                memory_by_document = defaultdict(list)
                for doc, score in memory_chunks:
                    doc_id = doc.get('metadata', {}).get('doc_id', '')
                    base_doc_id = self._extract_base_document_id(doc_id)
                    memory_by_document[base_doc_id].append((doc, score, doc_id))
                
                # Process each memory document
                for base_doc_id, memory_doc_chunks in memory_by_document.items():
                    self.logger.debug(f"[MEMORY_PRIORITY] Processing memory document {base_doc_id[:12]}... with {len(memory_doc_chunks)} chunks")
                    
                    if include_all_memory_chunks:
                        # Retrieve ALL chunks from this memory document for completeness
                        try:
                            all_memory_chunks = await self._retrieve_complete_document_chunks(
                                collection, base_doc_id, query_embedding, search_params, doc_filter, max_retrieval_chunks
                            )
                            
                            # Add all memory chunks with score boost
                            for chunk_doc, chunk_score in all_memory_chunks:
                                chunk_id = chunk_doc.get('metadata', {}).get('doc_id', '')
                                if chunk_id not in processed_chunk_ids:
                                    # Boost memory chunk score
                                    boosted_score = min(chunk_score + memory_score_boost, 1.0)
                                    enhanced_results.append((chunk_doc, boosted_score))
                                    processed_chunk_ids.add(chunk_id)
                                    self.logger.debug(f"[MEMORY_PRIORITY] Added complete memory chunk {chunk_id} with boosted score {boosted_score:.3f}")
                        except Exception as e:
                            self.logger.warning(f"[MEMORY_PRIORITY] Could not retrieve complete memory document {base_doc_id}: {e}")
                            # Fallback to just the matching chunks
                            for doc, score, doc_id in memory_doc_chunks:
                                if doc_id not in processed_chunk_ids:
                                    boosted_score = min(score + memory_score_boost, 1.0)
                                    enhanced_results.append((doc, boosted_score))
                                    processed_chunk_ids.add(doc_id)
                    else:
                        # Just add the matching memory chunks with boost
                        for doc, score, doc_id in memory_doc_chunks:
                            if doc_id not in processed_chunk_ids:
                                boosted_score = min(score + memory_score_boost, 1.0)
                                enhanced_results.append((doc, boosted_score))
                                processed_chunk_ids.add(doc_id)
            
            self.logger.debug(f"[MEMORY_PRIORITY] After memory processing: {len(enhanced_results)} chunks")
            
            # Step 3: Process document sources (if space remaining)
            remaining_capacity = max_retrieval_chunks - len(enhanced_results)
            if remaining_capacity > 0 and document_chunks:
                self.logger.debug(f"[MEMORY_PRIORITY] Processing document chunks with remaining capacity: {remaining_capacity}")
                
                # Group document chunks by document
                chunks_by_document = defaultdict(list)
                for doc, score in document_chunks:
                    doc_id = doc.get('metadata', {}).get('doc_id', '')
                    base_doc_id = self._extract_base_document_id(doc_id)
                    chunks_by_document[base_doc_id].append((doc, score, doc_id))
            
                # Calculate document-level scores for remaining document chunks
                document_scores = {}
                for base_doc_id, chunks in chunks_by_document.items():
                    if chunks:
                        # Calculate average score for the document
                        avg_score = sum(score for _, score, _ in chunks) / len(chunks)
                        # Weight by number of matching chunks
                        weighted_score = avg_score * min(len(chunks) / 3.0, 1.5)  # Cap multiplier at 1.5x
                        document_scores[base_doc_id] = weighted_score
                        
                        self.logger.debug(f"[DOCUMENT_AWARE] Document {base_doc_id[:12]}... score: {weighted_score:.3f} (avg: {avg_score:.3f}, chunks: {len(chunks)})")
                
                # Process documents in order of relevance (fill remaining capacity)
                for base_doc_id, doc_score in sorted(document_scores.items(), key=lambda x: x[1], reverse=True):
                    if len(enhanced_results) >= max_retrieval_chunks:
                        break
                        
                    document_chunks_list = chunks_by_document[base_doc_id]
                    
                    # Check if document exceeds completeness threshold
                    should_include_complete_document = doc_score >= document_completeness_threshold
                    
                    if should_include_complete_document:
                        self.logger.debug(f"[DOCUMENT_AWARE] Including complete document {base_doc_id[:12]}... (score: {doc_score:.3f} >= {document_completeness_threshold})")
                        
                        # Retrieve all chunks from this document
                        all_doc_chunks = await self._retrieve_complete_document_chunks(
                            collection, base_doc_id, query_embedding, search_params, doc_filter, remaining_capacity
                        )
                        
                        # Add all chunks from complete document
                        for chunk_doc, chunk_score in all_doc_chunks:
                            if len(enhanced_results) >= max_retrieval_chunks:
                                break
                            chunk_id = chunk_doc.get('metadata', {}).get('doc_id', '')
                            if chunk_id not in processed_chunk_ids:
                                enhanced_results.append((chunk_doc, chunk_score))
                                processed_chunk_ids.add(chunk_id)
                    else:
                        # Process individual matching chunks and their neighbors
                        for doc, score, doc_id in document_chunks_list:
                            if len(enhanced_results) >= max_retrieval_chunks:
                                break
                            if doc_id not in processed_chunk_ids:
                                enhanced_results.append((doc, score))
                                processed_chunk_ids.add(doc_id)
                                
                                # Retrieve neighboring chunks if enabled
                                if include_neighboring_chunks and len(enhanced_results) < max_retrieval_chunks:
                                    neighbor_chunks = await self._retrieve_neighboring_chunks(
                                        collection, doc_id, neighbor_chunk_radius, doc_filter
                                    )
                                
                                    for neighbor_doc, neighbor_score in neighbor_chunks:
                                        if len(enhanced_results) >= max_retrieval_chunks:
                                            break
                                        neighbor_id = neighbor_doc.get('metadata', {}).get('doc_id', '')
                                        if neighbor_id not in processed_chunk_ids:
                                            enhanced_results.append((neighbor_doc, neighbor_score))
                                            processed_chunk_ids.add(neighbor_id)
            
            # Step 4: Final sorting and output
            self.logger.debug(f"[MEMORY_PRIORITY] Final enhanced results: {len(enhanced_results)} chunks from {len(processed_chunk_ids)} unique chunks")
            
            # Sort by relevance score (memories will naturally rank higher due to score boost)
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            final_results = enhanced_results[:max_retrieval_chunks]
            
            # Debug: Log final ranking to verify memory prioritization
            memory_count = sum(1 for doc, score in final_results if self._is_memory_source(doc))
            self.logger.debug(f"[MEMORY_PRIORITY] Final ranking: {memory_count} memory chunks, {len(final_results) - memory_count} document chunks")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"[DOCUMENT_AWARE] Error in document-aware retrieval: {str(e)}")
            self.logger.error(f"[DOCUMENT_AWARE] Falling back to original results")
            return initial_results
    
    def _extract_base_document_id(self, doc_id: str) -> str:
        """
        Extract base document ID from chunk ID.
        
        For chunked IDs like '33997c75bf33_p1_c6', returns '33997c75bf33'.
        For memory UUIDs, returns the full UUID.
        """
        if not doc_id:
            return doc_id
        
        # Check if this looks like a UUID (memory ID)
        if len(doc_id) == 36 and doc_id.count('-') == 4:
            return doc_id
        
        # For document chunks, extract base ID before first underscore
        if '_' in doc_id:
            return doc_id.split('_')[0]
        
        return doc_id
    
    def _is_memory_source(self, chunk_data: Dict[str, Any]) -> bool:
        """
        Check if a chunk is from a memory source.
        
        Memory sources are identified by:
        1. doc_type: 'memory' in metadata
        2. doc_id in UUID format (36 chars with 4 dashes)
        
        Args:
            chunk_data: Document data dictionary with metadata
            
        Returns:
            True if this is a memory source, False otherwise
        """
        try:
            # Extract metadata from different possible structures
            metadata = {}
            if isinstance(chunk_data, dict):
                if 'metadata' in chunk_data:
                    metadata = chunk_data['metadata']
                else:
                    # Chunk_data might be the metadata itself
                    metadata = chunk_data
            elif hasattr(chunk_data, 'metadata'):
                metadata = chunk_data.metadata if isinstance(chunk_data.metadata, dict) else {}
            
            # Check doc_type first
            doc_type = metadata.get('doc_type', '')
            if doc_type == 'memory':
                return True
            
            # Check doc_id format (UUID pattern)
            doc_id = metadata.get('doc_id', '')
            if doc_id and len(doc_id) == 36 and doc_id.count('-') == 4:
                return True
            
            return False
        except Exception as e:
            self.logger.debug(f"[MEMORY_DETECTION] Error detecting memory source: {str(e)}")
            return False
    
    async def _retrieve_complete_document_chunks(
        self,
        collection,
        base_doc_id: str,
        query_embedding: List[float],
        search_params: Dict[str, Any],
        doc_filter: Optional[str],
        max_chunks: int
    ) -> List[tuple]:
        """
        Retrieve all chunks from a complete document.
        """
        try:
            # Create filter for all chunks from this document
            if len(base_doc_id) == 36 and base_doc_id.count('-') == 4:
                # Memory ID - exact match
                complete_doc_filter = f"doc_id == '{base_doc_id}'"
            else:
                # Document ID - prefix match for all chunks
                complete_doc_filter = f"doc_id like '{base_doc_id}%'"
            
            # Combine with existing doc_filter if present
            if doc_filter:
                complete_doc_filter = f"({doc_filter}) and ({complete_doc_filter})"
            
            # Search for all chunks from this document
            search_results = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=max_chunks,
                expr=complete_doc_filter,
                output_fields=["content", "doc_id", "source", "page", "doc_type", "uploaded_at", "section", "author", "hash"]
            )
            
            results = []
            for hits in search_results:
                for hit in hits:
                    content = hit.entity.get('content', '')
                    metadata = {
                        'doc_id': hit.entity.get('doc_id', ''),
                        'source': hit.entity.get('source', ''),
                        'page': hit.entity.get('page', 0),
                        'doc_type': hit.entity.get('doc_type', ''),
                        'uploaded_at': hit.entity.get('uploaded_at', ''),
                        'section': hit.entity.get('section', ''),
                        'author': hit.entity.get('author', ''),
                        'hash': hit.entity.get('hash', '')
                    }
                    
                    doc = {
                        'page_content': content,
                        'metadata': metadata
                    }
                    
                    results.append((doc, hit.score))
            
            self.logger.debug(f"[COMPLETE_DOC] Retrieved {len(results)} chunks for document {base_doc_id[:12]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"[COMPLETE_DOC] Error retrieving complete document {base_doc_id}: {str(e)}")
            return []
    
    async def _retrieve_neighboring_chunks(
        self,
        collection,
        chunk_id: str,
        radius: int,
        doc_filter: Optional[str]
    ) -> List[tuple]:
        """
        Retrieve neighboring chunks around a given chunk.
        
        For chunk IDs like 'doc_id_p1_c5', retrieves chunks c3, c4, c6, c7 (with radius=2).
        """
        try:
            import re
            
            neighbors = []
            
            # Parse chunk ID to extract base document and chunk info
            if '_' not in chunk_id:
                return neighbors  # Can't find neighbors for non-chunked IDs
            
            # Extract chunk number from IDs like 'doc_id_p1_c5'
            match = re.search(r'(.+_p\d+_c)(\d+)$', chunk_id)
            if not match:
                return neighbors
            
            chunk_prefix = match.group(1)  # 'doc_id_p1_c'
            chunk_num = int(match.group(2))  # 5
            
            # Generate neighboring chunk IDs
            neighbor_chunk_ids = []
            for offset in range(-radius, radius + 1):
                if offset == 0:  # Skip the original chunk
                    continue
                neighbor_num = chunk_num + offset
                if neighbor_num >= 0:  # Don't go below chunk 0
                    neighbor_id = f"{chunk_prefix}{neighbor_num}"
                    neighbor_chunk_ids.append(neighbor_id)
            
            if not neighbor_chunk_ids:
                return neighbors
            
            # Create filter for neighboring chunks
            neighbor_conditions = [f"doc_id == '{nid}'" for nid in neighbor_chunk_ids]
            neighbor_filter = " or ".join(neighbor_conditions)
            
            # Combine with existing doc_filter if present
            if doc_filter:
                neighbor_filter = f"({doc_filter}) and ({neighbor_filter})"
            
            # Search for neighboring chunks
            # Use the actual query embedding for consistency
            neighbor_embedding = query_embedding if query_embedding else [0.0] * 384
            
            search_results = collection.search(
                data=[neighbor_embedding],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=len(neighbor_chunk_ids),
                expr=neighbor_filter,
                output_fields=["content", "doc_id", "source", "page", "doc_type", "uploaded_at", "section", "author", "hash"]
            )
            
            for hits in search_results:
                for hit in hits:
                    content = hit.entity.get('content', '')
                    if content:  # Only include chunks with content
                        metadata = {
                            'doc_id': hit.entity.get('doc_id', ''),
                            'source': hit.entity.get('source', ''),
                            'page': hit.entity.get('page', 0),
                            'doc_type': hit.entity.get('doc_type', ''),
                            'uploaded_at': hit.entity.get('uploaded_at', ''),
                            'section': hit.entity.get('section', ''),
                            'author': hit.entity.get('author', ''),
                            'hash': hit.entity.get('hash', '')
                        }
                        
                        doc = {
                            'page_content': content,
                            'metadata': metadata
                        }
                        
                        # Assign a moderate score to neighboring chunks
                        neighbor_score = 0.7  # Fixed score for neighbors
                        neighbors.append((doc, neighbor_score))
            
            self.logger.debug(f"[NEIGHBORS] Retrieved {len(neighbors)} neighboring chunks for {chunk_id}")
            return neighbors
            
        except Exception as e:
            self.logger.error(f"[NEIGHBORS] Error retrieving neighbors for {chunk_id}: {str(e)}")
            return []

    async def _get_notebook_collections(
        self,
        db: Session,
        notebook_id: str,
        collection_filter: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Get collections and document IDs for a notebook.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            collection_filter: Optional filter for specific collections
            
        Returns:
            Dictionary mapping collection names to document ID lists
        """
        try:
            # Build query to get collections and document IDs from both documents and memories
            base_query = """
                SELECT DISTINCT milvus_collection, document_id, document_name, 'document' as source_type
                FROM notebook_documents
                WHERE notebook_id = :notebook_id 
                AND milvus_collection IS NOT NULL
                
                UNION ALL
                
                SELECT DISTINCT milvus_collection, memory_id as document_id, name as document_name, 'memory' as source_type  
                FROM notebook_memories
                WHERE notebook_id = :notebook_id 
                AND milvus_collection IS NOT NULL
            """
            
            params = {'notebook_id': notebook_id}
            
            if collection_filter:
                placeholders = ', '.join([f':col_{i}' for i in range(len(collection_filter))])
                base_query += f" AND milvus_collection IN ({placeholders})"
                for i, collection in enumerate(collection_filter):
                    params[f'col_{i}'] = collection
            
            query = text(base_query)
            result = db.execute(query, params)
            rows = result.fetchall()
            
            # Group by collection, preserving document names
            collections_map = {}
            document_names_map = {}  # Track document names separately
            for row in rows:
                collection = row.milvus_collection
                document_id = row.document_id
                document_name = row.document_name
                source_type = row.source_type
                
                if collection not in collections_map:
                    collections_map[collection] = []
                collections_map[collection].append(document_id)
                
                # Store document name with source type for later use
                document_names_map[document_id] = {
                    'name': document_name,
                    'type': source_type
                }
            
            self.logger.debug(f"Found collections for notebook {notebook_id}: {list(collections_map.keys())}")
            # Store document names map as instance variable for use in search
            self._current_document_names = document_names_map
            return collections_map
            
        except Exception as e:
            self.logger.error(f"Error getting notebook collections: {str(e)}")
            return {}
    
    async def _get_document_name(
        self,
        db: Session,
        notebook_id: str,
        document_id: str
    ) -> Optional[str]:
        """
        Get document or memory name from notebook_documents or notebook_memories table.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            document_id: Document ID or memory ID
            
        Returns:
            Document/memory name if found
        """
        try:
            # Try documents table first
            doc_query = text("""
                SELECT document_name
                FROM notebook_documents
                WHERE notebook_id = :notebook_id AND document_id = :document_id
            """)
            
            result = db.execute(doc_query, {
                'notebook_id': notebook_id,
                'document_id': document_id
            })
            
            row = result.fetchone()
            if row:
                return row.document_name
            
            # If not found in documents, try memories table
            memory_query = text("""
                SELECT name
                FROM notebook_memories
                WHERE notebook_id = :notebook_id AND memory_id = :memory_id
            """)
            
            result = db.execute(memory_query, {
                'notebook_id': notebook_id,
                'memory_id': document_id
            })
            
            row = result.fetchone()
            return row.name if row else None
            
        except Exception as e:
            self.logger.error(f"Error getting document/memory name: {str(e)}")
            return None
    
    async def get_available_collections(
        self,
        db: Session,
        notebook_id: str
    ) -> List[str]:
        """
        Get list of available collections for a notebook.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            
        Returns:
            List of collection names
        """
        try:
            query = text("""
                SELECT DISTINCT milvus_collection
                FROM notebook_documents
                WHERE notebook_id = :notebook_id 
                AND milvus_collection IS NOT NULL
                ORDER BY milvus_collection
            """)
            
            result = db.execute(query, {'notebook_id': notebook_id})
            rows = result.fetchall()
            
            return [row.milvus_collection for row in rows]
            
        except Exception as e:
            self.logger.error(f"Error getting available collections: {str(e)}")
            return []
    
    async def test_collection_connectivity(
        self,
        collection_name: str
    ) -> Dict[str, Any]:
        """
        Test connectivity to a Milvus collection.
        
        Args:
            collection_name: Collection name to test
            
        Returns:
            Test results
        """
        try:
            embedding_function = self._get_embedding_function()
            connection_args = self._get_milvus_connection_args()
            
            self.logger.info(f"Testing connectivity to collection {collection_name} with args: {connection_args}")
            
            # Try to connect directly to Milvus collection
            try:
                connections.connect(
                    alias=connection_args['alias'] + "_test",
                    uri=connection_args.get('uri'),
                    token=connection_args.get('token', ''),
                    host=connection_args.get('host'),
                    port=connection_args.get('port')
                )
                
                collection = Collection(collection_name, using=connection_args['alias'] + "_test")
                collection.load()
                self.logger.debug(f"[DEBUG] Successfully connected to collection {collection_name}")
            except Exception as connect_err:
                self.logger.error(f"[DEBUG] Failed to connect to collection: {str(connect_err)}")
                self.logger.error(f"[DEBUG] Connection traceback: {traceback.format_exc()}")
                raise
            
            # Try a simple test query
            self.logger.debug(f"[DEBUG] Running test connectivity query on collection {collection_name}")
            try:
                # Generate test query embedding
                if hasattr(embedding_function, 'embed_query'):
                    test_embedding = embedding_function.embed_query("test connectivity")
                elif hasattr(embedding_function, 'encode'):
                    test_embedding = embedding_function.encode(["test connectivity"])[0].tolist()
                else:
                    raise Exception(f"Unsupported embedding function type: {type(embedding_function)}")
                
                if not isinstance(test_embedding, list):
                    test_embedding = test_embedding.tolist()
                
                # Execute direct search
                search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                test_results = collection.search(
                    data=[test_embedding],
                    anns_field="vector",
                    param=search_params,
                    limit=1,
                    output_fields=["content", "doc_id"]
                )
                
                result_count = 0
                for hits in test_results:
                    result_count += len(hits)
                
                self.logger.debug(f"[DEBUG] Test query successful, got {result_count} results")
            except Exception as test_err:
                self.logger.error(f"[DEBUG] Test query failed: {str(test_err)}")
                self.logger.error(f"[DEBUG] Test query error type: {type(test_err).__name__}")
                self.logger.error(f"[DEBUG] Test query traceback: {traceback.format_exc()}")
                raise
            
            # Clean up test connection
            try:
                connections.disconnect(alias=connection_args['alias'] + "_test")
            except:
                pass  # Ignore cleanup errors
            
            return {
                'success': True,
                'collection': collection_name,
                'document_count': result_count,
                'connection_uri': connection_args.get('uri', 'unknown'),
                'message': 'Connection successful'
            }
            
        except Exception as e:
            self.logger.error(f"[DEBUG] Collection connectivity test failed: {str(e)}")
            self.logger.error(f"[DEBUG] Connectivity test error type: {type(e).__name__}")
            if isinstance(e, KeyError):
                self.logger.error(f"[DEBUG] KeyError during connectivity test - missing key: {str(e)}")
            self.logger.error(f"[DEBUG] Connectivity test traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'collection': collection_name,
                'document_count': 0,
                'connection_uri': connection_args.get('uri', 'unknown') if 'connection_args' in locals() else 'unknown',
                'error': str(e),
                'message': 'Connection failed'
            }
    
    def invalidate_cache(self):
        """Invalidate internal caches."""
        self._vector_settings = None
        self._embedding_settings = None
        self._embedding_function = None
        self._milvus_client = None
        self.logger.info("Notebook RAG service cache invalidated")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on RAG service components.
        
        Returns:
            Health status information
        """
        try:
            health_status = {
                'service': 'notebook_rag_service',
                'timestamp': datetime.now(),
                'components': {}
            }
            
            # Test vector database settings
            try:
                vector_settings = self._get_vector_settings()
                health_status['components']['vector_db'] = {
                    'status': 'healthy',
                    'host': vector_settings.get('host'),
                    'port': vector_settings.get('port')
                }
            except Exception as e:
                health_status['components']['vector_db'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Test embedding function
            try:
                embedding_function = self._get_embedding_function()
                health_status['components']['embeddings'] = {
                    'status': 'healthy',
                    'model': getattr(embedding_function, 'model_name', 'unknown')
                }
            except Exception as e:
                health_status['components']['embeddings'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Overall health
            component_statuses = [comp['status'] for comp in health_status['components'].values()]
            health_status['overall_status'] = 'healthy' if all(s == 'healthy' for s in component_statuses) else 'degraded'
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'service': 'notebook_rag_service',
                'timestamp': datetime.now(),
                'overall_status': 'error',
                'error': str(e)
            }