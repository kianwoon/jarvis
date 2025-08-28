"""
Notebook RAG service for querying documents within notebooks using Milvus vector store.
Integrates with existing vector database and embedding infrastructure.
"""

import logging
import traceback
import time
import hashlib
import re  # Needed for project extraction patterns
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.notebook_models import (
    NotebookRAGResponse, NotebookRAGSource, ProjectData
)
from app.services.ai_task_planner import TaskExecutionPlan, RetrievalStrategy, ai_task_planner
# from app.services.request_execution_state_tracker import (
#     check_operation_completed, mark_operation_completed, get_operation_result, ExecutionPhase
# )  # Removed - causing Redis async/sync errors
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.notebook_llm_settings_cache import get_notebook_llm_settings
from app.core.timeout_settings_cache import (
    get_list_cache_ttl, 
    get_notebook_rag_timeout,
    get_extraction_timeout,
    get_chunk_processing_timeout,
    calculate_dynamic_timeout
)
from app.core.redis_client import get_redis_client
from app.core.http_embedding_function import HTTPEmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import connections, Collection, utility
from app.core.notebook_llm_settings_cache import get_notebook_llm_full_config
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import json
import asyncio

logger = logging.getLogger(__name__)


class RetrievalIntensity(Enum):
    """Retrieval intensity levels for intelligent query processing."""
    MINIMAL = "minimal"      # Simple lookups: 5-10 sources, single strategy
    BALANCED = "balanced"    # Standard queries: 20 sources, 1-2 strategies  
    COMPREHENSIVE = "comprehensive"  # Complex analysis: 50+ sources, all strategies


@dataclass
class RetrievalPlan:
    """Plan for intelligent retrieval execution based on query complexity."""
    intensity: RetrievalIntensity
    max_sources: int
    use_multiple_strategies: bool
    use_full_scan: bool
    reasoning: str
    
    
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
        
        # Performance optimization: Cache LLM instances and configs
        self._cached_llm = None
        self._cached_llm_config_full = None
        self._cached_llm_config_hash = None
        
        # Timeout tracking for adaptive chunking
        self._recent_timeouts = []  # Track recent timeout events for adaptive chunking
        self._llm_cache_lock = asyncio.Lock()
        
        # Configuration cache to prevent repeated Redis calls
        self._cached_notebook_settings = None
        self._settings_cache_lock = asyncio.Lock()
    
    async def _get_cached_notebook_settings(self) -> dict:
        """Get cached notebook settings to prevent repeated Redis calls."""
        async with self._settings_cache_lock:
            if self._cached_notebook_settings is None:
                self._cached_notebook_settings = get_notebook_llm_settings()
                self.logger.debug("[SETTINGS_CACHE] Cached notebook settings to prevent repeated Redis calls")
            return self._cached_notebook_settings
    
    async def _get_cached_llm(self) -> Optional[OllamaLLM]:
        """
        Get cached LLM instance with configuration validation.
        Creates new instance only if config changed or none exists.
        Eliminates redundant Redis calls and LLM instantiations.
        """
        async with self._llm_cache_lock:
            try:
                # Return existing cached instance if we have one (skip config check during operation)
                if self._cached_llm and self._cached_llm_config_full:
                    return self._cached_llm
                
                # Only fetch config when no cache exists
                llm_config_full = get_notebook_llm_full_config()
                if not llm_config_full:
                    self.logger.warning("[LLM_CACHE] No notebook LLM config available")
                    return None
                
                # Create config hash for cache invalidation
                config_str = json.dumps(llm_config_full, sort_keys=True)
                config_hash = hashlib.md5(config_str.encode()).hexdigest()
                
                # Config changed or no cache - create new instance
                llm_config = llm_config_full.get('notebook_llm', {})
                if not llm_config:
                    self.logger.warning("[LLM_CACHE] Notebook LLM config not properly structured")
                    return None
                
                llm_config_obj = LLMConfig(
                    model_name=llm_config['model'],
                    temperature=float(llm_config['temperature']),
                    top_p=float(llm_config['top_p']),
                    max_tokens=int(llm_config['max_tokens'])
                )
                
                # Create new cached instance
                self._cached_llm = OllamaLLM(llm_config_obj)
                self._cached_llm_config_full = llm_config_full
                self._cached_llm_config_hash = config_hash
                
                self.logger.debug(f"[LLM_CACHE] Created new cached LLM instance with model={llm_config['model']}")
                return self._cached_llm
                
            except Exception as e:
                self.logger.error(f"[LLM_CACHE] Error getting cached LLM: {str(e)}", exc_info=True)
                self.logger.error(f"[LLM_CACHE] Context - model: {llm_config.get('model', 'unknown')}, endpoint: {llm_config.get('endpoint', 'unknown')}")
                # Clear broken cache
                self._cached_llm = None
                self._cached_llm_config_full = None
                self._cached_llm_config_hash = None
                return None
    
    def _clear_llm_cache(self):
        """Clear LLM cache - useful for testing or config changes"""
        self._cached_llm = None
        self._cached_llm_config_full = None
        self._cached_llm_config_hash = None
        self.logger.debug("[LLM_CACHE] Cleared LLM cache")
    
    async def _get_query_embedding(self, query: str, embedding_function: Any) -> List[float]:
        """
        Get query embedding.
        
        Args:
            query: Query string to embed
            embedding_function: Embedding function instance
            
        Returns:
            Query embedding vector
        """
        # Generate embedding
        try:
            if hasattr(embedding_function, 'embed_query'):
                # Check if it's async
                if asyncio.iscoroutinefunction(embedding_function.embed_query):
                    query_embedding = await embedding_function.embed_query(query)
                else:
                    query_embedding = embedding_function.embed_query(query)
            elif hasattr(embedding_function, 'encode'):
                # Check if it's async
                if asyncio.iscoroutinefunction(embedding_function.encode):
                    query_embedding = (await embedding_function.encode([query]))[0].tolist()
                else:
                    query_embedding = embedding_function.encode([query])[0].tolist()
            else:
                raise Exception(f"Unsupported embedding function type: {type(embedding_function)}")
            
            if not isinstance(query_embedding, list):
                query_embedding = query_embedding.tolist()
            
            return query_embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    async def _analyze_query_intent(self, query: str) -> dict:
        """
        Analyze query intent using AI-powered QueryIntentAnalyzer for intelligent understanding.
        
        Provides comprehensive intent analysis including:
        - Query type detection (comprehensive, filtered, specific)  
        - Quantity intent analysis (all, limited, few, single)
        - Confidence scoring and semantic understanding
        - Context-aware categorization with notebook context
        - Domain-specific intent recognition
        
        Args:
            query: The user query string
            
        Returns:
            dict: Query analysis results with intent classification and confidence
        """
        try:
            from app.services.query_intent_analyzer import analyze_query_intent
            
            # Use AI-powered intent analysis with notebook context
            intent_result = await analyze_query_intent(query)
            
            # Enhanced analysis with notebook-specific considerations
            wants_comprehensive = (
                intent_result.get('quantity_intent') == 'all' or 
                intent_result.get('scope') == 'comprehensive' or
                intent_result.get('completeness_preference') == 'thorough'
            )
            
            return {
                "wants_comprehensive": wants_comprehensive,
                "confidence": intent_result.get('confidence', 0.8),
                "query_type": intent_result.get('scope', 'filtered'),
                "quantity_intent": intent_result.get('quantity_intent', 'limited'),
                "user_type": intent_result.get('user_type', 'casual'),
                "completeness_preference": intent_result.get('completeness_preference', 'balanced'),
                "requires_deep_search": wants_comprehensive or intent_result.get('user_type') == 'researcher',
                "context": intent_result.get('context', {}),
                "reasoning": intent_result.get('reasoning', 'AI semantic analysis'),
                "urgency": intent_result.get('urgency', 'medium'),
                "ai_powered": True
            }
            
        except Exception as e:
            self.logger.warning(f"AI intent analysis failed, using enhanced semantic fallback: {str(e)}")
            
            # Enhanced semantic fallback with notebook context
            query_lower = query.lower()
            query_words = query.split()
            
            # Semantic indicators for comprehensive queries
            comprehensive_indicators = (
                any(word in query_lower for word in ['all', 'every', 'complete', 'comprehensive', 'overview', 'summary', 'entire']) or
                any(phrase in query_lower for phrase in ['give me everything', 'show me all', 'list everything', 'find all']) or
                len(query_words) > 10 or  # Complex queries often want comprehensive results
                any(word in query_lower for word in ['research', 'analyze', 'study', 'investigate'])
            )
            
            # Deep search indicators
            requires_deep_search = (
                comprehensive_indicators or
                any(word in query_lower for word in ['detailed', 'thorough', 'complete', 'comprehensive']) or
                '?' in query and len(query_words) > 5
            )
            
            return {
                "wants_comprehensive": comprehensive_indicators,
                "confidence": 0.7,  # Moderate confidence for enhanced semantic fallback
                "query_type": "comprehensive" if comprehensive_indicators else "filtered", 
                "quantity_intent": "all" if comprehensive_indicators else "limited",
                "requires_deep_search": requires_deep_search,
                "fallback_reason": str(e),
                "ai_powered": False,
                "reasoning": "Enhanced semantic fallback analysis"
            }
    
    async def plan_retrieval_strategy(self, message: str, intent: str) -> RetrievalPlan:
        """
        Phase 3: Intelligently plan retrieval approach based on query complexity.
        Following: Understand → Think → Plan → Do paradigm
        
        Determines HOW to retrieve when retrieval IS needed, choosing between:
        - MINIMAL: Simple lookups (5-10 sources, single strategy, fast)
        - BALANCED: Standard queries (20 sources, moderate strategies)  
        - COMPREHENSIVE: Complex analysis (50+ sources, full pipeline)
        
        Args:
            message: User query string
            intent: Intent classification result
            
        Returns:
            RetrievalPlan with optimal retrieval approach
        """
        message_lower = message.lower()
        
        # UNDERSTAND: Analyze query characteristics
        is_simple_lookup = (
            len(message.split()) <= 8 and
            any(word in message_lower for word in ['what is', 'tell me about', 'describe', 'explain'])
        )
        
        
        is_complex_analysis = any(word in message_lower for word in ['analyze', 'compare', 'relationship', 'pattern'])
        
        # THINK: Determine optimal approach based on complexity
        if is_simple_lookup:
            # PLAN: Minimal retrieval for simple questions
            return RetrievalPlan(
                intensity=RetrievalIntensity.MINIMAL,
                max_sources=8,
                use_multiple_strategies=False,
                use_full_scan=False,
                reasoning="Simple lookup query detected - using minimal retrieval for fast response"
            )
        
        elif is_complex_analysis:
            # PLAN: Comprehensive retrieval for complex analysis
            return RetrievalPlan(
                intensity=RetrievalIntensity.COMPREHENSIVE,
                max_sources=50,
                use_multiple_strategies=True,
                use_full_scan=True,
                reasoning="Complex analysis query detected - using comprehensive retrieval for complete results"
            )
        
        elif is_complex_analysis:
            # PLAN: Comprehensive retrieval for analysis
            return RetrievalPlan(
                intensity=RetrievalIntensity.COMPREHENSIVE,
                max_sources=40,
                use_multiple_strategies=True,
                use_full_scan=False,
                reasoning="Complex analysis query - using comprehensive retrieval with limited scan"
            )
        
        else:
            # PLAN: Balanced approach for general queries
            return RetrievalPlan(
                intensity=RetrievalIntensity.BALANCED,
                max_sources=20,
                use_multiple_strategies=True,
                use_full_scan=False,
                reasoning="General query - using balanced retrieval approach"
            )
    
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
                embedding_config = self._get_embedding_settings()
                
                # Check for HTTP embedding endpoint (match document.py pattern)
                embedding_endpoint = embedding_config.get('embedding_endpoint')
                embedding_model = embedding_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                
                if embedding_endpoint:
                    # Use HTTP embedding function (same as document.py)
                    try:
                        self._embedding_function = HTTPEmbeddingFunction(endpoint=embedding_endpoint)
                        self.logger.info(f"Initialized HTTP embedding function with endpoint: {embedding_endpoint}")
                    except Exception as http_err:
                        self.logger.error(f"[DEBUG] HTTPEmbeddingFunction creation failed: {str(http_err)}")
                        self.logger.error(f"[DEBUG] HTTPEmbeddingFunction traceback: {traceback.format_exc()}")
                        raise
                else:
                    # Use HuggingFace embeddings as fallback
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
            self.logger.error(f"Failed to get Milvus connection args: {str(e)}", exc_info=True)
            self.logger.error(f"[MILVUS_CONNECTION] Context - uri attempted: {uri if 'uri' in locals() else 'not_set'}, token present: {'token' in locals()}")
            # Fallback to legacy format for backwards compatibility
            vector_config = self._get_vector_settings()
            return {
                "host": vector_config.get('host', 'localhost'),
                "port": vector_config.get('port', 19530),
                "user": vector_config.get('user', ''),
                "password": vector_config.get('password', ''),
                "alias": "notebook_rag_fallback"
            }
    
    async def _perform_full_collection_scan(
        self,
        collection,
        doc_filter: str = None,
        limit: int = 10000,
        output_fields: List[str] = None
    ) -> List[dict]:
        """
        Perform full collection scan without similarity search filtering.
        
        This method bypasses semantic similarity by using collection.query() instead of
        collection.search(), ensuring ALL chunks are retrieved regardless of semantic
        similarity to the query. Essential for comprehensive analysis queries like
        "list all projects" that need to include older content (2000-2008) that may
        not match semantically.
        
        Args:
            collection: Milvus collection instance
            doc_filter: Optional document filter expression
            limit: Maximum number of results to return
            output_fields: Fields to include in results
            
        Returns:
            List of document chunks sorted by document recency
        """
        try:
            if output_fields is None:
                output_fields = ["content", "doc_id", "source", "page", "doc_type", "uploaded_at", "section", "author", "hash"]
            
            self.logger.info(f"[FULL_COLLECTION_SCAN] Performing full collection scan with filter: {doc_filter[:100] if doc_filter else 'None'}")
            
            # Use collection.query() with empty expression to get ALL documents
            # This bypasses similarity search limitations completely
            query_results = collection.query(
                expr=doc_filter if doc_filter else "",
                output_fields=output_fields,
                limit=limit
            )
            
            # Convert query results to consistent format
            scan_results = []
            for result in query_results:
                # Convert to format matching similarity search results
                scan_result = {
                    'content': result.get('content', ''),
                    'metadata': {
                        'doc_id': result.get('doc_id', ''),
                        'source': result.get('source', ''),
                        'page': result.get('page', 0),
                        'doc_type': result.get('doc_type', ''),
                        'uploaded_at': result.get('uploaded_at', ''),
                        'section': result.get('section', ''),
                        'author': result.get('author', ''),
                        'hash': result.get('hash', '')
                    },
                    'score': 1.0  # Uniform score since no similarity ranking
                }
                scan_results.append(scan_result)
            
            # Sort by document recency instead of similarity scores
            # Parse uploaded_at for proper date sorting
            def parse_upload_date(item):
                try:
                    uploaded_at = item['metadata'].get('uploaded_at', '')
                    if uploaded_at:
                        from datetime import datetime
                        return datetime.fromisoformat(uploaded_at.replace('Z', '+00:00'))
                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.debug(f"[SCAN_DATE_PARSE] Could not parse upload date for document: {uploaded_at}, error: {e}")
                except Exception as e:
                    self.logger.warning(f"[SCAN_DATE_PARSE] Unexpected error parsing upload date: {type(e).__name__}: {e}")
                return datetime.min  # Fallback for items without dates
            
            scan_results.sort(key=parse_upload_date, reverse=True)  # Newer first
            
            self.logger.info(f"[FULL_COLLECTION_SCAN] Retrieved {len(scan_results)} documents via full collection scan")
            return scan_results
            
        except Exception as e:
            self.logger.error(f"[FULL_COLLECTION_SCAN] Error performing full collection scan: {str(e)}")
            return []
    
    async def get_actual_content_count(
        self, 
        notebook_id: str, 
        query: str = None, 
        doc_filter: str = None,
        collection_name: str = None
    ) -> int:
        """
        Get exact count of matching content from Milvus.
        
        Args:
            notebook_id: Notebook ID
            query: Optional semantic query for filtering 
            doc_filter: Optional document filter expression
            collection_name: Optional specific collection to query
            
        Returns:
            Exact count of matching items
        """
        try:
            
            # Create cache key for this count request
            cache_params = {
                'notebook_id': notebook_id,
                'query': query or '',
                'doc_filter': doc_filter or '',
                'collection_name': collection_name or ''
            }
            cache_key_raw = f"notebook_content_count:{notebook_id}:" + hashlib.md5(
                str(cache_params).encode()
            ).hexdigest()
            
            # Check cache first
            try:
                redis_client = get_redis_client()
                if redis_client:
                    cached_count = redis_client.get(cache_key_raw)
                    if cached_count is not None:
                        count_value = int(cached_count)
                        return count_value
            except Exception as cache_err:
                self.logger.warning(f"[COUNT] Cache error (proceeding without cache): {str(cache_err)}")
            
            # Get collections to query
            db_session = None
            try:
                from app.core.db import SessionLocal
                db_session = SessionLocal()
                
                if collection_name:
                    # Query specific collection
                    collections_info = {collection_name: []}  # We'll get all docs from this collection
                    # Get document IDs for this notebook and collection
                    collections_info = await self._get_notebook_collections(db_session, notebook_id, [collection_name])
                else:
                    # Query all collections for this notebook
                    collections_info = await self._get_notebook_collections(db_session, notebook_id)
            finally:
                if db_session:
                    db_session.close()
            
            if not collections_info:
                return 0
            
            total_count = 0
            connection_args = self._get_milvus_connection_args()
            
            for collection_name, document_ids in collections_info.items():
                try:
                    
                    # Connect to Milvus collection
                    connections.connect(
                        alias=connection_args['alias'] + "_count",
                        uri=connection_args.get('uri'),
                        token=connection_args.get('token', ''),
                        host=connection_args.get('host'),
                        port=connection_args.get('port')
                    )
                    
                    collection = Collection(collection_name, using=connection_args['alias'] + "_count")
                    collection.load()
                    
                    # Build filter expression
                    count_filter = doc_filter
                    if document_ids:
                        # Create filter for notebook documents
                        conditions = []
                        for doc_id in document_ids:
                            # Check if this looks like a UUID (memory ID) - exact match
                            if len(doc_id) == 36 and doc_id.count('-') == 4:
                                conditions.append(f"doc_id == '{doc_id}'")
                            else:
                                # Document ID - use prefix matching for chunks
                                conditions.append(f"doc_id like '{doc_id}%'")
                        
                        notebook_filter = " or ".join(conditions)
                        
                        if count_filter:
                            count_filter = f"({count_filter}) and ({notebook_filter})"
                        else:
                            count_filter = notebook_filter
                    
                    # Get count of entities
                    if count_filter:
                        # For filtered counts, we need to do actual query and count results
                        # Use a reasonable limit to estimate count efficiently
                        results = collection.query(
                            expr=count_filter,
                            output_fields=["doc_id"],  # Minimal field to reduce data transfer
                            limit=10000  # Set high limit to get accurate count for most cases
                        )
                        collection_count = len(results)
                        
                        # If we hit the limit, we know there are at least this many
                        if len(results) == 10000:
                            self.logger.debug(f"[COUNT] Collection {collection_name}: {collection_count}+ entities (hit limit)")
                        else:
                            self.logger.debug(f"[COUNT] Collection {collection_name}: {collection_count} entities (exact count)")
                    else:
                        # For total count without filters, use num_entities (fast)
                        try:
                            collection_count = collection.num_entities
                            self.logger.debug(f"[COUNT] Collection {collection_name}: {collection_count} total entities")
                        except Exception as num_entities_error:
                            self.logger.warning(f"[COUNT] num_entities failed for {collection_name}, using query fallback: {num_entities_error}")
                            # Fallback to query approach
                            results = collection.query(
                                expr="",
                                output_fields=["doc_id"],
                                limit=10000
                            )
                            collection_count = len(results)
                            if len(results) == 10000:
                                self.logger.debug(f"[COUNT] Collection {collection_name}: {collection_count}+ entities (hit limit, fallback)")
                            else:
                                self.logger.debug(f"[COUNT] Collection {collection_name}: {collection_count} entities (exact count, fallback)")
                    total_count += collection_count
                    
                    
                    # Clean up connection
                    try:
                        connections.disconnect(alias=connection_args['alias'] + "_count")
                    except Exception as cleanup_err:
                        self.logger.debug(f"[COUNT_CLEANUP] Failed to disconnect from {collection_name}: {cleanup_err}")
                        
                except Exception as collection_err:
                    self.logger.warning(f"[COUNT] Error counting collection {collection_name}: {str(collection_err)}")
                    self.logger.warning(f"[COUNT] Collection error type: {type(collection_err).__name__}")
                    # Add debug info about the collection
                    try:
                        self.logger.warning(f"[COUNT] Collection {collection_name} had {len(document_ids)} document IDs to filter")
                        if document_ids:
                            self.logger.warning(f"[COUNT] Sample document IDs: {document_ids[:3]}")
                    except Exception as debug_err:
                        self.logger.debug(f"[COUNT] Failed to log debug info for {collection_name}: {debug_err}")
                    # Continue with other collections
                    continue
            
            # Cache the result with 5-minute TTL
            try:
                redis_client = get_redis_client()
                if redis_client:
                    cache_ttl = get_list_cache_ttl()  # 300 seconds (5 minutes)
                    redis_client.setex(cache_key_raw, cache_ttl, str(total_count))
            except Exception as cache_err:
                self.logger.warning(f"[COUNT] Failed to cache result: {str(cache_err)}")
            
            # Add detailed logging for debugging
            if total_count == 0 and len(collections_info) > 0:
                self.logger.warning(f"[COUNT] Suspicious: {len(collections_info)} collections found but total_count is 0")
                self.logger.warning(f"[COUNT] Collections checked: {list(collections_info.keys())}")
                for col_name, doc_ids in collections_info.items():
                    self.logger.warning(f"[COUNT] Collection {col_name} has document IDs: {doc_ids[:5]}{'...' if len(doc_ids) > 5 else ''}")
            
            self.logger.info(f"[COUNT] Notebook {notebook_id} has {total_count} matching items")
            return total_count
            
        except Exception as e:
            self.logger.error(f"[COUNT] Error getting content count for notebook {notebook_id}: {str(e)}")
            # Return conservative fallback estimate
            return 25  # Conservative estimate for planning purposes

    async def get_notebook_total_items(self, notebook_id: str) -> int:
        """
        Get total number of items in notebook (all collections).
        
        Args:
            notebook_id: Notebook ID
            
        Returns:
            Total count of all items across collections
        """
        return await self.get_actual_content_count(notebook_id)
    
    async def query_notebook_adaptive(
        self,
        db: Session,
        notebook_id: str,
        query: str,
        max_sources: Optional[int] = None,
        include_metadata: bool = True,
        collection_filter: Optional[List[str]] = None,
        force_simple_retrieval: bool = False
    ) -> NotebookRAGResponse:
        """
        Perform adaptive RAG query with AI-powered intent analysis and dynamic limit optimization.
        
        This method provides truly adaptive retrieval by:
        1. Using AI to understand user intent and scope preferences
        2. Dynamically counting available content in the notebook  
        3. Setting retrieval limits based on actual content availability
        4. Supporting unlimited retrieval for comprehensive queries when content supports it
        
        Args:
            db: Database session
            notebook_id: Notebook ID to query
            query: Search query  
            max_sources: Optional hint for maximum sources (will be optimized)
            include_metadata: Whether to include metadata
            collection_filter: Optional filter for specific collections
            force_simple_retrieval: If True, skip intelligent planning and use simple RAG only
            
        Returns:
            RAG query results with adaptive optimization applied
        """
        # Parameter validation guards
        validation_error = self._validate_query_parameters(db, notebook_id, query, max_sources, collection_filter)
        if validation_error:
            return validation_error
            
        try:
            self.logger.info(f"[ADAPTIVE_QUERY] Starting adaptive query for notebook {notebook_id}: '{query[:50]}...'")
            
            # Removed execution state tracking to fix Redis async/sync errors
            
            # Step 1: Analyze query intent using AI
            intent_analysis = await self._analyze_query_intent(query)
            wants_comprehensive = intent_analysis.get("wants_comprehensive", False)
            quantity_intent = intent_analysis.get("quantity_intent", "limited")
            confidence = intent_analysis.get("confidence", 0.5)
            
            self.logger.info(f"[ADAPTIVE_QUERY] Intent analysis - comprehensive: {wants_comprehensive}, quantity: {quantity_intent}, confidence: {confidence:.2f}")
            
            # Step 1.2: Check if query should use intelligent task planning
            # Use intelligent planning for complex queries that would benefit from multi-strategy retrieval
            # Skip if force_simple_retrieval is True (e.g., when called as timeout fallback)
            should_use_intelligent_planning = (
                not force_simple_retrieval and (
                    wants_comprehensive or 
                    quantity_intent == "all" or 
                    confidence < 0.7 or  # Low confidence suggests complex intent
                    len(query.split()) > 8  # Complex multi-part queries
                )
            )
            
            if should_use_intelligent_planning:
                self.logger.info(f"[ADAPTIVE_QUERY] Using intelligent task planning for complex query")
                
                # Removed execution state tracking to fix Redis async/sync errors
                
                try:
                    # Create AI task plan for this query
                    execution_plan = await ai_task_planner.understand_and_plan(query)
                    self.logger.info(f"[ADAPTIVE_QUERY] Generated plan: {execution_plan.intent_type} with {len(execution_plan.retrieval_strategies)} strategies")
                    
                    # Execute the intelligent plan
                    result = await self.execute_intelligent_plan(
                        db=db,
                        notebook_id=notebook_id,
                        plan=execution_plan,
                        include_metadata=include_metadata,
                        collection_filter=collection_filter
                    )
                    
                    self.logger.info(f"[ADAPTIVE_QUERY] Intelligent plan executed: {len(result.sources)} sources returned")
                    return result
                    
                except Exception as e:
                    self.logger.warning(f"[ADAPTIVE_QUERY] Intelligent planning failed, falling back to traditional approach: {str(e)}")
                    # Continue with traditional approach below
            
            
            # Step 2: Get actual content count for dynamic limit calculation
            total_available = await self.get_actual_content_count(notebook_id, query)
            self.logger.info(f"[ADAPTIVE_QUERY] Total available content: {total_available} items")
            
            # Step 3: Calculate adaptive retrieval limit based on intent and available content
            if max_sources is None:
                max_sources = 10  # Default starting point
            
            if (wants_comprehensive and quantity_intent == "all") or (intent_analysis.get('scope') == 'comprehensive' and intent_analysis.get('quantity') == 'all'):
                # User wants comprehensive results - provide all available content up to reasonable limits
                if total_available <= 50:
                    adaptive_limit = total_available  # Give them everything for small notebooks
                elif total_available <= 200:
                    adaptive_limit = min(total_available, int(total_available * 0.95))  # 95% of available
                elif total_available <= 500:
                    adaptive_limit = min(int(total_available * 0.8), 400)  # 80% up to 400
                else:
                    adaptive_limit = min(500, int(total_available * 0.6))  # 60% capped at 500
                
                self.logger.info(f"[ADAPTIVE_QUERY] Comprehensive query: providing {adaptive_limit} sources from {total_available} available")
                self.logger.info(f"[DEBUG_LIMIT] wants_comprehensive={wants_comprehensive}, quantity_intent={quantity_intent}, adaptive_limit={adaptive_limit}, max_sources={max_sources}, top_k={top_k}")
                
            elif quantity_intent == "limited" or intent_analysis.get("user_type") == "researcher":
                # Moderate coverage for researchers or limited queries
                if total_available <= 30:
                    adaptive_limit = total_available
                elif total_available <= 100:
                    adaptive_limit = min(int(total_available * 0.6), 60)
                else:
                    adaptive_limit = min(100, int(total_available * 0.4))
                    
            elif quantity_intent in ["few", "single"]:
                # Focused results for specific queries
                # But don't limit comprehensive scope queries even if quantity is "few"
                if intent_analysis.get('query_type') == 'comprehensive':
                    adaptive_limit = min(max_sources, total_available)
                else:
                    adaptive_limit = min(max_sources, 20, total_available)
                
            else:
                # Balanced approach for general queries
                if total_available <= 20:
                    adaptive_limit = total_available
                else:
                    base_multiplier = 3 if confidence > 0.8 else 2
                    adaptive_limit = min(max_sources * base_multiplier, int(total_available * 0.5), 150)
            
            # Apply confidence-based adjustments
            if confidence < 0.6:
                adaptive_limit = min(adaptive_limit, int(adaptive_limit * 0.8))
            elif confidence > 0.9:
                adaptive_limit = min(int(adaptive_limit * 1.3), total_available)
            
            # Ensure minimum viable results
            adaptive_limit = max(adaptive_limit, 3)
            
            self.logger.info(f"[ADAPTIVE_QUERY] Optimized retrieval limit: {adaptive_limit} (original: {max_sources}, available: {total_available})")
            
            # Step 4: Perform the query with optimized parameters
            result = await self.query_notebook(
                db=db,
                notebook_id=notebook_id,
                query=query,
                top_k=adaptive_limit,
                include_metadata=include_metadata,
                collection_filter=collection_filter
            )
            
            # Step 5: Enhance response with adaptive metadata
            if hasattr(result, 'metadata'):
                if result.metadata is None:
                    result.metadata = {}
            else:
                result.metadata = {}
                
            result.metadata.update({
                "adaptive_retrieval": True,
                "intent_analysis": {
                    "wants_comprehensive": wants_comprehensive,
                    "quantity_intent": quantity_intent,
                    "confidence": confidence,
                    "reasoning": intent_analysis.get("reasoning", ""),
                    "ai_powered": intent_analysis.get("ai_powered", False)
                },
                "dynamic_limits": {
                    "requested_max": max_sources,
                    "adaptive_limit": adaptive_limit,
                    "total_available": total_available,
                    "utilization_percent": round((len(result.sources) / total_available) * 100, 1) if total_available > 0 else 0
                }
            })
            
            self.logger.info(f"[ADAPTIVE_QUERY] Completed adaptive query: returned {len(result.sources)} sources with {result.metadata['dynamic_limits']['utilization_percent']}% utilization")
            
            return result
            
        except Exception as e:
            self.logger.error(f"[ADAPTIVE_QUERY] Adaptive query failed: {str(e)}", exc_info=True)
            
            # Defensive parameter validation and fallback
            try:
                # Conservative fallback based on typical query needs with parameter validation
                fallback_limit = self._get_safe_fallback_limit(max_sources)
                
                self.logger.info(f"[ADAPTIVE_QUERY] Using fallback query with limit: {fallback_limit}")
                return await self.query_notebook(
                    db=db,
                    notebook_id=notebook_id,
                    query=query,
                    top_k=fallback_limit,
                    include_metadata=include_metadata,
                    collection_filter=collection_filter
                )
            except Exception as fallback_error:
                self.logger.error(f"[ADAPTIVE_QUERY] Fallback query also failed: {fallback_error}", exc_info=True)
                # Return empty response instead of crashing
                return NotebookRAGResponse(
                    response="I apologize, but I encountered an error processing your query. Please try again with a simpler question.",
                    sources=[],
                    metadata={"error": "adaptive_query_fallback_failed", "original_error": str(e)}
                )
    
    def _get_safe_fallback_limit(self, max_sources: Optional[int]) -> int:
        """
        Safely calculate fallback limit with parameter validation.
        
        Args:
            max_sources: User-requested max sources (may be None or invalid)
            
        Returns:
            int: Safe fallback limit for query
        """
        try:
            # Validate and sanitize max_sources
            if max_sources is None:
                return 10  # Safe default
            
            if not isinstance(max_sources, int):
                self.logger.warning(f"[DEFENSIVE_CODING] max_sources is not int: {type(max_sources)}, using default")
                return 10
            
            if max_sources <= 0:
                self.logger.warning(f"[DEFENSIVE_CODING] max_sources is non-positive: {max_sources}, using default")
                return 10
            
            # Cap at reasonable limit to prevent system overload
            safe_limit = min(max_sources, 30)
            self.logger.debug(f"[DEFENSIVE_CODING] Calculated safe fallback limit: {safe_limit}")
            return safe_limit
            
        except Exception as e:
            self.logger.error(f"[DEFENSIVE_CODING] Error calculating fallback limit: {e}, using default")
            return 10
    
    def _validate_query_parameters(
        self, 
        db: Session, 
        notebook_id: str, 
        query: str, 
        max_sources: Optional[int], 
        collection_filter: Optional[List[str]]
    ) -> Optional[NotebookRAGResponse]:
        """
        Validate query parameters and return error response if invalid.
        
        Args:
            db: Database session
            notebook_id: Notebook ID
            query: Query string
            max_sources: Max sources parameter
            collection_filter: Collection filter
            
        Returns:
            NotebookRAGResponse with error if validation fails, None if valid
        """
        try:
            # Validate database session
            if db is None:
                self.logger.error("[VALIDATION] Database session is None")
                return NotebookRAGResponse(
                    response="Database connection error. Please try again.",
                    sources=[],
                    metadata={"error": "invalid_db_session"}
                )
            
            # Validate notebook_id
            if not notebook_id or not isinstance(notebook_id, str) or len(notebook_id.strip()) == 0:
                self.logger.error(f"[VALIDATION] Invalid notebook_id: {notebook_id}")
                return NotebookRAGResponse(
                    response="Invalid notebook identifier. Please check your request.",
                    sources=[],
                    metadata={"error": "invalid_notebook_id"}
                )
            
            # Validate query
            if not query or not isinstance(query, str) or len(query.strip()) == 0:
                self.logger.error(f"[VALIDATION] Invalid query: {query}")
                return NotebookRAGResponse(
                    response="Please provide a valid search query.",
                    sources=[],
                    metadata={"error": "invalid_query"}
                )
            
            # Validate max_sources if provided
            if max_sources is not None:
                if not isinstance(max_sources, int) or max_sources <= 0:
                    self.logger.warning(f"[VALIDATION] Invalid max_sources: {max_sources}, will use default")
                    # Don't return error, just log - will be handled by _get_safe_fallback_limit
                elif max_sources > 1000:  # Prevent abuse
                    self.logger.warning(f"[VALIDATION] max_sources too large: {max_sources}, will be capped")
            
            # Validate collection_filter if provided
            if collection_filter is not None:
                if not isinstance(collection_filter, list):
                    self.logger.warning(f"[VALIDATION] collection_filter is not list: {type(collection_filter)}")
                    # Don't error, just ignore the filter
                else:
                    # Filter out invalid collection names
                    valid_collections = []
                    for col in collection_filter:
                        if isinstance(col, str) and len(col.strip()) > 0:
                            valid_collections.append(col.strip())
                        else:
                            self.logger.warning(f"[VALIDATION] Invalid collection name in filter: {col}")
                    
                    if len(valid_collections) != len(collection_filter):
                        self.logger.info(f"[VALIDATION] Cleaned collection_filter: {len(collection_filter)} -> {len(valid_collections)}")
            
            # All validations passed
            return None
            
        except Exception as e:
            self.logger.error(f"[VALIDATION] Parameter validation error: {e}", exc_info=True)
            return NotebookRAGResponse(
                response="Parameter validation error. Please try again.",
                sources=[],
                metadata={"error": "validation_exception", "details": str(e)}
            )
    
    async def estimate_query_matches(
        self, 
        notebook_id: str, 
        query: str,
        confidence_threshold: float = 0.5
    ) -> dict:
        """
        Estimate how many items would match a semantic query.
        
        Args:
            notebook_id: Notebook ID
            query: Semantic query string
            confidence_threshold: Minimum confidence threshold for matches
            
        Returns:
            Dictionary with total_available, estimated_matches, confidence
        """
        try:
            
            # Get total available items
            total_available = await self.get_notebook_total_items(notebook_id)
            
            if total_available == 0:
                return {
                    "total_available": 0,
                    "estimated_matches": 0,
                    "confidence": 1.0,
                    "query": query
                }
            
            # For semantic queries, estimate based on query characteristics
            query_lower = query.lower()
            
            # Simple heuristic-based estimation
            if any(keyword in query_lower for keyword in ['all', 'every', 'complete', 'entire', 'full']):
                # Comprehensive queries likely match most content
                estimated_matches = int(total_available * 0.8)
                confidence = 0.9
            elif any(keyword in query_lower for keyword in ['list', 'show', 'find']):
                # Listing queries typically match moderate amounts
                estimated_matches = int(total_available * 0.6)
                confidence = 0.7
            elif len(query_lower.split()) <= 2:
                # Short queries tend to match more broadly
                estimated_matches = int(total_available * 0.4)
                confidence = 0.6
            else:
                # Specific/complex queries match fewer items
                estimated_matches = min(int(total_available * 0.2), 20)
                confidence = 0.5
            
            # Apply confidence threshold adjustment
            if confidence < confidence_threshold:
                # If confidence is low, be more conservative
                estimated_matches = min(estimated_matches, 10)
            
            result = {
                "total_available": total_available,
                "estimated_matches": max(1, estimated_matches),  # At least 1
                "confidence": confidence,
                "query": query
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error estimating query matches: {str(e)}")
            # Return conservative fallback based on typical notebook sizes
            return {
                "total_available": 30,  # Conservative estimate for small notebooks
                "estimated_matches": 5,
                "confidence": 0.3,
                "query": query
            }
    
    # REMOVED: query_notebook_comprehensive method - caused degraded results
    # Rolled back to restore previous working system with rich project details


    async def query_notebook(
        self,
        db: Session,
        notebook_id: str,
        query: str,
        top_k: int = 5,
        include_metadata: bool = True,
        collection_filter: Optional[List[str]] = None,
        skip_extraction: bool = False
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
                    collections_searched=[],
                    extracted_projects=None
                )
            
            # Query each collection and aggregate results
            all_sources = []
            collections_searched = set()
            
            embedding_function = self._get_embedding_function()
            connection_args = self._get_milvus_connection_args()
            
            for collection_name, document_ids in collections_info.items():
                try:
                    # Connect to Milvus using direct pymilvus client
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
                        # Successfully connected to collection
                    except Exception as milvus_err:
                        self.logger.error(f"[DEBUG] Failed to connect to Milvus collection: {str(milvus_err)}")
                        self.logger.error(f"[DEBUG] Milvus connection traceback: {traceback.format_exc()}")
                        raise
                    
                    # Get query embedding with execution state tracking
                    try:
                        query_embedding = await self._get_query_embedding(query, embedding_function)
                        
                        # Generated embedding successfully
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
                    
                    # Get configurable retrieval limits with dynamic content-aware defaults
                    try:
                        notebook_settings = await self._get_cached_notebook_settings()
                        base_max_retrieval_chunks = notebook_settings.get('notebook_llm', {}).get('max_retrieval_chunks', None)
                        retrieval_multiplier = notebook_settings.get('notebook_llm', {}).get('retrieval_multiplier', 3)
                        # Document-aware retrieval settings
                        include_neighboring_chunks = notebook_settings.get('notebook_llm', {}).get('include_neighboring_chunks', True)
                        neighbor_chunk_radius = notebook_settings.get('notebook_llm', {}).get('neighbor_chunk_radius', 2)
                        document_completeness_threshold = notebook_settings.get('notebook_llm', {}).get('document_completeness_threshold', 0.8)
                        enable_document_aware_retrieval = notebook_settings.get('notebook_llm', {}).get('enable_document_aware_retrieval', True)
                    except Exception as e:
                        self.logger.warning(f"Could not load notebook settings, using defaults: {e}")
                        base_max_retrieval_chunks = None  # Will be calculated dynamically
                        retrieval_multiplier = 3    # Default multiplier
                        # Document-aware retrieval defaults
                        include_neighboring_chunks = True
                        neighbor_chunk_radius = 2
                        document_completeness_threshold = 0.8
                        enable_document_aware_retrieval = True
                    
                    # Calculate dynamic max_retrieval_chunks based on actual content
                    if base_max_retrieval_chunks is None:
                        # Dynamic calculation based on available content
                        try:
                            total_notebook_items = await self.get_actual_content_count(notebook_id)
                            # GENERAL PURPOSE: Always use all available items when user wants completeness
                            max_retrieval_chunks = total_notebook_items
                            
                        except Exception as count_err:
                            self.logger.warning(f"Could not get content count for dynamic limits: {count_err}")
                            # GENERAL PURPOSE: If we can't count, retrieve without limits
                            max_retrieval_chunks = 10000  # High limit to ensure no artificial constraint
                    else:
                        max_retrieval_chunks = base_max_retrieval_chunks
                    
                    # Analyze query intent and adjust retrieval parameters intelligently
                    intent_analysis = await self._analyze_query_intent(query)
                    is_comprehensive_query = intent_analysis.get("wants_comprehensive", False)
                    
                    if is_comprehensive_query:
                        # AI-driven parameter optimization for comprehensive queries
                        original_multiplier = retrieval_multiplier
                        original_max_chunks = max_retrieval_chunks
                        
                        # Scale multiplier based on content size and intent confidence
                        confidence = intent_analysis.get("confidence", 0.8)
                        if confidence > 0.9:
                            retrieval_multiplier = max(retrieval_multiplier * 3, 6)  # High confidence = more aggressive
                        else:
                            retrieval_multiplier = max(retrieval_multiplier * 2, 4)  # Moderate confidence = moderate scaling
                        
                        # Scale max_chunks based on actual available content (no hardcoded caps)
                        try:
                            total_available = await self.get_actual_content_count(notebook_id)
                            # GENERAL PURPOSE: For comprehensive queries, use ALL available items regardless of size
                            max_retrieval_chunks = total_available
                                
                            self.logger.info(f"[COMPREHENSIVE_QUERY] Intelligent analysis detected comprehensive query: '{query[:100]}...' - Enhanced retrieval limits based on {total_available} available items: multiplier {original_multiplier}→{retrieval_multiplier}, max_chunks {original_max_chunks}→{max_retrieval_chunks}")
                        except Exception as count_err:
                            # GENERAL PURPOSE: If we can't count, remove all limits for comprehensive queries
                            max_retrieval_chunks = 10000  # High limit to ensure completeness
                            self.logger.warning(f"Could not get content count for comprehensive scaling, removing limits: {max_retrieval_chunks}")
                    
                    limit = min(top_k * retrieval_multiplier, max_retrieval_chunks)
                    
                    # Executing collection search with calculated limits
                    if doc_filter:
                        # Applying document filter
                        expr = doc_filter
                    else:
                        expr = ""
                    
                    try:
                        # Check if this is a comprehensive query that should bypass similarity search
                        quantity_intent = intent_analysis.get("quantity_intent", "limited")
                        should_use_full_scan = (
                            quantity_intent == "all" and 
                            is_comprehensive_query and
                            intent_analysis.get("confidence", 0.0) > 0.7
                        )
                        
                        if should_use_full_scan:
                            # Use full collection scan to bypass similarity limitations
                            self.logger.info(f"[FULL_COLLECTION_SCAN] Detected comprehensive query - bypassing similarity search for complete results")
                            # For comprehensive queries, remove limit to get ALL available documents
                            comprehensive_limit = max(limit, max_retrieval_chunks, 100)  # Ensure we get comprehensive results
                            self.logger.info(f"[FULL_COLLECTION_SCAN] Using comprehensive limit: {comprehensive_limit} (original: {limit})")
                            initial_results = await self._perform_full_collection_scan(
                                collection=collection,
                                doc_filter=doc_filter,
                                limit=comprehensive_limit,
                                output_fields=["content", "doc_id", "source", "page", "doc_type", "uploaded_at", "section", "author", "hash"]
                            )
                            self.logger.info(f"[BOTTLENECK_DEBUG] After full collection scan: {len(initial_results)} initial_results")
                        else:
                            # Use standard similarity search with Redis vector caching
                            
                            # Generate cache key for vector results
                            vector_cache_key = await self._get_vector_cache_key(notebook_id, query, limit)
                            
                            # Check cache first
                            cached_vector_results = await self._get_cached_vector_results(vector_cache_key)
                            
                            if cached_vector_results:
                                # Cache hit - use cached results
                                self.logger.info(f"[VECTOR_CACHE] Cache HIT - Retrieved {len(cached_vector_results)} cached vector results")
                                
                                # Convert cached results to our format
                                initial_results = []
                                for cached_hit in cached_vector_results:
                                    content = cached_hit.get('content', '')
                                    metadata = {
                                        'doc_id': cached_hit.get('doc_id', ''),
                                        'source': cached_hit.get('source', ''),
                                        'page': cached_hit.get('page', 0),
                                        'doc_type': cached_hit.get('doc_type', ''),
                                        'uploaded_at': cached_hit.get('uploaded_at', ''),
                                        'section': cached_hit.get('section', ''),
                                        'author': cached_hit.get('author', ''),
                                        'hash': cached_hit.get('hash', '')
                                    }
                                    
                                    # Create a document-like object that matches our processing logic
                                    doc = {
                                        'page_content': content,
                                        'metadata': metadata
                                    }
                                    
                                    score = cached_hit.get('distance', 0.0)
                                    initial_results.append((doc, score))
                            else:
                                # Cache miss - perform Milvus search
                                self.logger.info(f"[VECTOR_CACHE] Cache MISS - Performing Milvus search")
                                
                                search_results = collection.search(
                                    data=[query_embedding],
                                    anns_field="vector",
                                    param=search_params,
                                    limit=limit,
                                    expr=doc_filter,
                                    output_fields=["content", "doc_id", "source", "page", "doc_type", "uploaded_at", "section", "author", "hash"]
                                )
                                
                                # Cache the raw search results immediately after successful retrieval
                                await self._cache_vector_results(vector_cache_key, search_results, ttl=3600)
                                self.logger.info(f"[VECTOR_CACHE] Cached {len(search_results)} vector results")
                                
                                # Convert similarity search results to our format
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
                        
                        # Convert full collection scan results to processing format if needed
                        if should_use_full_scan and initial_results:
                            # Convert full scan results to the expected format (doc, score) tuples
                            converted_results = []
                            for result in initial_results:
                                if isinstance(result, dict) and 'content' in result:
                                    # Convert from full scan format to processing format
                                    doc = {
                                        'page_content': result['content'],
                                        'metadata': result['metadata']
                                    }
                                    score = result.get('score', 1.0)
                                    converted_results.append((doc, score))
                                else:
                                    # Already in correct format
                                    converted_results.append(result)
                            initial_results = converted_results
                            self.logger.info(f"[BOTTLENECK_DEBUG] After conversion: {len(initial_results)} initial_results")
                        
                        # Apply document-aware retrieval logic
                        if enable_document_aware_retrieval and initial_results:
                            self.logger.info(f"[BOTTLENECK_DEBUG] Before document-aware: {len(initial_results)} initial_results")
                            # Determine if this is a listing/comprehensive query for document-aware processing
                            is_listing_query = intent_analysis.get("quantity_intent") == "all" and is_comprehensive_query
                            
                            results = await self._apply_document_aware_retrieval(
                                initial_results, 
                                collection,
                                query_embedding,
                                search_params,
                                doc_filter,
                                include_neighboring_chunks,
                                neighbor_chunk_radius,
                                document_completeness_threshold,
                                max_retrieval_chunks,
                                query,
                                is_listing_query
                            )
                        else:
                            results = initial_results
                        
                        # Direct search completed successfully
                    except Exception as search_err:
                        self.logger.error(f"[DEBUG] Direct Milvus search failed: {str(search_err)}")
                        self.logger.error(f"[DEBUG] Search error type: {type(search_err).__name__}")
                        self.logger.error(f"[DEBUG] Search traceback: {traceback.format_exc()}")
                        raise
                    
                    collections_searched.add(collection_name)
                    self.logger.debug(f"Retrieved {len(results)} results from collection {collection_name}")
                    
                    # Convert results to our format
                    for i, (doc, score) in enumerate(results):
                        try:
                            # Processing result
                            
                            # Extract document metadata with robust handling
                            doc_metadata = {}
                            if hasattr(doc, 'metadata') and doc.metadata:
                                doc_metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
                            elif isinstance(doc, dict) and 'metadata' in doc:
                                doc_metadata = doc['metadata'] if isinstance(doc['metadata'], dict) else {}
                            
                            document_id = doc_metadata.get('doc_id', 'unknown')
                            
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
                                    continue
                            
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
                                            # Found exact memory match
                                            break
                                # Try prefix matching for document chunks 
                                elif '_' in document_id:
                                    # For document chunks like '33997c75bf33_p0_c0', try base ID '33997c75bf33'  
                                    base_id = document_id.split('_')[0]
                                    document_info = cached_names.get(base_id, {})
                                    if document_info:
                                        # Found chunk base match
                                        pass
                            
                            document_name = document_info.get('name')
                            source_type = document_info.get('type', 'document')
                            
                            # CRITICAL FIX: If memory source has no name, provide fallback to prevent filtering
                            if source_type == 'memory' and not document_name:
                                self.logger.warning(f"[MEMORY_FIX] Memory {document_id} has no name in cached results")
                                self.logger.warning(f"[MEMORY_FIX] Available cached names: {list(cached_names.keys())}")
                                document_name = f"Memory ({document_id[:8]}...)"
                            # Memory document name resolution completed
                            
                            
                            # Robust content extraction - handle different document formats
                            content = None
                            
                            try:
                                if hasattr(doc, 'page_content'):
                                    content = doc.page_content
                                elif hasattr(doc, 'content'):
                                    content = doc.content
                                elif hasattr(doc, 'text'):
                                    content = doc.text
                                elif isinstance(doc, dict):
                                    # Handle dict-like document objects
                                    content = doc.get('page_content') or doc.get('content') or doc.get('text', '')
                                    if not content and 'text' in doc:
                                        # Additional check for KeyError debugging
                                        try:
                                            content = doc['text']
                                        except KeyError as ke:
                                            self.logger.error(f"[DEBUG] KeyError accessing doc['text']: {str(ke)}")
                                            self.logger.error(f"[DEBUG] Doc keys at time of KeyError: {list(doc.keys())}")
                                            self.logger.error(f"[DEBUG] Full doc content: {str(doc)}")
                                            raise
                                else:
                                    # Fallback - convert to string
                                    content = str(doc)
                                
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
                            
                            # Create NotebookRAGSource
                            
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
                        
                        # Use configurable multiplier for source collection
                        try:
                            notebook_settings = await self._get_cached_notebook_settings()
                            collection_multiplier = notebook_settings.get('notebook_llm', {}).get('collection_multiplier', 4)
                        except Exception:
                            collection_multiplier = 4  # Default for comprehensive collection
                        
                        if len(all_sources) >= top_k * collection_multiplier:
                            break
                    
                    self.logger.debug(f"Found {len(results)} results from collection {collection_name}")
                    
                    # Clean up connection for this collection
                    try:
                        connections.disconnect(alias=connection_args['alias'])
                    except Exception as cleanup_err:
                        self.logger.debug(f"[QUERY_CLEANUP] Failed to disconnect from {collection_name}: {cleanup_err}")
                    
                except Exception as e:
                    self.logger.error(f"[DEBUG] Error querying collection {collection_name}: {str(e)}")
                    self.logger.error(f"[DEBUG] Collection query error type: {type(e).__name__}")
                    if isinstance(e, KeyError):
                        self.logger.error(f"[DEBUG] KeyError during collection query - missing key: {str(e)}")
                    self.logger.error(f"[DEBUG] Full error traceback for collection {collection_name}: {traceback.format_exc()}")
                    continue
            
            # Sort and filter sources
            
            # Sort all sources by score (higher is better for COSINE similarity)
            all_sources.sort(key=lambda x: x.score, reverse=True)
            
            # For comprehensive queries, don't limit to top_k to preserve all retrieved data
            if is_comprehensive_query and intent_analysis.get('query_type') == 'comprehensive':
                top_sources = all_sources  # Use all retrieved sources
                self.logger.info(f"[COMPREHENSIVE_QUERY] Using all {len(all_sources)} sources (bypassing top_k={top_k} limit)")
            else:
                # Take top_k results for regular queries
                top_sources = all_sources[:top_k]
            
            # Return final sources
            
            # Removed execution state tracking to fix Redis async/sync errors
            
            self.logger.info(f"Successfully queried notebook {notebook_id}, found {len(top_sources)} sources")
            
            # Extract structured professional data for comprehensive queries using AI detection
            extracted_projects = None
            
            if skip_extraction:
                self.logger.info(f"[SKIP_EXTRACTION] Skipping extraction for immediate response - {len(top_sources)} sources")
            else:
                should_extract = await self._should_extract_structured_data(query, top_sources)
                if should_extract:
                    self.logger.info(f"[STRUCTURED_EXTRACTION] Detected comprehensive query: '{query[:50]}...', extracting structured professional data")
                    
                    # Removed execution state tracking to fix Redis async/sync errors
                    
                    try:
                        extracted_projects = await self.extract_project_data(top_sources)
                        self.logger.info(f"[STRUCTURED_EXTRACTION] Successfully extracted {len(extracted_projects) if extracted_projects else 0} structured activities")
                    except Exception as e:
                        self.logger.error(f"[STRUCTURED_EXTRACTION] Extraction failed: {str(e)}")
                        extracted_projects = None
            
            return NotebookRAGResponse(
                notebook_id=notebook_id,
                query=query,
                sources=top_sources,
                total_sources=len(all_sources),
                queried_documents=len(set(source.document_id for source in all_sources)),
                collections_searched=list(collections_searched),
                extracted_projects=extracted_projects
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error querying notebook: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            if isinstance(e, KeyError):
                self.logger.error(f"KeyError details - missing key: {str(e)}")
                self.logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return empty result instead of raising
            return NotebookRAGResponse(
                notebook_id=notebook_id,
                query=query,
                sources=[],
                total_sources=0,
                queried_documents=0,
                collections_searched=[],
                extracted_projects=None
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
        max_retrieval_chunks: int,
        query: str = "",
        is_listing_query: bool = False
    ) -> List[tuple]:
        """
        Apply document-aware retrieval logic to initial results with memory-first prioritization.
        
        This method implements NotebookLM-style comprehensive information coverage by:
        1. Separating memory sources from document sources (memory-first priority)
        2. Grouping chunks by document and calculating document-level scores
        3. For listing queries: Retrieving individual memory chunks to preserve project granularity
        4. For regular queries: Retrieving ALL memory chunks for completeness (memories are user-curated)
        5. Retrieving neighboring chunks when a chunk matches
        6. Including complete documents when they exceed the completeness threshold
        7. Boosting memory chunk scores for final ranking
        
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
            query: Original query string for context
            is_listing_query: Whether this is a comprehensive listing query
            
        Returns:
            Enhanced results with document-aware retrieval and memory-first prioritization applied
        """
        try:
            from collections import defaultdict
            # TODO: Replace with intelligent pattern analysis when migrating to QueryIntentAnalyzer
            import re  # Local import until migration complete
            
            if not initial_results:
                return initial_results
            
            self.logger.info(f"[DOCUMENT_AWARE_DEBUG] Starting with {len(initial_results)} initial_results, max_retrieval_chunks={max_retrieval_chunks}, is_listing_query={is_listing_query}")
            
            
            # Load memory prioritization settings
            try:
                notebook_settings = await self._get_cached_notebook_settings()
                prioritize_memory_sources = notebook_settings.get('notebook_llm', {}).get('prioritize_memory_sources', True)
                memory_score_boost = notebook_settings.get('notebook_llm', {}).get('memory_score_boost', 0.2)
                include_all_memory_chunks = notebook_settings.get('notebook_llm', {}).get('include_all_memory_chunks', True)
            except Exception as e:
                self.logger.warning(f"Could not load memory prioritization settings, using defaults: {e}")
                prioritize_memory_sources = True
                memory_score_boost = 0.2
                include_all_memory_chunks = True
            
            # For comprehensive queries, adjust memory consolidation based on intent analysis
            if is_listing_query:
                include_all_memory_chunks = True  # Comprehensive queries need ALL memory chunks for completeness
                self.logger.info(f"[COMPREHENSIVE_QUERY] Intelligent analysis detected comprehensive query: '{query[:100]}...' - Enabling full memory consolidation to capture all available data")
            
            
            # Step 1: Separate memory sources from document sources
            memory_chunks = []
            document_chunks = []
            
            for doc, score in initial_results:
                if self._is_memory_source(doc):
                    memory_chunks.append((doc, score))
                else:
                    document_chunks.append((doc, score))
            
            self.logger.info(f"[DOCUMENT_AWARE_DEBUG] Separated: {len(memory_chunks)} memory_chunks, {len(document_chunks)} document_chunks")
            
            
            enhanced_results = []
            processed_chunk_ids = set()
            
            # Step 2: Process memory sources first (if prioritization enabled)
            if prioritize_memory_sources and memory_chunks:
                
                if include_all_memory_chunks:
                    # For comprehensive queries: Add ALL memory chunks directly without document grouping
                    # Memory chunks have unique UUIDs, so document grouping treats each as separate "document"
                    self.logger.info(f"[COMPREHENSIVE_MEMORY] Processing all {len(memory_chunks)} memory chunks directly for comprehensive query")
                    for doc, score in memory_chunks:
                        # Use unique identifier instead of shared doc_id to avoid deduplication
                        chunk_uuid = doc.get('id', '') or doc.get('metadata', {}).get('chunk_id', '') or f"mem_{len(enhanced_results)}"
                        if chunk_uuid not in processed_chunk_ids:
                            boosted_score = min(score + memory_score_boost, 1.0)
                            enhanced_results.append((doc, boosted_score))
                            processed_chunk_ids.add(chunk_uuid)
                            self.logger.debug(f"[COMPREHENSIVE_MEMORY] Added memory chunk {chunk_uuid} with score {boosted_score}")
                        else:
                            self.logger.debug(f"[COMPREHENSIVE_MEMORY] Skipped duplicate memory chunk {chunk_uuid}")
                else:
                    # Group memory chunks by document ID for selective processing
                    memory_by_document = defaultdict(list)
                    for doc, score in memory_chunks:
                        doc_id = doc.get('metadata', {}).get('doc_id', '')
                        base_doc_id = self._extract_base_document_id(doc_id)
                        memory_by_document[base_doc_id].append((doc, score, doc_id))
                    
                    # Process each memory document selectively
                    for base_doc_id, memory_doc_chunks in memory_by_document.items():
                        # For non-comprehensive queries: Just add the matching memory chunks with boost to preserve granularity
                        for doc, score, doc_id in memory_doc_chunks:
                            if doc_id not in processed_chunk_ids:
                                boosted_score = min(score + memory_score_boost, 1.0)
                                enhanced_results.append((doc, boosted_score))
                                processed_chunk_ids.add(doc_id)
            
            
            # Step 3: Process document sources (if space remaining)
            remaining_capacity = max_retrieval_chunks - len(enhanced_results)
            self.logger.info(f"[DOCUMENT_AWARE_DEBUG] After memory processing: {len(enhanced_results)} enhanced_results, remaining_capacity={remaining_capacity}, document_chunks={len(document_chunks)}")
            if remaining_capacity > 0 and document_chunks:
                
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
                        
                
                # Process documents in order of relevance (fill remaining capacity)
                for base_doc_id, doc_score in sorted(document_scores.items(), key=lambda x: x[1], reverse=True):
                    if len(enhanced_results) >= max_retrieval_chunks:
                        break
                        
                    document_chunks_list = chunks_by_document[base_doc_id]
                    
                    # Check if document exceeds completeness threshold
                    should_include_complete_document = doc_score >= document_completeness_threshold
                    
                    if should_include_complete_document:
                        
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
                                        collection, doc_id, neighbor_chunk_radius, doc_filter, query_embedding
                                    )
                                
                                    for neighbor_doc, neighbor_score in neighbor_chunks:
                                        if len(enhanced_results) >= max_retrieval_chunks:
                                            break
                                        neighbor_id = neighbor_doc.get('metadata', {}).get('doc_id', '')
                                        if neighbor_id not in processed_chunk_ids:
                                            enhanced_results.append((neighbor_doc, neighbor_score))
                                            processed_chunk_ids.add(neighbor_id)
            
            # Step 4: Final sorting and output
            
            # Sort by relevance score (memories will naturally rank higher due to score boost)
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            self.logger.info(f"[DOCUMENT_AWARE_DEBUG] Before final limiting: {len(enhanced_results)} enhanced_results")
            
            # For comprehensive/listing queries, don't limit results to preserve all data
            if is_listing_query:
                final_results = enhanced_results  # Keep all results for comprehensive queries
                self.logger.info(f"[DOCUMENT_AWARE] Comprehensive query: preserved all {len(enhanced_results)} results (bypassing max_retrieval_chunks={max_retrieval_chunks})")
            else:
                final_results = enhanced_results[:max_retrieval_chunks]
            
            # Debug: Log final ranking to verify memory prioritization
            memory_count = sum(1 for doc, score in final_results if self._is_memory_source(doc))
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in document-aware retrieval: {str(e)}")
            self.logger.error(f"Falling back to original results")
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
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving complete document {base_doc_id}: {str(e)}")
            return []
    
    async def _retrieve_neighboring_chunks(
        self,
        collection,
        chunk_id: str,
        radius: int,
        doc_filter: Optional[str],
        query_embedding: List[float]
    ) -> List[tuple]:
        """
        Retrieve neighboring chunks around a given chunk.
        
        For chunk IDs like 'doc_id_p1_c5', retrieves chunks c3, c4, c6, c7 (with radius=2).
        """
        try:
            # TODO: Replace with more robust chunk ID parsing when migrating to intelligent system
            import re  # Local import until migration complete
            
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

    async def extract_project_data(self, sources: List[NotebookRAGSource]) -> List[ProjectData]:
        """
        Extract structured professional activity data from retrieved chunks.
        
        This method parses chunks to identify ANY professional information and extract
        structured data including name, organization, years, and description.
        Works for projects, jobs, achievements, activities, or any professional entity.
        Missing metadata is marked as "Not specified" rather than excluding items.
        
        Args:
            sources: List of RAG sources containing professional information
            
        Returns:
            List of structured professional activity data with completed metadata
        """
        try:
            import re
            self.logger.info(f"[PROJECT_EXTRACTION] Starting extraction from {len(sources)} sources")
            self.logger.info(f"[EXTRACTION_DEBUG] Starting extraction with {len(sources)} sources")
            
            # Initialize professional activity collection
            raw_activities = []
            
            # Extract professional activities from sources (batch processing for efficiency)
            raw_activities = await self._extract_projects_from_sources_batch(sources)
            
            self.logger.info(f"[STRUCTURED_EXTRACTION] Found {len(raw_activities)} raw professional activities")
            self.logger.info(f"[EXTRACTION_DEBUG] Extracted {len(raw_activities)} raw projects before deduplication")
            if raw_activities[:3]:  # Log first 3 for debugging
                for i, proj in enumerate(raw_activities[:3]):
                    self.logger.info(f"[EXTRACTION_DEBUG] Sample {i+1}: {proj.name} at {proj.company} ({proj.year})")
            
            # Complete missing metadata using cross-referencing
            completed_activities = await self._complete_project_metadata(raw_activities, sources)
            self.logger.info(f"[EXTRACTION_DEBUG] After metadata completion: {len(completed_activities)} projects")
            
            # Deduplicate and merge similar activities
            self.logger.info(f"[EXTRACTION_DEBUG] Before deduplication: {len(completed_activities)} projects")
            final_activities = await self._deduplicate_and_merge_projects(completed_activities)
            self.logger.info(f"[EXTRACTION_DEBUG] After deduplication: {len(final_activities)} projects")
            
            # Sort activities chronologically by start year
            if final_activities:
                final_activities = self._sort_projects_chronologically(final_activities)
                self.logger.info(f"[STRUCTURED_EXTRACTION] Activities sorted chronologically (default behavior)")
            
            self.logger.info(f"[STRUCTURED_EXTRACTION] Final extraction: {len(final_activities)} unique professional activities")
            self.logger.info(f"[EXTRACTION_DEBUG] Final result: {len(final_activities)} projects after deduplication")
            if final_activities:
                self.logger.info(f"[EXTRACTION_DEBUG] Returning {len(final_activities)} ProjectData objects")
                # Log samples of final results
                for i, proj in enumerate(final_activities[:3]):
                    self.logger.info(f"[EXTRACTION_DEBUG] Final sample {i+1}: {proj.name} at {proj.company} ({proj.year})")
            else:
                self.logger.warning(f"[EXTRACTION_DEBUG] WARNING: Returning empty list despite extraction attempts")
            return final_activities
            
        except Exception as e:
            self.logger.error(f"[STRUCTURED_EXTRACTION] Error in professional activity extraction: {str(e)}")
            return []

    def _sort_projects_chronologically(self, projects: List[ProjectData]) -> List[ProjectData]:
        """
        Sort projects chronologically by start year.
        
        Args:
            projects: List of projects to sort
            
        Returns:
            List of projects sorted by start year (earliest first)
        """
        def extract_start_year(project):
            try:
                year_str = project.year
                if not year_str or year_str == "N/A" or year_str.strip() == "":
                    return 9999  # Put N/A years at the end
                
                # Extract first year from ranges like "2020-2021", "2020 – 2021", "2020-22", etc.
                import re
                year_match = re.search(r'(\d{4})', str(year_str))
                if year_match:
                    return int(year_match.group(1))
                    
                # Handle 2-digit years like "20-21" (assume 20xx)
                two_digit_match = re.search(r'(\d{2})-', str(year_str))
                if two_digit_match:
                    two_digit = int(two_digit_match.group(1))
                    # Assume 20xx for years 00-99
                    return 2000 + two_digit
                    
                return 9999  # Unrecognized format goes to end
            except (AttributeError, ValueError, TypeError):
                return 9999
        
        try:
            sorted_projects = sorted(projects, key=extract_start_year)
            self.logger.info(f"[PROJECT_SORTING] Sorted {len(projects)} projects chronologically")
            return sorted_projects
        except Exception as e:
            self.logger.warning(f"[PROJECT_SORTING] Error sorting projects: {str(e)}, returning original order")
            return projects

    async def _extract_projects_from_sources_batch(self, sources: List[NotebookRAGSource]) -> List[ProjectData]:
        """
        Extract professional activities using chunked batch processing.
        
        For large source sets (>12 sources), splits into chunks and processes in parallel.
        This prevents timeouts and improves performance for comprehensive queries.
        
        Args:
            sources: List of RAG sources to process
            
        Returns:
            List of all professional activities found across all sources
        """
        if not sources:
            return []
            
        try:
            source_count = len(sources)
            self.logger.info(f"[CHUNKED_EXTRACTION] Processing {source_count} sources")
            
            # Determine processing strategy based on source count with adaptive chunking
            # Base chunk size reduced to 5 for improved reliability and timeout avoidance
            base_chunk_size = 5  # Smaller chunks for better reliability
            
            # Adaptive chunking: reduce chunk size further if we detect timeout patterns
            if hasattr(self, '_recent_timeouts') and len(self._recent_timeouts) >= 2:
                # If we've had recent timeouts, use even smaller chunks
                chunk_size = max(3, base_chunk_size - len(self._recent_timeouts))  # Min 3, max 5
                self.logger.info(f"[ADAPTIVE_CHUNKING] Detected {len(self._recent_timeouts)} recent timeouts, reducing chunk size to {chunk_size}")
            else:
                chunk_size = base_chunk_size
            
            if source_count <= chunk_size:
                # Small batch - process normally
                self.logger.debug(f"[CHUNKED_EXTRACTION] Small batch ({source_count} sources) - processing directly")
                return await self._extract_projects_from_single_chunk(sources)
            
            # Large batch - split into chunks and process in parallel
            self.logger.info(f"[CHUNKED_EXTRACTION] Large batch ({source_count} sources) - splitting into chunks of {chunk_size}")
            
            chunks = [sources[i:i + chunk_size] for i in range(0, source_count, chunk_size)]
            num_chunks = len(chunks)
            self.logger.info(f"[CHUNKED_EXTRACTION] Created {num_chunks} chunks for better parallelization")
            
            # Calculate dynamic timeout for the overall operation
            dynamic_timeout = calculate_dynamic_timeout(source_count)
            self.logger.info(f"[CHUNKED_EXTRACTION] Using dynamic timeout: {dynamic_timeout}s for {source_count} sources")
            
            # Process chunks in parallel with individual timeouts
            chunk_tasks = []
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"[CHUNKED_EXTRACTION] Queuing chunk {i+1}/{num_chunks} with {len(chunk)} sources")
                task = self._extract_projects_from_chunk_with_timeout(
                    chunk, i+1, num_chunks, source_count
                )
                chunk_tasks.append(task)
            
            # Progressive processing with as_completed for better user experience
            import asyncio
            
            all_activities = []
            successful_chunks = 0
            failed_chunks = 0
            completed_chunks = 0
            
            self.logger.info(f"[PROGRESSIVE_EXTRACTION] Starting progressive processing of {len(chunks)} chunks")
            
            # Process chunks as they complete (progressive results)
            for completed_task in asyncio.as_completed(chunk_tasks):
                try:
                    result = await completed_task
                    completed_chunks += 1
                    
                    if isinstance(result, list):
                        all_activities.extend(result)
                        successful_chunks += 1
                        self.logger.info(f"[PROGRESSIVE_EXTRACTION] Chunk {completed_chunks}/{len(chunks)} completed: {len(result)} activities (+{len(all_activities)} total)")
                        
                        # Yield intermediate results every 2 chunks or when significant progress is made
                        if completed_chunks % 2 == 0 or len(all_activities) >= 10:
                            self.logger.debug(f"[PROGRESSIVE_EXTRACTION] Intermediate result: {len(all_activities)} activities from {completed_chunks} chunks")
                    else:
                        failed_chunks += 1
                        self.logger.warning(f"[PROGRESSIVE_EXTRACTION] Chunk {completed_chunks}/{len(chunks)} failed")
                        
                except Exception as e:
                    completed_chunks += 1
                    failed_chunks += 1
                    self.logger.error(f"[PROGRESSIVE_EXTRACTION] Chunk {completed_chunks}/{len(chunks)} failed: {str(e)}")
            
            self.logger.info(f"[PROGRESSIVE_EXTRACTION] All chunks completed: {successful_chunks}/{len(chunks)} successful, {len(all_activities)} total activities")
            
            if failed_chunks > 0:
                self.logger.warning(f"[CHUNKED_EXTRACTION] {failed_chunks} chunks failed - results may be incomplete")
            
            return all_activities
            
        except Exception as e:
            self.logger.error(f"[CHUNKED_EXTRACTION] Critical error in chunked processing: {str(e)}")
            # Fallback to individual processing
            self.logger.info("[CHUNKED_EXTRACTION] Falling back to individual source processing")
            return await self._extract_projects_from_sources_individual(sources)

    async def _extract_projects_from_single_chunk(self, sources: List[NotebookRAGSource]) -> List[ProjectData]:
        """Process a single chunk of sources (old batch method)"""
        try:
            self.logger.debug(f"[SINGLE_CHUNK] Processing {len(sources)} sources in single call")
            
            # Create combined content for AI analysis
            combined_content = ""
            for idx, source in enumerate(sources):
                content = source.content
                combined_content += f"\n--- SOURCE {idx + 1} START ---\n{content}\n--- SOURCE {idx + 1} END ---\n"
            
            # Process with single AI call
            all_activities = await self._extract_projects_from_batch_content(combined_content, sources)
            
            self.logger.debug(f"[SINGLE_CHUNK] Extracted {len(all_activities)} activities from {len(sources)} sources")
            return all_activities
            
        except Exception as e:
            self.logger.error(f"[SINGLE_CHUNK] Error processing chunk: {str(e)}")
            # Fallback to individual processing for this chunk
            return await self._extract_projects_from_sources_individual(sources)

    async def _extract_projects_from_chunk_with_timeout(self, chunk: List[NotebookRAGSource], chunk_num: int, total_chunks: int, total_source_count: int = None) -> List[ProjectData]:
        """Process a chunk with timeout protection and enhanced circuit breaker"""
        import asyncio
        import time
        
        start_time = time.time()
        
        try:
            # Use configurable chunk processing timeout with adaptive scaling
            base_chunk_timeout = get_chunk_processing_timeout()  # 120s from config
            
            # Adaptive timeout: increase for larger total datasets
            if total_source_count and total_source_count >= 50:
                # For large datasets (50+ sources), increase timeout per chunk
                adaptive_multiplier = min(1.5, 1 + (total_source_count - 50) / 100)
                chunk_timeout = int(base_chunk_timeout * adaptive_multiplier)
            else:
                # Standard timeout calculation: base + per-source allowance (increased)
                chunk_timeout = base_chunk_timeout + (len(chunk) * 10)  # 10 seconds per source (increased from 8)
            
            # Cap timeout to prevent excessive waits
            chunk_timeout = min(chunk_timeout, 300)  # Maximum 5 minutes per chunk
            
            self.logger.debug(f"[CHUNK_{chunk_num}] Processing {len(chunk)} sources with {chunk_timeout}s adaptive timeout")
            
            # Enhanced circuit breaker: analyze performance patterns for better reliability
            performance_mode = "normal"
            if hasattr(self, '_chunk_performance_history') and len(self._chunk_performance_history) > 0:
                avg_time = sum(self._chunk_performance_history) / len(self._chunk_performance_history)
                failure_rate = getattr(self, '_chunk_failure_count', 0) / max(len(self._chunk_performance_history), 1)
                
                # Activate fast mode if performance is degrading or failure rate is high
                if (avg_time > 90 and len(self._chunk_performance_history) >= 2) or failure_rate > 0.3:
                    performance_mode = "fast"
                    # Use more conservative timeout for fast mode but still reasonable for large datasets
                    chunk_timeout = min(chunk_timeout, max(120, len(chunk) * 12))  # At least 120s or 12s per source (increased)
                    self.logger.warning(f"[CIRCUIT_BREAKER] Activated fast mode - avg: {avg_time:.1f}s, failures: {failure_rate:.1%}")
                elif total_source_count and total_source_count >= 100:
                    # For very large datasets (100+ sources), use conservative approach from start
                    performance_mode = "conservative"
                    self.logger.info(f"[CIRCUIT_BREAKER] Using conservative mode for {total_source_count} sources")
            
            # Process chunk with timeout based on performance mode
            if performance_mode == "fast":
                # In fast mode, try individual processing first (more reliable)
                self.logger.debug(f"[CHUNK_{chunk_num}] Using fast mode - individual processing")
                result = await asyncio.wait_for(
                    self._extract_projects_from_sources_individual(chunk),
                    timeout=chunk_timeout
                )
            elif performance_mode == "conservative":
                # Conservative mode for very large datasets - use individual processing with extended timeouts
                self.logger.debug(f"[CHUNK_{chunk_num}] Using conservative mode - individual processing with extended timeout")
                result = await asyncio.wait_for(
                    self._extract_projects_from_sources_individual(chunk),
                    timeout=chunk_timeout
                )
            else:
                # Normal mode - use batch processing
                result = await asyncio.wait_for(
                    self._extract_projects_from_single_chunk(chunk),
                    timeout=chunk_timeout
                )
            
            # Track successful performance
            processing_time = time.time() - start_time
            if not hasattr(self, '_chunk_performance_history'):
                self._chunk_performance_history = []
            self._chunk_performance_history.append(processing_time)
            # Keep only last 5 chunk times for rolling average (increased from 3 for better analysis)
            if len(self._chunk_performance_history) > 5:
                self._chunk_performance_history.pop(0)
            
            self.logger.info(f"[CHUNK_{chunk_num}] Completed in {processing_time:.1f}s: {len(result)} activities extracted")
            return result
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            # Track timeout for adaptive chunking
            import time as time_module
            self._recent_timeouts.append(time_module.time())
            # Keep only recent timeouts (last 10 minutes)
            cutoff_time = time_module.time() - 600  # 10 minutes ago
            self._recent_timeouts = [t for t in self._recent_timeouts if t > cutoff_time]
            
            # Track failure for circuit breaker
            if not hasattr(self, '_chunk_failure_count'):
                self._chunk_failure_count = 0
            self._chunk_failure_count += 1
            
            self.logger.warning(f"[CHUNK_{chunk_num}] Timeout after {processing_time:.1f}s - falling back to individual processing")
            
            # Emergency fallback with adaptive timeout based on source count
            try:
                # Calculate adaptive emergency timeout based on total dataset size
                if total_source_count and total_source_count >= 50:
                    # For large datasets, give more time per source in emergency mode
                    emergency_timeout = len(chunk) * 20  # 20s per source for large datasets
                else:
                    # Standard emergency timeout
                    emergency_timeout = len(chunk) * 15  # 15s per source
                
                # Cap emergency timeout to prevent excessive waits
                emergency_timeout = min(emergency_timeout, 180)  # Max 3 minutes for emergency fallback
                result = await asyncio.wait_for(
                    self._extract_projects_from_sources_individual(chunk),
                    timeout=emergency_timeout
                )
                self.logger.info(f"[CHUNK_{chunk_num}] Emergency fallback succeeded: {len(result)} activities")
                return result
            except asyncio.TimeoutError:
                self.logger.error(f"[CHUNK_{chunk_num}] Emergency fallback also timed out - returning empty")
                return []
        except Exception as e:
            processing_time = time.time() - start_time
            # Track failure for circuit breaker
            if not hasattr(self, '_chunk_failure_count'):
                self._chunk_failure_count = 0
            self._chunk_failure_count += 1
            
            self.logger.error(f"[CHUNK_{chunk_num}] Failed after {processing_time:.1f}s with error: {str(e)}")
            return []

    async def _extract_projects_from_sources_individual(self, sources: List[NotebookRAGSource]) -> List[ProjectData]:
        """Fallback individual processing method with timeout protection"""
        import asyncio
        
        all_activities = []
        
        # Process sources individually with per-source timeout
        for i, source in enumerate(sources):
            try:
                # Use configurable per-source timeout for better scalability
                per_source_timeout = get_extraction_timeout() // 6  # ~15s (90s / 6 sources)
                activities_in_source = await asyncio.wait_for(
                    self._extract_projects_from_content(source),
                    timeout=per_source_timeout
                )
                all_activities.extend(activities_in_source)
                self.logger.debug(f"[INDIVIDUAL_{i+1}] Extracted {len(activities_in_source)} activities")
                
            except asyncio.TimeoutError:
                self.logger.warning(f"[INDIVIDUAL_{i+1}] Source processing timeout after 15s - skipping")
                continue
            except Exception as e:
                self.logger.warning(f"[INDIVIDUAL_{i+1}] Source processing failed: {str(e)} - skipping")
                continue
        
        self.logger.info(f"[INDIVIDUAL_PROCESSING] Completed: {len(all_activities)} activities from {len(sources)} sources")
        return all_activities

    async def _extract_projects_from_batch_content(self, combined_content: str, sources: List[NotebookRAGSource]) -> List[ProjectData]:
        """
        Extract activities from combined batch content using cached LLM.
        Processes multiple sources in a single AI call for maximum efficiency.
        """
        try:
            # Check cache first (using combined content hash)
            cache_key = await self._get_extraction_cache_key(combined_content)
            cached_activities = await self._get_cached_ai_extraction(cache_key)
            if cached_activities is not None:
                self.logger.debug(f"[BATCH_AI] Using cached batch extraction with {len(cached_activities)} activities")
                return cached_activities

            # Get cached LLM instance
            llm = await self._get_cached_llm()
            if not llm:
                self.logger.warning("[BATCH_AI] No notebook LLM available for batch extraction")
                return []

            # Create batch extraction prompt
            prompt = await self._build_batch_project_extraction_prompt(combined_content, len(sources))
            
            # Single AI call for chunk with increased timeout for reliable extraction
            chunk_size = len(sources)
            ai_timeout = min(180, max(90, chunk_size * 10))  # 10s per source, 90-180s range (increased)
            self.logger.debug(f"[BATCH_AI] Using increased timeout of {ai_timeout}s for {chunk_size} sources")
            
            response = await llm.generate(prompt, timeout=ai_timeout, task_type="batch_extraction")
            ai_response = response.text
            
            # Parse AI response into ProjectData objects
            batch_activities = await self._parse_batch_ai_extraction_response(ai_response, sources)
            
            # Cache the results
            await self._cache_ai_extraction(cache_key, batch_activities)
            
            self.logger.debug(f"[BATCH_AI] Batch AI extraction found {len(batch_activities)} activities")
            return batch_activities
            
        except Exception as e:
            self.logger.error(f"[BATCH_AI] Error in batch AI extraction: {str(e)}")
            self.logger.error(f"[BATCH_AI] Full traceback: {traceback.format_exc()}")
            return []

    async def _build_batch_project_extraction_prompt(self, combined_content: str, source_count: int) -> str:
        """Build optimized, streamlined prompt for fast extraction"""
        # Optimized prompt that focuses on speed and essential information only
        return f"""Extract projects and experiences from {source_count} sources:

{combined_content}

Extract as JSON array - name, company, year, brief description only:
[{{"name": "Project Name", "company": "Company", "year": "2024", "description": "Brief summary"}}]

Rules:
- Every substantial project/role from ALL sources
- Keep descriptions under 100 characters
- Use "Unknown" for missing info
- JSON only, no extra text

JSON:"""

    async def _parse_batch_ai_extraction_response(self, ai_response: str, sources: List[NotebookRAGSource]) -> List[ProjectData]:
        """Parse batch AI response into ProjectData objects"""
        try:
            # Reuse existing parsing logic but handle batch format
            # For simplicity, create a dummy source for batch processing
            dummy_source = sources[0] if sources else NotebookRAGSource(
                content="", similarity_score=0.0, document_id="batch", chunk_index=0
            )
            
            return await self._parse_ai_extraction_response(ai_response, dummy_source)
            
        except Exception as e:
            self.logger.error(f"[BATCH_PARSE] Error parsing batch AI response: {str(e)}")
            return []

    async def _extract_projects_from_content(self, source: NotebookRAGSource) -> List[ProjectData]:
        """
        Extract professional activity information from a single content source using hybrid approach (minimal structure + AI).
        
        Args:
            source: RAG source to extract professional activities from
            
        Returns:
            List of professional activities found in the content
        """
        try:
            content = source.content
            activities = []
            
            # Always use AI-first approach with minimal structure detection as fallback
            ai_activities = await self._extract_projects_with_ai(source)
            
            if ai_activities and len(ai_activities) >= 2:
                # AI found good results, use them
                activities = ai_activities
                self.logger.debug(f"[STRUCTURED_EXTRACTION] AI extraction found {len(ai_activities)} activities")
            else:
                # AI didn't find much, try minimal structural patterns as fallback
                self.logger.info(f"[STRUCTURED_EXTRACTION] AI found only {len(ai_activities)} activities, trying minimal structure fallback")
                structure_activities = await self._extract_projects_with_regex(source)
                
                # Merge AI and structure results
                activities = await self._merge_and_deduplicate_projects(ai_activities, structure_activities)
                self.logger.debug(f"[STRUCTURED_EXTRACTION] After AI+structure merge: {len(activities)} activities")
            
            return activities
            
        except Exception as e:
            self.logger.error(f"[STRUCTURED_EXTRACTION] Error extracting from content: {str(e)}")
            return []
    
    async def _extract_projects_with_regex(self, source: NotebookRAGSource) -> List[ProjectData]:
        """
        Minimal structure-based extraction with universal patterns only.
        Most extraction work is now delegated to AI for generalization.
        """
        try:
            import re
            content = source.content
            projects = []
            
            # MINIMAL universal structure patterns only - no specific terminology
            # These patterns focus on document structure, not content-specific terms
            universal_patterns = [
                # Bullet points or numbered lists (universal structure)
                r'[•\-*]\s*([^\n]{15,200})',
                # Sentences with years (universal temporal indicators) 
                r'([^.!?\n]{20,200}(?:19|20)\d{2}[^.!?\n]{0,100})',
                # Capitalized phrases (universal proper noun structure)
                r'\b([A-Z][^.!?\n]{20,150})',
            ]
            
            # Process each pattern with minimal filtering
            for pattern in universal_patterns:
                matches = re.finditer(pattern, content, re.DOTALL)
                for match in matches:
                    text_fragment = match.group(1).strip()
                    
                    # Universal cleanup (structure-based, not content-specific)
                    text_fragment = re.sub(r'[\r\n]+', ' ', text_fragment)
                    text_fragment = re.sub(r'\s+', ' ', text_fragment).strip()
                    
                    # Only filter by length - no content assumptions
                    if len(text_fragment) < 15 or len(text_fragment) > 300:
                        continue
                    
                    # Extract context for AI analysis
                    context_start = max(0, match.start() - 300)
                    context_end = min(len(content), match.end() + 300)
                    context = content[context_start:context_end]
                    
                    # Create minimal project with AI-friendly metadata
                    project = ProjectData(
                        name=text_fragment[:100],  # Use fragment as name candidate
                        company="TBD",  # Let AI determine
                        year="TBD",     # Let AI determine
                        description=text_fragment,
                        source_chunk_id=source.metadata.get('chunk_id'),
                        confidence_score=0.3,  # Low confidence - needs AI refinement
                        metadata={
                            'source_document': source.document_name,
                            'extraction_method': 'minimal_structure',
                            'context': context[:500],  # Provide context for AI
                            'requires_ai_processing': True
                        }
                    )
                    
                    projects.append(project)
            
            self.logger.debug(f"[MINIMAL_EXTRACTION] Found {len(projects)} structural candidates for AI processing")
            return projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error in minimal extraction: {str(e)}")
            return []
    
    async def _extract_projects_with_ai(self, source: NotebookRAGSource) -> List[ProjectData]:
        """
        Extract projects using AI-powered analysis when regex patterns fail.
        """
        try:
            content = source.content
            
            # Check cache first
            cache_key = await self._get_extraction_cache_key(content)
            cached_projects = await self._get_cached_ai_extraction(cache_key)
            if cached_projects is not None:
                self.logger.debug(f"[PROJECT_EXTRACTION] Using cached AI extraction with {len(cached_projects)} projects")
                return cached_projects
            
            # Get cached LLM instance (eliminates redundant config fetches and instantiations)
            llm = await self._get_cached_llm()
            if not llm:
                self.logger.warning("[PROJECT_EXTRACTION] No notebook LLM available for AI extraction")
                return []
            
            # Create extraction prompt
            prompt = await self._build_project_extraction_prompt(content)
            
            # Get AI response
            response = await llm.generate(prompt)
            ai_response = response.text
            
            # Parse AI response into ProjectData objects
            ai_projects = await self._parse_ai_extraction_response(ai_response, source)
            
            # Cache the results
            await self._cache_ai_extraction(cache_key, ai_projects)
            
            self.logger.debug(f"[PROJECT_EXTRACTION] AI extraction found {len(ai_projects)} projects")
            return ai_projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error in AI extraction: {str(e)}")
            return []
    
    async def _build_project_extraction_prompt(self, content: str) -> str:
        """
        Build intelligent prompt for general-purpose entity extraction.
        Now handles any type of professional activity, not just "projects".
        """
        # Truncate content if too long to avoid token limits
        max_content_length = 4000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = f"""
Analyze the following text and identify ALL professional activities, work experiences, achievements, and notable activities mentioned. Be comprehensive and identify any structured professional information regardless of format or language.

Extrract ANY type of professional activity including but not limited to:
- Work projects, assignments, initiatives, implementations
- Employment positions, roles, responsibilities
- Business ventures, startups, companies founded or managed
- Consulting work, freelance projects, contract work
- Research activities, academic work, publications
- Products built, systems developed, solutions delivered
- Leadership roles, team management, mentoring
- Training provided, courses taught, presentations given
- Awards, certifications, achievements, recognitions
- Volunteer work with professional relevance
- Personal projects with career impact
- Any organized professional activity with measurable outcomes

For each activity found, provide:
- name: Clear, descriptive name of the activity (extract from context, don't assume format)
- company: Organization, employer, client, or context where this occurred
- year: Time period when this occurred (extract any temporal information available)
- description: What was accomplished, built, achieved, or delivered
- confidence: Your confidence in this extraction (0.0-1.0)

IMPORTANT GUIDELINES:
- Extract information AS-IS from the text, don't impose format assumptions
- Work with ANY language, culture, or professional context
- Identify activities even if they don't use standard business terminology  
- Don't require specific keywords like "project" or "developed"
- Extract from any document type (resume, bio, portfolio, narrative, etc.)
- Include activities that show professional growth or skill demonstration

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "name": "Activity Name",
    "company": "Organization/Context", 
    "year": "Time Period",
    "description": "What was accomplished or delivered",
    "confidence": 0.9
  }}
]

Text to analyze:
{content}
"""
        return prompt
    
    async def _parse_ai_extraction_response(self, response: str, source: NotebookRAGSource) -> List[ProjectData]:
        """
        Parse AI response into ProjectData objects.
        """
        try:
            # Clean response - extract JSON from response
            response = response.strip()
            
            # Find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                self.logger.warning("[PROJECT_EXTRACTION] No JSON array found in AI response")
                return []
            
            json_str = response[start_idx:end_idx + 1]
            
            # Parse JSON
            try:
                projects_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"[PROJECT_EXTRACTION] JSON decode error: {str(e)}")
                return []
            
            if not isinstance(projects_data, list):
                self.logger.warning("[PROJECT_EXTRACTION] AI response is not a list")
                return []
            
            projects = []
            for item in projects_data:
                if not isinstance(item, dict):
                    continue
                
                # Validate required fields
                if not all(key in item for key in ['name', 'company', 'year', 'description']):
                    continue
                
                # Create ProjectData object
                project = ProjectData(
                    name=str(item['name'])[:100],  # Limit length
                    company=str(item['company'])[:100],
                    year=str(item['year'])[:50],
                    description=str(item['description'])[:500],
                    source_chunk_id=source.metadata.get('chunk_id'),
                    confidence_score=float(item.get('confidence', 0.8)),
                    metadata={
                        'source_document': source.document_name,
                        'extraction_method': 'ai',
                        'ai_confidence': float(item.get('confidence', 0.8))
                    }
                )
                projects.append(project)
            
            return projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error parsing AI response: {str(e)}")
            return []
    
    async def _merge_and_deduplicate_projects(self, regex_projects: List[ProjectData], ai_projects: List[ProjectData]) -> List[ProjectData]:
        """
        Merge and deduplicate projects from regex and AI extraction.
        """
        try:
            # Start with AI projects (higher confidence for completeness)
            merged_projects = list(ai_projects)
            
            # Add unique regex projects
            for regex_project in regex_projects:
                is_duplicate = False
                
                for existing_project in merged_projects:
                    # Check for duplicates using name similarity
                    if self._are_projects_similar(regex_project.name, existing_project.name):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged_projects.append(regex_project)
            
            self.logger.debug(f"[PROJECT_EXTRACTION] Merged {len(ai_projects)} AI + {len(regex_projects)} regex = {len(merged_projects)} unique projects")
            return merged_projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error merging projects: {str(e)}")
            return regex_projects  # Fallback to regex results
    
    def _are_projects_similar(self, name1: str, name2: str) -> bool:
        """
        Check if two project names are similar enough to be considered duplicates.
        """
        try:
            # Normalize names for comparison
            norm1 = re.sub(r'[^a-zA-Z0-9\s]', '', name1.lower()).strip()
            norm2 = re.sub(r'[^a-zA-Z0-9\s]', '', name2.lower()).strip()
            
            # Check exact match
            if norm1 == norm2:
                return True
            
            # Check if one is contained in the other (with length threshold)
            if len(norm1) > 5 and len(norm2) > 5:
                if norm1 in norm2 or norm2 in norm1:
                    return True
            
            # Simple word overlap check
            words1 = set(norm1.split())
            words2 = set(norm2.split())
            
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1.intersection(words2))
                total_unique = len(words1.union(words2))
                similarity = overlap / total_unique if total_unique > 0 else 0
                
                return similarity > 0.6  # 60% word overlap threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error comparing project names: {str(e)}")
            return False
    
    async def _get_extraction_cache_key(self, content: str) -> str:
        """
        Generate cache key based on content hash.
        """
        try:
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            return f"project_extraction:{content_hash}"
        except Exception:
            # Fallback to timestamp-based key
            return f"project_extraction:{int(time.time())}"
    
    async def _get_cached_ai_extraction(self, cache_key: str) -> Optional[List[ProjectData]]:
        """
        Get cached AI extraction results.
        """
        try:
            redis_client = get_redis_client()
            if not redis_client:
                return None
            
            cached_data = redis_client.get(cache_key)
            if not cached_data:
                return None
            
            # Parse cached JSON - handle bytes if necessary
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode('utf-8')
            projects_data = json.loads(cached_data)
            projects = []
            
            for item in projects_data:
                project = ProjectData(
                    name=item['name'],
                    company=item['company'],
                    year=item['year'],
                    description=item['description'],
                    source_chunk_id=item.get('source_chunk_id'),
                    confidence_score=float(item['confidence_score']),
                    metadata=item.get('metadata', {})
                )
                projects.append(project)
            
            return projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error getting cached extraction: {str(e)}")
            return None
    
    async def _cache_ai_extraction(self, cache_key: str, projects: List[ProjectData]) -> None:
        """
        Cache AI extraction results.
        """
        try:
            redis_client = get_redis_client()
            if not redis_client:
                return
            
            # Convert projects to JSON-serializable format
            projects_data = []
            for project in projects:
                project_dict = {
                    'name': project.name,
                    'company': project.company,
                    'year': project.year,
                    'description': project.description,
                    'source_chunk_id': project.source_chunk_id,
                    'confidence_score': project.confidence_score,
                    'metadata': project.metadata
                }
                projects_data.append(project_dict)
            
            # Cache with reasonable TTL (2 hours)
            cache_ttl = 7200
            redis_client.setex(cache_key, cache_ttl, json.dumps(projects_data))
            
            self.logger.debug(f"[PROJECT_EXTRACTION] Cached {len(projects)} projects with key: {cache_key}")
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error caching extraction: {str(e)}")

    async def _get_vector_cache_key(self, notebook_id: str, query: str, max_sources: int) -> str:
        """
        Generate unique cache key for vector retrieval results.
        
        Args:
            notebook_id: The notebook ID
            query: The search query
            max_sources: Maximum number of sources to retrieve
            
        Returns:
            Cache key string
        """
        try:
            # Create hash from query for consistent key generation
            query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()[:12]  # Use first 12 chars
            return f"notebook_vectors:{notebook_id}:{query_hash}:{max_sources}"
        except Exception:
            # Fallback to timestamp-based key if hashing fails
            return f"notebook_vectors:{notebook_id}:{int(time.time())}:{max_sources}"

    async def _cache_vector_results(self, cache_key: str, raw_results: list, ttl: int = 3600) -> None:
        """
        Store raw Milvus vector search results in cache.
        
        Args:
            cache_key: Cache key for storage
            raw_results: Raw results from Milvus search (list of hit objects)
            ttl: Time to live in seconds (default: 3600 = 1 hour)
        """
        try:
            redis_client = get_redis_client()
            if not redis_client:
                return
            
            # Convert Milvus results to JSON-serializable format
            serializable_results = []
            for result in raw_results:
                if hasattr(result, '__iter__'):  # Handle search results which are lists of hits
                    for hit in result:
                        hit_data = {
                            'content': hit.entity.get('content', ''),
                            'doc_id': hit.entity.get('doc_id', ''),
                            'source': hit.entity.get('source', ''),
                            'page': hit.entity.get('page', 0),
                            'doc_type': hit.entity.get('doc_type', ''),
                            'uploaded_at': hit.entity.get('uploaded_at', ''),
                            'section': hit.entity.get('section', ''),
                            'author': hit.entity.get('author', ''),
                            'hash': hit.entity.get('hash', ''),
                            'distance': getattr(hit, 'distance', 0.0),
                            'id': getattr(hit, 'id', '')
                        }
                        serializable_results.append(hit_data)
                else:
                    # Handle individual hit objects
                    hit_data = {
                        'content': result.entity.get('content', ''),
                        'doc_id': result.entity.get('doc_id', ''),
                        'source': result.entity.get('source', ''),
                        'page': result.entity.get('page', 0),
                        'doc_type': result.entity.get('doc_type', ''),
                        'uploaded_at': result.entity.get('uploaded_at', ''),
                        'section': result.entity.get('section', ''),
                        'author': result.entity.get('author', ''),
                        'hash': result.entity.get('hash', ''),
                        'distance': getattr(result, 'distance', 0.0),
                        'id': getattr(result, 'id', '')
                    }
                    serializable_results.append(hit_data)
            
            # Cache the results
            redis_client.setex(cache_key, ttl, json.dumps(serializable_results))
            
            self.logger.debug(f"[VECTOR_CACHE] Cached {len(serializable_results)} vector results with key: {cache_key}")
            
        except Exception as e:
            self.logger.error(f"[VECTOR_CACHE] Error caching vector results: {str(e)}")

    async def _get_cached_vector_results(self, cache_key: str) -> Optional[list]:
        """
        Retrieve cached vector search results.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            List of cached results or None if not found/expired
        """
        try:
            redis_client = get_redis_client()
            if not redis_client:
                return None
            
            cached_data = redis_client.get(cache_key)
            if not cached_data:
                return None
            
            # Parse cached JSON - handle bytes if necessary
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode('utf-8')
            
            results = json.loads(cached_data)
            
            self.logger.debug(f"[VECTOR_CACHE] Retrieved {len(results)} cached vector results with key: {cache_key}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"[VECTOR_CACHE] Error getting cached vector results: {str(e)}")
            return None

    async def invalidate_vector_cache(self, notebook_id: Optional[str] = None) -> int:
        """
        Invalidate vector cache entries.
        
        Args:
            notebook_id: Optional specific notebook ID to invalidate. If None, clears all vector caches.
            
        Returns:
            Number of cache entries invalidated
        """
        try:
            redis_client = get_redis_client()
            if not redis_client:
                return 0
            
            if notebook_id:
                # Clear cache entries for specific notebook
                pattern = f"notebook_vectors:{notebook_id}:*"
                keys = redis_client.keys(pattern)
                if keys:
                    count = redis_client.delete(*keys)
                    self.logger.info(f"[VECTOR_CACHE] Invalidated {count} vector cache entries for notebook {notebook_id}")
                    return count
            else:
                # Clear all vector caches
                pattern = "notebook_vectors:*"
                keys = redis_client.keys(pattern)
                if keys:
                    count = redis_client.delete(*keys)
                    self.logger.info(f"[VECTOR_CACHE] Invalidated all {count} vector cache entries")
                    return count
                    
            return 0
                    
        except Exception as e:
            self.logger.error(f"[VECTOR_CACHE] Error invalidating cache: {str(e)}")
            return 0

    async def _extract_metadata_with_ai(self, context: str, entity_name: str) -> dict:
        """Extract metadata using AI instead of hardcoded patterns."""
        try:
            # Use AI to extract company and year information
            prompt = f"""
Analyze this text and extract metadata for the professional activity: "{entity_name}"

Text: {context[:1000]}

Return ONLY a JSON object with:
{{
  "company": "Organization/employer/client name or 'Not specified'",
  "year": "Time period or 'Not specified'"
}}
"""
            
            # Get cached LLM instance for metadata extraction
            try:
                llm = await self._get_cached_llm()
                if llm:
                    response = await llm.generate(prompt)
                    
                    # Parse AI response
                    response_text = response.text.strip()
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = response_text[start_idx:end_idx + 1]
                        metadata = json.loads(json_str)
                        return {
                            'company': metadata.get('company', 'Not specified'),
                            'year': metadata.get('year', 'Not specified')
                        }
            except Exception as e:
                self.logger.debug(f"[AI_METADATA] AI extraction failed: {str(e)}")
            
            # Fallback to "Not specified" instead of hardcoded patterns
            return {'company': 'Not specified', 'year': 'Not specified'}
            
        except Exception as e:
            self.logger.error(f"[AI_METADATA] Error in AI metadata extraction: {str(e)}")
            return {'company': 'Not specified', 'year': 'Not specified'}
    
    async def _should_extract_structured_data(self, query: str, sources: List[NotebookRAGSource]) -> bool:
        """
        Use AI to determine if query requires structured professional data extraction.
        Replaces hardcoded keyword matching with intelligent analysis.
        """
        try:
            # Quick heuristic checks first (universal patterns)
            if len(query.strip()) < 3:
                return False
            
            # Check if sources contain structured professional information
            has_professional_content = False
            for source in sources[:3]:  # Sample first few sources
                content_sample = source.content[:500].lower()
                # Look for universal professional indicators (not language-specific)
                if any(indicator in content_sample for indicator in [
                    '20', '19',  # Years (universal)
                    '\n-', '\n•', '\n*',  # Lists (universal structure)
                    ':', ';'  # Structured text indicators
                ]):
                    has_professional_content = True
                    break
            
            if not has_professional_content:
                return False
            
            # Use AI for intelligent query analysis with cached LLM
            try:
                llm = await self._get_cached_llm()
                if llm:
                    prompt = f"""
Analyze this query and determine if it's asking for comprehensive professional information that would benefit from structured extraction.

Query: "{query}"

Return ONLY "true" if the query is:
- Asking for lists, summaries, or comprehensive information
- Requesting professional background, experience, or activities
- Wanting to know about accomplishments, work history, or capabilities
- Seeking structured information rather than specific details

Return ONLY "false" if the query is:
- Asking for specific details about one particular thing
- A narrow technical question
- Looking for definitions or explanations

Return only: true or false
"""
                    response = await llm.generate(prompt)
                    result = response.text.strip().lower()
                    return result == 'true'
                        
            except Exception as e:
                self.logger.debug(f"[QUERY_ANALYSIS] AI analysis failed: {str(e)}")
            
            # Fallback to conservative approach - only extract for obviously comprehensive queries
            query_lower = query.lower()
            comprehensive_indicators = [
                'list', 'all', 'summary', 'overview', 'tell me about',
                'what', 'who', 'where', 'when', 'how many',
                'background', 'experience', 'history',
                'can you', 'please', 'show me'
            ]
            
            return any(indicator in query_lower for indicator in comprehensive_indicators)
            
        except Exception as e:
            self.logger.error(f"[QUERY_ANALYSIS] Error in query analysis: {str(e)}")
            return False  # Conservative fallback

    async def _complete_project_metadata(self, projects: List[ProjectData], sources: List[NotebookRAGSource]) -> List[ProjectData]:
        """
        Complete missing project metadata by cross-referencing with other chunks.
        
        Args:
            projects: List of projects with potentially missing metadata
            sources: All available sources for cross-referencing
            
        Returns:
            Projects with completed metadata where possible
        """
        try:
            completed = []
            
            for project in projects:
                updated_project = project.copy()
                
                # If company or year is missing, try to find it in related chunks using AI
                if project.company in ["N/A", "TBD"] or project.year in ["N/A", "TBD"]:
                    # Look for the project name or similar description in other sources
                    for source in sources:
                        if await self._is_related_content(project.name, project.description, source.content):
                            # Use AI to extract missing metadata from related source
                            metadata = await self._extract_metadata_with_ai(source.content, project.name)
                            
                            if project.company in ["N/A", "TBD"] and metadata.get('company') != 'Not specified':
                                updated_project.company = metadata['company']
                                updated_project.confidence_score += 0.1
                            
                            if project.year in ["N/A", "TBD"] and metadata.get('year') != 'Not specified':
                                updated_project.year = metadata['year']
                                updated_project.confidence_score += 0.1
                            
                            # Break if we found both
                            if (updated_project.company not in ["N/A", "TBD"] and 
                                updated_project.year not in ["N/A", "TBD"]):
                                break
                
                completed.append(updated_project)
            
            return completed
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error completing metadata: {str(e)}")
            return projects

    async def _is_related_content(self, project_name: str, project_description: str, content: str) -> bool:
        """
        Check if content is related to the project by looking for similar terms.
        
        Args:
            project_name: Name of the project
            project_description: Description of the project
            content: Content to check for similarity
            
        Returns:
            True if content appears related to the project
        """
        try:
            import re
            
            # Extract key terms from project name and description
            text_to_check = f"{project_name} {project_description}".lower()
            key_words = re.findall(r'\b\w{4,}\b', text_to_check)  # Words 4+ chars
            key_words = [w for w in key_words if w not in ['with', 'using', 'from', 'this', 'that', 'have', 'will', 'been', 'were', 'they']]
            
            content_lower = content.lower()
            
            # Count matches
            matches = sum(1 for word in key_words if word in content_lower)
            
            # Consider related if significant overlap
            threshold = max(1, len(key_words) * 0.3)  # At least 30% of key words
            return matches >= threshold
            
        except Exception:
            return False

    async def _deduplicate_and_merge_projects(self, projects: List[ProjectData]) -> List[ProjectData]:
        """
        Identify and merge duplicate projects, keeping the most complete information.
        
        Args:
            projects: List of projects that may contain duplicates
            
        Returns:
            Deduplicated list with merged project information
        """
        try:
            if not projects:
                return []
            
            # Group similar projects
            project_groups = []
            
            for project in projects:
                # Find if this project belongs to an existing group
                added_to_group = False
                
                for group in project_groups:
                    if await self._are_similar_projects(project, group[0]):
                        group.append(project)
                        added_to_group = True
                        break
                
                if not added_to_group:
                    project_groups.append([project])
            
            # Merge each group into a single project
            merged_projects = []
            for group in project_groups:
                merged = await self._merge_project_group(group)
                merged_projects.append(merged)
            
            return merged_projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error deduplicating projects: {str(e)}")
            return projects

    async def _are_similar_projects(self, project1: ProjectData, project2: ProjectData) -> bool:
        """
        Check if two projects are similar enough to be considered duplicates.
        
        Args:
            project1: First project to compare
            project2: Second project to compare
            
        Returns:
            True if projects are similar enough to merge
        """
        try:
            import re
            
            # Normalize names for comparison
            name1 = re.sub(r'[^\w\s]', '', project1.name.lower()).strip()
            name2 = re.sub(r'[^\w\s]', '', project2.name.lower()).strip()
            
            # Check for exact or very similar names
            if name1 == name2:
                return True
            
            # Check for significant word overlap in names
            words1 = set(name1.split())
            words2 = set(name2.split())
            
            if len(words1) > 0 and len(words2) > 0:
                overlap = len(words1.intersection(words2))
                min_length = min(len(words1), len(words2))
                
                # If 70%+ of words overlap and same company
                if overlap / min_length >= 0.7 and project1.company == project2.company:
                    return True
            
            # Check for substring relationships
            if len(name1) >= 10 and len(name2) >= 10:
                if name1 in name2 or name2 in name1:
                    # Same company strengthens the similarity
                    if project1.company == project2.company or project1.company == "N/A" or project2.company == "N/A":
                        return True
            
            return False
            
        except Exception:
            return False

    async def _merge_project_group(self, group: List[ProjectData]) -> ProjectData:
        """
        Merge a group of similar projects into a single project with the best available information.
        
        Args:
            group: List of similar projects to merge
            
        Returns:
            Single merged project with combined information
        """
        try:
            if len(group) == 1:
                return group[0]
            
            # Start with the project that has the highest confidence score
            base_project = max(group, key=lambda p: p.confidence_score)
            
            # Merge information from other projects
            merged_company = base_project.company
            merged_year = base_project.year
            merged_description = base_project.description
            merged_metadata = base_project.metadata or {}
            
            # Collect information from all projects
            all_companies = [p.company for p in group if p.company and p.company != "N/A"]
            all_years = [p.year for p in group if p.year and p.year != "N/A"]
            all_descriptions = [p.description for p in group]
            
            # Use the best available company
            if merged_company == "N/A" and all_companies:
                merged_company = max(all_companies, key=lambda c: sum(1 for p in group if p.company == c))
            
            # Use the best available year
            if merged_year == "N/A" and all_years:
                merged_year = max(all_years, key=lambda y: sum(1 for p in group if p.year == y))
            
            # Use the longest description
            merged_description = max(all_descriptions, key=len)
            
            # Merge metadata
            for project in group:
                if project.metadata:
                    merged_metadata.update(project.metadata)
            
            # Add merge information
            merged_metadata['merged_from'] = len(group)
            merged_metadata['source_chunks'] = [p.source_chunk_id for p in group if p.source_chunk_id]
            
            return ProjectData(
                name=base_project.name,
                company=merged_company,
                year=merged_year,
                description=merged_description,
                source_chunk_id=base_project.source_chunk_id,
                confidence_score=min(1.0, base_project.confidence_score + 0.1 * (len(group) - 1)),
                metadata=merged_metadata
            )
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error merging project group: {str(e)}")
            return group[0] if group else None

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
            except Exception as cleanup_err:
                self.logger.debug(f"[TEST_CLEANUP] Failed to disconnect test connection for {collection_name}: {cleanup_err}")
            
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
    
    def invalidate_count_cache(self, notebook_id: str = None):
        """
        Invalidate content count cache for specific notebook or all notebooks.
        
        Args:
            notebook_id: Optional specific notebook ID to invalidate. If None, clears all count caches.
        """
        try:
            redis_client = get_redis_client()
            if not redis_client:
                return
            
            if notebook_id:
                # Clear cache entries for specific notebook
                pattern = f"notebook_content_count:{notebook_id}:*"
                keys = redis_client.keys(pattern)
                if keys:
                    redis_client.delete(*keys)
                    self.logger.info(f"[COUNT_CACHE] Invalidated {len(keys)} count cache entries for notebook {notebook_id}")
            else:
                # Clear all notebook count caches
                pattern = "notebook_content_count:*"
                keys = redis_client.keys(pattern)
                if keys:
                    redis_client.delete(*keys)
                    self.logger.info(f"[COUNT_CACHE] Invalidated all {len(keys)} count cache entries")
                    
        except Exception as e:
            self.logger.warning(f"[COUNT_CACHE] Error invalidating count cache: {str(e)}")
    
    async def should_use_progressive_loading(
        self,
        query: str,
        intent_analysis: dict,
        total_available: int,
        max_sources: int
    ) -> bool:
        """
        Determine if progressive loading is beneficial for a query.
        
        Uses progressive loading when:
        - Query requires >200 results from >500 available items
        - Comprehensive queries on very large notebooks
        - Complex research queries requiring deep coverage
        
        Args:
            query: User query string
            intent_analysis: Query intent analysis results
            total_available: Total items available in notebook
            max_sources: Maximum sources requested
            
        Returns:
            True if progressive loading should be used
        """
        try:
            # Check if this is a comprehensive query requiring many results
            wants_comprehensive = intent_analysis.get("wants_comprehensive", False)
            quantity_intent = intent_analysis.get("quantity_intent", "limited")
            requires_deep_search = intent_analysis.get("requires_deep_search", False)
            
            # Progressive loading thresholds
            MIN_AVAILABLE_FOR_PROGRESSIVE = 500  # Only use for very large notebooks
            MIN_REQUESTED_FOR_PROGRESSIVE = 200  # Only when requesting many results
            
            # Use progressive loading for comprehensive queries on large datasets
            if (wants_comprehensive and 
                quantity_intent == "all" and 
                total_available > MIN_AVAILABLE_FOR_PROGRESSIVE and 
                max_sources > MIN_REQUESTED_FOR_PROGRESSIVE):
                
                self.logger.info(f"[PROGRESSIVE_LOADING] Enabled for comprehensive query: "
                              f"available={total_available}, requested={max_sources}")
                return True
            
            # Use for research queries on large datasets
            if (requires_deep_search and 
                total_available > MIN_AVAILABLE_FOR_PROGRESSIVE and 
                max_sources > MIN_REQUESTED_FOR_PROGRESSIVE / 2):  # Lower threshold for research
                
                self.logger.info(f"[PROGRESSIVE_LOADING] Enabled for research query: "
                              f"available={total_available}, requested={max_sources}")
                return True
            
            self.logger.debug(f"[PROGRESSIVE_LOADING] Not beneficial: "
                            f"comprehensive={wants_comprehensive}, deep_search={requires_deep_search}, "
                            f"available={total_available}, requested={max_sources}")
            return False
            
        except Exception as e:
            self.logger.error(f"[PROGRESSIVE_LOADING] Error determining if progressive loading needed: {str(e)}")
            return False

    async def execute_minimal_retrieval(self, message: str, notebook_id: str, db: Session) -> NotebookRAGResponse:
        """
        Execute minimal retrieval for simple lookups.
        Fast, focused retrieval using single strategy with limited sources.
        """
        self.logger.info(f"[MINIMAL_RETRIEVAL] Executing fast retrieval for simple lookup")
        
        try:
            # Use simplified retrieval with limited scope
            sources = await self._execute_single_strategy_retrieval(
                notebook_id=notebook_id,
                query=message,
                max_sources=8,
                threshold=0.3
            )
            
            self.logger.info(f"[MINIMAL_RETRIEVAL] Retrieved {len(sources)} sources")
            
            return NotebookRAGResponse(
                notebook_id=notebook_id,
                query=message,
                sources=sources,
                total_sources=len(sources),
                queried_documents=1,  # Minimal scan
                collections_searched=["notebook_chunks"]
            )
            
        except Exception as e:
            self.logger.error(f"[MINIMAL_RETRIEVAL] Error in minimal retrieval: {str(e)}")
            raise

    async def execute_balanced_retrieval(self, message: str, notebook_id: str, db: Session) -> NotebookRAGResponse:
        """
        Execute balanced retrieval for standard queries.
        Moderate resource usage with good result quality.
        """
        self.logger.info(f"[BALANCED_RETRIEVAL] Executing balanced retrieval")
        
        try:
            # Use moderate scope retrieval
            sources = await self._execute_dual_strategy_retrieval(
                notebook_id=notebook_id,
                query=message,
                max_sources=20,
                threshold=0.3
            )
            
            self.logger.info(f"[BALANCED_RETRIEVAL] Retrieved {len(sources)} sources")
            
            return NotebookRAGResponse(
                notebook_id=notebook_id,
                query=message,
                sources=sources,
                total_sources=len(sources),
                queried_documents=min(5, len(sources)),  # Moderate scan
                collections_searched=["notebook_chunks"]
            )
            
        except Exception as e:
            self.logger.error(f"[BALANCED_RETRIEVAL] Error in balanced retrieval: {str(e)}")
            raise

    async def execute_comprehensive_retrieval(self, message: str, notebook_id: str, db: Session, max_sources: int = 50) -> NotebookRAGResponse:
        """
        Execute comprehensive retrieval for complex analysis queries.
        Uses existing full pipeline with all strategies and extensive scanning.
        """
        self.logger.info(f"[COMPREHENSIVE_RETRIEVAL] Executing full comprehensive retrieval")
        
        try:
            # Use existing comprehensive approach - delegate to multi_stage_retrieval
            intent_analysis = await self._analyze_query_intent(message)
            
            # Get total available documents for progressive loading decision
            total_available = await self._get_total_available_documents(notebook_id)
            
            # Use generator to get the final comprehensive result
            final_response = None
            async for response in self.multi_stage_retrieval(
                notebook_id=notebook_id,
                query=message,
                intent_analysis=intent_analysis,
                total_available=total_available,
                max_sources=max_sources
            ):
                final_response = response  # Keep the latest/final response
                
            return final_response
            
        except Exception as e:
            self.logger.error(f"[COMPREHENSIVE_RETRIEVAL] Error in comprehensive retrieval: {str(e)}")
            raise

    async def _execute_single_strategy_retrieval(self, notebook_id: str, query: str, max_sources: int, threshold: float) -> List[NotebookRAGSource]:
        """Execute single strategy retrieval for minimal approach."""
        try:
            # Simplified single-strategy retrieval
            vector_settings = self._get_vector_settings()
            embedding_function = self._get_embedding_function()
            
            # Get embedding with execution state tracking  
            query_embedding = await self._get_query_embedding(query, embedding_function)
            
            # Simple Milvus search with single strategy
            collection_name = f"notebook_{notebook_id}_chunks"
            collection = Collection(collection_name)
            
            search_results = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=max_sources,
                expr=f'score >= {threshold}'
            )
            
            sources = []
            for hit in search_results[0]:
                if hit.score >= threshold:
                    sources.append(NotebookRAGSource(
                        content=hit.entity.get('content', ''),
                        metadata=hit.entity.get('metadata', {}),
                        score=hit.score,
                        document_id=hit.entity.get('document_id', ''),
                        document_name=hit.entity.get('document_name', ''),
                        collection=collection_name
                    ))
            
            return sources
            
        except Exception as e:
            self.logger.error(f"[SINGLE_STRATEGY] Error: {str(e)}")
            return []

    async def _execute_dual_strategy_retrieval(self, notebook_id: str, query: str, max_sources: int, threshold: float) -> List[NotebookRAGSource]:
        """Execute dual strategy retrieval for balanced approach."""
        try:
            # Execute two complementary retrieval strategies
            strategy1_sources = await self._execute_single_strategy_retrieval(
                notebook_id, query, max_sources // 2, threshold
            )
            
            # Second strategy with slightly different parameters
            strategy2_sources = await self._execute_single_strategy_retrieval(
                notebook_id, f"{query} project work", max_sources // 2, threshold + 0.05
            )
            
            # Combine and deduplicate
            all_sources = strategy1_sources + strategy2_sources
            seen_content = set()
            unique_sources = []
            
            for source in all_sources:
                content_key = source.content[:100]  # Use first 100 chars as key
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_sources.append(source)
            
            # Limit to max_sources and sort by score
            unique_sources.sort(key=lambda x: x.score, reverse=True)
            return unique_sources[:max_sources]
            
        except Exception as e:
            self.logger.error(f"[DUAL_STRATEGY] Error: {str(e)}")
            return []

    async def multi_stage_retrieval(
        self,
        notebook_id: str,
        query: str,
        intent_analysis: dict,
        total_available: int,
        max_sources: int
    ) -> AsyncGenerator[NotebookRAGResponse, None]:
        """
        Multi-stage retrieval for large datasets with progressive loading.
        
        Implements progressive loading strategy:
        1. Quick initial batch (50-100 items) - yields within 2-3 seconds  
        2. Progressive loading of remaining items in batches
        3. Background loading with progress updates
        4. Cache-aware optimization
        5. Circuit breaker for error recovery
        6. Memory usage monitoring
        
        Args:
            notebook_id: Notebook ID
            query: User query string
            intent_analysis: Query intent analysis results
            total_available: Total items available in notebook
            max_sources: Maximum sources to retrieve
            
        Yields:
            NotebookRAGResponse objects for each stage
        """
        import psutil
        import os
        
        # Circuit breaker state
        consecutive_failures = 0
        max_consecutive_failures = 3
        initial_memory_usage = None
        memory_limit_exceeded = False
        
        try:
            self.logger.info(f"[MULTI_STAGE] Starting multi-stage retrieval for notebook {notebook_id}")
            self.logger.info(f"[MULTI_STAGE] Query: '{query[:100]}...', max_sources: {max_sources}, available: {total_available}")
            
            # Monitor initial memory usage
            try:
                process = psutil.Process(os.getpid())
                initial_memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                self.logger.info(f"[MEMORY_MONITOR] Initial memory usage: {initial_memory_usage:.1f} MB")
            except Exception as mem_err:
                self.logger.warning(f"[MEMORY_MONITOR] Could not get initial memory usage: {str(mem_err)}")
                initial_memory_usage = None
            
            # Calculate batch sizes based on total requirements
            initial_batch_size = max(int(max_sources * 0.1), 50)  # 10% or minimum 50
            initial_batch_size = min(initial_batch_size, 100)  # Cap at 100 for quick response
            
            subsequent_batch_size = min(100, max(50, int(max_sources * 0.15)))  # 15% or 50-100
            
            self.logger.info(f"[MULTI_STAGE] Batch sizes: initial={initial_batch_size}, subsequent={subsequent_batch_size}")
            
            # Stage 1: Quick initial batch
            from app.core.db import SessionLocal
            db = SessionLocal()
            try:
                # Check for high-confidence comprehensive queries that need deterministic results
                wants_comprehensive = intent_analysis.get("wants_comprehensive", False)
                confidence = intent_analysis.get("confidence", 0)
                
                # Route to comprehensive mode for high-confidence comprehensive queries
                if wants_comprehensive and confidence > 0.8:
                    self.logger.info(f"[MULTI_STAGE_COMPREHENSIVE] Using deterministic comprehensive mode (confidence: {confidence:.2f})")
                    
                    initial_response = await self.query_notebook_adaptive(
                        db=db,
                        notebook_id=notebook_id,
                        query=query,
                        max_sources=initial_batch_size,
                        include_metadata=True,
                        collection_filter=collection_filter
                    )
                    
                    self.logger.info(f"[MULTI_STAGE_COMPREHENSIVE] Retrieved {len(initial_response.sources)} sources deterministically")
                else:
                    initial_response = await self.query_notebook_adaptive(
                        db=db,
                        notebook_id=notebook_id,
                        query=query,
                        max_sources=initial_batch_size,
                        include_metadata=True
                    )
                
                # Add stage metadata
                if not hasattr(initial_response, 'metadata') or not initial_response.metadata:
                    initial_response.metadata = {}
                
                initial_response.metadata.update({
                    "multi_stage": {
                        "stage": "initial",
                        "stage_number": 1,
                        "batch_size": initial_batch_size,
                        "total_stages_planned": max(2, int(max_sources / subsequent_batch_size) + 1),
                        "progress_percent": round((len(initial_response.sources) / max_sources) * 100, 1),
                        "more_available": len(initial_response.sources) < max_sources
                    }
                })
                
                self.logger.info(f"[MULTI_STAGE] Stage 1 completed: {len(initial_response.sources)} sources")
                yield initial_response
                
                # Stage 2+: Progressive loading of remaining results
                if len(initial_response.sources) < max_sources:
                    remaining_needed = max_sources - len(initial_response.sources)
                    retrieved_so_far = len(initial_response.sources)
                    stage_number = 2
                    
                    # Get unique document IDs from initial batch to avoid duplicates
                    processed_doc_ids = {source.document_id for source in initial_response.sources}
                    
                    while remaining_needed > 0 and retrieved_so_far < total_available and not memory_limit_exceeded:
                        # Circuit breaker check
                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.warning(f"[MULTI_STAGE] Circuit breaker triggered after {consecutive_failures} failures, stopping progressive loading")
                            break
                        
                        # Memory monitoring
                        if initial_memory_usage:
                            try:
                                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                                memory_increase = current_memory - initial_memory_usage
                                memory_increase_percent = (memory_increase / initial_memory_usage) * 100 if initial_memory_usage > 0 else 0
                                
                                # Log memory usage every few stages
                                if stage_number % 3 == 0:
                                    self.logger.debug(f"[MEMORY_MONITOR] Stage {stage_number}: Current memory: {current_memory:.1f} MB, "
                                                    f"increase: +{memory_increase:.1f} MB ({memory_increase_percent:.1f}%)")
                                
                                # Memory limit protection
                                if memory_increase_percent > 150:  # More than 150% increase
                                    self.logger.warning(f"[MEMORY_MONITOR] Memory usage increased by {memory_increase_percent:.1f}%, "
                                                       f"stopping progressive loading to prevent memory exhaustion")
                                    memory_limit_exceeded = True
                                    break
                                    
                            except Exception as mem_err:
                                self.logger.debug(f"[MEMORY_MONITOR] Memory check error: {str(mem_err)}")
                        
                        batch_size = min(subsequent_batch_size, remaining_needed)
                        
                        try:
                            # Progressive batch retrieval with offset simulation
                            # Since we can't easily offset in vector search, we retrieve more and filter
                            expanded_batch_size = min(retrieved_so_far + batch_size + 20, total_available)
                            
                            # Use same mode as initial response for consistency
                            if wants_comprehensive and confidence > 0.8:
                                # For comprehensive mode, we already got all results in stage 1
                                # No need for additional batches - return empty to exit loop
                                batch_response = NotebookRAGResponse(
                                    notebook_id=notebook_id,
                                    query=query,
                                    sources=[],
                                    total_sources=0,
                                    queried_documents=0,
                                    collections_searched=[]
                                )
                                self.logger.info(f"[MULTI_STAGE_COMPREHENSIVE] All results retrieved in stage 1, skipping additional batches")
                            else:
                                batch_response = await self.query_notebook_adaptive(
                                    db=db,
                                    notebook_id=notebook_id,
                                    query=query,
                                    max_sources=expanded_batch_size,
                                    include_metadata=True
                                )
                            
                            # Reset failure counter on successful retrieval
                            consecutive_failures = 0
                            
                            # Filter out already-processed documents
                            new_sources = []
                            for source in batch_response.sources:
                                if source.document_id not in processed_doc_ids:
                                    new_sources.append(source)
                                    processed_doc_ids.add(source.document_id)
                                    if len(new_sources) >= batch_size:
                                        break
                            
                            if not new_sources:
                                self.logger.info(f"[MULTI_STAGE] No new sources in stage {stage_number}, stopping")
                                break
                                
                        except Exception as batch_err:
                            consecutive_failures += 1
                            self.logger.error(f"[MULTI_STAGE] Error in stage {stage_number} (failure {consecutive_failures}/{max_consecutive_failures}): {str(batch_err)}")
                            
                            if consecutive_failures >= max_consecutive_failures:
                                self.logger.error(f"[MULTI_STAGE] Max failures reached, stopping progressive loading")
                                break
                            
                            # Continue to next iteration to retry or stop
                            continue
                        
                        # Create response for this stage
                        stage_response = NotebookRAGResponse(
                            notebook_id=notebook_id,
                            query=query,
                            sources=new_sources,
                            total_sources=len(new_sources),
                            queried_documents=len(set(s.document_id for s in new_sources)),
                            collections_searched=batch_response.collections_searched
                        )
                        
                        retrieved_so_far += len(new_sources)
                        remaining_needed -= len(new_sources)
                        
                        # Add stage metadata
                        stage_response.metadata = {
                            "multi_stage": {
                                "stage": "progressive",
                                "stage_number": stage_number,
                                "batch_size": len(new_sources),
                                "total_retrieved_so_far": retrieved_so_far,
                                "progress_percent": round((retrieved_so_far / max_sources) * 100, 1),
                                "more_available": remaining_needed > 0 and retrieved_so_far < total_available
                            }
                        }
                        
                        self.logger.info(f"[MULTI_STAGE] Stage {stage_number} completed: "
                                       f"{len(new_sources)} new sources, {retrieved_so_far} total")
                        yield stage_response
                        
                        stage_number += 1
                        
                        # Add small delay to prevent overwhelming the system
                        import asyncio
                        await asyncio.sleep(0.1)
                        
                        # Safety break to prevent infinite loops
                        if stage_number > 10:
                            self.logger.warning(f"[MULTI_STAGE] Reached maximum stages limit (10), stopping")
                            break
                
            finally:
                db.close()
                
                # Final memory monitoring
                if initial_memory_usage:
                    try:
                        final_memory = process.memory_info().rss / 1024 / 1024  # MB
                        total_memory_increase = final_memory - initial_memory_usage
                        total_memory_increase_percent = (total_memory_increase / initial_memory_usage) * 100 if initial_memory_usage > 0 else 0
                        self.logger.info(f"[MEMORY_MONITOR] Final memory usage: {final_memory:.1f} MB, "
                                        f"total increase: +{total_memory_increase:.1f} MB ({total_memory_increase_percent:.1f}%)")
                    except Exception as final_mem_err:
                        self.logger.debug(f"[MEMORY_MONITOR] Final memory check error: {str(final_mem_err)}")
                
            completion_message = f"[MULTI_STAGE] Multi-stage retrieval completed for notebook {notebook_id}"
            if memory_limit_exceeded:
                completion_message += " (stopped due to memory limit)"
            elif consecutive_failures >= max_consecutive_failures:
                completion_message += " (stopped due to circuit breaker)"
            
            self.logger.info(completion_message)
            
        except Exception as e:
            self.logger.error(f"[MULTI_STAGE] Error in multi-stage retrieval: {str(e)}")
            # Fallback to single-stage retrieval
            try:
                from app.core.db import SessionLocal
                db = SessionLocal()
                try:
                    fallback_response = await self.query_notebook_adaptive(
                        db=db,
                        notebook_id=notebook_id,
                        query=query,
                        max_sources=min(max_sources, 100),  # Conservative fallback
                        include_metadata=True
                    )
                    
                    if not hasattr(fallback_response, 'metadata') or not fallback_response.metadata:
                        fallback_response.metadata = {}
                    
                    fallback_response.metadata.update({
                        "multi_stage": {
                            "stage": "fallback",
                            "error": str(e),
                            "fallback_used": True
                        }
                    })
                    
                    yield fallback_response
                finally:
                    db.close()
            except Exception as fallback_error:
                self.logger.error(f"[MULTI_STAGE] Fallback also failed: {str(fallback_error)}")
                raise

    async def execute_intelligent_plan(
        self,
        db: Session,
        notebook_id: str,
        plan: TaskExecutionPlan,
        include_metadata: bool = True,
        collection_filter: Optional[List[str]] = None
    ) -> NotebookRAGResponse:
        """
        Execute an AI-generated task plan with multi-strategy retrieval.
        
        This method implements intelligent retrieval execution by:
        1. Running multiple retrieval strategies from the plan
        2. Combining results intelligently with deduplication
        3. Returning comprehensive results with metadata
        
        Args:
            db: Database session
            notebook_id: Notebook ID to query
            plan: TaskExecutionPlan containing strategies and requirements
            include_metadata: Whether to include metadata in results
            collection_filter: Optional filter for specific collections
            
        Returns:
            Comprehensive RAG query results from multiple strategies
        """
        try:
            self.logger.info(f"[INTELLIGENT_PLAN] Executing plan with {len(plan.retrieval_strategies)} strategies for notebook {notebook_id}")
            
            # Execute all retrieval strategies concurrently
            strategy_results = []
            for i, strategy in enumerate(plan.retrieval_strategies):
                self.logger.info(f"[INTELLIGENT_PLAN] Executing strategy {i+1}: {strategy.description}")
                
                try:
                    result = await self._execute_retrieval_strategy(
                        db=db,
                        notebook_id=notebook_id,
                        strategy=strategy,
                        include_metadata=include_metadata,
                        collection_filter=collection_filter
                    )
                    strategy_results.append({
                        'strategy': strategy,
                        'result': result,
                        'strategy_index': i
                    })
                    self.logger.info(f"[INTELLIGENT_PLAN] Strategy {i+1} returned {len(result.sources)} sources")
                    
                except Exception as e:
                    self.logger.warning(f"[INTELLIGENT_PLAN] Strategy {i+1} failed: {str(e)}")
                    # Continue with other strategies
                    continue
            
            if not strategy_results:
                self.logger.warning(f"[INTELLIGENT_PLAN] All strategies failed, returning empty result")
                return NotebookRAGResponse(
                    notebook_id=notebook_id,
                    query="list_all_projects",
                    sources=[],
                    total_sources=0,
                    queried_documents=0,
                    collections_searched=[]
                )
            
            # Combine results intelligently
            combined_result = self._combine_strategy_results(strategy_results, plan)
            
            # Plan execution metadata would be stored elsewhere if needed
            # combined_result has all required fields already
            
            self.logger.info(f"[INTELLIGENT_PLAN] Combined result: {len(combined_result.sources)} sources from {len(strategy_results)} successful strategies")
            return combined_result
            
        except Exception as e:
            self.logger.error(f"[INTELLIGENT_PLAN] Plan execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    async def _execute_retrieval_strategy(
        self,
        db: Session,
        notebook_id: str,
        strategy: RetrievalStrategy,
        include_metadata: bool = True,
        collection_filter: Optional[List[str]] = None
    ) -> NotebookRAGResponse:
        """
        Execute a single retrieval strategy.
        
        Args:
            db: Database session
            notebook_id: Notebook ID to query
            strategy: RetrievalStrategy to execute
            include_metadata: Whether to include metadata
            collection_filter: Optional filter for specific collections
            
        Returns:
            Raw results from this strategy
        """
        try:
            # Check if this is a comprehensive query and override max_chunks if needed
            intent_analysis = await self._analyze_query_intent(strategy.query)
            is_comprehensive = (intent_analysis.get('query_type') == 'comprehensive' and 
                               intent_analysis.get('quantity_intent') == 'all')
            self.logger.info(f"[COMPREHENSIVE_STRATEGY_DEBUG] Intent: query_type={intent_analysis.get('query_type')}, quantity_intent={intent_analysis.get('quantity_intent')}, is_comprehensive={is_comprehensive}")
            
            # For comprehensive queries, override strategy max_chunks to ensure all data is retrieved
            effective_top_k = strategy.max_chunks
            if is_comprehensive:
                try:
                    total_available = await self.get_actual_content_count(notebook_id)
                    effective_top_k = min(total_available, 100)  # Use all available, capped at reasonable limit
                    self.logger.info(f"[COMPREHENSIVE_STRATEGY] Override strategy max_chunks from {strategy.max_chunks} to {effective_top_k} for comprehensive query")
                except Exception as e:
                    effective_top_k = 100  # Fallback to high limit
                    self.logger.warning(f"Could not get content count for strategy override: {e}")
            
            # Use the existing query_notebook method with strategy parameters
            result = await self.query_notebook(
                db=db,
                notebook_id=notebook_id,
                query=strategy.query,
                top_k=effective_top_k,
                include_metadata=include_metadata,
                collection_filter=collection_filter
            )
            
            # Filter results based on strategy threshold if needed
            if strategy.threshold > 0:
                filtered_sources = []
                for source in result.sources:
                    # Assume score is stored in metadata or use a default scoring
                    score = source.metadata.get('score', 1.0) if source.metadata else 1.0
                    if score >= strategy.threshold:
                        filtered_sources.append(source)
                
                result.sources = filtered_sources
                result.total_sources = len(filtered_sources)
            
            # Strategy metadata would be stored elsewhere if needed
            # result already contains all required fields
            
            return result
            
        except Exception as e:
            self.logger.error(f"[STRATEGY_EXEC] Failed to execute strategy '{strategy.description}': {str(e)}")
            raise

    def _combine_strategy_results(
        self,
        strategy_results: List[Dict[str, Any]],
        plan: TaskExecutionPlan
    ) -> NotebookRAGResponse:
        """
        Combine results from multiple strategies intelligently.
        
        This method:
        1. Deduplicates based on content similarity
        2. Preserves diversity of sources
        3. Respects the plan's verification rules
        
        Args:
            strategy_results: List of results from different strategies
            plan: Original execution plan with requirements
            
        Returns:
            Combined and deduplicated results
        """
        try:
            all_sources = []
            seen_contents = set()
            
            # Collect all sources with deduplication
            for strategy_data in strategy_results:
                result = strategy_data['result']
                strategy = strategy_data['strategy']
                strategy_index = strategy_data['strategy_index']
                
                for source in result.sources:
                    # Create content hash for deduplication
                    content_hash = hashlib.md5(source.content.encode()).hexdigest()
                    
                    if plan.verification.check_for_duplicates:
                        if content_hash in seen_contents:
                            continue
                        seen_contents.add(content_hash)
                    
                    # Add strategy source info to metadata
                    if source.metadata is None:
                        source.metadata = {}
                    
                    source.metadata.update({
                        'source_strategy_index': strategy_index,
                        'source_strategy_description': strategy.description,
                        'source_strategy_query': strategy.query
                    })
                    
                    all_sources.append(source)
            
            # Sort sources by relevance/score if available
            try:
                all_sources.sort(key=lambda x: x.metadata.get('score', 0.0) if x.metadata else 0.0, reverse=True)
            except (TypeError, AttributeError, KeyError) as e:
                self.logger.warning(f"[COMBINE_RESULTS] Failed to sort sources by score: {e}, keeping original order")
            except Exception as e:
                self.logger.error(f"[COMBINE_RESULTS] Unexpected error sorting sources: {type(e).__name__}: {e}")
            
            # Apply verification rules
            if len(all_sources) < plan.verification.min_expected_results:
                self.logger.warning(f"[COMBINE_RESULTS] Got {len(all_sources)} sources, expected minimum {plan.verification.min_expected_results}")
            
            # Limit results if needed (preserve top results)
            max_results = 500  # Reasonable limit
            if len(all_sources) > max_results:
                all_sources = all_sources[:max_results]
                self.logger.info(f"[COMBINE_RESULTS] Limited results to {max_results} sources")
            
            # Combine extracted projects from all strategies
            all_extracted_projects = []
            for sr in strategy_results:
                if hasattr(sr['result'], 'extracted_projects') and sr['result'].extracted_projects:
                    all_extracted_projects.extend(sr['result'].extracted_projects)
            
            # Deduplicate projects by name and company
            unique_projects = {}
            for project in all_extracted_projects:
                key = f"{project.name}:{project.company}"
                if key not in unique_projects:
                    unique_projects[key] = project
            
            combined_extracted_projects = list(unique_projects.values()) if unique_projects else None
            
            return NotebookRAGResponse(
                sources=all_sources,
                total_sources=len(all_sources),
                notebook_id=strategy_results[0]['result'].notebook_id,
                query=strategy_results[0]['result'].query,
                queried_documents=sum(sr['result'].queried_documents for sr in strategy_results),
                collections_searched=list(set(col for sr in strategy_results for col in sr['result'].collections_searched)),
                extracted_projects=combined_extracted_projects
            )
            
        except Exception as e:
            self.logger.error(f"[COMBINE_RESULTS] Failed to combine results: {str(e)}")
            # Return the first successful result as fallback
            if strategy_results:
                return strategy_results[0]['result']
            else:
                return NotebookRAGResponse(
                    notebook_id="unknown",
                    query="unknown",
                    sources=[],
                    total_sources=0,
                    queried_documents=0,
                    collections_searched=[]
                )

    async def execute_correction_strategies(
        self,
        db: Session,
        notebook_id: str,
        original_response: NotebookRAGResponse,
        correction_strategies: List
    ) -> NotebookRAGResponse:
        """
        Execute correction strategies to improve incomplete results.
        Called by verification loop when initial results are insufficient.
        
        Args:
            db: Database session
            notebook_id: Notebook ID to query
            original_response: Original RAG response that needs improvement
            correction_strategies: List of RetrievalStrategy objects with correction approaches
            
        Returns:
            NotebookRAGResponse with improved results, or original response if correction fails
        """
        try:
            self.logger.info(f"[CORRECTION] Executing {len(correction_strategies)} correction strategies for notebook {notebook_id}")
            
            if not correction_strategies:
                self.logger.warning("[CORRECTION] No correction strategies provided, returning original response")
                return original_response
            
            # Track all sources found during correction
            correction_sources = []
            seen_document_ids = set()
            
            # Keep track of original sources to avoid duplicates
            original_document_ids = {source.document_id for source in original_response.sources}
            
            # Execute each correction strategy
            for i, strategy in enumerate(correction_strategies):
                try:
                    self.logger.info(f"[CORRECTION] Executing strategy {i+1}/{len(correction_strategies)}: {strategy.description}")
                    
                    # Execute the strategy using the existing query_notebook method with strategy parameters
                    strategy_response = await self.query_notebook(
                        db=db,
                        notebook_id=notebook_id,
                        query=strategy.query,
                        top_k=strategy.max_chunks,
                        include_metadata=True
                    )
                    
                    if strategy_response and strategy_response.sources:
                        self.logger.info(f"[CORRECTION] Strategy {i+1} found {len(strategy_response.sources)} sources")
                        
                        # Filter sources by threshold and avoid duplicates
                        for source in strategy_response.sources:
                            # Apply threshold filter
                            if source.score >= strategy.threshold:
                                # Avoid duplicates with original response and within correction results
                                if (source.document_id not in original_document_ids and 
                                    source.document_id not in seen_document_ids):
                                    correction_sources.append(source)
                                    seen_document_ids.add(source.document_id)
                    else:
                        self.logger.info(f"[CORRECTION] Strategy {i+1} found no sources")
                        
                except Exception as e:
                    self.logger.error(f"[CORRECTION] Strategy {i+1} failed: {str(e)}")
                    continue
            
            # Combine original and correction sources
            if correction_sources:
                # Sort correction sources by score (descending)
                correction_sources.sort(key=lambda x: x.score, reverse=True)
                
                # Combine with original sources
                combined_sources = list(original_response.sources) + correction_sources
                
                # Sort combined results by score
                combined_sources.sort(key=lambda x: x.score, reverse=True)
                
                self.logger.info(f"[CORRECTION] Successfully improved results: {len(original_response.sources)} → {len(combined_sources)} total sources")
                
                # Return improved response
                return NotebookRAGResponse(
                    notebook_id=notebook_id,
                    query=original_response.query,
                    sources=combined_sources,
                    total_sources=len(combined_sources),
                    queried_documents=original_response.queried_documents + len(seen_document_ids),
                    collections_searched=getattr(original_response, 'collections_searched', [])
                )
            else:
                self.logger.warning("[CORRECTION] No additional sources found through correction strategies")
                return original_response
                
        except Exception as e:
            self.logger.error(f"[CORRECTION] Failed to execute correction strategies: {str(e)}")
            # Return original response as fallback
            return original_response

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


class ConversationContextManager:
    """
    Manages conversation context and caches retrieval results to avoid repeated queries.
    Implements Phase 2 of intelligent message handling with Redis-based caching.
    """
    
    def __init__(self):
        """Initialize ConversationContextManager with Redis client."""
        self.logger = logging.getLogger(__name__)
        self.context_ttl = 3600  # 1 hour cache TTL for conversation intelligence
        self.redis_client = None
        
    def _get_redis_client(self):
        """Get Redis client using existing infrastructure."""
        if self.redis_client is None:
            self.redis_client = get_redis_client(decode_responses=True)
        return self.redis_client
        
    def _get_cache_key(self, conversation_id: str) -> str:
        """Generate consistent cache key for conversation context."""
        return f"conversation_context:{conversation_id}"
        
    def _get_metadata_key(self, conversation_id: str) -> str:
        """Generate cache key for conversation metadata."""
        return f"conversation_meta:{conversation_id}"
    
    def _format_cache_age(self, age_seconds: int) -> str:
        """
        Format cache age in human-readable format.
        
        Args:
            age_seconds: Age in seconds
            
        Returns:
            Human-readable age string (e.g., "5 minutes ago", "2 hours ago")
        """
        if age_seconds < 60:
            return f"{age_seconds} seconds ago"
        elif age_seconds < 3600:  # Less than 1 hour
            minutes = age_seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif age_seconds < 86400:  # Less than 1 day
            hours = age_seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:  # 1 day or more
            days = age_seconds // 86400
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    async def cache_retrieval_context(self, conversation_id: str, context: Dict[str, Any]) -> bool:
        """
        Cache retrieval results and context for reuse in follow-up questions.
        
        Args:
            conversation_id: Unique conversation identifier
            context: Dictionary containing retrieval context with keys:
                - sources: List of retrieved sources
                - query: Original query that triggered retrieval
                - extracted_entities: Extracted entities/projects from query
                - timestamp: When retrieval was performed
                - metadata: Additional retrieval metadata
        
        Returns:
            bool: True if caching succeeded, False otherwise
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                self.logger.warning("Redis not available, skipping context caching")
                return False
            
            # Prepare cache data with timestamp
            cache_data = {
                **context,
                'cached_at': datetime.now().isoformat(),
                'cache_ttl': self.context_ttl
            }
            
            # Store in Redis with TTL
            cache_key = self._get_cache_key(conversation_id)
            redis_client.setex(
                cache_key,
                self.context_ttl,
                json.dumps(cache_data, default=str)
            )
            
            # Store metadata separately for quick checks
            metadata = {
                'query': context.get('query', '')[:100],  # Truncated query for metadata
                'source_count': len(context.get('sources', [])),
                'cached_at': cache_data['cached_at'],
                'has_context': True
            }
            
            metadata_key = self._get_metadata_key(conversation_id)
            redis_client.setex(
                metadata_key,
                self.context_ttl,
                json.dumps(metadata)
            )
            
            self.logger.info(f"[CACHE] Cached context for conversation {conversation_id}: "
                           f"{len(context.get('sources', []))} sources, TTL={self.context_ttl}s")
            return True
            
        except Exception as e:
            self.logger.error(f"[CACHE] Failed to cache retrieval context: {str(e)}")
            return False
    
    async def get_cached_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached context if still valid.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Cached context dictionary or None if not found/expired
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return None
                
            cache_key = self._get_cache_key(conversation_id)
            cached_data = redis_client.get(cache_key)
            
            if cached_data:
                # Handle both bytes and string responses from Redis
                if isinstance(cached_data, bytes):
                    cached_data = cached_data.decode('utf-8')
                
                context = json.loads(cached_data)
                self.logger.info(f"[CACHE] Retrieved cached context for conversation {conversation_id}: "
                               f"{len(context.get('sources', []))} sources")
                return context
                
            return None
            
        except Exception as e:
            self.logger.error(f"[CACHE] Failed to retrieve cached context: {str(e)}")
            return None
    
    async def has_recent_context(self, conversation_id: str, max_age_minutes: int = 5) -> bool:
        """
        Check if we have recent retrieval context within the specified time window.
        
        Args:
            conversation_id: Unique conversation identifier
            max_age_minutes: Maximum age in minutes to consider "recent"
            
        Returns:
            bool: True if recent context exists, False otherwise
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return False
                
            # Check metadata first (lighter operation)
            metadata_key = self._get_metadata_key(conversation_id)
            metadata_json = redis_client.get(metadata_key)
            
            if not metadata_json:
                return False
                
            # Handle both bytes and string responses from Redis
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode('utf-8')
                
            metadata = json.loads(metadata_json)
            cached_at_str = metadata.get('cached_at')
            
            if not cached_at_str:
                return False
                
            cached_at = datetime.fromisoformat(cached_at_str)
            age_minutes = (datetime.now() - cached_at).total_seconds() / 60
            
            has_recent = age_minutes <= max_age_minutes
            
            if has_recent:
                self.logger.info(f"[CACHE] Found recent context for conversation {conversation_id}: "
                               f"{age_minutes:.1f} minutes old, {metadata.get('source_count', 0)} sources")
            
            return has_recent
            
        except Exception as e:
            self.logger.error(f"[CACHE] Failed to check recent context: {str(e)}")
            return False
    
    async def invalidate_context(self, conversation_id: str) -> bool:
        """
        Clear cached context (for new topics or explicit invalidation).
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            bool: True if invalidation succeeded, False otherwise
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return False
                
            cache_key = self._get_cache_key(conversation_id)
            metadata_key = self._get_metadata_key(conversation_id)
            
            # Delete both context and metadata
            deleted_count = redis_client.delete(cache_key, metadata_key)
            
            self.logger.info(f"[CACHE] Invalidated context for conversation {conversation_id}: "
                           f"{deleted_count} keys deleted")
            return deleted_count > 0
            
        except Exception as e:
            self.logger.error(f"[CACHE] Failed to invalidate context: {str(e)}")
            return False
            
    async def get_cache_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics for monitoring and UI display.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Dictionary with cache statistics including:
            - status: 'cached', 'no_cache', 'expired', 'redis_unavailable'
            - cache_age_seconds: Age in seconds
            - cache_age_human: Human-readable age (e.g., "5 minutes ago")
            - original_query: Full original query text
            - source_count: Number of cached sources
            - cached_at: ISO timestamp when cached
            - ttl_seconds: Time to live remaining
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                self.logger.warning(f"[CACHE] Redis client unavailable for conversation {conversation_id}")
                return {
                    'status': 'redis_unavailable',
                    'conversation_id': conversation_id,
                    'has_context': False,
                    'error': 'Redis connection unavailable'
                }
                
            metadata_key = self._get_metadata_key(conversation_id)
            metadata_json = redis_client.get(metadata_key)
            
            if not metadata_json:
                self.logger.debug(f"[CACHE] No cache metadata found for conversation {conversation_id}")
                return {
                    'status': 'no_cache',
                    'conversation_id': conversation_id,
                    'has_context': False,
                    'cache_age_seconds': None,
                    'cache_age_human': 'No cache',
                    'original_query': None,
                    'source_count': 0,
                    'cached_at': None,
                    'ttl_seconds': -1
                }
                
            # Handle both bytes and string responses from Redis
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode('utf-8')
                
            metadata = json.loads(metadata_json)
            
            # Calculate cache age and human-readable format
            cached_at_str = metadata.get('cached_at')
            cache_age_seconds = None
            cache_age_human = 'Unknown age'
            
            if cached_at_str:
                try:
                    cached_at = datetime.fromisoformat(cached_at_str)
                    cache_age_seconds = int((datetime.now() - cached_at).total_seconds())
                    cache_age_human = self._format_cache_age(cache_age_seconds)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"[CACHE] Invalid cached_at format '{cached_at_str}': {e}")
                    cache_age_human = 'Invalid timestamp'
            
            # Get TTL for cache expiration check
            cache_key = self._get_cache_key(conversation_id)
            ttl_seconds = redis_client.ttl(cache_key)
            
            # Determine cache status
            if ttl_seconds <= 0:
                status = 'expired'
                self.logger.info(f"[CACHE] Cache expired for conversation {conversation_id} (TTL: {ttl_seconds})")
            else:
                status = 'cached'
                self.logger.debug(f"[CACHE] Active cache for conversation {conversation_id} (TTL: {ttl_seconds}s, Age: {cache_age_human})")
            
            # Get original query from cache context if available
            original_query = None
            try:
                cache_data_json = redis_client.get(cache_key)
                if cache_data_json:
                    if isinstance(cache_data_json, bytes):
                        cache_data_json = cache_data_json.decode('utf-8')
                    cache_data = json.loads(cache_data_json)
                    original_query = cache_data.get('query', metadata.get('query', ''))
            except Exception as e:
                self.logger.warning(f"[CACHE] Could not retrieve original query: {e}")
                original_query = metadata.get('query', '')
            
            result = {
                'status': status,
                'conversation_id': conversation_id,
                'has_context': status in ['cached', 'expired'],
                'cache_age_seconds': cache_age_seconds,
                'cache_age_human': cache_age_human,
                'original_query': original_query or '',
                'source_count': metadata.get('source_count', 0),
                'cached_at': cached_at_str,
                'ttl_seconds': ttl_seconds
            }
            
            self.logger.info(f"[CACHE] Cache stats for {conversation_id}: {status}, {result['source_count']} sources, {cache_age_human}")
            return result
            
        except Exception as e:
            self.logger.error(f"[CACHE] Failed to get cache stats for {conversation_id}: {str(e)}")
            self.logger.error(f"[CACHE] Cache stats error traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'conversation_id': conversation_id,
                'has_context': False,
                'error': str(e),
                'cache_age_seconds': None,
                'cache_age_human': 'Error retrieving age',
                'original_query': None,
                'source_count': 0,
                'cached_at': None,
                'ttl_seconds': -1
            }
    
    async def store_conversation_response(self, conversation_id: str, user_message: str, ai_response: str, sources: List[Dict] = None) -> bool:
        """
        Store the conversation response for LLM-driven conversation intelligence.
        
        This allows the LLM to see what it previously said and make intelligent decisions
        about whether to retrieve new data or work with existing context.
        
        Args:
            conversation_id: Unique conversation identifier
            user_message: The user's message
            ai_response: The AI's complete response
            sources: Optional sources used in the response
            
        Returns:
            bool: True if storage succeeded, False otherwise
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                self.logger.warning("Redis not available, skipping conversation response storage")
                return False
            
            # Prepare conversation response data
            response_data = {
                'user_message': user_message,
                'ai_response': ai_response,
                'sources_count': len(sources) if sources else 0,
                'timestamp': datetime.now().isoformat(),
                'ttl': self.context_ttl
            }
            
            # Store with shortened response for memory efficiency if too long
            if len(ai_response) > 2000:
                response_data['ai_response_full'] = ai_response
                response_data['ai_response'] = ai_response[:2000] + "... [truncated]"
            
            # Store in Redis with TTL
            response_key = f"conversation_response:{conversation_id}"
            redis_client.setex(
                response_key,
                self.context_ttl,
                json.dumps(response_data, default=str)
            )
            
            self.logger.info(f"[CONVERSATION_MEMORY] Stored response for conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"[CONVERSATION_MEMORY] Failed to store conversation response: {str(e)}")
            return False
    
    async def get_conversation_context(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the last conversation exchange for LLM context.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Dictionary with last conversation context or None if not found
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return None
                
            response_key = f"conversation_response:{conversation_id}"
            response_data = redis_client.get(response_key)
            
            if response_data:
                # Handle both bytes and string responses from Redis
                if isinstance(response_data, bytes):
                    response_data = response_data.decode('utf-8')
                
                context = json.loads(response_data)
                self.logger.info(f"[CONVERSATION_MEMORY] Retrieved conversation context for {conversation_id}")
                return context
                
            return None
            
        except Exception as e:
            self.logger.error(f"[CONVERSATION_MEMORY] Failed to retrieve conversation context: {str(e)}")
            return None


class IntelligentRoutingMetrics:
    """
    Comprehensive metrics collection system for intelligent routing decisions.
    
    Tracks efficiency and performance of the intelligent message handling system:
    - Routing decisions and their frequency
    - Cache effectiveness and hit rates  
    - Retrieval planning optimization
    - Performance timings by operation type
    - Resource savings and efficiency gains
    """
    
    def __init__(self):
        """Initialize metrics collector with Redis client."""
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.metrics_ttl = 86400  # 24 hours storage
        
    def _get_redis_client(self):
        """Get Redis client using existing infrastructure."""
        if self.redis_client is None:
            self.redis_client = get_redis_client(decode_responses=True)
        return self.redis_client
    
    async def log_routing_decision(self, conversation_id: str, decision_data: Dict[str, Any]) -> bool:
        """
        Log routing decision with context for efficiency analysis.
        
        Args:
            conversation_id: Unique conversation identifier
            decision_data: Dictionary containing decision details
            
        Returns:
            bool: True if logged successfully
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return False
                
            # Create timestamped decision entry
            decision_entry = {
                **decision_data,
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'date_hour': datetime.now().strftime('%Y-%m-%d-%H')  # For hourly aggregation
            }
            
            # Store individual decision
            decision_key = f"routing_decision:{conversation_id}:{int(time.time()*1000)}"
            redis_client.setex(decision_key, self.metrics_ttl, json.dumps(decision_entry))
            
            # Update hourly aggregation counters
            hour_key = f"routing_stats:{decision_entry['date_hour']}"
            routing_type = decision_data.get('routing_decision', 'unknown')
            
            # Increment counters for this hour
            redis_client.hincrby(hour_key, f"total_messages", 1)
            redis_client.hincrby(hour_key, f"routing_{routing_type}", 1)
            redis_client.expire(hour_key, self.metrics_ttl)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"[METRICS] Failed to log routing decision: {str(e)}")
            return False
    
    async def log_cache_event(self, conversation_id: str, event_type: str, hit: bool, time_saved: float = 0) -> bool:
        """
        Log cache hit/miss events with timing data.
        
        Args:
            conversation_id: Unique conversation identifier
            event_type: Type of cache event ('context_reference', 'retrieval_cache', etc.)
            hit: True if cache hit, False if miss
            time_saved: Estimated time saved in seconds (for hits)
            
        Returns:
            bool: True if logged successfully
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return False
                
            # Create cache event entry
            cache_entry = {
                'conversation_id': conversation_id,
                'event_type': event_type,
                'hit': hit,
                'time_saved': time_saved if hit else 0,
                'timestamp': datetime.now().isoformat(),
                'date_hour': datetime.now().strftime('%Y-%m-%d-%H')
            }
            
            # Store individual cache event
            cache_key = f"cache_event:{conversation_id}:{int(time.time()*1000)}"
            redis_client.setex(cache_key, self.metrics_ttl, json.dumps(cache_entry))
            
            # Update hourly cache statistics
            hour_key = f"cache_stats:{cache_entry['date_hour']}"
            redis_client.hincrby(hour_key, "total_cache_attempts", 1)
            
            if hit:
                redis_client.hincrby(hour_key, "cache_hits", 1)
                if time_saved > 0:
                    # Track time savings (stored as milliseconds for precision)
                    redis_client.hincrbyfloat(hour_key, "total_time_saved", time_saved)
            else:
                redis_client.hincrby(hour_key, "cache_misses", 1)
                
            redis_client.expire(hour_key, self.metrics_ttl)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"[METRICS] Failed to log cache event: {str(e)}")
            return False
    
    async def log_retrieval_plan(self, conversation_id: str, plan: RetrievalPlan, execution_time: float) -> bool:
        """
        Log retrieval planning decisions and execution timing.
        
        Args:
            conversation_id: Unique conversation identifier
            plan: RetrievalPlan instance with intensity and strategy details
            execution_time: Time taken to execute the plan in seconds
            
        Returns:
            bool: True if logged successfully
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return False
                
            # Create retrieval plan entry
            plan_entry = {
                'conversation_id': conversation_id,
                'intensity': plan.intensity.value,
                'max_sources': plan.max_sources,
                'use_multiple_strategies': plan.use_multiple_strategies,
                'reasoning': plan.reasoning[:200],  # Truncate for storage
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'date_hour': datetime.now().strftime('%Y-%m-%d-%H')
            }
            
            # Store individual plan execution
            plan_key = f"retrieval_plan:{conversation_id}:{int(time.time()*1000)}"
            redis_client.setex(plan_key, self.metrics_ttl, json.dumps(plan_entry))
            
            # Update hourly retrieval statistics  
            hour_key = f"retrieval_stats:{plan_entry['date_hour']}"
            intensity = plan.intensity.value
            
            redis_client.hincrby(hour_key, "total_retrievals", 1)
            redis_client.hincrby(hour_key, f"intensity_{intensity}", 1)
            redis_client.hincrbyfloat(hour_key, f"total_time_{intensity}", execution_time)
            redis_client.hincrbyfloat(hour_key, "total_execution_time", execution_time)
            redis_client.expire(hour_key, self.metrics_ttl)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"[METRICS] Failed to log retrieval plan: {str(e)}")
            return False
    
    async def log_performance_timing(self, operation_type: str, conversation_id: str, execution_time: float, metadata: Optional[Dict] = None) -> bool:
        """
        Log performance timing for different operation types.
        
        Args:
            operation_type: Type of operation ('simple_response', 'cached_response', etc.)
            conversation_id: Unique conversation identifier
            execution_time: Time taken in seconds
            metadata: Optional additional metadata
            
        Returns:
            bool: True if logged successfully
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return False
                
            # Create timing entry
            timing_entry = {
                'operation_type': operation_type,
                'conversation_id': conversation_id,
                'execution_time': execution_time,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
                'date_hour': datetime.now().strftime('%Y-%m-%d-%H')
            }
            
            # Store individual timing
            timing_key = f"performance:{conversation_id}:{int(time.time()*1000)}"
            redis_client.setex(timing_key, self.metrics_ttl, json.dumps(timing_entry))
            
            # Update hourly performance statistics
            hour_key = f"performance_stats:{timing_entry['date_hour']}"
            redis_client.hincrby(hour_key, f"count_{operation_type}", 1)
            redis_client.hincrbyfloat(hour_key, f"total_time_{operation_type}", execution_time)
            redis_client.expire(hour_key, self.metrics_ttl)
            
            return True
            
        except Exception as e:
            self.logger.warning(f"[METRICS] Failed to log performance timing: {str(e)}")
            return False
    
    async def get_efficiency_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive efficiency summary for the past N hours.
        
        Args:
            hours: Number of hours to analyze (default 24)
            
        Returns:
            Dict with routing efficiency, cache effectiveness, and performance metrics
        """
        try:
            redis_client = self._get_redis_client()
            if not redis_client:
                return {'status': 'redis_unavailable'}
            
            # Calculate time range
            now = datetime.now()
            hour_keys = []
            
            for i in range(hours):
                hour_time = now - timedelta(hours=i)
                hour_key = hour_time.strftime('%Y-%m-%d-%H')
                hour_keys.append(hour_key)
            
            # Aggregate routing statistics
            routing_stats = {}
            cache_stats = {}
            retrieval_stats = {}
            performance_stats = {}
            
            for hour_key in hour_keys:
                # Get routing data
                routing_data = redis_client.hgetall(f"routing_stats:{hour_key}")
                for key, value in routing_data.items():
                    routing_stats[key] = routing_stats.get(key, 0) + int(value)
                
                # Get cache data
                cache_data = redis_client.hgetall(f"cache_stats:{hour_key}")
                for key, value in cache_data.items():
                    if key == 'total_time_saved':
                        cache_stats[key] = cache_stats.get(key, 0) + float(value)
                    else:
                        cache_stats[key] = cache_stats.get(key, 0) + int(value)
                
                # Get retrieval data
                retrieval_data = redis_client.hgetall(f"retrieval_stats:{hour_key}")
                for key, value in retrieval_data.items():
                    if 'time' in key:
                        retrieval_stats[key] = retrieval_stats.get(key, 0) + float(value)
                    else:
                        retrieval_stats[key] = retrieval_stats.get(key, 0) + int(value)
                
                # Get performance data
                perf_data = redis_client.hgetall(f"performance_stats:{hour_key}")
                for key, value in perf_data.items():
                    if 'time' in key:
                        performance_stats[key] = performance_stats.get(key, 0) + float(value)
                    else:
                        performance_stats[key] = performance_stats.get(key, 0) + int(value)
            
            # Calculate derived metrics
            total_messages = routing_stats.get('total_messages', 0)
            if total_messages == 0:
                return {'status': 'no_data', 'period_hours': hours}
            
            # Routing efficiency calculation
            simple_responses = routing_stats.get('routing_simple_response', 0) + routing_stats.get('routing_simple_response_for_greeting', 0) + routing_stats.get('routing_simple_response_for_acknowledgment', 0)
            cached_responses = routing_stats.get('routing_cached_response', 0) + routing_stats.get('routing_cached_context_reuse', 0)
            efficiency_saves = simple_responses + cached_responses
            efficiency_rate = (efficiency_saves / total_messages * 100) if total_messages > 0 else 0
            
            # Cache effectiveness
            total_cache_attempts = cache_stats.get('total_cache_attempts', 0)
            cache_hits = cache_stats.get('cache_hits', 0)
            cache_hit_rate = (cache_hits / total_cache_attempts * 100) if total_cache_attempts > 0 else 0
            avg_time_saved = (cache_stats.get('total_time_saved', 0) / cache_hits) if cache_hits > 0 else 0
            
            # Performance comparison
            def safe_avg(total_time, count):
                return (total_time / count) if count > 0 else 0
                
            simple_avg = safe_avg(performance_stats.get('total_time_simple_response', 0), performance_stats.get('count_simple_response', 1))
            cached_avg = safe_avg(performance_stats.get('total_time_cached_response', 0), performance_stats.get('count_cached_response', 1))
            minimal_avg = safe_avg(retrieval_stats.get('total_time_minimal', 0), retrieval_stats.get('intensity_minimal', 1))
            comprehensive_avg = safe_avg(retrieval_stats.get('total_time_comprehensive', 0), retrieval_stats.get('intensity_comprehensive', 1))
            
            # Calculate overall speedup
            baseline_time = comprehensive_avg if comprehensive_avg > 0 else 5.0  # Fallback baseline
            actual_avg_time = (performance_stats.get('total_time_simple_response', 0) + 
                             performance_stats.get('total_time_cached_response', 0) + 
                             retrieval_stats.get('total_execution_time', 0)) / max(total_messages, 1)
            speedup = ((baseline_time - actual_avg_time) / baseline_time * 100) if baseline_time > 0 else 0
            
            return {
                'status': 'success',
                'period_hours': hours,
                'total_messages_analyzed': total_messages,
                'routing_efficiency': {
                    'total_messages': total_messages,
                    'simple_responses': simple_responses,
                    'cached_responses': cached_responses, 
                    'retrieval_required': total_messages - efficiency_saves,
                    'efficiency_rate': f"{efficiency_rate:.1f}%",
                    'messages_saved_from_expensive_ops': efficiency_saves
                },
                'cache_effectiveness': {
                    'total_attempts': total_cache_attempts,
                    'hits': cache_hits,
                    'misses': cache_stats.get('cache_misses', 0),
                    'hit_rate': f"{cache_hit_rate:.1f}%",
                    'avg_time_saved_per_hit': f"{avg_time_saved:.2f}s",
                    'total_time_saved': f"{cache_stats.get('total_time_saved', 0):.1f}s"
                },
                'retrieval_distribution': {
                    'minimal': retrieval_stats.get('intensity_minimal', 0),
                    'balanced': retrieval_stats.get('intensity_balanced', 0), 
                    'comprehensive': retrieval_stats.get('intensity_comprehensive', 0),
                    'total_retrievals': retrieval_stats.get('total_retrievals', 0)
                },
                'performance_impact': {
                    'avg_simple_response_time': f"{simple_avg:.2f}s",
                    'avg_cached_response_time': f"{cached_avg:.2f}s",
                    'avg_minimal_retrieval_time': f"{minimal_avg:.2f}s", 
                    'avg_comprehensive_retrieval_time': f"{comprehensive_avg:.2f}s",
                    'overall_speedup': f"{max(speedup, 0):.1f}%"
                },
                'resource_savings': {
                    'avoided_database_queries': efficiency_saves,
                    'avoided_vector_searches': efficiency_saves,
                    'estimated_compute_savings': f"{efficiency_rate * 0.8:.1f}%"  # Conservative estimate
                },
                'generated_at': now.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[METRICS] Failed to get efficiency summary: {str(e)}")
            return {'status': 'error', 'error': str(e)}


# Performance timing decorator
def track_execution_time(operation_type: str):
    """
    Decorator to track operation execution time for metrics collection.
    
    Args:
        operation_type: Type of operation being tracked ('simple_response', 'cached_response', etc.)
    """
    from functools import wraps
    import time
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log timing data
                logger.info(f"[PERFORMANCE] {operation_type} completed in {execution_time:.3f}s")
                
                # Add timing to response metadata if it's a dictionary with metadata
                if isinstance(result, dict) and 'metadata' in result:
                    if 'performance' not in result['metadata']:
                        result['metadata']['performance'] = {}
                    result['metadata']['performance']['execution_time'] = execution_time
                    result['metadata']['performance']['operation_type'] = operation_type
                
                # Extract conversation_id for metrics logging
                conversation_id = None
                if isinstance(result, dict):
                    conversation_id = result.get('conversation_id')
                elif len(args) > 0 and hasattr(args[0], 'get'):
                    conversation_id = args[0].get('conversation_id')
                
                # Log performance timing to metrics
                if conversation_id:
                    try:
                        metrics = IntelligentRoutingMetrics()
                        await metrics.log_performance_timing(
                            operation_type=operation_type,
                            conversation_id=conversation_id,
                            execution_time=execution_time
                        )
                    except Exception as metrics_err:
                        logger.warning(f"[METRICS] Failed to log timing: {str(metrics_err)}")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"[PERFORMANCE] {operation_type} failed after {execution_time:.3f}s: {str(e)}")
                raise
        return wrapper
    return decorator


# Initialize service instances
notebook_rag_service = NotebookRAGService()
conversation_context_manager = ConversationContextManager()
intelligent_routing_metrics = IntelligentRoutingMetrics()