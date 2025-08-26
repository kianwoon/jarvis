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

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.notebook_models import (
    NotebookRAGResponse, NotebookRAGSource, ProjectData
)
from app.services.ai_task_planner import TaskExecutionPlan, RetrievalStrategy, ai_task_planner
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.notebook_llm_settings_cache import get_notebook_llm_settings
from app.core.timeout_settings_cache import get_list_cache_ttl
from app.core.redis_client import get_redis_client
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
                    except:
                        pass
                        
                except Exception as collection_err:
                    self.logger.warning(f"[COUNT] Error counting collection {collection_name}: {str(collection_err)}")
                    self.logger.warning(f"[COUNT] Collection error type: {type(collection_err).__name__}")
                    # Add debug info about the collection
                    try:
                        self.logger.warning(f"[COUNT] Collection {collection_name} had {len(document_ids)} document IDs to filter")
                        if document_ids:
                            self.logger.warning(f"[COUNT] Sample document IDs: {document_ids[:3]}")
                    except:
                        pass
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
        collection_filter: Optional[List[str]] = None
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
            
        Returns:
            RAG query results with adaptive optimization applied
        """
        try:
            self.logger.info(f"[ADAPTIVE_QUERY] Starting adaptive query for notebook {notebook_id}: '{query[:50]}...'")
            
            # Step 1: Analyze query intent using AI
            intent_analysis = await self._analyze_query_intent(query)
            wants_comprehensive = intent_analysis.get("wants_comprehensive", False)
            quantity_intent = intent_analysis.get("quantity_intent", "limited")
            confidence = intent_analysis.get("confidence", 0.5)
            enumeration_mode = intent_analysis.get("enumeration_mode", False)
            enumeration_target = intent_analysis.get("enumeration_target")
            
            self.logger.info(f"[ADAPTIVE_QUERY] Intent analysis - comprehensive: {wants_comprehensive}, quantity: {quantity_intent}, confidence: {confidence:.2f}, enumeration: {enumeration_mode}, target: {enumeration_target}")
            
            # Step 1.2: Check if query should use intelligent task planning
            # Use intelligent planning for complex queries that would benefit from multi-strategy retrieval
            should_use_intelligent_planning = (
                wants_comprehensive or 
                quantity_intent == "all" or 
                confidence < 0.7 or  # Low confidence suggests complex intent
                enumeration_mode or
                len(query.split()) > 8  # Complex multi-part queries
            )
            
            if should_use_intelligent_planning:
                self.logger.info(f"[ADAPTIVE_QUERY] Using intelligent task planning for complex query")
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
            
            # Step 1.5: Route to enumeration method if enumeration mode is detected
            if enumeration_mode:
                self.logger.info(f"[ADAPTIVE_QUERY] Routing to enumeration retrieval for comprehensive listing")
                return await self.query_notebook_enumeration(
                    db=db,
                    notebook_id=notebook_id,
                    query=query,
                    enumeration_target=enumeration_target,
                    max_sources=max_sources,
                    include_metadata=include_metadata,
                    collection_filter=collection_filter
                )
            
            # Step 2: Get actual content count for dynamic limit calculation
            total_available = await self.get_actual_content_count(notebook_id, query)
            self.logger.info(f"[ADAPTIVE_QUERY] Total available content: {total_available} items")
            
            # Step 3: Calculate adaptive retrieval limit based on intent and available content
            if max_sources is None:
                max_sources = 10  # Default starting point
            
            if wants_comprehensive and quantity_intent == "all":
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
            self.logger.error(f"[ADAPTIVE_QUERY] Adaptive query failed: {str(e)}")
            # Fallback to standard query with conservative limits
            # Conservative fallback based on typical query needs
            fallback_limit = min(max_sources or 10, 30)
            return await self.query_notebook(
                db=db,
                notebook_id=notebook_id,
                query=query,
                top_k=fallback_limit,
                include_metadata=include_metadata,
                collection_filter=collection_filter
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

    async def query_notebook_enumeration(
        self,
        db: Session,
        notebook_id: str,
        query: str,
        enumeration_target: str = None,
        max_sources: Optional[int] = None,
        include_metadata: bool = True,
        collection_filter: Optional[List[str]] = None
    ) -> NotebookRAGResponse:
        """
        Perform enumeration-focused retrieval for queries requiring comprehensive listing.
        
        This method uses a different strategy than semantic similarity search:
        1. Uses multiple broad search terms to ensure recall over precision
        2. Retrieves larger initial batches to capture all relevant items
        3. Combines results from multiple query variations
        4. Focuses on completeness rather than semantic relevance scoring
        
        Designed for queries like "list all projects" or "show all companies" where
        the user wants exhaustive results regardless of specific query wording.
        
        Args:
            db: Database session
            notebook_id: Notebook ID to query
            query: Original search query
            enumeration_target: Type of items being enumerated (e.g., "projects", "companies")
            max_sources: Maximum sources to return (defaults to larger values for enumeration)
            include_metadata: Whether to include metadata
            collection_filter: Optional filter for specific collections
            
        Returns:
            RAG query results optimized for enumeration completeness
        """
        try:
            self.logger.info(f"[ENUMERATION] Starting enumeration query for notebook {notebook_id}: '{query[:50]}...', target: {enumeration_target}")
            
            # Get available content count for enumeration planning
            total_available = await self.get_actual_content_count(notebook_id, query)
            
            # Set enumeration-friendly defaults
            if max_sources is None:
                # For enumeration, default to capturing more content
                if total_available <= 100:
                    max_sources = total_available  # Get everything for small notebooks
                elif total_available <= 300:
                    max_sources = min(int(total_available * 0.9), 250)  # 90% for medium
                else:
                    max_sources = min(int(total_available * 0.8), 400)  # 80% for large, capped
            
            self.logger.info(f"[ENUMERATION] Enumeration limit: {max_sources} from {total_available} available")
            
            # Generate multiple query variations for comprehensive coverage
            query_variations = await self._generate_enumeration_queries(query, enumeration_target)
            self.logger.info(f"[ENUMERATION] Generated {len(query_variations)} query variations")
            
            # Collect results from all query variations
            all_enumeration_sources = []
            seen_document_ids = set()
            
            # Get notebook collections once
            collections_info = await self._get_notebook_collections(db, notebook_id, collection_filter)
            if not collections_info:
                self.logger.warning(f"[ENUMERATION] No collections found for notebook {notebook_id}")
                return NotebookRAGResponse(
                    notebook_id=notebook_id,
                    query=query,
                    sources=[],
                    total_sources=0,
                    queried_documents=0,
                    collections_searched=[],
                    extracted_projects=None
                )
            
            collections_searched = set()
            embedding_function = self._get_embedding_function()
            connection_args = self._get_milvus_connection_args()
            
            # Execute each query variation
            for i, variant_query in enumerate(query_variations):
                try:
                    self.logger.debug(f"[ENUMERATION] Executing variant {i+1}/{len(query_variations)}: '{variant_query[:30]}...'")
                    
                    # Use larger retrieval limits for enumeration to ensure coverage
                    variant_limit = min(max_sources * 2, 500)  # Allow 2x limit per variant, capped
                    
                    # Query each collection for this variant
                    for collection_name, document_ids in collections_info.items():
                        try:
                            # Connect to Milvus
                            connections.connect(
                                alias=connection_args['alias'] + f"_enum_{i}",
                                uri=connection_args.get('uri'),
                                token=connection_args.get('token', ''),
                                host=connection_args.get('host'),
                                port=connection_args.get('port')
                            )
                            
                            collection = Collection(collection_name, using=connection_args['alias'] + f"_enum_{i}")
                            collection.load()
                            
                            # Get query embedding
                            if hasattr(embedding_function, 'embed_query'):
                                variant_embedding = embedding_function.embed_query(variant_query)
                            elif hasattr(embedding_function, 'encode'):
                                variant_embedding = embedding_function.encode([variant_query])[0].tolist()
                            else:
                                raise Exception(f"Unsupported embedding function type: {type(embedding_function)}")
                            
                            if not isinstance(variant_embedding, list):
                                variant_embedding = variant_embedding.tolist()
                            
                            # Build document filter
                            doc_filter = None
                            if document_ids:
                                conditions = []
                                for doc_id in document_ids:
                                    if len(doc_id) == 36 and doc_id.count('-') == 4:
                                        conditions.append(f"doc_id == '{doc_id}'")
                                    else:
                                        conditions.append(f"doc_id like '{doc_id}%'")
                                doc_filter = " or ".join(conditions)
                            
                            # Execute search with enumeration-friendly parameters
                            search_params = {
                                "metric_type": "COSINE",
                                "params": {"nprobe": 16}  # Higher nprobe for better recall in enumeration
                            }
                            
                            variant_results = collection.search(
                                data=[variant_embedding],
                                anns_field="vector",
                                param=search_params,
                                limit=variant_limit,
                                expr=doc_filter,
                                output_fields=["content", "doc_id", "source", "page", "doc_type", "uploaded_at", "section", "author", "hash"]
                            )
                            
                            # Process variant results
                            for hits in variant_results:
                                for hit in hits:
                                    doc_id = hit.entity.get('doc_id', '')
                                    
                                    # Skip if we've already seen this document
                                    if doc_id in seen_document_ids:
                                        continue
                                    
                                    # Extract content and metadata
                                    content = hit.entity.get('content', '')
                                    if not content:
                                        continue
                                    
                                    metadata = {
                                        'doc_id': doc_id,
                                        'source': hit.entity.get('source', ''),
                                        'page': hit.entity.get('page', 0),
                                        'doc_type': hit.entity.get('doc_type', ''),
                                        'uploaded_at': hit.entity.get('uploaded_at', ''),
                                        'section': hit.entity.get('section', ''),
                                        'author': hit.entity.get('author', ''),
                                        'hash': hit.entity.get('hash', '')
                                    }
                                    
                                    # Get document name
                                    cached_names = getattr(self, '_current_document_names', {})
                                    document_info = cached_names.get(doc_id, {})
                                    if not document_info and '_' in doc_id:
                                        base_id = doc_id.split('_')[0]
                                        document_info = cached_names.get(base_id, {})
                                    
                                    document_name = document_info.get('name')
                                    source_type = document_info.get('type', 'document')
                                    
                                    # Create source with enumeration context
                                    source = NotebookRAGSource(
                                        content=content,
                                        metadata=metadata if include_metadata else {},
                                        score=float(hit.score),
                                        document_id=doc_id,
                                        document_name=document_name,
                                        collection=collection_name,
                                        source_type=source_type
                                    )
                                    
                                    all_enumeration_sources.append(source)
                                    seen_document_ids.add(doc_id)
                                    
                                    # Break if we have enough results
                                    if len(all_enumeration_sources) >= max_sources * 3:  # Allow 3x for deduplication
                                        break
                                
                                if len(all_enumeration_sources) >= max_sources * 3:
                                    break
                            
                            collections_searched.add(collection_name)
                            
                            # Clean up connection
                            try:
                                connections.disconnect(alias=connection_args['alias'] + f"_enum_{i}")
                            except:
                                pass
                                
                        except Exception as collection_err:
                            self.logger.warning(f"[ENUMERATION] Error in collection {collection_name}, variant {i}: {str(collection_err)}")
                            continue
                            
                except Exception as variant_err:
                    self.logger.warning(f"[ENUMERATION] Error in variant {i}: {str(variant_err)}")
                    continue
                    
                # Break if we have enough results across all variants
                if len(all_enumeration_sources) >= max_sources * 2:
                    break
            
            # Sort by score and apply enhanced deduplication
            all_enumeration_sources.sort(key=lambda x: x.score, reverse=True)
            
            # Enhanced deduplication with overlap detection and document-aware filtering
            final_sources = []
            seen_content_hashes = set()
            seen_doc_chunks = {}  # doc_id -> set of content snippets
            
            import hashlib
            
            for source in all_enumeration_sources:
                # Multi-level deduplication approach
                
                # 1. Exact content hash (first 200 chars)
                content_hash = hashlib.md5(source.content[:200].encode()).hexdigest()
                if content_hash in seen_content_hashes:
                    continue
                
                # 2. Document-aware chunk overlap detection
                doc_id = getattr(source, 'doc_id', source.metadata.get('doc_id', ''))
                if doc_id:
                    # Create content signature for overlap detection (middle portion)
                    content_signature = source.content[50:150].strip() if len(source.content) > 200 else source.content.strip()
                    
                    if doc_id in seen_doc_chunks:
                        # Check for significant overlap with existing chunks from same document
                        is_overlapping = False
                        for existing_signature in seen_doc_chunks[doc_id]:
                            # Calculate similarity based on common words
                            existing_words = set(existing_signature.lower().split())
                            current_words = set(content_signature.lower().split())
                            
                            if len(existing_words) > 0 and len(current_words) > 0:
                                overlap_ratio = len(existing_words & current_words) / len(existing_words | current_words)
                                if overlap_ratio > 0.6:  # 60% overlap threshold
                                    is_overlapping = True
                                    break
                        
                        if is_overlapping:
                            continue
                        
                        seen_doc_chunks[doc_id].add(content_signature)
                    else:
                        seen_doc_chunks[doc_id] = {content_signature}
                
                # 3. Content diversity check - ensure we get varied content types
                content_words = set(source.content.lower().split()[:20])  # First 20 words
                is_diverse = True
                
                # Check against recent sources for diversity
                if len(final_sources) >= 3:
                    recent_sources = final_sources[-3:]  # Check last 3 sources
                    for recent in recent_sources:
                        recent_words = set(recent.content.lower().split()[:20])
                        if len(content_words & recent_words) / max(len(content_words), len(recent_words), 1) > 0.8:
                            is_diverse = False
                            break
                
                if not is_diverse:
                    continue
                
                # Source passes all deduplication checks
                final_sources.append(source)
                seen_content_hashes.add(content_hash)
                    
                if len(final_sources) >= max_sources:
                    break
            
            # Retrieval validation and logging
            unique_docs = len(seen_doc_chunks)
            avg_chunks_per_doc = len(final_sources) / max(unique_docs, 1)
            
            self.logger.info(f"[ENUMERATION] Enumeration complete: {len(final_sources)} final sources from {len(all_enumeration_sources)} candidates")
            self.logger.info(f"[ENUMERATION] Retrieved from {unique_docs} unique documents (avg {avg_chunks_per_doc:.1f} chunks/doc)")
            
            # Log retrieval diversity for validation
            if final_sources:
                source_types = {}
                for source in final_sources:
                    source_type = source.metadata.get('source', 'unknown')[:20]  # First 20 chars of source
                    source_types[source_type] = source_types.get(source_type, 0) + 1
                
                self.logger.info(f"[ENUMERATION] Content diversity: {len(source_types)} different source types found")
            
            # Extract structured project data if this appears to be a project-related query
            extracted_projects = None
            if any(term in query.lower() for term in ['project', 'projects', 'work', 'experience', 'portfolio', 'built', 'developed', 'created']):
                self.logger.info("[ENUMERATION] Detected project-related query, extracting structured data")
                extracted_projects = await self.extract_project_data(final_sources)
                self.logger.info(f"[ENUMERATION] Extracted {len(extracted_projects) if extracted_projects else 0} structured projects")
            
            return NotebookRAGResponse(
                notebook_id=notebook_id,
                query=query,
                sources=final_sources,
                total_sources=len(all_enumeration_sources),
                queried_documents=len(seen_document_ids),
                collections_searched=list(collections_searched),
                extracted_projects=extracted_projects,
                metadata={
                    "enumeration_mode": True,
                    "enumeration_target": enumeration_target,
                    "query_variations": len(query_variations),
                    "deduplication_applied": True,
                    "original_candidates": len(all_enumeration_sources),
                    "unique_documents_retrieved": unique_docs,
                    "avg_chunks_per_document": round(avg_chunks_per_doc, 2),
                    "content_diversity_score": len(source_types) if final_sources else 0,
                    "deduplication_removed": len(all_enumeration_sources) - len(final_sources),
                    "retrieval_coverage": "comprehensive",
                    "structured_extraction_applied": extracted_projects is not None,
                    "extracted_projects_count": len(extracted_projects) if extracted_projects else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"[ENUMERATION] Error in enumeration query: {str(e)}")
            # Fallback to regular adaptive query
            self.logger.info(f"[ENUMERATION] Falling back to adaptive query")
            return await self.query_notebook_adaptive(
                db=db,
                notebook_id=notebook_id,
                query=query,
                max_sources=max_sources,
                include_metadata=include_metadata,
                collection_filter=collection_filter
            )

    async def _generate_enumeration_queries(self, original_query: str, enumeration_target: str = None) -> List[str]:
        """
        Generate multiple query variations for enumeration to ensure comprehensive coverage.
        Enhanced to capture ALL projects regardless of terminology used.
        
        Args:
            original_query: The original user query
            enumeration_target: The type of items being enumerated
            
        Returns:
            List of query variations to run
        """
        query_variations = [original_query]  # Always include original
        
        # Comprehensive base terms for maximum coverage
        project_terms = [
            # Core project terms
            "project", "projects", "initiative", "initiatives", "solution", "solutions",
            "platform", "platforms", "system", "systems", "application", "applications", 
            "app", "apps", "framework", "frameworks", "tool", "tools",
            
            # Action-based terms (what was done)
            "developed", "built", "implemented", "created", "designed", "architected",
            "delivered", "led", "managed", "worked", "launched", "deployed",
            
            # Context terms
            "client", "company", "business", "enterprise", "startup", "organization",
            "work", "development", "deliverables", "portfolio", "experience",
            
            # Industry-specific terms
            "mobile", "web", "digital", "AI", "analytics", "data", "cloud",
            "software", "product", "technology", "tech"
        ]
        
        # Semantic variations for different project descriptions
        semantic_variations = {
            "projects": ["initiatives", "solutions", "work", "development", "deliverables", "systems", "applications"],
            "developed": ["built", "created", "implemented", "designed", "architected", "delivered"],
            "system": ["platform", "solution", "application", "framework", "tool"],
            "client": ["company", "business", "organization", "enterprise", "customer"]
        }
        
        # Add target-specific comprehensive terms
        if enumeration_target:
            target_lower = enumeration_target.lower()
            if "project" in target_lower:
                project_terms.extend([
                    "portfolio", "case study", "example", "implementation", "deployment",
                    "prototype", "MVP", "product", "service", "integration", "migration"
                ])
            elif "company" in target_lower or "client" in target_lower:
                project_terms.extend([
                    "firm", "corporation", "startup", "agency", "consultancy",
                    "partner", "customer", "account", "engagement"
                ])
        
        # Generate comprehensive query combinations
        # 1. High-impact combinations (2-3 terms that work well together)
        high_impact_combos = [
            "project developed",
            "solution built",
            "system implemented", 
            "application created",
            "platform designed",
            "client work",
            "business solution",
            "technology project",
            "software development",
            "digital platform",
            "enterprise system",
            "mobile application"
        ]
        
        # 2. Single high-value terms for maximum breadth
        high_value_singles = [
            "project", "solution", "system", "application", "platform",
            "developed", "built", "implemented", "created", "designed",
            "client", "business", "work", "technology", "software"
        ]
        
        # 3. Context-aware queries
        context_queries = [
            "worked on", "responsible for", "led development",
            "delivered solution", "built system", "created application"
        ]
        
        # 4. Industry and domain queries
        domain_queries = [
            "mobile app", "web platform", "data system", "AI solution",
            "cloud platform", "enterprise software", "digital tool"
        ]
        
        # Combine all variations with smart prioritization
        query_variations.extend(high_impact_combos)
        query_variations.extend(high_value_singles[:8])  # Top 8 singles
        query_variations.extend(context_queries[:4])     # Top 4 context
        query_variations.extend(domain_queries[:4])      # Top 4 domain
        
        # Add semantic variations of the original query
        original_words = original_query.lower().split()
        for word in original_words:
            if word in semantic_variations:
                for variation in semantic_variations[word][:2]:  # Top 2 variations per word
                    varied_query = original_query.lower().replace(word, variation)
                    if varied_query != original_query.lower():
                        query_variations.append(varied_query)
        
        # Remove duplicates while preserving order and relevance
        seen = set()
        deduplicated = []
        for q in query_variations:
            q_clean = q.strip().lower()
            if q_clean not in seen and q_clean and len(q_clean) > 1:
                deduplicated.append(q.strip())
                seen.add(q_clean)
        
        # Log query generation for validation
        self.logger.info(f"[ENUMERATION] Generated {len(deduplicated)} unique query variations from {len(query_variations)} candidates")
        
        # Validate coverage quality of generated queries
        final_queries = deduplicated[:12]  # Increased from 8 to 12 for better coverage
        coverage_analysis = self._validate_query_coverage(final_queries)
        
        self.logger.info(f"[ENUMERATION] Query coverage analysis: {coverage_analysis['overall_coverage_percentage']}% overall ({coverage_analysis['coverage_quality']})")
        
        # Log any coverage gaps for improvement
        for category, stats in coverage_analysis['category_coverage'].items():
            if stats['percentage'] < 70:  # Log categories with less than 70% coverage
                self.logger.warning(f"[ENUMERATION] Low coverage in {category}: {stats['percentage']:.1f}% (missing: {stats['missing'][:3]})")
        
        return final_queries

    def _validate_query_coverage(self, queries: List[str]) -> dict:
        """
        Validate that generated queries provide comprehensive coverage for different project types.
        
        Args:
            queries: List of generated queries
            
        Returns:
            Dictionary with coverage analysis
        """
        coverage_categories = {
            "action_verbs": ["developed", "built", "implemented", "created", "designed", "delivered", "led"],
            "project_types": ["project", "solution", "system", "application", "platform", "tool"],
            "contexts": ["client", "business", "work", "company", "enterprise"],
            "industries": ["mobile", "web", "digital", "AI", "cloud", "software", "data"],
            "outcomes": ["launched", "deployed", "delivered", "completed", "managed"]
        }
        
        coverage_scores = {}
        for category, terms in coverage_categories.items():
            covered_terms = set()
            for query in queries:
                query_lower = query.lower()
                for term in terms:
                    if term in query_lower:
                        covered_terms.add(term)
            
            coverage_scores[category] = {
                "covered": len(covered_terms),
                "total": len(terms),
                "percentage": (len(covered_terms) / len(terms)) * 100,
                "missing": [term for term in terms if term not in covered_terms]
            }
        
        # Overall coverage score
        total_covered = sum(score["covered"] for score in coverage_scores.values())
        total_possible = sum(score["total"] for score in coverage_scores.values())
        overall_coverage = (total_covered / total_possible) * 100
        
        return {
            "overall_coverage_percentage": round(overall_coverage, 1),
            "category_coverage": coverage_scores,
            "total_queries": len(queries),
            "coverage_quality": "excellent" if overall_coverage > 80 else "good" if overall_coverage > 60 else "needs_improvement"
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
                    
                    # Get query embedding
                    try:
                        if hasattr(embedding_function, 'embed_query'):
                            query_embedding = embedding_function.embed_query(query)
                        elif hasattr(embedding_function, 'encode'):
                            query_embedding = embedding_function.encode([query])[0].tolist()
                        else:
                            raise Exception(f"Unsupported embedding function type: {type(embedding_function)}")
                        
                        if not isinstance(query_embedding, list):
                            query_embedding = query_embedding.tolist()
                        
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
                        notebook_settings = get_notebook_llm_settings()
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
                            if total_notebook_items <= 100:
                                max_retrieval_chunks = total_notebook_items  # Use all for small notebooks
                            elif total_notebook_items <= 300:
                                max_retrieval_chunks = min(int(total_notebook_items * 0.8), total_notebook_items)  # 80%
                            else:
                                max_retrieval_chunks = min(int(total_notebook_items * 0.6), 500)  # 60% capped at 500
                            
                        except Exception as count_err:
                            self.logger.warning(f"Could not get content count for dynamic limits: {count_err}")
                            max_retrieval_chunks = 150  # Conservative fallback that's not hardcoded to specific use case
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
                            if total_available <= 200:
                                max_retrieval_chunks = total_available  # Use all available for small collections
                            elif total_available <= 600:
                                max_retrieval_chunks = min(int(total_available * 0.9), total_available)  # 90% for medium collections
                            else:
                                max_retrieval_chunks = min(int(total_available * 0.7), 1000)  # 70% for large collections, capped at 1000
                                
                            self.logger.info(f"[COMPREHENSIVE_QUERY] Intelligent analysis detected comprehensive query: '{query[:100]}...' - Enhanced retrieval limits based on {total_available} available items: multiplier {original_multiplier}→{retrieval_multiplier}, max_chunks {original_max_chunks}→{max_retrieval_chunks}")
                        except Exception as count_err:
                            # Fallback scaling if counting fails
                            max_retrieval_chunks = max(max_retrieval_chunks * 2, original_max_chunks * 3)
                            self.logger.warning(f"Could not get content count for comprehensive scaling, using fallback: {max_retrieval_chunks}")
                    
                    limit = min(top_k * retrieval_multiplier, max_retrieval_chunks)
                    
                    # Executing collection search with calculated limits
                    if doc_filter:
                        # Applying document filter
                        expr = doc_filter
                    else:
                        expr = ""
                    
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
            
            # Sort and filter sources
            
            # Sort all sources by score (higher is better for COSINE similarity)
            all_sources.sort(key=lambda x: x.score, reverse=True)
            
            # Take top_k results
            top_sources = all_sources[:top_k]
            
            # Return final sources
            
            self.logger.info(f"Successfully queried notebook {notebook_id}, found {len(top_sources)} sources")
            
            # Set extracted_projects to None - rely on LLM's natural language understanding
            # instead of broken regex-based extraction that generates garbage data
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
                import traceback
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
            
            # For comprehensive queries, adjust memory consolidation based on intent analysis
            if is_listing_query:
                include_all_memory_chunks = False
                self.logger.info(f"[COMPREHENSIVE_QUERY] Intelligent analysis detected comprehensive query: '{query[:100]}...' - Adjusting memory consolidation to preserve individual item visibility")
            
            
            # Step 1: Separate memory sources from document sources
            memory_chunks = []
            document_chunks = []
            
            for doc, score in initial_results:
                if self._is_memory_source(doc):
                    memory_chunks.append((doc, score))
                else:
                    document_chunks.append((doc, score))
            
            
            enhanced_results = []
            processed_chunk_ids = set()
            
            # Step 2: Process memory sources first (if prioritization enabled)
            if prioritize_memory_sources and memory_chunks:
                
                # Group memory chunks by document ID
                memory_by_document = defaultdict(list)
                for doc, score in memory_chunks:
                    doc_id = doc.get('metadata', {}).get('doc_id', '')
                    base_doc_id = self._extract_base_document_id(doc_id)
                    memory_by_document[base_doc_id].append((doc, score, doc_id))
                
                # Process each memory document
                for base_doc_id, memory_doc_chunks in memory_by_document.items():
                    
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
                        except Exception as e:
                            self.logger.warning(f"Could not retrieve complete memory document {base_doc_id}: {e}")
                            # Fallback to just the matching chunks
                            for doc, score, doc_id in memory_doc_chunks:
                                if doc_id not in processed_chunk_ids:
                                    boosted_score = min(score + memory_score_boost, 1.0)
                                    enhanced_results.append((doc, boosted_score))
                                    processed_chunk_ids.add(doc_id)
                    else:
                        # For comprehensive queries: Just add the matching memory chunks with boost to preserve granularity
                        for doc, score, doc_id in memory_doc_chunks:
                            if doc_id not in processed_chunk_ids:
                                boosted_score = min(score + memory_score_boost, 1.0)
                                enhanced_results.append((doc, boosted_score))
                                processed_chunk_ids.add(doc_id)
            
            
            # Step 3: Process document sources (if space remaining)
            remaining_capacity = max_retrieval_chunks - len(enhanced_results)
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
        Extract structured project data from retrieved chunks.
        
        This method parses chunks to identify project information and extract
        structured data including name, company, years, and description.
        Missing metadata is marked as "N/A" rather than excluding projects.
        
        Args:
            sources: List of RAG sources containing project information
            
        Returns:
            List of structured project data with completed metadata
        """
        try:
            import re
            self.logger.info(f"[PROJECT_EXTRACTION] Starting extraction from {len(sources)} sources")
            
            # Initialize project collection
            raw_projects = []
            
            # Extract projects from each source
            for source in sources:
                projects_in_source = await self._extract_projects_from_content(source)
                raw_projects.extend(projects_in_source)
            
            self.logger.info(f"[PROJECT_EXTRACTION] Found {len(raw_projects)} raw projects")
            
            # Complete missing metadata using cross-referencing
            completed_projects = await self._complete_project_metadata(raw_projects, sources)
            
            # Deduplicate and merge similar projects
            final_projects = await self._deduplicate_and_merge_projects(completed_projects)
            
            self.logger.info(f"[PROJECT_EXTRACTION] Final extraction: {len(final_projects)} unique projects")
            return final_projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error in project extraction: {str(e)}")
            return []

    async def _extract_projects_from_content(self, source: NotebookRAGSource) -> List[ProjectData]:
        """
        Extract project information from a single content source using patterns and NLP.
        
        Args:
            source: RAG source to extract projects from
            
        Returns:
            List of projects found in the content
        """
        try:
            import re
            content = source.content
            projects = []
            
            # Common project indicators and patterns
            project_patterns = [
                # Direct project mentions
                r'(?:project|initiative|program|solution|implementation|development|system|platform|application|tool|service|product)[\s\-:]+([^.\n]{10,100})',
                # Work descriptions that indicate projects
                r'(?:built|developed|created|implemented|designed|architected|delivered|deployed|launched)[\s\-:]+([^.\n]{10,100})',
                # Led/managed project patterns
                r'(?:led|managed|oversaw|directed|coordinated)[\s\-:]+([^.\n]{10,150})',
                # Experience with specific technologies/solutions
                r'(?:worked on|involved in|contributed to|participated in)[\s\-:]+([^.\n]{10,100})'
            ]
            
            # Company name patterns
            company_patterns = [
                r'(?:at|for|with|@)\s+([A-Z][a-zA-Z\s&\.]{2,30}(?:Inc|LLC|Corp|Ltd|Company|Group|Solutions|Technologies|Systems|Consulting)?)',
                r'([A-Z][a-zA-Z\s&\.]{2,30}(?:Inc|LLC|Corp|Ltd|Company|Group|Solutions|Technologies|Systems|Consulting))',
                r'(?:Company|Organization|Employer|Client):\s*([A-Z][a-zA-Z\s&\.]{2,40})'
            ]
            
            # Year patterns - flexible to capture various formats
            year_patterns = [
                r'(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2}|\s*[-–]\s*present)?',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}(?:\s*[-–]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}|\s*[-–]\s*present)?',
                r'\b(?:19|20)\d{2}\b'
            ]
            
            # Extract potential projects
            for pattern in project_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    project_desc = match.group(1).strip()
                    
                    # Clean up the description
                    project_desc = re.sub(r'[\r\n]+', ' ', project_desc)
                    project_desc = re.sub(r'\s+', ' ', project_desc).strip()
                    
                    if len(project_desc) < 10:  # Skip very short matches
                        continue
                    
                    # Extract project name from description (first few words)
                    name_match = re.match(r'^([^,;.!?]{5,50})', project_desc)
                    project_name = name_match.group(1).strip() if name_match else project_desc[:50]
                    
                    # Try to find company in surrounding context
                    context_start = max(0, match.start() - 200)
                    context_end = min(len(content), match.end() + 200)
                    context = content[context_start:context_end]
                    
                    company = await self._extract_company_from_context(context, company_patterns)
                    year = await self._extract_year_from_context(context, year_patterns)
                    
                    project = ProjectData(
                        name=project_name,
                        company=company if company else "N/A",
                        year=year if year else "N/A",
                        description=project_desc,
                        source_chunk_id=source.metadata.get('chunk_id'),
                        confidence_score=0.7,  # Base confidence
                        metadata={
                            'source_document': source.document_name,
                            'extraction_pattern': pattern[:20] + "...",
                            'context_length': len(context)
                        }
                    )
                    
                    projects.append(project)
            
            self.logger.debug(f"[PROJECT_EXTRACTION] Extracted {len(projects)} projects from source")
            return projects
            
        except Exception as e:
            self.logger.error(f"[PROJECT_EXTRACTION] Error extracting from content: {str(e)}")
            return []

    async def _extract_company_from_context(self, context: str, company_patterns: List[str]) -> Optional[str]:
        """Extract company name from context using patterns."""
        try:
            import re
            for pattern in company_patterns:
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    company = match.group(1).strip()
                    # Clean up company name
                    company = re.sub(r'\s+', ' ', company).strip()
                    if len(company) >= 3 and not company.lower() in ['the', 'and', 'with', 'for']:
                        return company
            return None
        except Exception:
            return None

    async def _extract_year_from_context(self, context: str, year_patterns: List[str]) -> Optional[str]:
        """Extract year information from context using patterns."""
        try:
            import re
            for pattern in year_patterns:
                matches = re.finditer(pattern, context, re.IGNORECASE)
                for match in matches:
                    year = match.group(0).strip()
                    # Validate year is reasonable
                    if re.search(r'(?:19|20)\d{2}', year):
                        return year
            return None
        except Exception:
            return None

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
                
                # If company or year is missing, try to find it in related chunks
                if project.company == "N/A" or project.year == "N/A":
                    # Look for the project name or similar description in other sources
                    for source in sources:
                        if await self._is_related_content(project.name, project.description, source.content):
                            # Try to extract missing metadata from this related source
                            if project.company == "N/A":
                                company_patterns = [
                                    r'(?:at|for|with|@)\s+([A-Z][a-zA-Z\s&\.]{2,30}(?:Inc|LLC|Corp|Ltd|Company|Group|Solutions|Technologies|Systems|Consulting)?)',
                                    r'([A-Z][a-zA-Z\s&\.]{2,30}(?:Inc|LLC|Corp|Ltd|Company|Group|Solutions|Technologies|Systems|Consulting))',
                                ]
                                company = await self._extract_company_from_context(source.content, company_patterns)
                                if company:
                                    updated_project.company = company
                                    updated_project.confidence_score += 0.1
                            
                            if project.year == "N/A":
                                year_patterns = [
                                    r'(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2}|\s*[-–]\s*present)?',
                                    r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}(?:\s*[-–]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(?:19|20)\d{2}|\s*[-–]\s*present)?'
                                ]
                                year = await self._extract_year_from_context(source.content, year_patterns)
                                if year:
                                    updated_project.year = year
                                    updated_project.confidence_score += 0.1
                            
                            # Break if we found both
                            if updated_project.company != "N/A" and updated_project.year != "N/A":
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
                # Check for enumeration mode or high-confidence comprehensive queries that need deterministic results
                wants_comprehensive = intent_analysis.get("wants_comprehensive", False)
                confidence = intent_analysis.get("confidence", 0)
                enumeration_mode = intent_analysis.get("enumeration_mode", False)
                enumeration_target = intent_analysis.get("enumeration_target")
                
                # Route to enumeration mode for enumeration queries
                if enumeration_mode:
                    self.logger.info(f"[MULTI_STAGE_ENUMERATION] Using enumeration mode for comprehensive listing")
                    initial_response = await self.query_notebook_enumeration(
                        db=db,
                        notebook_id=notebook_id,
                        query=query,
                        enumeration_target=enumeration_target,
                        max_sources=initial_batch_size,
                        include_metadata=True,
                        collection_filter=collection_filter
                    )
                # Route to comprehensive mode for high-confidence comprehensive queries
                elif wants_comprehensive and confidence > 0.8:
                    self.logger.info(f"[MULTI_STAGE_COMPREHENSIVE] Using deterministic comprehensive mode (confidence: {confidence:.2f})")
                    
                    # Determine content type from query
                    query_lower = query.lower()
                    if any(keyword in query_lower for keyword in ['project', 'company', 'companies', 'business', 'startup', 'firm', 'organization']):
                        content_type = "projects"
                    else:
                        content_type = "general"
                    
                    initial_response = await self.query_notebook_comprehensive(
                        db=db,
                        notebook_id=notebook_id,
                        query=query,
                        content_type=content_type,
                        include_metadata=True
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
                            if enumeration_mode:
                                # For enumeration mode, we already got all results in stage 1
                                # No need for additional batches - return empty to exit loop
                                batch_response = NotebookRAGResponse(
                                    notebook_id=notebook_id,
                                    query=query,
                                    sources=[],
                                    total_sources=0,
                                    queried_documents=0,
                                    collections_searched=[]
                                )
                                self.logger.info(f"[MULTI_STAGE_ENUMERATION] All enumeration results retrieved in stage 1, skipping additional batches")
                            elif wants_comprehensive and confidence > 0.8:
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
            # Use the existing query_notebook method with strategy parameters
            result = await self.query_notebook(
                db=db,
                notebook_id=notebook_id,
                query=strategy.query,
                top_k=strategy.max_chunks,
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
            except:
                # If sorting fails, keep original order
                pass
            
            # Apply verification rules
            if len(all_sources) < plan.verification.min_expected_results:
                self.logger.warning(f"[COMBINE_RESULTS] Got {len(all_sources)} sources, expected minimum {plan.verification.min_expected_results}")
            
            # Limit results if needed (preserve top results)
            max_results = 500  # Reasonable limit
            if len(all_sources) > max_results:
                all_sources = all_sources[:max_results]
                self.logger.info(f"[COMBINE_RESULTS] Limited results to {max_results} sources")
            
            return NotebookRAGResponse(
                sources=all_sources,
                total_sources=len(all_sources),
                notebook_id=strategy_results[0]['result'].notebook_id,
                query=strategy_results[0]['result'].query,
                queried_documents=sum(sr['result'].queried_documents for sr in strategy_results),
                collections_searched=list(set(col for sr in strategy_results for col in sr['result'].collections_searched))
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