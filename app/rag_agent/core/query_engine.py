"""
Query execution engine with collection search capabilities

This module handles the actual execution of searches against collections
using the existing RAG infrastructure while providing enhanced query processing.
"""

import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from app.rag_agent.utils.types import (
    CollectionResult, Source, SearchContext, StepResult, StepType
)
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.collection_registry_cache import get_collection_config
from app.langchain.service import handle_rag_query, keyword_search_milvus
from langchain_community.vectorstores import Milvus

logger = logging.getLogger(__name__)


class QueryEngine:
    """Execute searches against collections with optimization and result processing"""
    
    def __init__(self):
        self.vector_db_settings = None
        self.embedding_function = None
        self._settings_refresh_time = None
        
    async def execute_collection_search(
        self,
        collection_name: str,
        query: str,
        search_strategy: str = "auto",
        max_results: int = 10,
        similarity_threshold: float = 0.7,
        context: Optional[SearchContext] = None
    ) -> CollectionResult:
        """
        Execute search against a specific collection
        
        Args:
            collection_name: Name of collection to search
            query: Search query (possibly refined for this collection)
            search_strategy: "semantic", "keyword", "hybrid", or "auto"
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity threshold
            context: Search context for optimization
            
        Returns:
            CollectionResult with sources and metadata
        """
        start_time = time.time()
        
        try:
            # Get collection configuration
            collection_config = get_collection_config(collection_name)
            if not collection_config:
                logger.error(f"Collection {collection_name} not found")
                return self._create_empty_result(collection_name, query, "Collection not found")
            
            # Initialize vector database settings
            await self._ensure_vector_db_settings()
            
            # Determine optimal search strategy
            if search_strategy == "auto":
                search_strategy = self._determine_optimal_strategy(
                    collection_config, query, context
                )
            
            # Execute search based on strategy
            if search_strategy == "hybrid":
                sources = await self._execute_hybrid_search(
                    collection_name, query, max_results, similarity_threshold
                )
            elif search_strategy == "keyword":
                sources = await self._execute_keyword_search(
                    collection_name, query, max_results
                )
            else:  # semantic or fallback
                sources = await self._execute_semantic_search(
                    collection_name, query, max_results, similarity_threshold
                )
            
            # Calculate relevance score
            relevance_score = self._calculate_collection_relevance(sources, query)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"Collection {collection_name} search completed: "
                       f"{len(sources)} results in {execution_time}ms")
            
            return CollectionResult(
                collection_name=collection_name,
                sources=sources,
                relevance_score=relevance_score,
                search_strategy=search_strategy,
                query_used=query,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Error searching collection {collection_name}: {e}")
            
            return CollectionResult(
                collection_name=collection_name,
                sources=[],
                relevance_score=0.0,
                search_strategy=search_strategy,
                query_used=query,
                execution_time_ms=execution_time
            )
    
    async def execute_parallel_searches(
        self,
        search_requests: List[Dict[str, Any]],
        context: Optional[SearchContext] = None
    ) -> List[CollectionResult]:
        """
        Execute multiple collection searches in parallel
        
        Args:
            search_requests: List of search request dicts with collection_name, query, etc.
            context: Search context
            
        Returns:
            List of CollectionResult objects
        """
        # Create tasks for parallel execution
        tasks = []
        for request in search_requests:
            task = self.execute_collection_search(
                collection_name=request["collection_name"],
                query=request["query"],
                search_strategy=request.get("search_strategy", "auto"),
                max_results=request.get("max_results", 10),
                similarity_threshold=request.get("similarity_threshold", 0.7),
                context=context
            )
            tasks.append(task)
        
        # Execute all searches in parallel
        logger.info(f"Executing {len(tasks)} parallel collection searches")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel search {i} failed: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _execute_semantic_search(
        self,
        collection_name: str,
        query: str,
        max_results: int,
        similarity_threshold: float
    ) -> List[Source]:
        """Execute semantic vector search"""
        
        try:
            # Use existing RAG infrastructure
            docs, sources, confidence = handle_rag_query(
                question=query,
                thinking=False,
                collections=[collection_name],
                collection_strategy="specific"
            )
            
            # Convert to Source objects
            source_objects = []
            for i, (doc, source_info) in enumerate(zip(docs, sources)):
                # Extract metadata from document
                metadata = getattr(doc, 'metadata', {})
                
                source_obj = Source(
                    collection_name=collection_name,
                    document_id=metadata.get('doc_id', f'doc_{i}'),
                    content=doc.page_content if hasattr(doc, 'page_content') else str(doc),
                    score=source_info.get('score', confidence),
                    metadata=metadata,
                    page=metadata.get('page'),
                    section=metadata.get('section')
                )
                
                # Filter by similarity threshold
                if source_obj.score >= similarity_threshold:
                    source_objects.append(source_obj)
                
                # Limit results
                if len(source_objects) >= max_results:
                    break
            
            return source_objects
            
        except Exception as e:
            logger.error(f"Semantic search failed for {collection_name}: {e}")
            return []
    
    async def _execute_keyword_search(
        self,
        collection_name: str,
        query: str,
        max_results: int
    ) -> List[Source]:
        """Execute keyword-based search using BM25"""
        
        try:
            # Get vector DB settings for keyword search
            vector_db_settings = get_vector_db_settings()
            milvus_config = vector_db_settings.get('milvus', {})
            
            uri = milvus_config.get('MILVUS_URI')
            token = milvus_config.get('MILVUS_TOKEN', '')
            
            if not uri:
                logger.warning("No Milvus URI configured for keyword search")
                return []
            
            # Use existing keyword search function
            keyword_results = keyword_search_milvus(
                query=query,
                collection_name=collection_name,
                uri=uri,
                token=token,
                limit=max_results
            )
            
            # Convert to Source objects
            source_objects = []
            for result in keyword_results:
                source_obj = Source(
                    collection_name=collection_name,
                    document_id=result.get('doc_id', result.get('id', 'unknown')),
                    content=result.get('content', ''),
                    score=result.get('score', 0.5),
                    metadata=result,
                    page=result.get('page'),
                    section=result.get('section')
                )
                source_objects.append(source_obj)
            
            return source_objects
            
        except Exception as e:
            logger.error(f"Keyword search failed for {collection_name}: {e}")
            return []
    
    async def _execute_hybrid_search(
        self,
        collection_name: str,
        query: str,
        max_results: int,
        similarity_threshold: float
    ) -> List[Source]:
        """Execute hybrid search combining semantic and keyword results"""
        
        try:
            # Execute both searches in parallel
            semantic_task = self._execute_semantic_search(
                collection_name, query, max_results, similarity_threshold
            )
            keyword_task = self._execute_keyword_search(
                collection_name, query, max_results
            )
            
            semantic_results, keyword_results = await asyncio.gather(
                semantic_task, keyword_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(semantic_results, Exception):
                logger.error(f"Semantic search failed: {semantic_results}")
                semantic_results = []
            
            if isinstance(keyword_results, Exception):
                logger.error(f"Keyword search failed: {keyword_results}")
                keyword_results = []
            
            # Merge and deduplicate results
            merged_results = self._merge_search_results(
                semantic_results, keyword_results, max_results
            )
            
            return merged_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed for {collection_name}: {e}")
            return []
    
    def _merge_search_results(
        self,
        semantic_results: List[Source],
        keyword_results: List[Source],
        max_results: int
    ) -> List[Source]:
        """Merge and rank results from semantic and keyword searches"""
        
        # Create lookup for deduplication
        seen_content = set()
        merged_results = []
        
        # Weight semantic results higher
        semantic_weight = 0.7
        keyword_weight = 0.3
        
        # Process semantic results first
        for source in semantic_results:
            content_hash = hash(source.content[:100])  # Hash first 100 chars for dedup
            if content_hash not in seen_content:
                source.score = source.score * semantic_weight
                merged_results.append(source)
                seen_content.add(content_hash)
        
        # Add unique keyword results
        for source in keyword_results:
            content_hash = hash(source.content[:100])
            if content_hash not in seen_content:
                source.score = source.score * keyword_weight
                merged_results.append(source)
                seen_content.add(content_hash)
        
        # Sort by combined score and limit
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results[:max_results]
    
    def _determine_optimal_strategy(
        self,
        collection_config: Dict,
        query: str,
        context: Optional[SearchContext] = None
    ) -> str:
        """Determine optimal search strategy based on query and collection"""
        
        # Get collection search config
        search_config = collection_config.get('search_config', {})
        default_strategy = search_config.get('strategy', 'balanced')
        
        # Collection type preferences
        collection_type = collection_config.get('collection_type', '')
        
        if collection_type in ['regulatory_compliance', 'policies_procedures']:
            # Exact terms important for compliance
            if any(word in query.lower() for word in ['policy', 'rule', 'requirement', 'regulation']):
                return "keyword"
        
        elif collection_type in ['technical_docs', 'api_documentation']:
            # Technical queries often need exact matches
            if any(word in query.lower() for word in ['api', 'function', 'code', 'method']):
                return "keyword"
        
        # Query analysis
        query_lower = query.lower()
        
        # Prefer semantic for conceptual queries
        if any(word in query_lower for word in ['how', 'why', 'what', 'explain', 'describe']):
            return "semantic"
        
        # Prefer keyword for specific term searches
        if any(word in query_lower for word in ['find', 'search', 'locate', 'specific']):
            return "keyword"
        
        # Use hybrid for complex queries
        if len(query.split()) > 8:
            return "hybrid"
        
        # Default to collection preference
        strategy_map = {
            'balanced': 'hybrid',
            'precision': 'keyword',
            'recall': 'semantic'
        }
        
        return strategy_map.get(default_strategy, 'hybrid')
    
    def _calculate_collection_relevance(self, sources: List[Source], query: str) -> float:
        """Calculate overall relevance score for collection results"""
        
        if not sources:
            return 0.0
        
        # Average score of top results
        top_scores = [source.score for source in sources[:5]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Boost for number of relevant results
        result_count_boost = min(len(sources) / 10, 0.2)
        
        # Boost for content diversity (different documents)
        unique_docs = len(set(source.document_id for source in sources))
        diversity_boost = min(unique_docs / 5, 0.1)
        
        final_score = avg_score + result_count_boost + diversity_boost
        return min(final_score, 1.0)
    
    def _create_empty_result(
        self,
        collection_name: str,
        query: str,
        reason: str
    ) -> CollectionResult:
        """Create empty result for failed searches"""
        
        return CollectionResult(
            collection_name=collection_name,
            sources=[],
            relevance_score=0.0,
            search_strategy="failed",
            query_used=query,
            execution_time_ms=0
        )
    
    async def _ensure_vector_db_settings(self):
        """Ensure vector DB settings are loaded and current"""
        current_time = datetime.now()
        
        if (not self.vector_db_settings or
            not self._settings_refresh_time or
            (current_time - self._settings_refresh_time).total_seconds() > 300):
            
            self.vector_db_settings = get_vector_db_settings()
            self._settings_refresh_time = current_time
            
            # Initialize embedding function if needed
            if not self.embedding_function:
                try:
                    # Lazy import to avoid circular dependency
                    from app.api.v1.endpoints.document import HTTPEmbeddingFunction
                    from app.core.embedding_settings_cache import get_embedding_settings
                    
                    # Get embedding endpoint from settings
                    embedding_settings = get_embedding_settings()
                    embedding_endpoint = embedding_settings.get('embedding_endpoint', 'http://qwen-embedder:8050/embed')
                    
                    # Handle Docker vs local environment
                    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
                    if not is_docker and 'qwen-embedder:8050' in embedding_endpoint:
                        embedding_endpoint = embedding_endpoint.replace('qwen-embedder:8050', 'localhost:8050')
                    
                    # Remove '/embed' suffix if present for HTTPEmbeddingFunction
                    if embedding_endpoint.endswith('/embed'):
                        embedding_endpoint = embedding_endpoint[:-6]
                    
                    self.embedding_function = HTTPEmbeddingFunction(embedding_endpoint)
                except Exception as e:
                    logger.warning(f"Failed to initialize embedding function: {e}")


# Utility functions for direct use
async def search_collection(
    collection_name: str,
    query: str,
    strategy: str = "auto",
    max_results: int = 10
) -> List[Source]:
    """Quick utility function to search a single collection"""
    
    engine = QueryEngine()
    result = await engine.execute_collection_search(
        collection_name=collection_name,
        query=query,
        search_strategy=strategy,
        max_results=max_results
    )
    return result.sources