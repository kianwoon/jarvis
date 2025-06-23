"""
Standalone RAG Interface

This is the main interface for the agent-based RAG system. It provides a clean,
simple API that can be used by chat systems, multi-agent frameworks, or any
other application requiring intelligent knowledge retrieval.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator, List, Union

from app.rag_agent.utils.types import (
    RAGResponse, RAGOptions, SearchContext, RAGStreamChunk
)
from app.rag_agent.core.rag_orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)


class StandaloneRAGInterface:
    """
    Clean interface for standalone RAG module usage
    
    This is the main entry point for all RAG operations. It handles:
    - Simple query processing
    - Streaming responses
    - Context management
    - Performance monitoring
    """
    
    def __init__(self):
        self.orchestrator = RAGOrchestrator()
        self._performance_stats = {
            "total_queries": 0,
            "avg_response_time_ms": 0,
            "success_rate": 0.0
        }
    
    async def query(
        self,
        query: str,
        context: Optional[SearchContext] = None,
        options: Optional[RAGOptions] = None
    ) -> RAGResponse:
        """
        Process a RAG query and return comprehensive results
        
        Args:
            query: User query string
            context: Search context with user information and constraints
            options: Processing options and configuration
            
        Returns:
            RAGResponse with answer, sources, and execution metadata
            
        Example:
            rag = StandaloneRAGInterface()
            
            # Simple query
            response = await rag.query("What is our data retention policy?")
            
            # Query with context
            context = SearchContext(
                user_id="user123",
                domain="compliance",
                required_accuracy="high"
            )
            response = await rag.query("What are KYC requirements?", context=context)
            
            # Query with options
            options = RAGOptions(
                max_iterations=2,
                include_execution_trace=True,
                confidence_threshold=0.8
            )
            response = await rag.query("How do I configure API access?", options=options)
        """
        
        try:
            # Set default options if not provided
            if options is None:
                options = RAGOptions()
            
            # Validate query
            if not query or not query.strip():
                return RAGResponse(
                    content="Please provide a valid question or query.",
                    sources=[],
                    confidence_score=0.0,
                    processing_time_ms=0
                )
            
            # Process through orchestrator
            response = await self.orchestrator.process_query(
                query=query.strip(),
                context=context,
                options=options
            )
            
            # Update performance stats
            self._update_performance_stats(response.processing_time_ms, response.confidence_score > 0.3)
            
            logger.info(f"RAG query processed: {len(response.content)} chars, "
                       f"{len(response.sources)} sources, "
                       f"confidence: {response.confidence_score:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query processing: {e}")
            return RAGResponse(
                content=f"I encountered an error while processing your query: {str(e)}. Please try again.",
                sources=[],
                confidence_score=0.0,
                processing_time_ms=0
            )
    
    async def query_stream(
        self,
        query: str,
        context: Optional[SearchContext] = None,
        options: Optional[RAGOptions] = None
    ) -> AsyncGenerator[RAGStreamChunk, None]:
        """
        Streaming interface for real-time RAG responses
        
        Args:
            query: User query string
            context: Search context
            options: Processing options
            
        Yields:
            RAGStreamChunk objects with progressive results
            
        Example:
            rag = StandaloneRAGInterface()
            
            async for chunk in rag.query_stream("Explain our privacy policy"):
                if chunk.chunk_type == "status":
                    print(f"Status: {chunk.content}")
                elif chunk.chunk_type == "content":
                    print(f"Content: {chunk.content}")
                    if chunk.is_final:
                        print(f"Sources: {len(chunk.sources)}")
        """
        
        try:
            # Set default options for streaming
            if options is None:
                options = RAGOptions(stream=True)
            else:
                options.stream = True
            
            # Validate query
            if not query or not query.strip():
                yield RAGStreamChunk(
                    content="Please provide a valid question or query.",
                    chunk_type="error",
                    is_final=True
                )
                return
            
            # Stream through orchestrator
            async for chunk in self.orchestrator.process_query_stream(
                query=query.strip(),
                context=context,
                options=options
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming RAG query: {e}")
            yield RAGStreamChunk(
                content=f"Error processing query: {str(e)}",
                chunk_type="error",
                is_final=True
            )
    
    async def batch_query(
        self,
        queries: List[str],
        context: Optional[SearchContext] = None,
        options: Optional[RAGOptions] = None,
        max_concurrent: int = 3
    ) -> List[RAGResponse]:
        """
        Process multiple queries in parallel with concurrency control
        
        Args:
            queries: List of query strings
            context: Shared search context
            options: Processing options
            max_concurrent: Maximum concurrent queries
            
        Returns:
            List of RAGResponse objects in same order as input queries
        """
        
        if not queries:
            return []
        
        logger.info(f"Processing {len(queries)} queries in parallel (max_concurrent={max_concurrent})")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_query(query: str, index: int) -> tuple:
            async with semaphore:
                try:
                    response = await self.query(query, context, options)
                    return index, response
                except Exception as e:
                    logger.error(f"Error processing batch query {index}: {e}")
                    return index, RAGResponse(
                        content=f"Error processing query: {str(e)}",
                        sources=[],
                        confidence_score=0.0,
                        processing_time_ms=0
                    )
        
        # Create tasks for all queries
        tasks = [
            process_single_query(query, i) 
            for i, query in enumerate(queries)
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by original index and extract responses
        sorted_results = sorted(
            [r for r in results if not isinstance(r, Exception)],
            key=lambda x: x[0]
        )
        
        responses = [result[1] for result in sorted_results]
        
        logger.info(f"Batch processing completed: {len(responses)} responses")
        return responses
    
    def create_context(
        self,
        user_id: Optional[str] = None,
        domain: str = "general",
        urgency_level: str = "normal",
        required_accuracy: str = "high",
        conversation_history: Optional[List[Dict]] = None,
        user_permissions: Optional[List[str]] = None
    ) -> SearchContext:
        """
        Convenience method to create SearchContext
        
        Args:
            user_id: User identifier for access control
            domain: Domain context (e.g., "compliance", "technical", "general")
            urgency_level: "low", "normal", "high" - affects processing strategy
            required_accuracy: "low", "medium", "high" - affects confidence thresholds
            conversation_history: Previous conversation messages
            user_permissions: List of user permissions for collection access
            
        Returns:
            SearchContext object
        """
        
        return SearchContext(
            user_id=user_id,
            domain=domain,
            urgency_level=urgency_level,
            required_accuracy=required_accuracy,
            conversation_history=conversation_history or [],
            user_permissions=user_permissions or []
        )
    
    def create_options(
        self,
        max_iterations: int = 3,
        stream: bool = False,
        include_sources: bool = True,
        include_execution_trace: bool = False,
        confidence_threshold: float = 0.6,
        max_results_per_collection: int = 10,
        execution_timeout_ms: int = 30000
    ) -> RAGOptions:
        """
        Convenience method to create RAGOptions
        
        Args:
            max_iterations: Maximum search iterations for refinement
            stream: Enable streaming responses
            include_sources: Include source references in response
            include_execution_trace: Include detailed execution trace
            confidence_threshold: Minimum confidence for results
            max_results_per_collection: Maximum results per collection
            execution_timeout_ms: Timeout for entire execution
            
        Returns:
            RAGOptions object
        """
        
        return RAGOptions(
            max_iterations=max_iterations,
            stream=stream,
            include_sources=include_sources,
            include_execution_trace=include_execution_trace,
            confidence_threshold=confidence_threshold,
            max_results_per_collection=max_results_per_collection,
            execution_timeout_ms=execution_timeout_ms
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health status of RAG system components
        
        Returns:
            Dict with health status and performance metrics
        """
        
        try:
            # Test basic functionality with a simple query
            test_response = await self.query(
                "test query for health check",
                options=RAGOptions(max_iterations=1, execution_timeout_ms=5000)
            )
            
            # Get orchestrator performance stats
            orchestrator_stats = self.orchestrator.get_performance_stats()
            
            # Check collection availability
            collections_available = len(self.orchestrator.tool_registry.get_collection_names())
            
            health_status = {
                "status": "healthy" if test_response.processing_time_ms < 10000 else "degraded",
                "timestamp": asyncio.get_event_loop().time(),
                "performance": {
                    **self._performance_stats,
                    **orchestrator_stats
                },
                "collections_available": collections_available,
                "components": {
                    "orchestrator": "healthy",
                    "llm_router": "healthy",
                    "query_engine": "healthy",
                    "result_fusion": "healthy"
                }
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "timestamp": asyncio.get_event_loop().time(),
                "error": str(e),
                "components": {
                    "orchestrator": "unknown",
                    "llm_router": "unknown", 
                    "query_engine": "unknown",
                    "result_fusion": "unknown"
                }
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        return {
            "interface_stats": self._performance_stats.copy(),
            "orchestrator_stats": self.orchestrator.get_performance_stats()
        }
    
    def _update_performance_stats(self, processing_time_ms: int, success: bool):
        """Update internal performance statistics"""
        
        self._performance_stats["total_queries"] += 1
        
        # Update average response time
        current_avg = self._performance_stats["avg_response_time_ms"]
        total_queries = self._performance_stats["total_queries"]
        
        self._performance_stats["avg_response_time_ms"] = (
            (current_avg * (total_queries - 1) + processing_time_ms) / total_queries
        )
        
        # Update success rate
        if success:
            current_success_rate = self._performance_stats["success_rate"]
            self._performance_stats["success_rate"] = (
                (current_success_rate * (total_queries - 1) + 1.0) / total_queries
            )


# Convenience functions for direct use
async def rag_query(
    query: str,
    user_id: Optional[str] = None,
    domain: str = "general",
    include_sources: bool = True
) -> RAGResponse:
    """
    Quick utility function for simple RAG queries
    
    Args:
        query: User query
        user_id: Optional user ID for access control
        domain: Domain context
        include_sources: Whether to include sources
        
    Returns:
        RAGResponse object
    """
    
    rag = StandaloneRAGInterface()
    
    context = None
    if user_id:
        context = SearchContext(user_id=user_id, domain=domain)
    
    options = RAGOptions(include_sources=include_sources)
    
    return await rag.query(query, context, options)


async def rag_query_stream(
    query: str,
    user_id: Optional[str] = None,
    domain: str = "general"
) -> AsyncGenerator[RAGStreamChunk, None]:
    """
    Quick utility function for streaming RAG queries
    
    Args:
        query: User query
        user_id: Optional user ID for access control
        domain: Domain context
        
    Yields:
        RAGStreamChunk objects
    """
    
    rag = StandaloneRAGInterface()
    
    context = None
    if user_id:
        context = SearchContext(user_id=user_id, domain=domain)
    
    async for chunk in rag.query_stream(query, context):
        yield chunk