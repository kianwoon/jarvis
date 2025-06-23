"""
Main RAG Orchestrator for LLM-driven intelligent routing and execution

This is the core orchestration engine that coordinates LLM routing, collection searches,
result fusion, and iterative refinement for optimal RAG performance.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime

from app.rag_agent.utils.types import (
    RAGResponse, RAGOptions, SearchContext, ExecutionTrace, 
    StepResult, CollectionResult, Source, RAGStreamChunk,
    ExecutionStrategy, FusionMethod
)
from app.rag_agent.routers.llm_router import LLMRouter
from app.rag_agent.routers.tool_registry import get_collection_tool_registry
from app.rag_agent.core.query_engine import QueryEngine
from app.rag_agent.core.result_fusion import ResultFusion

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Main orchestrator for LLM-driven RAG routing and execution
    
    This class coordinates the entire RAG pipeline:
    1. LLM-based routing to determine optimal collections
    2. Parallel/sequential search execution
    3. Result fusion and ranking
    4. Iterative refinement based on quality analysis
    """
    
    def __init__(self):
        self.llm_router = LLMRouter()
        self.tool_registry = get_collection_tool_registry()
        self.query_engine = QueryEngine()
        self.result_fusion = ResultFusion()
        
        # Performance tracking
        self._execution_stats = {
            "total_queries": 0,
            "avg_execution_time_ms": 0,
            "success_rate": 0.0
        }
    
    async def process_query(
        self,
        query: str,
        context: Optional[SearchContext] = None,
        options: Optional[RAGOptions] = None
    ) -> RAGResponse:
        """
        Main entry point for RAG query processing
        
        Args:
            query: User query string
            context: Search context with user information
            options: Processing options and configuration
            
        Returns:
            RAGResponse with results, sources, and execution metadata
        """
        start_time = time.time()
        execution_id = str(uuid.uuid4())
        
        # Set default options
        if options is None:
            options = RAGOptions()
        
        logger.info(f"[{execution_id}] Processing RAG query: {query[:100]}...")
        
        try:
            # Phase 1: LLM-based routing
            routing_decision = await self.llm_router.route_query(
                query=query,
                context=context,
                max_collections=options.max_results_per_collection // 2  # Conservative limit
            )
            
            if not routing_decision.selected_collections:
                logger.warning(f"[{execution_id}] No collections selected by router")
                return self._create_empty_response(
                    "No relevant collections found for your query. Please try rephrasing your question.",
                    execution_id
                )
            
            logger.info(f"[{execution_id}] Router selected {len(routing_decision.selected_collections)} collections: "
                       f"{routing_decision.selected_collections}")
            
            # Phase 2: Execute searches based on strategy
            step_results = await self._execute_search_strategy(
                routing_decision=routing_decision,
                context=context,
                options=options,
                execution_id=execution_id
            )
            
            if not step_results or not any(sr.collection_results for sr in step_results):
                logger.warning(f"[{execution_id}] No results from any collection searches")
                return self._create_empty_response(
                    "No relevant information found in the knowledge base for your query.",
                    execution_id
                )
            
            # Phase 3: Fuse and rank results
            final_response = await self.result_fusion.fuse_results(
                step_results=step_results,
                original_query=query,
                routing_decision=routing_decision,
                context=context,
                options=options
            )
            
            # Phase 4: Build execution trace if requested
            execution_trace = None
            if options.include_execution_trace:
                execution_trace = self._build_execution_trace(
                    execution_id, step_results, routing_decision, start_time
                )
            
            # Calculate final processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Build final response
            response = RAGResponse(
                content=final_response.content,
                sources=final_response.sources,
                confidence_score=final_response.confidence_score,
                execution_trace=execution_trace,
                collections_searched=routing_decision.selected_collections,
                query_refinements=list(routing_decision.query_refinements.values()),
                processing_time_ms=processing_time_ms,
                fusion_method=final_response.fusion_method
            )
            
            # Update performance stats
            self._update_performance_stats(processing_time_ms, True)
            
            logger.info(f"[{execution_id}] RAG query completed in {processing_time_ms}ms "
                       f"with confidence {response.confidence_score:.2f}")
            
            return response
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[{execution_id}] Error processing RAG query: {e}")
            
            # Update performance stats
            self._update_performance_stats(processing_time_ms, False)
            
            return RAGResponse(
                content=f"I encountered an error while processing your query: {str(e)}. Please try again or rephrase your question.",
                sources=[],
                confidence_score=0.0,
                processing_time_ms=processing_time_ms
            )
    
    async def process_query_stream(
        self,
        query: str,
        context: Optional[SearchContext] = None,
        options: Optional[RAGOptions] = None
    ) -> AsyncGenerator[RAGStreamChunk, None]:
        """
        Streaming version of query processing for real-time feedback
        
        Yields:
            RAGStreamChunk objects with progressive results
        """
        if options is None:
            options = RAGOptions(stream=True)
        
        execution_id = str(uuid.uuid4())
        
        try:
            # Yield initial status
            yield RAGStreamChunk(
                content="🔍 Analyzing query and selecting relevant knowledge sources...",
                chunk_type="status"
            )
            
            # Phase 1: Routing
            routing_decision = await self.llm_router.route_query(query, context)
            
            if routing_decision.selected_collections:
                collections_text = ", ".join(routing_decision.selected_collections)
                yield RAGStreamChunk(
                    content=f"📚 Searching collections: {collections_text}",
                    chunk_type="status"
                )
            
            # Phase 2: Execute searches with progress updates
            step_results = []
            
            if routing_decision.execution_strategy == ExecutionStrategy.PARALLEL_SEARCH:
                # Parallel execution with progress
                async for chunk in self._execute_parallel_with_progress(
                    routing_decision, context, options, execution_id
                ):
                    if chunk.chunk_type == "result":
                        step_results.extend(chunk.step_info.get("results", []))
                    yield chunk
            else:
                # Sequential execution
                step_results = await self._execute_search_strategy(
                    routing_decision, context, options, execution_id
                )
                
                yield RAGStreamChunk(
                    content="🔗 Combining results from multiple sources...",
                    chunk_type="status"
                )
            
            # Phase 3: Final fusion and response
            if step_results and any(sr.collection_results for sr in step_results):
                final_response = await self.result_fusion.fuse_results(
                    step_results, query, routing_decision, context, options
                )
                
                # Yield final content
                yield RAGStreamChunk(
                    content=final_response.content,
                    sources=final_response.sources,
                    chunk_type="content",
                    is_final=True
                )
            else:
                yield RAGStreamChunk(
                    content="No relevant information found in the knowledge base.",
                    chunk_type="content",
                    is_final=True
                )
                
        except Exception as e:
            logger.error(f"[{execution_id}] Error in streaming query: {e}")
            yield RAGStreamChunk(
                content=f"Error processing query: {str(e)}",
                chunk_type="error",
                is_final=True
            )
    
    async def _execute_search_strategy(
        self,
        routing_decision,
        context: Optional[SearchContext],
        options: RAGOptions,
        execution_id: str
    ) -> List[StepResult]:
        """Execute search strategy based on routing decision"""
        
        strategy = routing_decision.execution_strategy
        
        if strategy == ExecutionStrategy.PARALLEL_SEARCH:
            return await self._execute_parallel_searches(
                routing_decision, context, options, execution_id
            )
        elif strategy == ExecutionStrategy.ITERATIVE_REFINEMENT:
            return await self._execute_iterative_searches(
                routing_decision, context, options, execution_id
            )
        else:  # SINGLE_COLLECTION or fallback
            return await self._execute_single_search(
                routing_decision, context, options, execution_id
            )
    
    async def _execute_parallel_searches(
        self,
        routing_decision,
        context: Optional[SearchContext],
        options: RAGOptions,
        execution_id: str
    ) -> List[StepResult]:
        """Execute searches on multiple collections in parallel"""
        
        search_requests = []
        for collection_name in routing_decision.selected_collections:
            refined_query = routing_decision.query_refinements.get(
                collection_name, routing_decision.query_refinements.get("default", "")
            )
            
            search_requests.append({
                "collection_name": collection_name,
                "query": refined_query,
                "search_strategy": "auto",
                "max_results": options.max_results_per_collection,
                "similarity_threshold": options.confidence_threshold
            })
        
        logger.info(f"[{execution_id}] Executing {len(search_requests)} parallel searches")
        
        # Execute all searches in parallel
        collection_results = await self.query_engine.execute_parallel_searches(
            search_requests, context
        )
        
        # Wrap in StepResult
        step_result = StepResult(
            step_id=f"{execution_id}_parallel",
            step_type="parallel_search",
            collection_results=collection_results,
            success_score=self._calculate_step_success_score(collection_results),
            execution_time_ms=sum(cr.execution_time_ms for cr in collection_results)
        )
        
        return [step_result]
    
    async def _execute_single_search(
        self,
        routing_decision,
        context: Optional[SearchContext],
        options: RAGOptions,
        execution_id: str
    ) -> List[StepResult]:
        """Execute search on a single collection"""
        
        if not routing_decision.selected_collections:
            return []
        
        collection_name = routing_decision.selected_collections[0]
        refined_query = routing_decision.query_refinements.get(
            collection_name, routing_decision.query_refinements.get("default", "")
        )
        
        logger.info(f"[{execution_id}] Executing single search on {collection_name}")
        
        collection_result = await self.query_engine.execute_collection_search(
            collection_name=collection_name,
            query=refined_query,
            search_strategy="auto",
            max_results=options.max_results_per_collection,
            similarity_threshold=options.confidence_threshold,
            context=context
        )
        
        step_result = StepResult(
            step_id=f"{execution_id}_single",
            step_type="single_search",
            collection_results=[collection_result],
            success_score=collection_result.relevance_score,
            execution_time_ms=collection_result.execution_time_ms
        )
        
        return [step_result]
    
    async def _execute_iterative_searches(
        self,
        routing_decision,
        context: Optional[SearchContext],
        options: RAGOptions,
        execution_id: str
    ) -> List[StepResult]:
        """Execute iterative searches with refinement"""
        
        step_results = []
        collections_to_search = routing_decision.selected_collections.copy()
        
        # Start with primary collection
        primary_collection = collections_to_search.pop(0) if collections_to_search else None
        if not primary_collection:
            return []
        
        logger.info(f"[{execution_id}] Starting iterative search with {primary_collection}")
        
        # Initial search
        initial_query = routing_decision.query_refinements.get(primary_collection, "")
        initial_result = await self.query_engine.execute_collection_search(
            collection_name=primary_collection,
            query=initial_query,
            search_strategy="auto",
            max_results=options.max_results_per_collection,
            context=context
        )
        
        step_results.append(StepResult(
            step_id=f"{execution_id}_iter_0",
            step_type="initial_search",
            collection_results=[initial_result],
            success_score=initial_result.relevance_score,
            execution_time_ms=initial_result.execution_time_ms
        ))
        
        # Iterative searches on remaining collections if initial results are insufficient
        if (initial_result.relevance_score < 0.6 and 
            len(initial_result.sources) < 3 and 
            collections_to_search):
            
            # Search additional collections
            for i, collection_name in enumerate(collections_to_search[:2]):  # Limit iterations
                refined_query = routing_decision.query_refinements.get(collection_name, initial_query)
                
                logger.info(f"[{execution_id}] Iterative search {i+1} on {collection_name}")
                
                additional_result = await self.query_engine.execute_collection_search(
                    collection_name=collection_name,
                    query=refined_query,
                    search_strategy="auto",
                    max_results=options.max_results_per_collection,
                    context=context
                )
                
                step_results.append(StepResult(
                    step_id=f"{execution_id}_iter_{i+1}",
                    step_type="iterative_search",
                    collection_results=[additional_result],
                    success_score=additional_result.relevance_score,
                    execution_time_ms=additional_result.execution_time_ms
                ))
                
                # Stop if we have sufficient results
                total_sources = sum(len(sr.collection_results[0].sources) for sr in step_results)
                if total_sources >= options.max_results_per_collection:
                    break
        
        return step_results
    
    async def _execute_parallel_with_progress(
        self,
        routing_decision,
        context: Optional[SearchContext],
        options: RAGOptions,
        execution_id: str
    ) -> AsyncGenerator[RAGStreamChunk, None]:
        """Execute parallel searches with progress updates"""
        
        # Create search tasks
        search_tasks = {}
        for collection_name in routing_decision.selected_collections:
            refined_query = routing_decision.query_refinements.get(collection_name, "")
            
            task = self.query_engine.execute_collection_search(
                collection_name=collection_name,
                query=refined_query,
                search_strategy="auto",
                max_results=options.max_results_per_collection,
                context=context
            )
            search_tasks[collection_name] = task
        
        # Wait for results with progress updates
        completed_results = []
        
        for collection_name, task in search_tasks.items():
            yield RAGStreamChunk(
                content=f"🔍 Searching {collection_name}...",
                chunk_type="status"
            )
            
            try:
                result = await task
                completed_results.append(result)
                
                if result.sources:
                    yield RAGStreamChunk(
                        content=f"✅ Found {len(result.sources)} results in {collection_name}",
                        chunk_type="status"
                    )
                else:
                    yield RAGStreamChunk(
                        content=f"⚠️ No results found in {collection_name}",
                        chunk_type="status"
                    )
                    
            except Exception as e:
                logger.error(f"Search failed for {collection_name}: {e}")
                yield RAGStreamChunk(
                    content=f"❌ Search failed for {collection_name}",
                    chunk_type="status"
                )
        
        # Yield final results
        yield RAGStreamChunk(
            content="",
            chunk_type="result",
            step_info={"results": [StepResult(
                step_id=f"{execution_id}_parallel_stream",
                step_type="parallel_search",
                collection_results=completed_results,
                success_score=self._calculate_step_success_score(completed_results),
                execution_time_ms=sum(cr.execution_time_ms for cr in completed_results)
            )]}
        )
    
    def _calculate_step_success_score(self, collection_results: List[CollectionResult]) -> float:
        """Calculate success score for a step based on collection results"""
        
        if not collection_results:
            return 0.0
        
        # Average relevance scores weighted by result count
        total_weight = 0
        weighted_score = 0
        
        for result in collection_results:
            weight = len(result.sources) + 1  # +1 to avoid zero weight
            weighted_score += result.relevance_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _build_execution_trace(
        self,
        execution_id: str,
        step_results: List[StepResult],
        routing_decision,
        start_time: float
    ) -> ExecutionTrace:
        """Build execution trace for debugging and analysis"""
        
        total_time_ms = int((time.time() - start_time) * 1000)
        
        collections_searched = []
        query_refinements = []
        
        for step_result in step_results:
            for collection_result in step_result.collection_results:
                if collection_result.collection_name not in collections_searched:
                    collections_searched.append(collection_result.collection_name)
                if collection_result.query_used not in query_refinements:
                    query_refinements.append(collection_result.query_used)
        
        return ExecutionTrace(
            plan_id=execution_id,
            steps_executed=step_results,
            total_time_ms=total_time_ms,
            collections_searched=collections_searched,
            query_refinements=query_refinements,
            final_strategy=routing_decision.execution_strategy
        )
    
    def _create_empty_response(self, message: str, execution_id: str) -> RAGResponse:
        """Create empty response for failed queries"""
        
        return RAGResponse(
            content=message,
            sources=[],
            confidence_score=0.0,
            collections_searched=[],
            query_refinements=[],
            processing_time_ms=0
        )
    
    def _update_performance_stats(self, processing_time_ms: int, success: bool):
        """Update internal performance statistics"""
        
        self._execution_stats["total_queries"] += 1
        
        # Update average execution time
        current_avg = self._execution_stats["avg_execution_time_ms"]
        total_queries = self._execution_stats["total_queries"]
        
        self._execution_stats["avg_execution_time_ms"] = (
            (current_avg * (total_queries - 1) + processing_time_ms) / total_queries
        )
        
        # Update success rate
        if success:
            current_success_rate = self._execution_stats["success_rate"]
            self._execution_stats["success_rate"] = (
                (current_success_rate * (total_queries - 1) + 1.0) / total_queries
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self._execution_stats.copy()