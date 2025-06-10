"""
Self-Reflective RAG Service

This module integrates self-reflection capabilities into the existing RAG pipeline,
providing an enhanced version that can evaluate and iteratively improve responses.
"""

import logging
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime

from app.rag.response_quality_evaluator import ResponseQualityEvaluator
from app.rag.retrieval_quality_monitor import RetrievalQualityMonitor
from app.rag.iterative_refinement_engine import IterativeRefinementEngine
from app.rag.reflection_orchestrator import (
    ReflectionOrchestrator, ReflectionContext, ReflectionMode, ReflectionResult
)
from app.langchain.enhanced_rag_service import enhanced_rag_answer
from app.core.llm_settings_cache import get_llm_settings_by_model

logger = logging.getLogger(__name__)


class SelfReflectiveRAGService:
    """
    Enhanced RAG service with self-reflection capabilities
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the self-reflective RAG service
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize reflection components
        self.response_evaluator = ResponseQualityEvaluator()
        self.retrieval_monitor = RetrievalQualityMonitor()
        self.refinement_engine = None  # Will be initialized with retriever/generator
        self.orchestrator = None  # Will be initialized on first use
        
        # Cache for reflection results
        self._reflection_cache = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "enable_reflection": True,
            "reflection_mode": ReflectionMode.BALANCED,
            "cache_reflections": True,
            "cache_ttl_seconds": 3600,
            "min_query_length_for_reflection": 10,
            "reflection_model": None,  # Use same model as generation
            "streaming_reflection": False,  # Stream intermediate results
            "reflection_thresholds": {
                "always_reflect_below": 0.6,
                "never_reflect_above": 0.9,
                "consider_reflection_between": (0.6, 0.9)
            }
        }
    
    async def self_reflective_rag_answer(
        self,
        query: str,
        collection_names: List[str],
        model_name: str,
        conversation_history: Optional[List[Dict]] = None,
        stream: bool = True,
        enable_reflection: Optional[bool] = None,
        reflection_mode: Optional[ReflectionMode] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Enhanced RAG answer with self-reflection
        
        Args:
            query: User query
            collection_names: Vector collections to search
            model_name: LLM model to use
            conversation_history: Previous conversation
            stream: Whether to stream the response
            enable_reflection: Override default reflection setting
            reflection_mode: Override default reflection mode
            **kwargs: Additional arguments for RAG
            
        Yields:
            Response chunks with reflection metadata
        """
        # Check if reflection should be enabled
        if enable_reflection is None:
            enable_reflection = self.config["enable_reflection"]
        
        if reflection_mode is None:
            reflection_mode = self.config["reflection_mode"]
        
        # For short queries, skip reflection
        if len(query.strip()) < self.config["min_query_length_for_reflection"]:
            enable_reflection = False
        
        # Generate initial response using existing RAG
        initial_response = ""
        retrieved_documents = []
        
        # Collect the initial response
        async for chunk in enhanced_rag_answer(
            query=query,
            collection_names=collection_names,
            model_name=model_name,
            conversation_history=conversation_history,
            stream=False,  # Need complete response for reflection
            **kwargs
        ):
            if isinstance(chunk, dict):
                initial_response = chunk.get("answer", "")
                retrieved_documents = chunk.get("sources", [])
                metadata = chunk.get("metadata", {})
            else:
                initial_response += chunk
        
        # If reflection is disabled, stream the initial response
        if not enable_reflection:
            if stream:
                for i in range(0, len(initial_response), 10):
                    yield initial_response[i:i+10]
            else:
                yield {
                    "answer": initial_response,
                    "sources": retrieved_documents,
                    "metadata": metadata,
                    "reflection_performed": False
                }
            return
        
        # Check cache
        cache_key = self._get_cache_key(query, collection_names, model_name)
        if self.config["cache_reflections"] and cache_key in self._reflection_cache:
            cached_result = self._reflection_cache[cache_key]
            if self._is_cache_valid(cached_result):
                logger.info(f"Using cached reflection for query: {query[:50]}...")
                if stream:
                    yield from self._stream_response(cached_result["response"])
                else:
                    yield cached_result
                return
        
        # Perform reflection
        try:
            reflection_result = await self._perform_reflection(
                query=query,
                initial_response=initial_response,
                retrieved_documents=retrieved_documents,
                conversation_history=conversation_history,
                model_name=model_name,
                mode=reflection_mode
            )
            
            # Cache result if enabled
            if self.config["cache_reflections"]:
                self._cache_result(cache_key, reflection_result)
            
            # Stream or return the reflected response
            if stream:
                # Stream with metadata
                if self.config["streaming_reflection"]:
                    yield from self._stream_with_reflection_info(reflection_result)
                else:
                    yield from self._stream_response(reflection_result.final_response)
            else:
                yield {
                    "answer": reflection_result.final_response,
                    "sources": retrieved_documents,
                    "metadata": {
                        **metadata,
                        "reflection_performed": True,
                        "quality_score": reflection_result.quality_score,
                        "improvements": reflection_result.improvements_made,
                        "reflection_time_ms": reflection_result.reflection_metrics.total_time_ms
                    }
                }
                
        except Exception as e:
            logger.error(f"Reflection failed, using initial response: {str(e)}")
            # Fallback to initial response
            if stream:
                yield from self._stream_response(initial_response)
            else:
                yield {
                    "answer": initial_response,
                    "sources": retrieved_documents,
                    "metadata": {
                        **metadata,
                        "reflection_performed": False,
                        "reflection_error": str(e)
                    }
                }
    
    async def _perform_reflection(
        self,
        query: str,
        initial_response: str,
        retrieved_documents: List[Dict],
        conversation_history: Optional[List[Dict]],
        model_name: str,
        mode: ReflectionMode
    ) -> ReflectionResult:
        """Perform the reflection process"""
        # Initialize orchestrator if needed
        if self.orchestrator is None:
            self._initialize_orchestrator(model_name)
        
        # Create reflection context
        context = ReflectionContext(
            query=query,
            initial_response=initial_response,
            retrieved_documents=retrieved_documents,
            conversation_history=conversation_history,
            metadata={"model": model_name}
        )
        
        # Check if reflection is needed based on initial quality
        quick_eval = await self._quick_quality_check(
            query, initial_response, retrieved_documents
        )
        
        thresholds = self.config["reflection_thresholds"]
        if quick_eval > thresholds["never_reflect_above"]:
            logger.info(f"Skipping reflection - high quality: {quick_eval:.2f}")
            return self._create_skip_reflection_result(
                initial_response, quick_eval, "High initial quality"
            )
        
        force_reflection = quick_eval < thresholds["always_reflect_below"]
        
        # Perform reflection
        logger.info(f"Performing {mode.value} reflection for query: {query[:50]}...")
        result = await self.orchestrator.reflect_and_improve(
            context=context,
            mode=mode,
            force_refinement=force_reflection
        )
        
        return result
    
    def _initialize_orchestrator(self, model_name: str):
        """Initialize the reflection orchestrator with model-specific settings"""
        # Get LLM settings
        llm_settings = get_llm_settings_by_model(model_name)
        
        # Create retriever and generator wrappers
        retriever = RAGRetrieverWrapper()
        generator = RAGGeneratorWrapper(model_name, llm_settings)
        
        # Initialize refinement engine
        self.refinement_engine = IterativeRefinementEngine(
            retriever=retriever,
            generator=generator,
            llm_client=generator  # Use same for LLM tasks
        )
        
        # Initialize orchestrator
        self.orchestrator = ReflectionOrchestrator(
            response_evaluator=self.response_evaluator,
            retrieval_monitor=self.retrieval_monitor,
            refinement_engine=self.refinement_engine
        )
    
    async def _quick_quality_check(
        self, query: str, response: str, documents: List[Dict]
    ) -> float:
        """Perform a quick quality check to determine if reflection is needed"""
        # Simple heuristics for quick check
        score = 1.0
        
        # Check response length
        if len(response) < 50:
            score -= 0.3
        
        # Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words) / max(len(query_words), 1)
        if overlap < 0.2:
            score -= 0.3
        
        # Check document relevance
        if documents:
            avg_score = sum(d.get('score', 0.5) for d in documents) / len(documents)
            if avg_score < 0.7:
                score -= 0.2
        
        return max(0.0, score)
    
    def _create_skip_reflection_result(
        self, response: str, quality: float, reason: str
    ) -> ReflectionResult:
        """Create a reflection result when skipping reflection"""
        from app.rag.reflection_orchestrator import ReflectionMetrics
        
        return ReflectionResult(
            final_response=response,
            quality_score=quality,
            improvements_made=[f"Skipped: {reason}"],
            reflection_metrics=ReflectionMetrics(
                total_time_ms=0,
                evaluation_time_ms=0,
                refinement_time_ms=0,
                iterations_performed=0,
                quality_improvement=0,
                strategies_attempted=[],
                final_confidence=quality
            ),
            evaluation_results=None,
            retrieval_assessment=None,
            refinement_details=None,
            success=True,
            metadata={"skipped": True, "reason": reason}
        )
    
    def _get_cache_key(
        self, query: str, collections: List[str], model: str
    ) -> str:
        """Generate cache key for reflection results"""
        import hashlib
        key_parts = [query, ','.join(sorted(collections)), model]
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_result: Dict) -> bool:
        """Check if cached result is still valid"""
        if "timestamp" not in cached_result:
            return False
        
        age_seconds = (datetime.now() - cached_result["timestamp"]).total_seconds()
        return age_seconds < self.config["cache_ttl_seconds"]
    
    def _cache_result(self, key: str, result: ReflectionResult):
        """Cache reflection result"""
        self._reflection_cache[key] = {
            "response": result.final_response,
            "quality": result.quality_score,
            "improvements": result.improvements_made,
            "timestamp": datetime.now()
        }
    
    async def _stream_response(self, response: str):
        """Stream response in chunks"""
        chunk_size = 10
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]
    
    async def _stream_with_reflection_info(self, result: ReflectionResult):
        """Stream response with reflection information"""
        # Send initial metadata
        yield {
            "type": "reflection_start",
            "quality_score": result.quality_score,
            "improvements": len(result.improvements_made)
        }
        
        # Stream the response
        chunk_size = 10
        for i in range(0, len(result.final_response), chunk_size):
            yield {
                "type": "content",
                "chunk": result.final_response[i:i + chunk_size]
            }
        
        # Send final metadata
        yield {
            "type": "reflection_complete",
            "metrics": {
                "total_time_ms": result.reflection_metrics.total_time_ms,
                "iterations": result.reflection_metrics.iterations_performed,
                "improvement": result.reflection_metrics.quality_improvement
            }
        }


class RAGRetrieverWrapper:
    """Wrapper to integrate with existing RAG retriever"""
    
    async def retrieve(self, query: str) -> List[Dict]:
        """Retrieve documents for query"""
        # This would integrate with existing Milvus retrieval
        # For now, return placeholder
        return [
            {"id": "1", "content": f"Retrieved document for: {query}", "score": 0.8}
        ]


class RAGGeneratorWrapper:
    """Wrapper to integrate with existing RAG generator"""
    
    def __init__(self, model_name: str, settings: Dict):
        self.model_name = model_name
        self.settings = settings
    
    async def generate(
        self, query: str, documents: List[Dict], history: Optional[List[Dict]]
    ) -> str:
        """Generate response using existing RAG pipeline"""
        # This would integrate with existing generation logic
        # For now, return placeholder
        context = '\n'.join(d.get('content', '') for d in documents)
        return f"Generated response for '{query}' using context from {len(documents)} documents"
    
    async def complete(self, prompt: str) -> str:
        """Complete a prompt for refinement tasks"""
        # This would use the LLM for completion
        return f"Completion for: {prompt}"


# Convenience function for backward compatibility
async def self_reflective_rag_answer(
    query: str,
    collection_names: List[str],
    model_name: str,
    conversation_history: Optional[List[Dict]] = None,
    stream: bool = True,
    enable_reflection: bool = True,
    reflection_mode: Optional[ReflectionMode] = None,
    **kwargs
) -> AsyncGenerator[str, None]:
    """
    Convenience function for self-reflective RAG
    """
    service = SelfReflectiveRAGService()
    
    async for chunk in service.self_reflective_rag_answer(
        query=query,
        collection_names=collection_names,
        model_name=model_name,
        conversation_history=conversation_history,
        stream=stream,
        enable_reflection=enable_reflection,
        reflection_mode=reflection_mode,
        **kwargs
    ):
        yield chunk