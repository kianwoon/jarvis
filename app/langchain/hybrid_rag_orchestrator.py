"""
Hybrid RAG orchestrator for intelligent routing between temporary and persistent RAG.
Implements strategy pattern for different blending approaches and result fusion.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.services.temporary_document_service import get_temporary_document_service
from app.core.in_memory_rag_settings import get_in_memory_rag_settings

logger = logging.getLogger(__name__)

class RAGStrategy(Enum):
    """Available RAG routing strategies."""
    TEMP_PRIORITY = "temp_priority"  # Try temp docs first, fallback to persistent
    PERSISTENT_PRIORITY = "persistent_priority"  # Try persistent first, fallback to temp
    PARALLEL_FUSION = "parallel_fusion"  # Query both and blend results
    ADAPTIVE = "adaptive"  # Dynamically choose based on content and context

class RAGMode(Enum):
    """RAG execution modes."""
    TEMP_ONLY = "temp_only"
    PERSISTENT_ONLY = "persistent_only"
    HYBRID = "hybrid"

@dataclass
class RAGSource:
    """Information about a RAG source."""
    source_type: str  # "temp_documents", "persistent_rag", "in_memory_rag"
    priority: float
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResult:
    """Result from a RAG query."""
    sources: List[Dict[str, Any]]
    total_chunks: int
    processing_time_ms: float
    source_info: Dict[str, Any]
    score: float = 0.0
    rank: int = 0

@dataclass
class HybridRAGResponse:
    """Response from hybrid RAG orchestrator."""
    query: str
    final_sources: List[Dict[str, Any]]
    strategy_used: str
    sources_queried: List[str]
    fusion_metadata: Dict[str, Any]
    total_processing_time_ms: float
    conversation_id: str

class RAGStrategyInterface(ABC):
    """Abstract interface for RAG strategies."""
    
    @abstractmethod
    async def execute(
        self, 
        query: str, 
        conversation_id: str, 
        available_sources: List[RAGSource],
        **kwargs
    ) -> HybridRAGResponse:
        """Execute the RAG strategy."""
        pass

class TempPriorityStrategy(RAGStrategyInterface):
    """Strategy that prioritizes temporary documents over persistent RAG."""
    
    async def execute(
        self, 
        query: str, 
        conversation_id: str, 
        available_sources: List[RAGSource],
        **kwargs
    ) -> HybridRAGResponse:
        """Execute temp priority strategy."""
        start_time = datetime.now()
        sources_queried = []
        final_sources = []
        fusion_metadata = {}
        
        try:
            config = get_in_memory_rag_settings()
            temp_service = get_temporary_document_service()
            top_k = kwargs.get('top_k', config.max_results_per_query)
            
            # First, try temporary documents
            temp_source = next((s for s in available_sources if s.source_type == "temp_documents"), None)
            if temp_source and temp_source.enabled:
                sources_queried.append("temp_documents")
                
                temp_result = await temp_service.query_documents(
                    conversation_id=conversation_id,
                    query=query,
                    use_in_memory_rag=True,
                    fallback_to_temp_docs=True,
                    top_k=top_k
                )
                
                if temp_result.get('sources'):
                    # Filter by minimum score threshold
                    filtered_sources = [
                        source for source in temp_result['sources']
                        if source.get('score', 0) >= config.min_temp_doc_score
                    ]
                    
                    if filtered_sources:
                        final_sources = filtered_sources
                        fusion_metadata = {
                            'primary_source': 'temp_documents',
                            'temp_doc_results': len(filtered_sources),
                            'temp_doc_processing_time': temp_result.get('processing_time_ms', 0)
                        }
                        
                        logger.info(f"Temp priority strategy found {len(filtered_sources)} results from temp docs")
                        
                        processing_time = (datetime.now() - start_time).total_seconds() * 1000
                        return HybridRAGResponse(
                            query=query,
                            final_sources=final_sources,
                            strategy_used=RAGStrategy.TEMP_PRIORITY.value,
                            sources_queried=sources_queried,
                            fusion_metadata=fusion_metadata,
                            total_processing_time_ms=processing_time,
                            conversation_id=conversation_id
                        )
            
            # Fallback to persistent RAG if enabled
            persistent_source = next((s for s in available_sources if s.source_type == "persistent_rag"), None)
            if persistent_source and persistent_source.enabled and config.fallback_to_persistent:
                sources_queried.append("persistent_rag")
                
                # Call persistent RAG service
                persistent_result = await self._query_persistent_rag(
                    query, conversation_id, top_k, **kwargs
                )
                
                if persistent_result:
                    final_sources = persistent_result.get('sources', [])
                    fusion_metadata = {
                        'primary_source': 'persistent_rag',
                        'fallback_reason': 'no_temp_doc_results',
                        'persistent_rag_results': len(final_sources)
                    }
                    
                    logger.info(f"Temp priority strategy fell back to persistent RAG: {len(final_sources)} results")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return HybridRAGResponse(
                query=query,
                final_sources=final_sources,
                strategy_used=RAGStrategy.TEMP_PRIORITY.value,
                sources_queried=sources_queried,
                fusion_metadata=fusion_metadata,
                total_processing_time_ms=processing_time,
                conversation_id=conversation_id
            )
            
        except Exception as e:
            logger.error(f"Temp priority strategy failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return HybridRAGResponse(
                query=query,
                final_sources=[],
                strategy_used=RAGStrategy.TEMP_PRIORITY.value,
                sources_queried=sources_queried,
                fusion_metadata={'error': str(e)},
                total_processing_time_ms=processing_time,
                conversation_id=conversation_id
            )
    
    async def _query_persistent_rag(
        self, 
        query: str, 
        conversation_id: str, 
        top_k: int,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Query persistent RAG system."""
        try:
            # Import here to avoid circular imports
            from app.langchain.service import handle_rag_query
            
            # Use existing RAG service
            context, documents = handle_rag_query(
                question=query,
                thinking=kwargs.get('thinking', False),
                collections=kwargs.get('collections', None),
                collection_strategy=kwargs.get('collection_strategy', 'auto')
            )
            
            if documents:
                # Convert to standard format
                sources = []
                for i, doc in enumerate(documents[:top_k]):
                    if isinstance(doc, dict):
                        sources.append({
                            'content': doc.get('text', ''),
                            'metadata': doc.get('metadata', {}),
                            'score': doc.get('score', 0.5),
                            'rank': i + 1,
                            'source_type': 'persistent_rag'
                        })
                
                return {
                    'sources': sources,
                    'total_chunks': len(sources),
                    'processing_time_ms': 0  # Not tracked in existing system
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Persistent RAG query failed: {e}")
            return None

class ParallelFusionStrategy(RAGStrategyInterface):
    """Strategy that queries both temp and persistent RAG in parallel and fuses results."""
    
    async def execute(
        self, 
        query: str, 
        conversation_id: str, 
        available_sources: List[RAGSource],
        **kwargs
    ) -> HybridRAGResponse:
        """Execute parallel fusion strategy."""
        start_time = datetime.now()
        sources_queried = []
        final_sources = []
        fusion_metadata = {}
        
        try:
            config = get_in_memory_rag_settings()
            top_k = kwargs.get('top_k', config.max_results_per_query)
            
            # Prepare parallel queries
            query_tasks = []
            
            # Add temp documents query if available
            temp_source = next((s for s in available_sources if s.source_type == "temp_documents"), None)
            if temp_source and temp_source.enabled:
                sources_queried.append("temp_documents")
                temp_service = get_temporary_document_service()
                query_tasks.append(
                    temp_service.query_documents(
                        conversation_id=conversation_id,
                        query=query,
                        use_in_memory_rag=True,
                        fallback_to_temp_docs=True,
                        top_k=top_k
                    )
                )
            else:
                query_tasks.append(asyncio.create_task(self._empty_result()))
            
            # Add persistent RAG query if available
            persistent_source = next((s for s in available_sources if s.source_type == "persistent_rag"), None)
            if persistent_source and persistent_source.enabled:
                sources_queried.append("persistent_rag")
                query_tasks.append(
                    self._query_persistent_rag_async(query, conversation_id, top_k, **kwargs)
                )
            else:
                query_tasks.append(asyncio.create_task(self._empty_result()))
            
            # Execute queries in parallel
            results = await asyncio.gather(*query_tasks, return_exceptions=True)
            
            temp_result = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None
            persistent_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
            
            # Fuse results
            final_sources, fusion_stats = self._fuse_results(
                temp_result=temp_result,
                persistent_result=persistent_result,
                config=config,
                top_k=top_k
            )
            
            fusion_metadata = {
                'strategy': 'parallel_fusion',
                'temp_doc_count': fusion_stats['temp_count'],
                'persistent_count': fusion_stats['persistent_count'],
                'final_count': len(final_sources),
                'fusion_algorithm': 'weighted_score',
                'temp_weight': config.temp_doc_priority_weight,
                'persistent_weight': config.persistent_rag_weight
            }
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Parallel fusion strategy: {len(final_sources)} final results from {len(sources_queried)} sources")
            
            return HybridRAGResponse(
                query=query,
                final_sources=final_sources,
                strategy_used=RAGStrategy.PARALLEL_FUSION.value,
                sources_queried=sources_queried,
                fusion_metadata=fusion_metadata,
                total_processing_time_ms=processing_time,
                conversation_id=conversation_id
            )
            
        except Exception as e:
            logger.error(f"Parallel fusion strategy failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return HybridRAGResponse(
                query=query,
                final_sources=[],
                strategy_used=RAGStrategy.PARALLEL_FUSION.value,
                sources_queried=sources_queried,
                fusion_metadata={'error': str(e)},
                total_processing_time_ms=processing_time,
                conversation_id=conversation_id
            )
    
    async def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for disabled sources."""
        return {'sources': [], 'total_chunks': 0, 'processing_time_ms': 0}
    
    async def _query_persistent_rag_async(
        self, 
        query: str, 
        conversation_id: str, 
        top_k: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Async wrapper for persistent RAG query."""
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._query_persistent_rag_sync,
                query, conversation_id, top_k, kwargs
            )
            return result or {'sources': [], 'total_chunks': 0, 'processing_time_ms': 0}
        except Exception as e:
            logger.error(f"Async persistent RAG query failed: {e}")
            return {'sources': [], 'total_chunks': 0, 'processing_time_ms': 0}
    
    def _query_persistent_rag_sync(
        self, 
        query: str, 
        conversation_id: str, 
        top_k: int,
        kwargs: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Synchronous persistent RAG query."""
        try:
            from app.langchain.service import handle_rag_query
            
            context, documents = handle_rag_query(
                question=query,
                thinking=kwargs.get('thinking', False),
                collections=kwargs.get('collections', None),
                collection_strategy=kwargs.get('collection_strategy', 'auto')
            )
            
            if documents:
                sources = []
                for i, doc in enumerate(documents[:top_k]):
                    if isinstance(doc, dict):
                        sources.append({
                            'content': doc.get('text', ''),
                            'metadata': doc.get('metadata', {}),
                            'score': doc.get('score', 0.5),
                            'rank': i + 1,
                            'source_type': 'persistent_rag'
                        })
                
                return {
                    'sources': sources,
                    'total_chunks': len(sources),
                    'processing_time_ms': 0
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Sync persistent RAG query failed: {e}")
            return None
    
    def _fuse_results(
        self,
        temp_result: Optional[Dict[str, Any]],
        persistent_result: Optional[Dict[str, Any]],
        config,
        top_k: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Fuse results from multiple sources using weighted scoring."""
        try:
            all_sources = []
            temp_count = 0
            persistent_count = 0
            
            # Add temp document sources with priority weight
            if temp_result and temp_result.get('sources'):
                for source in temp_result['sources']:
                    weighted_source = source.copy()
                    original_score = source.get('score', 0.5)
                    weighted_source['score'] = original_score * config.temp_doc_priority_weight
                    weighted_source['original_score'] = original_score
                    weighted_source['weight_applied'] = config.temp_doc_priority_weight
                    weighted_source['source_type'] = 'temp_documents'
                    all_sources.append(weighted_source)
                temp_count = len(temp_result['sources'])
            
            # Add persistent RAG sources with persistent weight
            if persistent_result and persistent_result.get('sources'):
                for source in persistent_result['sources']:
                    weighted_source = source.copy()
                    original_score = source.get('score', 0.5)
                    weighted_source['score'] = original_score * config.persistent_rag_weight
                    weighted_source['original_score'] = original_score
                    weighted_source['weight_applied'] = config.persistent_rag_weight
                    weighted_source['source_type'] = 'persistent_rag'
                    all_sources.append(weighted_source)
                persistent_count = len(persistent_result['sources'])
            
            # Sort by weighted score and take top-k
            all_sources.sort(key=lambda x: x.get('score', 0), reverse=True)
            final_sources = all_sources[:top_k]
            
            # Update ranks
            for i, source in enumerate(final_sources):
                source['rank'] = i + 1
                source['fusion_rank'] = i + 1
            
            fusion_stats = {
                'temp_count': temp_count,
                'persistent_count': persistent_count,
                'total_before_fusion': len(all_sources),
                'final_count': len(final_sources)
            }
            
            return final_sources, fusion_stats
            
        except Exception as e:
            logger.error(f"Result fusion failed: {e}")
            return [], {'error': str(e), 'temp_count': 0, 'persistent_count': 0}

class HybridRAGOrchestrator:
    """Orchestrates hybrid RAG operations with intelligent routing and result fusion."""
    
    def __init__(self):
        self.config = get_in_memory_rag_settings()
        self.strategies = {
            RAGStrategy.TEMP_PRIORITY: TempPriorityStrategy(),
            RAGStrategy.PARALLEL_FUSION: ParallelFusionStrategy(),
        }
        self.default_strategy = RAGStrategy.TEMP_PRIORITY
    
    async def query(
        self,
        query: str,
        conversation_id: str,
        strategy: Optional[RAGStrategy] = None,
        mode: RAGMode = RAGMode.HYBRID,
        **kwargs
    ) -> HybridRAGResponse:
        """
        Execute hybrid RAG query with specified strategy.
        
        Args:
            query: Query string
            conversation_id: Conversation ID
            strategy: RAG strategy to use
            mode: RAG execution mode
            **kwargs: Additional parameters
            
        Returns:
            Hybrid RAG response
        """
        try:
            # Determine strategy
            if strategy is None:
                strategy = self._select_adaptive_strategy(query, conversation_id, **kwargs)
            
            # Get available sources based on mode
            available_sources = await self._get_available_sources(conversation_id, mode)
            
            # Execute strategy
            strategy_impl = self.strategies.get(strategy)
            if not strategy_impl:
                logger.warning(f"Strategy {strategy} not implemented, using default")
                strategy_impl = self.strategies[self.default_strategy]
            
            result = await strategy_impl.execute(
                query=query,
                conversation_id=conversation_id,
                available_sources=available_sources,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid RAG orchestrator failed: {e}")
            return HybridRAGResponse(
                query=query,
                final_sources=[],
                strategy_used="error",
                sources_queried=[],
                fusion_metadata={'error': str(e)},
                total_processing_time_ms=0,
                conversation_id=conversation_id
            )
    
    async def _get_available_sources(
        self, 
        conversation_id: str, 
        mode: RAGMode
    ) -> List[RAGSource]:
        """Get available RAG sources based on mode and conversation state."""
        sources = []
        
        if mode in [RAGMode.TEMP_ONLY, RAGMode.HYBRID]:
            # Check if temp documents are available
            try:
                temp_service = get_temporary_document_service()
                temp_docs = await temp_service.get_conversation_documents(conversation_id)
                active_docs = [doc for doc in temp_docs if doc.get('is_included', False)]
                
                if active_docs:
                    sources.append(RAGSource(
                        source_type="temp_documents",
                        priority=self.config.temp_doc_priority_weight,
                        enabled=True,
                        metadata={'active_document_count': len(active_docs)}
                    ))
            except Exception as e:
                logger.warning(f"Failed to check temp documents: {e}")
        
        if mode in [RAGMode.PERSISTENT_ONLY, RAGMode.HYBRID]:
            # Persistent RAG is always available
            sources.append(RAGSource(
                source_type="persistent_rag",
                priority=self.config.persistent_rag_weight,
                enabled=True,
                metadata={'type': 'milvus_qdrant'}
            ))
        
        return sources
    
    def _select_adaptive_strategy(
        self, 
        query: str, 
        conversation_id: str, 
        **kwargs
    ) -> RAGStrategy:
        """Select strategy adaptively based on query and context."""
        try:
            # For now, use simple heuristics
            # In future, this could use ML models or more sophisticated logic
            
            # If user explicitly requests parallel processing
            if kwargs.get('parallel_processing', False):
                return RAGStrategy.PARALLEL_FUSION
            
            # If temp documents are prioritized in config
            if self.config.temp_doc_priority_weight > self.config.persistent_rag_weight:
                return RAGStrategy.TEMP_PRIORITY
            
            # Default strategy
            return self.default_strategy
            
        except Exception as e:
            logger.warning(f"Adaptive strategy selection failed: {e}")
            return self.default_strategy
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return [strategy.value for strategy in self.strategies.keys()]
    
    def get_config(self) -> Dict[str, Any]:
        """Get orchestrator configuration."""
        return {
            'default_strategy': self.default_strategy.value,
            'available_strategies': self.get_available_strategies(),
            'temp_doc_priority_weight': self.config.temp_doc_priority_weight,
            'persistent_rag_weight': self.config.persistent_rag_weight,
            'fallback_to_persistent': self.config.fallback_to_persistent,
            'min_temp_doc_score': self.config.min_temp_doc_score
        }

# Global orchestrator instance
_orchestrator_instance: Optional[HybridRAGOrchestrator] = None

def get_hybrid_rag_orchestrator() -> HybridRAGOrchestrator:
    """Get the global hybrid RAG orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = HybridRAGOrchestrator()
    return _orchestrator_instance