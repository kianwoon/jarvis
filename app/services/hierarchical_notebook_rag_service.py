"""
Hierarchical Notebook RAG Service with Multi-Stage Retrieval and Context Management

This service implements Google NotebookLM-like retrieval with:
1. Document-level filtering using summaries
2. Smart chunk retrieval with reranking  
3. Dynamic context window management
4. Token budget system

Leverages existing reranking infrastructure from standard chat.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.notebook_models import NotebookRAGResponse, NotebookRAGSource
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.embedding_settings_cache import get_embedding_settings
from app.core.notebook_llm_settings_cache import get_notebook_llm_settings
from app.core.rag_settings_cache import get_rag_settings
from app.services.notebook_rag_service import NotebookRAGService
from app.api.v1.endpoints.document import HTTPEmbeddingFunction
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import connections, Collection, utility

logger = logging.getLogger(__name__)

class TokenBudget:
    """Manages token allocation for context window optimization"""
    
    def __init__(self, context_window: int = 131072, response_reserve: int = 8192):
        self.context_window = context_window
        self.response_reserve = response_reserve
        self.system_prompt_tokens = 1000  # Estimated
        self.query_tokens = 0
        self.available_for_retrieval = context_window - response_reserve - self.system_prompt_tokens
        self.used_tokens = 0
        
    def set_query_tokens(self, tokens: int):
        """Set tokens used by the query"""
        self.query_tokens = tokens
        self.available_for_retrieval = self.context_window - self.response_reserve - self.system_prompt_tokens - tokens
        
    def estimate_chunk_tokens(self, content: str) -> int:
        """Estimate tokens in chunk content (rough approximation)"""
        # Simple approximation: 1 token ≈ 4 characters for English text
        return len(content) // 4
        
    def can_add_chunk(self, content: str) -> bool:
        """Check if chunk can be added without exceeding budget"""
        chunk_tokens = self.estimate_chunk_tokens(content)
        return (self.used_tokens + chunk_tokens) <= self.available_for_retrieval
        
    def add_chunk(self, content: str) -> bool:
        """Add chunk to budget if possible"""
        chunk_tokens = self.estimate_chunk_tokens(content)
        if (self.used_tokens + chunk_tokens) <= self.available_for_retrieval:
            self.used_tokens += chunk_tokens
            return True
        return False
        
    def get_remaining_budget(self) -> int:
        """Get remaining token budget for retrieval"""
        return max(0, self.available_for_retrieval - self.used_tokens)
        
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics"""
        return {
            "context_window": self.context_window,
            "response_reserve": self.response_reserve,
            "query_tokens": self.query_tokens,
            "retrieval_used": self.used_tokens,
            "retrieval_available": self.available_for_retrieval,
            "remaining": self.get_remaining_budget(),
            "utilization_percent": int((self.used_tokens / self.available_for_retrieval) * 100) if self.available_for_retrieval > 0 else 0
        }


class HierarchicalNotebookRAGService(NotebookRAGService):
    """
    Enhanced NotebookRAG service with hierarchical retrieval and context optimization.
    Extends the existing NotebookRAGService to leverage all existing infrastructure.
    """
    
    def __init__(self):
        super().__init__()
        self.token_budget = None
        self.rag_settings = None
        
    async def query_with_context_optimization(
        self,
        notebook_id: str,
        query: str,
        top_k: int = 10,
        include_metadata: bool = True,
        collection_filter: Optional[List[str]] = None,
        db: Optional[Session] = None
    ) -> NotebookRAGResponse:
        """
        Enhanced query method with context optimization and hierarchical retrieval
        """
        try:
            # Initialize settings
            notebook_settings = get_notebook_llm_settings()
            context_window = notebook_settings.get('notebook_llm', {}).get('context_window_size', 131072)
            
            # Initialize token budget
            self.token_budget = TokenBudget(context_window=context_window)
            self.token_budget.set_query_tokens(self.token_budget.estimate_chunk_tokens(query))
            
            # Get RAG settings for reranking
            self.rag_settings = get_rag_settings()
            reranking_config = self.rag_settings.get('reranking', {})
            
            # Determine retrieval strategy based on query complexity and available budget
            strategy = self._determine_retrieval_strategy(query, top_k)
            
            if strategy == 'hierarchical':
                return await self._hierarchical_retrieval(
                    notebook_id, query, top_k, include_metadata, collection_filter, db
                )
            else:
                # Fallback to optimized standard retrieval
                return await self._optimized_standard_retrieval(
                    notebook_id, query, top_k, include_metadata, collection_filter, db
                )
                
        except Exception as e:
            logger.error(f"Context-optimized query failed: {str(e)}")
            # Fallback to original method
            return await super().query_notebook(db, notebook_id, query, top_k, include_metadata, collection_filter)
    
    def _determine_retrieval_strategy(self, query: str, top_k: int) -> str:
        """Determine optimal retrieval strategy based on query and context"""
        query_tokens = self.token_budget.estimate_chunk_tokens(query)
        available_budget = self.token_budget.get_remaining_budget()
        
        # If query is complex or we have limited budget, use hierarchical approach
        if query_tokens > 500 or available_budget < 20000 or top_k > 15:
            return 'hierarchical'
        else:
            return 'optimized_standard'
    
    async def _hierarchical_retrieval(
        self,
        notebook_id: str,
        query: str,
        top_k: int,
        include_metadata: bool,
        collection_filter: Optional[List[str]],
        db: Optional[Session]
    ) -> NotebookRAGResponse:
        """
        Multi-stage hierarchical retrieval:
        1. Document-level filtering (if summaries exist)
        2. Chunk retrieval from relevant documents
        3. Reranking with existing infrastructure
        4. Context-budget-aware selection
        """
        
        # Stage 1: Document-level filtering (placeholder for now - would need document summaries)
        # For now, use existing document-aware retrieval but with stricter limits
        relevant_documents = await self._identify_relevant_documents(notebook_id, query, db)
        
        # Stage 2: Smart chunk retrieval with higher initial limit
        initial_limit = min(50, top_k * 5)  # Retrieve more candidates for better reranking
        
        candidates_response = await super().query_notebook(
            db, notebook_id, query, initial_limit, include_metadata, collection_filter
        )
        
        if not candidates_response.sources:
            return candidates_response
            
        # Stage 3: Apply existing reranking infrastructure
        reranked_sources = await self._apply_reranking(query, candidates_response.sources)
        
        # Stage 4: Context-budget-aware selection
        final_sources = self._select_sources_by_budget(reranked_sources, top_k)
        
        # Create optimized response with required fields
        return NotebookRAGResponse(
            notebook_id=notebook_id,
            query=query,
            sources=final_sources,
            total_sources=len(final_sources),
            queried_documents=candidates_response.queried_documents,
            collections_searched=candidates_response.collections_searched,
            metadata={
                "strategy": "hierarchical",
                "token_budget": self.token_budget.get_usage_stats(),
                "reranked_from": len(reranked_sources),
                "initial_candidates": len(candidates_response.sources)
            }
        )
    
    async def _optimized_standard_retrieval(
        self,
        notebook_id: str,
        query: str,
        top_k: int,
        include_metadata: bool,
        collection_filter: Optional[List[str]],
        db: Optional[Session]
    ) -> NotebookRAGResponse:
        """
        Optimized version of standard retrieval with context awareness
        """
        # Use smaller initial limit to fit budget better
        adjusted_top_k = min(top_k, 20)  # Cap at reasonable limit
        
        response = await super().query_notebook(
            db, notebook_id, query, adjusted_top_k, include_metadata, collection_filter
        )
        
        if response.sources:
            # Apply budget-aware filtering
            response.sources = self._select_sources_by_budget(response.sources, top_k)
            
            # Add optimization metadata
            if not response.metadata:
                response.metadata = {}
            response.metadata.update({
                "strategy": "optimized_standard", 
                "token_budget": self.token_budget.get_usage_stats()
            })
        
        # Ensure required fields are set
        response.notebook_id = notebook_id
        response.query = query
        
        return response
    
    async def _identify_relevant_documents(self, notebook_id: str, query: str, db: Optional[Session]) -> List[str]:
        """
        Identify most relevant documents (placeholder for document summary filtering)
        For now, returns all documents - in future would query document summaries
        """
        # TODO: Implement document summary-based filtering
        # This would involve:
        # 1. Query against document summary embeddings
        # 2. Return top 5-10 most relevant document IDs
        # 3. Use these IDs to constrain chunk retrieval
        
        return []  # Empty list means no document filtering
    
    async def _apply_reranking(self, query: str, sources: List[NotebookRAGSource]) -> List[NotebookRAGSource]:
        """
        Apply existing reranking infrastructure to notebook sources
        """
        try:
            # Check if reranking is enabled
            reranking_config = self.rag_settings.get('reranking', {})
            if not reranking_config.get('enable_qwen_reranker', True):
                return sources
            
            # Import the existing reranking function
            from app.langchain.service import apply_reranking_to_sources
            
            # Convert NotebookRAGSource to format expected by reranker
            rerank_sources = []
            for source in sources:
                rerank_source = {
                    'content': source.content,
                    'score': source.score,
                    'metadata': {
                        'source': source.document_name or source.document_id,
                        'page': getattr(source.metadata, 'page', 0) if hasattr(source, 'metadata') and source.metadata else 0
                    }
                }
                rerank_sources.append(rerank_source)
            
            # Apply reranking
            reranked = apply_reranking_to_sources(query, rerank_sources)
            
            # Convert back to NotebookRAGSource
            reranked_notebook_sources = []
            for rerank_source in reranked:
                # Find the original source to preserve all its attributes
                original_source = None
                for orig in sources:
                    if orig.content == rerank_source['content']:
                        original_source = orig
                        break
                
                if original_source:
                    # Create new source with updated score but preserve all other attributes
                    notebook_source = NotebookRAGSource(
                        content=original_source.content,
                        metadata=original_source.metadata,
                        score=rerank_source['score'],
                        document_id=original_source.document_id,
                        document_name=original_source.document_name,
                        collection=original_source.collection,
                        source_type=original_source.source_type
                    )
                    reranked_notebook_sources.append(notebook_source)
                else:
                    # Fallback if original source not found
                    notebook_source = NotebookRAGSource(
                        content=rerank_source['content'],
                        metadata={},
                        score=rerank_source['score'],
                        document_id='unknown',
                        document_name=rerank_source.get('metadata', {}).get('source', ''),
                        collection=None,
                        source_type='document'
                    )
                    reranked_notebook_sources.append(notebook_source)
            
            logger.info(f"Reranked {len(sources)} sources to {len(reranked_notebook_sources)} using existing infrastructure")
            return reranked_notebook_sources
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original sources: {str(e)}")
            return sources
    
    def _select_sources_by_budget(self, sources: List[NotebookRAGSource], max_sources: int) -> List[NotebookRAGSource]:
        """
        Select sources that fit within token budget, prioritizing relevance
        """
        selected_sources = []
        
        for source in sources:
            # Check if we can add this source within budget and limits
            if len(selected_sources) >= max_sources:
                break
                
            if self.token_budget.can_add_chunk(source.content):
                if self.token_budget.add_chunk(source.content):
                    selected_sources.append(source)
            else:
                # Budget exceeded - stop adding sources
                logger.info(f"Token budget reached. Selected {len(selected_sources)} of {len(sources)} sources")
                break
        
        logger.info(f"Selected {len(selected_sources)} sources within budget. {self.token_budget.get_usage_stats()}")
        return selected_sources
    
    async def get_context_stats(self, notebook_id: str, query: str) -> Dict[str, Any]:
        """
        Get context usage statistics for a hypothetical query
        """
        # Initialize budget
        notebook_settings = get_notebook_llm_settings()
        context_window = notebook_settings.get('notebook_llm', {}).get('context_window_size', 131072)
        
        token_budget = TokenBudget(context_window=context_window)
        token_budget.set_query_tokens(token_budget.estimate_chunk_tokens(query))
        
        # Get current retrieval settings
        max_retrieval_chunks = notebook_settings.get('notebook_llm', {}).get('max_retrieval_chunks', 200)
        chunk_size = 1000  # Default chunk size
        
        # Calculate theoretical usage
        theoretical_max_tokens = max_retrieval_chunks * (chunk_size // 4)  # Rough token estimate
        
        return {
            "context_window": context_window,
            "query_tokens": token_budget.query_tokens,
            "available_for_retrieval": token_budget.available_for_retrieval,
            "current_max_chunks": max_retrieval_chunks,
            "theoretical_max_tokens": theoretical_max_tokens,
            "would_exceed_budget": theoretical_max_tokens > token_budget.available_for_retrieval,
            "recommended_max_chunks": max(1, token_budget.available_for_retrieval // (chunk_size // 4)),
            "optimization_needed": theoretical_max_tokens > token_budget.available_for_retrieval
        }


# Factory function to get the service
def get_hierarchical_notebook_rag_service() -> HierarchicalNotebookRAGService:
    """Get singleton instance of hierarchical notebook RAG service"""
    if not hasattr(get_hierarchical_notebook_rag_service, '_instance'):
        get_hierarchical_notebook_rag_service._instance = HierarchicalNotebookRAGService()
    return get_hierarchical_notebook_rag_service._instance