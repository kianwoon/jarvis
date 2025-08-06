"""
RAG MCP Service - Standalone document retrieval service for agents
================================================================

Provides intelligent document search capabilities as an MCP tool.
Auto-detects collections, performs hybrid search, returns structured results.
"""

import time
import logging
import hashlib
import json as json_module
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.core.llm_settings_cache import get_llm_settings
from app.core.rag_settings_cache import get_collection_selection_settings, get_document_retrieval_settings
from app.core.collection_registry_cache import get_all_collections
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class RAGSearchRequest(BaseModel):
    """Request model for RAG search"""
    query: str
    collections: Optional[List[str]] = None
    max_documents: Optional[int] = None  # Will use settings default if not provided
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
        self.cache_prefix = "rag_mcp:"
        self._redis_client = None
    
    def _get_cache_client(self):
        """Get Redis client for caching"""
        if self._redis_client is None:
            self._redis_client = get_redis_client()
        return self._redis_client
    
    def _generate_cache_key(self, query: str, collections: List[str], max_docs: int) -> str:
        """Generate cache key from query parameters"""
        # Handle case where query might be passed as dict
        if isinstance(query, dict):
            query = query.get('query', '')
        
        # Ensure query is string
        if not isinstance(query, str):
            query = str(query) if query else ''
            
        # Create a deterministic hash of the query parameters
        cache_data = {
            "query": query.lower().strip(),
            "collections": sorted(collections) if collections else [],
            "max_docs": max_docs
        }
        cache_str = json_module.dumps(cache_data, sort_keys=True)
        query_hash = hashlib.md5(cache_str.encode()).hexdigest()
        return f"{self.cache_prefix}{query_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        client = self._get_cache_client()
        if not client:
            return None
        
        try:
            cached_data = client.get(cache_key)
            if cached_data:
                logger.info(f"RAG MCP: Cache hit for key {cache_key}")
                return json_module.loads(cached_data)
        except Exception as e:
            logger.warning(f"RAG MCP: Cache retrieval error: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any], ttl_hours: int):
        """Cache the search result"""
        client = self._get_cache_client()
        if not client:
            return
        
        try:
            ttl_seconds = ttl_hours * 3600
            client.setex(cache_key, ttl_seconds, json_module.dumps(result))
            logger.info(f"RAG MCP: Cached result with key {cache_key}, TTL {ttl_hours}h")
        except Exception as e:
            logger.warning(f"RAG MCP: Cache storage error: {e}")
    
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
            
            # Get performance settings for caching
            from app.core.rag_settings_cache import get_performance_settings
            perf_settings = get_performance_settings()
            cache_enabled = perf_settings.get('enable_caching', True)
            cache_ttl_hours = perf_settings.get('cache_ttl_hours', 2)
            
            # Check cache if enabled
            cache_hit = False
            if cache_enabled:
                # Get max documents for cache key
                doc_settings = get_document_retrieval_settings()
                max_docs = request.max_documents or doc_settings.get('max_documents_mcp', 8)
                
                # Generate cache key before determining collections
                temp_collections = request.collections or []
                cache_key = self._generate_cache_key(request.query, temp_collections, max_docs)
                
                # Check cache
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    cache_hit = True
                    cached_result['search_metadata']['cache_hit'] = True
                    cached_result['search_metadata']['execution_time_ms'] = int((time.time() - start_time) * 1000)
                    return self._format_jsonrpc_response(cached_result, None)
            
            # Get collections to search
            collections_to_search = await self._determine_collections(
                request.query, 
                request.collections
            )
            
            # Execute RAG search using existing infrastructure (local import to avoid circular dependency)
            from app.langchain.service import handle_rag_query
            context, sources = handle_rag_query(
                question=request.query,
                thinking=False,  # Agents don't need thinking mode
                collections=collections_to_search,
                collection_strategy="specific" if request.collections else "auto"
            )
            
            # Get max documents from settings if not provided
            doc_settings = get_document_retrieval_settings()
            max_docs = request.max_documents or doc_settings.get('max_documents_mcp', 8)
            
            # Process and format results
            documents = self._format_documents(
                sources, 
                max_docs,
                request.include_content
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Build search metadata
            from app.core.rag_settings_cache import get_search_strategy_settings, get_performance_settings
            search_settings = get_search_strategy_settings()
            perf_settings = get_performance_settings()
            
            search_metadata = {
                "vector_search_results": len(sources) if sources else 0,
                "keyword_search_results": len(sources) if sources else 0,  # Both contribute in hybrid search
                "hybrid_fusion_applied": search_settings.get('search_strategy', 'auto') in ['auto', 'hybrid'],
                "collections_auto_detected": not bool(request.collections),
                "cache_hit": cache_hit,
                "similarity_threshold_used": self._get_similarity_threshold(),
                "search_strategy": search_settings.get('search_strategy', 'auto'),
                "semantic_weight": search_settings.get('semantic_weight', 0.7),
                "keyword_weight": search_settings.get('keyword_weight', 0.3),
                "caching_enabled": perf_settings.get('enable_caching', True)
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
            
            # Cache the result if caching is enabled
            if cache_enabled and not cache_hit:
                # Regenerate cache key with actual collections searched
                final_cache_key = self._generate_cache_key(request.query, collections_to_search, max_docs)
                self._cache_result(final_cache_key, response.dict(), cache_ttl_hours)
            
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
        """Determine which collections to search with smart matching"""
        
        if specified_collections:
            # Apply smart collection matching to handle agent inference errors
            from app.core.rag_fallback import smart_collection_matching
            matched_collections = smart_collection_matching(specified_collections)
            
            if matched_collections:
                logger.info(f"RAG MCP: Smart matched collections: {specified_collections} -> {matched_collections}")
                return matched_collections
            else:
                logger.warning(f"RAG MCP: No collections matched for {specified_collections}, falling back")
                return ["default_knowledge"]
        
        # Auto-detect collections based on query content
        return await self._auto_detect_collections(query)
    
    async def _auto_detect_collections(self, query: str) -> List[str]:
        """Auto-detect best collections for the query using LLM"""
        # Get collection selection settings
        selection_settings = get_collection_selection_settings()
        
        # Check if LLM selection is enabled
        if not selection_settings.get('enable_llm_selection', True):
            # Fallback to simple selection based on description overlap
            return self._simple_collection_selection(query)
        
        # Add timeout check to prevent hanging using centralized timeout config
        import asyncio
        from app.core.timeout_settings_cache import get_timeout_value
        
        try:
            # Use centralized timeout configuration
            llm_timeout = get_timeout_value("llm_ai", "llm_inference_timeout", 60)
            result = await asyncio.wait_for(
                self._try_llm_selection(query, selection_settings),
                timeout=llm_timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"RAG MCP: LLM selection timed out after {llm_timeout}s, using simple fallback")
            return self._simple_collection_selection(query)
    
    async def _try_llm_selection(self, query: str, selection_settings: Dict) -> List[str]:
        """Try LLM-based collection selection with proper error handling"""
        # Get available collections
        collections = get_all_collections()
        if not collections:
            return selection_settings.get('fallback_collections', ["default_knowledge"])
        
        try:
            # Check cache first if enabled
            if selection_settings.get('cache_selections', True):
                # Handle case where query might be passed as dict
                query_str = query.get('query', '') if isinstance(query, dict) else str(query)
                cache_key = f"{self.cache_prefix}collection_sel:{hashlib.md5(query_str.lower().encode()).hexdigest()}"
                client = self._get_cache_client()
                if client:
                    cached = client.get(cache_key)
                    if cached:
                        logger.info(f"RAG MCP: Using cached collection selection for query")
                        return json_module.loads(cached)
            
            # Format collections for prompt
            collections_info = []
            for coll in collections:
                name = coll.get('collection_name', '')
                desc = coll.get('description', 'No description')
                doc_count = coll.get('document_count', 0)
                collections_info.append(f"- {name}: {desc} ({doc_count} documents)")
            
            collections_text = "\n".join(collections_info)
            
            # Build prompt from template
            prompt_template = selection_settings.get('selection_prompt_template', 
                "Given the following query and available collections, determine which collections are most relevant to search:\n\nQuery: {query}\n\nAvailable Collections:\n{collections}\n\nReturn only the collection names that are relevant, separated by commas.")
            
            prompt = prompt_template.format(query=query, collections=collections_text)
            logger.debug(f"RAG MCP: LLM selection prompt length: {len(prompt)} chars")
            
            # Make LLM call - use sync version in thread context
            import asyncio
            try:
                # Check if we're in a thread with a new event loop (not the main loop)
                import threading
                if threading.current_thread() != threading.main_thread():
                    logger.info("RAG MCP: Using sync LLM call (in thread context)")
                    response = self._call_llm_for_selection_sync(prompt)
                else:
                    response = await self._call_llm_for_selection(prompt)
            except Exception as e:
                logger.warning(f"RAG MCP: Falling back to sync LLM call due to: {e}")
                response = self._call_llm_for_selection_sync(prompt)
            
            # Parse response
            selected = self._parse_llm_collection_response(response, collections)
            
            # Limit to max collections
            max_collections = selection_settings.get('max_collections', 3)
            selected = selected[:max_collections]
            
            # Cache the result if enabled
            if selection_settings.get('cache_selections', True) and client and selected:
                client.setex(cache_key, 3600, json_module.dumps(selected))  # Cache for 1 hour
            
            return selected if selected else selection_settings.get('fallback_collections', ["default_knowledge"])
            
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
            logger.error(f"RAG MCP: LLM collection selection failed: {error_msg}")
            return selection_settings.get('fallback_collections', ["default_knowledge"])
    
    async def _call_llm_for_selection(self, prompt: str) -> str:
        """Make LLM call for collection selection"""
        import httpx
        import json
        import os
        
        # Get timeout from performance settings
        from app.core.rag_settings_cache import get_performance_settings
        perf_settings = get_performance_settings()
        timeout_seconds = perf_settings.get('connection_timeout_s', 30)
        
        # Use Ollama directly like the rest of the system - use main LLM config, not base settings
        from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
        llm_settings = get_llm_settings()
        main_llm_config = get_main_llm_full_config(llm_settings)
        model_name = main_llm_config.get("model", "qwen3:30b-a3b")
        
        # Determine Ollama URL based on environment
        # Check if we're running inside Docker
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        # Use environment variable if set, otherwise use appropriate default
        if os.environ.get("OLLAMA_BASE_URL"):
            ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
        else:
            # Use appropriate default based on environment
            ollama_base_url = "http://ollama:11434" if in_docker else "http://localhost:11434"
        
        llm_api_url = f"{ollama_base_url}/api/generate"
        
        # Use low temperature for more deterministic selection
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # Get complete response at once
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 200,  # Collection names should be short
                "num_ctx": llm_settings.get("context_length", 32768)
            }
        }
        
        response_text = ""
        try:
            logger.info(f"RAG MCP: Making async LLM selection call to {llm_api_url} (base: {ollama_base_url}, docker: {in_docker}) with timeout {timeout_seconds}s")
            logger.debug(f"RAG MCP: Payload: {json.dumps(payload, indent=2)}")
            
            # Use async httpx client
            logger.info("RAG MCP: Creating async httpx client...")
            # Keep using localhost - it works in both Docker and local environments
            # The backend API is accessible on localhost:8000 inside the container
            
            async with httpx.AsyncClient(timeout=float(timeout_seconds)) as client:
                logger.info("RAG MCP: Async client created, making POST request...")
                response = await client.post(llm_api_url, json=payload)
                response.raise_for_status()
                logger.debug(f"RAG MCP: LLM selection response status: {response.status_code}")
                
                # Parse Ollama JSON response
                response_data = response.json()
                response_text = response_data.get("response", "")
                
                # Extract actual response if wrapped in thinking tags
                import re
                if '<think>' in response_text:
                    # Try to extract content after </think>
                    parts = re.split(r'</think>', response_text, 1)
                    if len(parts) > 1:
                        response_text = parts[1].strip()
                
                logger.debug(f"RAG MCP: LLM selection response length: {len(response_text)}")
        except httpx.ConnectError as e:
            logger.error(f"RAG MCP: Failed to connect to LLM API at {llm_api_url}: {str(e)}")
            raise Exception(f"Failed to connect to LLM API: {str(e)}")
        except httpx.TimeoutException as e:
            logger.warning(f"RAG MCP: LLM selection timeout after {timeout_seconds}s: {str(e)}")
            raise Exception(f"LLM call timed out after {timeout_seconds} seconds")
        except httpx.HTTPStatusError as e:
            logger.error(f"RAG MCP: LLM selection HTTP error {e.response.status_code}: {str(e)}")
            raise Exception(f"LLM HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"RAG MCP: LLM selection unexpected error: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"RAG MCP: Traceback: {traceback.format_exc()}")
            raise Exception(f"LLM call failed: {type(e).__name__}")
        
        if not response_text:
            logger.warning("RAG MCP: LLM selection returned empty response")
            raise Exception("LLM returned empty response")
        
        return response_text.strip()
    
    def _call_llm_for_selection_sync(self, prompt: str) -> str:
        """Make synchronous LLM call for collection selection"""
        import httpx
        import json
        import os
        
        # Get timeout from performance settings
        from app.core.rag_settings_cache import get_performance_settings
        perf_settings = get_performance_settings()
        timeout_seconds = perf_settings.get('connection_timeout_s', 30)
        
        # Use Ollama directly like the rest of the system - use main LLM config, not base settings
        from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
        llm_settings = get_llm_settings()
        main_llm_config = get_main_llm_full_config(llm_settings)
        model_name = main_llm_config.get("model", "qwen3:30b-a3b")
        
        # Determine Ollama URL based on environment
        # Check if we're running inside Docker
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        # Use environment variable if set, otherwise use appropriate default
        if os.environ.get("OLLAMA_BASE_URL"):
            ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
        else:
            # Use appropriate default based on environment
            ollama_base_url = "http://ollama:11434" if in_docker else "http://localhost:11434"
        
        llm_api_url = f"{ollama_base_url}/api/generate"
        
        # Use low temperature for more deterministic selection
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,  # Get complete response at once
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 200,  # Collection names should be short
                "num_ctx": llm_settings.get("context_length", 32768)
            }
        }
        
        response_text = ""
        try:
            logger.info(f"RAG MCP: Making sync LLM selection call to {llm_api_url} (base: {ollama_base_url}, docker: {in_docker}) with timeout {timeout_seconds}s")
            logger.debug(f"RAG MCP: Payload: {json.dumps(payload, indent=2)}")
            # Use synchronous call like standard chat
            with httpx.Client(timeout=float(timeout_seconds)) as client:
                response = client.post(llm_api_url, json=payload)
                response.raise_for_status()
                logger.debug(f"RAG MCP: LLM selection response status: {response.status_code}")
                
                # Parse Ollama JSON response
                response_data = response.json()
                response_text = response_data.get("response", "")
                
                # Extract actual response if wrapped in thinking tags
                import re
                if '<think>' in response_text:
                    # Try to extract content after </think>
                    parts = re.split(r'</think>', response_text, 1)
                    if len(parts) > 1:
                        response_text = parts[1].strip()
                
                logger.debug(f"RAG MCP: LLM selection response length: {len(response_text)}")
        except httpx.TimeoutException as e:
            logger.warning(f"RAG MCP: LLM selection timeout after {timeout_seconds}s: {str(e)}")
            raise Exception(f"LLM call timed out after {timeout_seconds} seconds")
        except httpx.HTTPStatusError as e:
            logger.error(f"RAG MCP: LLM selection HTTP error {e.response.status_code}: {str(e)}")
            raise Exception(f"LLM HTTP error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"RAG MCP: LLM selection unexpected error: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"RAG MCP: Traceback: {traceback.format_exc()}")
            raise Exception(f"LLM call failed: {type(e).__name__}")
        
        if not response_text:
            logger.warning("RAG MCP: LLM selection returned empty response")
            raise Exception("LLM returned empty response")
        
        return response_text.strip()
    
    def _parse_llm_collection_response(self, response: str, available_collections: List[Dict]) -> List[str]:
        """Parse LLM response to extract collection names"""
        # Get all available collection names
        available_names = [c.get('collection_name', '') for c in available_collections]
        
        # Clean and split response
        response = response.lower().strip()
        
        # Handle different response formats
        selected = []
        
        # Try comma-separated format first
        parts = [p.strip() for p in response.split(',')]
        
        # Also try newline-separated
        if len(parts) == 1 and '\n' in response:
            parts = [p.strip() for p in response.split('\n')]
        
        # Match against available collections (case-insensitive)
        for part in parts:
            # Clean up common formatting
            part = part.strip('- ').strip('* ').strip()
            
            # Find matching collection
            for available_name in available_names:
                if available_name.lower() == part or part in available_name.lower():
                    if available_name not in selected:
                        selected.append(available_name)
                    break
        
        return selected
    
    def _simple_collection_selection(self, query: str) -> List[str]:
        """Simple fallback collection selection based on description overlap"""
        collections = get_all_collections()
        if not collections:
            return ["default_knowledge"]
        
        # Handle case where query might be passed as dict
        query_str = query.get('query', '') if isinstance(query, dict) else str(query)
        query_lower = query_str.lower()
        selected = []
        
        for collection in collections:
            collection_name = collection.get('collection_name', '')
            collection_desc = collection.get('description', '').lower()
            
            # Check description overlap
            if collection_desc and any(word in query_lower for word in collection_desc.split() if len(word) > 3):
                selected.append(collection_name)
        
        # Get fallback collections from settings
        selection_settings = get_collection_selection_settings()
        return selected if selected else selection_settings.get('fallback_collections', ["default_knowledge"])
    
    def _format_documents(self, sources: List[Dict], max_docs: int, include_content: bool) -> List[Dict[str, Any]]:
        """Format source documents for agent consumption"""
        
        if not sources:
            return []
        
        documents = []
        total_docs = min(len(sources), max_docs)
        
        for i, source in enumerate(sources[:max_docs]):
            # Use actual score from source if available, otherwise fall back to position-based score
            # Check for score in multiple possible locations
            actual_score = source.get('score', source.get('relevance_score', None))
            
            # Debug logging to trace score availability
            if i < 3:  # Log first 3 documents
                logger.info(f"[RAG MCP DEBUG] Doc {i}: file={source.get('file', 'Unknown')[:50]}, score={actual_score}")
            
            if actual_score is not None:
                position_score = float(actual_score)
            else:
                # Fall back to position-based score if no actual score available
                # Use exponential decay: 1.0 for first doc, decreasing for subsequent docs
                position_score = 1.0 * (0.9 ** i)
            
            # Extract metadata from source
            file_path = source.get("file", "Unknown")
            collection_name = source.get("collection", "default_knowledge")
            page_num = source.get("page")
            
            # Generate chunk ID from file path and page
            chunk_id = f"{collection_name}:{file_path.split('/')[-1]}:p{page_num if page_num else 0}"
            
            # Get current timestamp for document retrieval time
            from datetime import datetime, timezone
            retrieval_time = datetime.now(timezone.utc).isoformat()
            
            doc = {
                "id": f"doc_{i+1}",
                "source": file_path,
                "relevance_score": round(position_score, 3),
                "collection": collection_name,
                "metadata": {
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "position_in_results": i + 1,
                    "total_results": total_docs,
                    "retrieved_at": retrieval_time,
                    "content_length": len(source.get("content", "")) if "content" in source else None
                }
            }
            
            if include_content:
                doc["content"] = source.get("content", "")
            
            documents.append(doc)
        
        return documents
    
    def _get_similarity_threshold(self) -> float:
        """Get similarity threshold from settings"""
        doc_settings = get_document_retrieval_settings()
        return doc_settings.get('similarity_threshold', 1.5)
    
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

async def execute_rag_search(query: str, collections: List[str] = None, max_documents: int = None, include_content: bool = True) -> Dict[str, Any]:
    """Main entry point for MCP tool execution"""
    
    # Handle case where query is passed as dict (from recent tool calling changes)
    if isinstance(query, dict):
        query = query.get('query', '')
    
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query) if query else ''
    
    request = RAGSearchRequest(
        query=query,
        collections=collections,
        max_documents=max_documents,
        include_content=include_content
    )
    
    return await rag_service.search_documents(request)

def execute_rag_search_sync(query: str, collections: List[str] = None, max_documents: int = None, include_content: bool = True) -> Dict[str, Any]:
    """Synchronous entry point for MCP tool execution - truly synchronous implementation"""
    import time
    
    # Handle case where query is passed as dict (from recent tool calling changes)
    if isinstance(query, dict):
        query = query.get('query', '')
    
    # Ensure query is a string
    if not isinstance(query, str):
        query = str(query) if query else ''
    
    # Create a new service instance that will use sync methods
    sync_service = RAGMCPService()
    sync_service._use_sync_mode = True  # Flag to force sync operations
    
    request = RAGSearchRequest(
        query=query,
        collections=collections,
        max_documents=max_documents,
        include_content=include_content
    )
    
    # Call a synchronous version of search
    start_time = time.time()
    
    try:
        logger.info(f"RAG MCP Sync: Processing query: {request.query[:100]}...")
        
        # Manually execute the search logic synchronously
        # This bypasses all async operations
        
        # Get performance settings for caching
        from app.core.rag_settings_cache import get_performance_settings
        perf_settings = get_performance_settings()
        cache_enabled = perf_settings.get('enable_caching', True)
        cache_ttl_hours = perf_settings.get('cache_ttl_hours', 2)
        
        # Check cache if enabled
        cache_hit = False
        if cache_enabled:
            # Get max documents for cache key
            from app.core.rag_settings_cache import get_document_retrieval_settings
            doc_settings = get_document_retrieval_settings()
            max_docs = request.max_documents or doc_settings.get('max_documents_mcp', 8)
            
            # Generate cache key
            temp_collections = request.collections or []
            cache_key = sync_service._generate_cache_key(request.query, temp_collections, max_docs)
            
            # Check cache
            cached_result = sync_service._get_cached_result(cache_key)
            if cached_result:
                cache_hit = True
                cached_result['search_metadata']['cache_hit'] = True
                cached_result['search_metadata']['execution_time_ms'] = int((time.time() - start_time) * 1000)
                return sync_service._format_jsonrpc_response(cached_result, None)
        
        # Get collections to search - use sync version
        collections_to_search = _determine_collections_sync(
            request.query, 
            request.collections,
            sync_service
        )
        
        # Execute RAG search using existing infrastructure
        from app.langchain.service import handle_rag_query
        context, sources = handle_rag_query(
            question=request.query,
            thinking=False,  # Agents don't need thinking mode
            collections=collections_to_search,
            collection_strategy="specific" if request.collections else "auto"
        )
        
        # Get max documents from settings if not provided
        from app.core.rag_settings_cache import get_document_retrieval_settings
        doc_settings = get_document_retrieval_settings()
        max_docs = request.max_documents or doc_settings.get('max_documents_mcp', 8)
        
        # Process and format results
        documents = sync_service._format_documents(
            sources, 
            max_docs,
            request.include_content
        )
        
        execution_time = int((time.time() - start_time) * 1000)
        
        # Build search metadata
        from app.core.rag_settings_cache import get_search_strategy_settings, get_performance_settings
        search_settings = get_search_strategy_settings()
        perf_settings = get_performance_settings()
        
        search_metadata = {
            "vector_search_results": len(sources) if sources else 0,
            "keyword_search_results": len(sources) if sources else 0,
            "hybrid_fusion_applied": search_settings.get('search_strategy', 'auto') in ['auto', 'hybrid'],
            "collections_auto_detected": not bool(request.collections),
            "cache_hit": cache_hit,
            "similarity_threshold_used": sync_service._get_similarity_threshold(),
            "search_strategy": search_settings.get('search_strategy', 'auto'),
            "semantic_weight": search_settings.get('semantic_weight', 0.7),
            "keyword_weight": search_settings.get('keyword_weight', 0.3),
            "caching_enabled": perf_settings.get('enable_caching', True),
            "execution_time_ms": execution_time
        }
        
        response = {
            "success": True,
            "query": request.query,
            "collections_searched": collections_to_search,
            "total_documents_found": len(sources) if sources else 0,
            "documents_returned": len(documents),
            "execution_time_ms": execution_time,
            "documents": documents,
            "search_metadata": search_metadata
        }
        
        logger.info(f"RAG MCP Sync: Found {len(documents)} documents in {execution_time}ms")
        
        # Cache the result if caching is enabled
        if cache_enabled and not cache_hit:
            # Regenerate cache key with actual collections searched
            final_cache_key = sync_service._generate_cache_key(request.query, collections_to_search, max_docs)
            sync_service._cache_result(final_cache_key, response, cache_ttl_hours)
        
        return sync_service._format_jsonrpc_response(response, None)
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        logger.error(f"RAG MCP Sync: Search failed: {str(e)}")
        
        error_response = {
            "success": False,
            "query": request.query,
            "collections_searched": [],
            "total_documents_found": 0,
            "documents_returned": 0,
            "execution_time_ms": execution_time,
            "documents": [],
            "search_metadata": {},
            "error": str(e)
        }
        
        return sync_service._format_jsonrpc_response(None, str(e))


def _determine_collections_sync(query: str, specified_collections: Optional[List[str]], service: RAGMCPService) -> List[str]:
    """Synchronous version of collection determination with smart matching"""
    
    if specified_collections:
        # Apply smart collection matching to handle agent inference errors
        from app.core.rag_fallback import smart_collection_matching
        matched_collections = smart_collection_matching(specified_collections)
        
        if matched_collections:
            logger.info(f"RAG MCP Sync: Smart matched collections: {specified_collections} -> {matched_collections}")
            return matched_collections
        else:
            logger.warning(f"RAG MCP Sync: No collections matched for {specified_collections}, falling back")
            return ["default_knowledge"]
    
    # Auto-detect collections based on query content
    # Get collection selection settings
    from app.core.rag_settings_cache import get_collection_selection_settings
    selection_settings = get_collection_selection_settings()
    
    # Check if LLM selection is enabled
    if not selection_settings.get('enable_llm_selection', True):
        # Fallback to simple selection based on description overlap
        return service._simple_collection_selection(query)
    
    # Try LLM selection using sync method
    try:
        # Get available collections
        from app.core.collection_registry_cache import get_all_collections
        collections = get_all_collections()
        if not collections:
            return selection_settings.get('fallback_collections', ["default_knowledge"])
        
        # Check cache first if enabled
        if selection_settings.get('cache_selections', True):
            # Handle case where query might be passed as dict
            query_str = query.get('query', '') if isinstance(query, dict) else str(query)
            cache_key = f"{service.cache_prefix}collection_sel:{hashlib.md5(query_str.lower().encode()).hexdigest()}"
            client = service._get_cache_client()
            if client:
                cached = client.get(cache_key)
                if cached:
                    logger.info(f"RAG MCP Sync: Using cached collection selection for query")
                    return json_module.loads(cached)
        
        # Format collections for prompt
        collections_info = []
        for coll in collections:
            name = coll.get('collection_name', '')
            desc = coll.get('description', 'No description')
            doc_count = coll.get('document_count', 0)
            collections_info.append(f"- {name}: {desc} ({doc_count} documents)")
        
        collections_text = "\n".join(collections_info)
        
        # Build prompt from template
        prompt_template = selection_settings.get('selection_prompt_template', 
            "Given the following query and available collections, determine which collections are most relevant to search:\n\nQuery: {query}\n\nAvailable Collections:\n{collections}\n\nReturn only the collection names that are relevant, separated by commas.")
        
        prompt = prompt_template.format(query=query, collections=collections_text)
        logger.debug(f"RAG MCP Sync: LLM selection prompt length: {len(prompt)} chars")
        
        # Make LLM call using sync method
        logger.info("RAG MCP Sync: Using sync LLM call for collection selection")
        response = service._call_llm_for_selection_sync(prompt)
        
        # Parse response
        selected = service._parse_llm_collection_response(response, collections)
        
        # Limit to max collections
        max_collections = selection_settings.get('max_collections', 3)
        selected = selected[:max_collections]
        
        # Cache the result if enabled
        if selection_settings.get('cache_selections', True) and client and selected:
            client.setex(cache_key, 3600, json_module.dumps(selected))  # Cache for 1 hour
        
        return selected if selected else selection_settings.get('fallback_collections', ["default_knowledge"])
        
    except Exception as e:
        error_msg = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
        logger.error(f"RAG MCP Sync: LLM collection selection failed: {error_msg}")
        return selection_settings.get('fallback_collections', ["default_knowledge"])