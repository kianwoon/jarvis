"""
Query Router: Integration layer between query classifier and processing pipelines.

This module integrates the query classifier with existing endpoints to route
queries appropriately based on their classification.
"""

import json
import httpx
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, List
from app.langchain.query_classifier import classify_query, QueryType, ClassificationResult
from app.langchain.service import (
    handle_rag_query, classify_query_type, get_conversation_history,
    store_conversation_message, build_prompt
)
from app.core.llm_settings_cache import get_llm_settings
from app.core.mcp_tools_cache import get_enabled_mcp_tools, execute_mcp_tool
from app.langchain.multi_agent_system_simple import MultiAgentSystem
import logging

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Routes queries to appropriate processing pipelines based on classification.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.llm_cfg = get_llm_settings()
        self.available_tools = get_enabled_mcp_tools()
    
    async def route_query(
        self, 
        query: str, 
        thinking: bool = False,
        stream: bool = True,
        collections: Optional[List[str]] = None,
        collection_strategy: str = "auto"
    ) -> AsyncGenerator[str, None]:
        """
        Route a query to the appropriate processing pipeline.
        
        Args:
            query: The user's query
            thinking: Whether to enable thinking mode
            stream: Whether to stream responses
            collections: Optional list of collections to search
            collection_strategy: Collection search strategy
            
        Yields:
            JSON-formatted response chunks
        """
        # Prepare context for classification
        context = {
            "has_collections": bool(collections),
            "conversation_history": get_conversation_history(self.conversation_id) if self.conversation_id else None,
            "available_tools": list(self.available_tools.keys()) if self.available_tools else []
        }
        
        # Classify the query
        classification = classify_query(query, context, use_llm_fallback=True)
        
        logger.info(f"Query classified as {classification.primary_type.value} "
                   f"with confidence {classification.confidence:.2f}")
        
        # Store classification metadata
        if self.conversation_id:
            store_conversation_message(self.conversation_id, "system", 
                                     f"Query classified as: {classification.primary_type.value}")
        
        # Route based on classification
        if classification.primary_type == QueryType.LARGE_GENERATION:
            async for chunk in self._handle_large_generation(query, classification):
                yield chunk
                
        elif classification.primary_type == QueryType.TOOLS:
            async for chunk in self._handle_tool_query(query, classification):
                yield chunk
                
        elif classification.primary_type == QueryType.RAG:
            async for chunk in self._handle_rag_query(
                query, thinking, collections, collection_strategy, classification
            ):
                yield chunk
                
        elif classification.primary_type == QueryType.HYBRID:
            async for chunk in self._handle_hybrid_query(
                query, thinking, collections, collection_strategy, classification
            ):
                yield chunk
                
        else:  # QueryType.DIRECT_LLM
            async for chunk in self._handle_direct_llm_query(query, thinking, classification):
                yield chunk
    
    async def _handle_large_generation(
        self, 
        query: str, 
        classification: ClassificationResult
    ) -> AsyncGenerator[str, None]:
        """Handle large generation requests"""
        # Extract target count from metadata or analyze query
        from app.langchain.service import detect_large_output_potential
        analysis = detect_large_output_potential(query)
        target_count = analysis.get("estimated_items", 100)
        
        # Use multi-agent system for large generation
        system = MultiAgentSystem(conversation_id=self.conversation_id)
        
        async for event in system.stream_large_generation_events(
            query=query,
            target_count=target_count,
            chunk_size=None  # Auto-calculated
        ):
            yield json.dumps(event, ensure_ascii=False) + "\n"
    
    async def _handle_tool_query(
        self, 
        query: str, 
        classification: ClassificationResult
    ) -> AsyncGenerator[str, None]:
        """Handle queries that need tool execution"""
        required_tools = classification.metadata.get("required_tools", [])
        
        # First, try to execute required tools
        tool_results = {}
        for tool_name in required_tools:
            if tool_name in self.available_tools:
                try:
                    # Execute tool (this would need proper implementation)
                    result = await self._execute_tool(tool_name, query)
                    tool_results[tool_name] = result
                    
                    # Stream tool execution status
                    yield json.dumps({
                        "type": "tool_execution",
                        "tool": tool_name,
                        "status": "completed"
                    }) + "\n"
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    yield json.dumps({
                        "type": "tool_error",
                        "tool": tool_name,
                        "error": str(e)
                    }) + "\n"
        
        # Build context with tool results
        tool_context = self._format_tool_results(tool_results)
        
        # Generate response using LLM with tool results
        prompt = self._build_tool_response_prompt(query, tool_context)
        
        async for chunk in self._stream_llm_response(prompt):
            yield chunk
    
    async def _handle_rag_query(
        self,
        query: str,
        thinking: bool,
        collections: Optional[List[str]],
        collection_strategy: str,
        classification: ClassificationResult
    ) -> AsyncGenerator[str, None]:
        """Handle RAG queries"""
        # Get RAG context
        rag_context, _ = handle_rag_query(query, thinking, collections, collection_strategy)
        
        if rag_context:
            yield json.dumps({
                "type": "rag_context",
                "found": True,
                "context_length": len(rag_context)
            }) + "\n"
            
            # Build prompt with RAG context
            prompt = self._build_rag_prompt(query, rag_context)
        else:
            yield json.dumps({
                "type": "rag_context",
                "found": False
            }) + "\n"
            
            # Fall back to direct LLM if no context found
            prompt = self._build_direct_prompt(query)
        
        # Stream LLM response
        async for chunk in self._stream_llm_response(prompt, thinking):
            yield chunk
    
    async def _handle_hybrid_query(
        self,
        query: str,
        thinking: bool,
        collections: Optional[List[str]],
        collection_strategy: str,
        classification: ClassificationResult
    ) -> AsyncGenerator[str, None]:
        """Handle hybrid queries that need multiple approaches"""
        secondary_types = classification.secondary_types or []
        
        # Execute each component
        combined_context = []
        
        for query_type, confidence in secondary_types:
            if query_type == QueryType.RAG and confidence > 0.3:
                rag_context, _ = handle_rag_query(query, thinking, collections, collection_strategy)
                if rag_context:
                    combined_context.append(f"Document Context:\n{rag_context}")
                    
            elif query_type == QueryType.TOOLS and confidence > 0.3:
                # Execute relevant tools
                tool_results = await self._execute_relevant_tools(query)
                if tool_results:
                    combined_context.append(f"Tool Results:\n{self._format_tool_results(tool_results)}")
        
        # Build comprehensive prompt
        prompt = self._build_hybrid_prompt(query, "\n\n".join(combined_context))
        
        # Stream response
        async for chunk in self._stream_llm_response(prompt, thinking):
            yield chunk
    
    async def _handle_direct_llm_query(
        self,
        query: str,
        thinking: bool,
        classification: ClassificationResult
    ) -> AsyncGenerator[str, None]:
        """Handle direct LLM queries"""
        # Build simple prompt
        prompt = self._build_direct_prompt(query)
        
        # Stream LLM response
        async for chunk in self._stream_llm_response(prompt, thinking):
            yield chunk
    
    async def _execute_tool(self, tool_name: str, query: str) -> Dict[str, Any]:
        """Execute a specific tool"""
        # This would integrate with the MCP tool execution system
        # For now, return a placeholder
        return {
            "tool": tool_name,
            "result": f"Tool {tool_name} execution result for: {query}",
            "status": "completed"
        }
    
    async def _execute_relevant_tools(self, query: str) -> Dict[str, Any]:
        """Execute all relevant tools for a query"""
        results = {}
        # Analyze query to determine which tools to execute
        # This is a simplified implementation
        for tool_name, tool_info in self.available_tools.items():
            # Simple keyword matching (could be improved)
            tool_keywords = tool_info.get('keywords', [])
            if any(keyword in query.lower() for keyword in tool_keywords):
                results[tool_name] = await self._execute_tool(tool_name, query)
        return results
    
    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """Format tool results for LLM context"""
        formatted = []
        for tool_name, result in tool_results.items():
            formatted.append(f"[{tool_name}]: {result.get('result', 'No result')}")
        return "\n".join(formatted)
    
    def _build_tool_response_prompt(self, query: str, tool_context: str) -> str:
        """Build prompt for tool-based responses"""
        conversation_context = ""
        if self.conversation_id:
            conversation_context = get_conversation_history(self.conversation_id)
        
        prompt = f"""Based on the following tool execution results, answer the user's question.

Tool Results:
{tool_context}

{conversation_context}

User Question: {query}

Provide a helpful and accurate response based on the tool results."""
        
        return build_prompt(prompt, is_internal=False)
    
    def _build_rag_prompt(self, query: str, rag_context: str) -> str:
        """Build prompt for RAG-based responses"""
        conversation_context = ""
        if self.conversation_id:
            conversation_context = get_conversation_history(self.conversation_id)
        
        prompt = f"""Answer the following question based on the provided context.

Context from documents:
{rag_context}

{conversation_context}

Question: {query}

Provide a comprehensive answer based on the context. If the context doesn't contain relevant information, indicate that."""
        
        return build_prompt(prompt, is_internal=False)
    
    def _build_hybrid_prompt(self, query: str, combined_context: str) -> str:
        """Build prompt for hybrid responses"""
        conversation_context = ""
        if self.conversation_id:
            conversation_context = get_conversation_history(self.conversation_id)
        
        prompt = f"""Answer the following question using all available information.

Available Information:
{combined_context}

{conversation_context}

Question: {query}

Synthesize the information from all sources to provide a comprehensive answer."""
        
        return build_prompt(prompt, is_internal=False)
    
    def _build_direct_prompt(self, query: str) -> str:
        """Build prompt for direct LLM responses"""
        conversation_context = ""
        if self.conversation_id:
            conversation_context = get_conversation_history(self.conversation_id)
        
        prompt = f"""{conversation_context}

{query}"""
        
        return build_prompt(prompt, is_internal=False)
    
    async def _stream_llm_response(self, prompt: str, thinking: bool = False) -> AsyncGenerator[str, None]:
        """Stream response from LLM"""
        llm_api_url = "http://localhost:8000/api/v1/generate_stream"
        
        mode_key = "thinking_mode" if thinking else "non_thinking_mode"
        model = self.llm_cfg.get(mode_key, self.llm_cfg.get("model"))
        
        payload = {
            "prompt": prompt,
            "model": model,
            "temperature": self.llm_cfg.get("temperature", 0.7),
            "max_tokens": self.llm_cfg.get("max_tokens", 8192),
            "top_p": self.llm_cfg.get("top_p", 0.9)
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", llm_api_url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        token = line.replace("data: ", "")
                        yield json.dumps({"type": "token", "content": token}) + "\n"


# Integration function for existing RAG endpoint
def create_enhanced_rag_answer(
    question: str,
    thinking: bool = False,
    stream: bool = True,
    conversation_id: Optional[str] = None,
    use_classifier: bool = True,
    collections: Optional[List[str]] = None,
    collection_strategy: str = "auto"
) -> AsyncGenerator[str, None]:
    """
    Enhanced RAG answer function with query classification.
    
    This can be used as a drop-in replacement for the existing rag_answer function
    with the additional benefit of intelligent query routing.
    """
    if use_classifier:
        router = QueryRouter(conversation_id=conversation_id)
        return router.route_query(
            query=question,
            thinking=thinking,
            stream=stream,
            collections=collections,
            collection_strategy=collection_strategy
        )
    else:
        # Fall back to original implementation
        from app.langchain.service import rag_answer
        return rag_answer(
            question=question,
            thinking=thinking,
            stream=stream,
            conversation_id=conversation_id,
            collections=collections,
            collection_strategy=collection_strategy
        )