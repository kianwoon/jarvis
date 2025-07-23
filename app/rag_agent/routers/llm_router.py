"""
LLM-based routing for intelligent collection selection

This module uses LLM function calling to intelligently route queries
to the most appropriate collections with optimized search parameters.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import httpx

from app.rag_agent.utils.types import (
    RoutingDecision, ExecutionStrategy, SearchContext
)
from app.rag_agent.utils.prompt_templates import PromptTemplates
from app.rag_agent.routers.tool_registry import get_collection_tool_registry
from app.core.llm_settings_cache import get_llm_settings

logger = logging.getLogger(__name__)


class LLMRouter:
    """LLM-based routing for collection selection and query optimization"""
    
    def __init__(self):
        self.prompt_templates = PromptTemplates()
        self.tool_registry = get_collection_tool_registry()
        self._llm_settings = None
        self._settings_refresh_time = None
        
    async def route_query(
        self,
        query: str,
        context: Optional[SearchContext] = None,
        max_collections: int = 5
    ) -> RoutingDecision:
        """
        Use LLM to intelligently route query to appropriate collections
        
        Args:
            query: User query to route
            context: Search context with user information
            max_collections: Maximum number of collections to select
            
        Returns:
            RoutingDecision with selected collections and strategies
        """
        try:
            # Get available tools based on user context
            available_tools = self.tool_registry.get_available_tools(context)
            
            if not available_tools:
                logger.warning("No available collections for routing")
                return self._create_fallback_decision(query)
            
            # Limit tools if too many (to avoid token limits)
            if len(available_tools) > 15:
                available_tools = self._prioritize_tools(available_tools, query, context)[:15]
            
            # Build routing prompt
            routing_prompt = self.prompt_templates.build_routing_prompt(
                query=query,
                available_tools=available_tools,
                context=context
            )
            
            # Call LLM with function calling
            llm_response = await self._call_llm_with_tools(
                prompt=routing_prompt,
                tools=available_tools,
                context=context
            )
            
            # Parse LLM response into routing decision
            routing_decision = self._parse_llm_response(llm_response, query)
            
            # Validate and optimize the decision
            routing_decision = self._validate_and_optimize_decision(
                routing_decision, context, max_collections
            )
            
            logger.info(f"Routed query to {len(routing_decision.selected_collections)} collections: "
                       f"{routing_decision.selected_collections}")
            
            return routing_decision
            
        except Exception as e:
            logger.error(f"Error in LLM routing: {e}")
            return self._create_fallback_decision(query)
    
    async def _call_llm_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        context: Optional[SearchContext] = None
    ) -> Dict[str, Any]:
        """Call LLM with function calling capability"""
        
        # Get LLM settings
        llm_settings = self._get_llm_settings()
        
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent knowledge router. Use the provided functions to search relevant collections based on the user's query. You can call multiple functions if the query requires information from multiple sources."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Prepare request payload
        payload = {
            "model": llm_settings.get("model_name", "qwen-chat"),
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.1,  # Low temperature for consistent routing
            "max_tokens": 1500
        }
        
        # Add any context-specific parameters
        if context and context.urgency_level == "high":
            payload["temperature"] = 0.05  # Even more deterministic for urgent queries
        
        try:
            # Call LLM endpoint - use model_server from settings
            model_server = llm_settings.get("model_server", "http://localhost:11434").strip()
            llm_endpoint = f"{model_server}/v1/chat/completions"
            
            # Use centralized timeout configuration
            from app.core.timeout_settings_cache import get_timeout_value
            http_timeout = get_timeout_value("api_network", "http_request_timeout", 30)
            
            async with httpx.AsyncClient(timeout=http_timeout) as client:
                response = await client.post(
                    llm_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                result = response.json()
                logger.debug(f"LLM routing response: {result}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to call LLM for routing: {e}")
            raise
    
    def _parse_llm_response(self, llm_response: Dict, original_query: str) -> RoutingDecision:
        """Parse LLM response into structured routing decision"""
        
        try:
            # Extract message and tool calls
            message = llm_response.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            reasoning = message.get("content", "No reasoning provided")
            
            if not tool_calls:
                logger.warning("No tool calls in LLM response, using fallback")
                return self._create_fallback_decision(original_query, reasoning)
            
            # Extract collections and refinements from tool calls
            selected_collections = []
            query_refinements = {}
            execution_strategy = ExecutionStrategy.SINGLE_COLLECTION
            
            for tool_call in tool_calls:
                function_info = tool_call.get("function", {})
                function_name = function_info.get("name", "")
                function_args = function_info.get("arguments", "{}")
                
                # Parse function arguments
                try:
                    args = json.loads(function_args) if isinstance(function_args, str) else function_args
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse function arguments: {function_args}")
                    continue
                
                # Extract collection name from function name
                if function_name.startswith("search_"):
                    collection_name = function_name[7:]  # Remove "search_" prefix
                    selected_collections.append(collection_name)
                    
                    # Extract refined query
                    refined_query = args.get("query", original_query)
                    query_refinements[collection_name] = refined_query
            
            # Determine execution strategy based on number of collections
            if len(selected_collections) == 1:
                execution_strategy = ExecutionStrategy.SINGLE_COLLECTION
            elif len(selected_collections) <= 3:
                execution_strategy = ExecutionStrategy.PARALLEL_SEARCH
            else:
                execution_strategy = ExecutionStrategy.ITERATIVE_REFINEMENT
            
            # Calculate confidence based on tool call quality
            confidence_score = self._calculate_routing_confidence(
                tool_calls, selected_collections, original_query
            )
            
            return RoutingDecision(
                selected_collections=selected_collections,
                query_refinements=query_refinements,
                execution_strategy=execution_strategy,
                confidence_score=confidence_score,
                reasoning=reasoning,
                tool_calls=tool_calls
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_fallback_decision(original_query, str(e))
    
    def _calculate_routing_confidence(
        self,
        tool_calls: List[Dict],
        selected_collections: List[str],
        original_query: str
    ) -> float:
        """Calculate confidence score for routing decision"""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence for valid tool calls
        if tool_calls and len(tool_calls) > 0:
            confidence += 0.2
        
        # Boost confidence for reasonable number of collections
        if 1 <= len(selected_collections) <= 3:
            confidence += 0.2
        elif len(selected_collections) > 5:
            confidence -= 0.1  # Too many collections might be less focused
        
        # Boost confidence if query refinements are meaningful
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            function_args = function_info.get("arguments", "{}")
            
            try:
                args = json.loads(function_args) if isinstance(function_args, str) else function_args
                refined_query = args.get("query", "")
                
                # Check if query was meaningfully refined
                if refined_query and refined_query != original_query:
                    if len(refined_query) > len(original_query) * 0.8:  # Substantial refinement
                        confidence += 0.1
            except:
                continue
        
        return min(confidence, 1.0)
    
    def _prioritize_tools(
        self,
        available_tools: List[Dict],
        query: str,
        context: Optional[SearchContext] = None
    ) -> List[Dict]:
        """Prioritize tools based on query relevance"""
        
        query_lower = query.lower()
        scored_tools = []
        
        for tool in available_tools:
            function_info = tool.get("function", {})
            description = function_info.get("description", "").lower()
            
            # Score based on keyword matches
            score = 0
            
            # Check for domain-specific keywords
            if any(word in description for word in ["regulatory", "compliance", "policy"]):
                if any(word in query_lower for word in ["policy", "regulation", "compliance", "rule"]):
                    score += 3
            
            if any(word in description for word in ["technical", "api", "system"]):
                if any(word in query_lower for word in ["technical", "api", "code", "system"]):
                    score += 3
            
            if any(word in description for word in ["product", "customer", "support"]):
                if any(word in query_lower for word in ["product", "customer", "support", "help"]):
                    score += 3
            
            # General relevance scoring
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in description:
                    score += 1
            
            scored_tools.append((tool, score))
        
        # Sort by score and return tools
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in scored_tools]
    
    def _validate_and_optimize_decision(
        self,
        decision: RoutingDecision,
        context: Optional[SearchContext] = None,
        max_collections: int = 5
    ) -> RoutingDecision:
        """Validate and optimize routing decision"""
        
        # Validate collection access
        if context:
            accessible_collections = self.tool_registry.validate_collection_access(
                decision.selected_collections, context
            )
            
            if len(accessible_collections) != len(decision.selected_collections):
                logger.warning(f"Some collections not accessible, filtering: {decision.selected_collections} -> {accessible_collections}")
                
                # Update decision with accessible collections only
                decision.selected_collections = accessible_collections
                decision.query_refinements = {
                    k: v for k, v in decision.query_refinements.items() 
                    if k in accessible_collections
                }
        
        # Limit number of collections
        if len(decision.selected_collections) > max_collections:
            logger.info(f"Limiting collections from {len(decision.selected_collections)} to {max_collections}")
            
            # Keep top collections (assume they're already prioritized by LLM)
            decision.selected_collections = decision.selected_collections[:max_collections]
            decision.query_refinements = {
                k: v for k, v in decision.query_refinements.items()
                if k in decision.selected_collections
            }
        
        # Adjust execution strategy based on final collection count
        if len(decision.selected_collections) == 1:
            decision.execution_strategy = ExecutionStrategy.SINGLE_COLLECTION
        elif len(decision.selected_collections) <= 2:
            decision.execution_strategy = ExecutionStrategy.PARALLEL_SEARCH
        
        return decision
    
    def _create_fallback_decision(self, query: str, reason: str = "LLM routing failed") -> RoutingDecision:
        """Create fallback routing decision when LLM routing fails"""
        
        # Try to suggest collections based on keywords
        query_keywords = query.lower().split()
        suggested_collections = self.tool_registry.get_collection_suggestions(query_keywords)
        
        if not suggested_collections:
            # Use default collection if available
            suggested_collections = ["default_knowledge"]
        
        # Limit to 2 collections for fallback
        suggested_collections = suggested_collections[:2]
        
        return RoutingDecision(
            selected_collections=suggested_collections,
            query_refinements={col: query for col in suggested_collections},
            execution_strategy=ExecutionStrategy.SINGLE_COLLECTION if len(suggested_collections) == 1 else ExecutionStrategy.PARALLEL_SEARCH,
            confidence_score=0.3,  # Low confidence for fallback
            reasoning=f"Fallback routing: {reason}. Using keyword-based suggestion: {suggested_collections}"
        )
    
    def _get_llm_settings(self) -> Dict[str, Any]:
        """Get LLM settings with caching"""
        current_time = datetime.now()
        
        # Refresh settings every 5 minutes
        if (not self._llm_settings or 
            not self._settings_refresh_time or 
            (current_time - self._settings_refresh_time).total_seconds() > 300):
            
            try:
                self._llm_settings = get_llm_settings()
                self._settings_refresh_time = current_time
            except Exception as e:
                logger.error(f"Failed to get LLM settings: {e}")
                # Use fallback settings
                self._llm_settings = {
                    "model_name": "qwen-chat",
                    "endpoint": "http://localhost:8080/v1/chat/completions"
                }
        
        return self._llm_settings


# Utility function for quick routing
async def route_query_to_collections(
    query: str,
    context: Optional[SearchContext] = None,
    max_collections: int = 3
) -> Tuple[List[str], Dict[str, str]]:
    """
    Quick utility function to route query to collections
    
    Returns:
        Tuple of (selected_collections, query_refinements)
    """
    router = LLMRouter()
    decision = await router.route_query(query, context, max_collections)
    return decision.selected_collections, decision.query_refinements