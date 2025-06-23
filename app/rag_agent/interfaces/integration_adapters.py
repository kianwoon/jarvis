"""
Integration adapters for seamless integration with existing systems

These adapters provide compatibility layers for different types of systems
that want to use the RAG agent module.
"""

import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass

from app.rag_agent.interfaces.rag_interface import StandaloneRAGInterface
from app.rag_agent.utils.types import RAGOptions, SearchContext, RAGStreamChunk

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Standard chat message format"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Standard chat response format"""
    message: str
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class AgentContext:
    """Context for multi-agent systems"""
    agent_id: str
    role: str
    current_task: str
    execution_context: Dict[str, Any]


@dataclass
class AgentToolResult:
    """Result from agent tool execution"""
    success: bool
    content: str
    confidence: float
    sources: List[Dict[str, Any]]
    execution_trace: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None


class StandardChatAdapter:
    """
    Adapter for integration with standard chat interfaces
    
    This adapter provides a simple interface for chat systems that want to
    leverage the RAG agent for knowledge-based responses.
    """
    
    def __init__(self):
        self.rag_interface = StandaloneRAGInterface()
        self._conversation_contexts = {}  # conversation_id -> context
    
    async def process_chat_message(
        self,
        message: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """
        Process chat message through RAG system
        
        Args:
            message: User message content
            conversation_id: Unique conversation identifier
            user_id: User identifier for access control
            user_context: Additional user context
            
        Returns:
            ChatResponse with answer and sources
        """
        
        try:
            # Build search context from chat context
            search_context = self._build_search_context(
                user_id, user_context, conversation_id
            )
            
            # Configure RAG options for chat (optimized for speed)
            rag_options = RAGOptions(
                max_iterations=2,  # Faster for chat
                stream=False,
                include_sources=True,
                confidence_threshold=0.5,  # Lower threshold for chat
                execution_timeout_ms=15000  # 15 second timeout
            )
            
            # Process through RAG
            rag_response = await self.rag_interface.query(
                query=message,
                context=search_context,
                options=rag_options
            )
            
            # Update conversation context
            self._update_conversation_context(conversation_id, message, rag_response)
            
            # Convert to chat response format
            return ChatResponse(
                message=rag_response.content,
                sources=self._convert_sources_to_chat_format(rag_response.sources),
                confidence=rag_response.confidence_score,
                metadata={
                    "collections_searched": rag_response.collections_searched,
                    "processing_time_ms": rag_response.processing_time_ms,
                    "query_refinements": rag_response.query_refinements
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return ChatResponse(
                message="I encountered an error while processing your message. Please try again.",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def process_chat_message_stream(
        self,
        message: str,
        conversation_id: str,
        user_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming version for real-time chat responses
        
        Yields:
            Dict with chat-formatted streaming chunks
        """
        
        try:
            search_context = self._build_search_context(
                user_id, user_context, conversation_id
            )
            
            rag_options = RAGOptions(
                stream=True,
                max_iterations=2,
                include_sources=True
            )
            
            async for chunk in self.rag_interface.query_stream(
                query=message,
                context=search_context,
                options=rag_options
            ):
                # Convert to chat streaming format
                yield {
                    "type": chunk.chunk_type,
                    "content": chunk.content,
                    "sources": self._convert_sources_to_chat_format(chunk.sources),
                    "is_final": chunk.is_final,
                    "metadata": chunk.step_info
                }
                
        except Exception as e:
            logger.error(f"Error in streaming chat message: {e}")
            yield {
                "type": "error",
                "content": f"Error processing message: {str(e)}",
                "sources": [],
                "is_final": True,
                "metadata": {}
            }
    
    def _build_search_context(
        self,
        user_id: Optional[str],
        user_context: Optional[Dict[str, Any]],
        conversation_id: str
    ) -> SearchContext:
        """Build search context from chat context"""
        
        # Get conversation history
        conversation_history = self._conversation_contexts.get(conversation_id, [])
        
        # Extract domain and other context
        domain = "general"
        user_permissions = []
        
        if user_context:
            domain = user_context.get("domain", "general")
            user_permissions = user_context.get("permissions", [])
        
        return SearchContext(
            user_id=user_id,
            conversation_id=conversation_id,
            domain=domain,
            urgency_level="normal",
            required_accuracy="medium",  # Balanced for chat
            conversation_history=conversation_history,
            user_permissions=user_permissions
        )
    
    def _convert_sources_to_chat_format(self, sources) -> List[Dict[str, Any]]:
        """Convert RAG sources to chat-friendly format"""
        
        chat_sources = []
        for source in sources:
            chat_source = {
                "collection": source.collection_name.replace('_', ' ').title(),
                "document": source.document_id,
                "content_preview": source.content[:150] + "..." if len(source.content) > 150 else source.content,
                "confidence": source.score,
                "metadata": source.metadata
            }
            
            if source.page:
                chat_source["page"] = source.page
            if source.section:
                chat_source["section"] = source.section
                
            chat_sources.append(chat_source)
        
        return chat_sources
    
    def _update_conversation_context(
        self,
        conversation_id: str,
        user_message: str,
        rag_response
    ):
        """Update conversation context with new exchange"""
        
        if conversation_id not in self._conversation_contexts:
            self._conversation_contexts[conversation_id] = []
        
        context = self._conversation_contexts[conversation_id]
        
        # Add user message
        context.append({
            "role": "user",
            "content": user_message,
            "timestamp": str(int(__import__("time").time()))
        })
        
        # Add assistant response
        context.append({
            "role": "assistant",
            "content": rag_response.content,
            "timestamp": str(int(__import__("time").time())),
            "confidence": rag_response.confidence_score,
            "collections": rag_response.collections_searched
        })
        
        # Keep only last 10 exchanges (20 messages)
        if len(context) > 20:
            context = context[-20:]
        
        self._conversation_contexts[conversation_id] = context


class MultiAgentAdapter:
    """
    Adapter for integration with multi-agent systems
    
    This adapter provides tools that can be used by AI agents in multi-agent
    frameworks like LangGraph, CrewAI, or AutoGen.
    """
    
    def __init__(self):
        self.rag_interface = StandaloneRAGInterface()
    
    async def knowledge_search_tool(
        self,
        query: str,
        agent_context: AgentContext,
        tool_config: Optional[Dict[str, Any]] = None
    ) -> AgentToolResult:
        """
        Knowledge search tool for multi-agent systems
        
        Args:
            query: Search query from agent
            agent_context: Context about the calling agent
            tool_config: Tool-specific configuration
            
        Returns:
            AgentToolResult with structured response
        """
        
        try:
            # Parse tool configuration
            config = tool_config or {}
            
            # Build search context for agent
            search_context = SearchContext(
                user_id=agent_context.agent_id,
                domain=self._extract_domain_from_agent_role(agent_context.role),
                urgency_level=config.get("urgency", "normal"),
                required_accuracy=config.get("accuracy", "high"),
                conversation_history=[],  # Agents typically don't have chat history
                user_permissions=config.get("permissions", [])
            )
            
            # Configure RAG options for agent usage
            rag_options = RAGOptions(
                max_iterations=config.get("max_iterations", 3),
                stream=False,
                include_sources=True,
                include_execution_trace=True,  # Agents benefit from execution trace
                confidence_threshold=config.get("confidence_threshold", 0.7),
                execution_timeout_ms=config.get("timeout_ms", 20000)
            )
            
            # Process query
            rag_response = await self.rag_interface.query(
                query=query,
                context=search_context,
                options=rag_options
            )
            
            # Determine success based on confidence
            success = rag_response.confidence_score >= rag_options.confidence_threshold
            
            # Generate agent-specific recommendations
            recommendations = self._generate_agent_recommendations(
                rag_response, agent_context, success
            )
            
            return AgentToolResult(
                success=success,
                content=rag_response.content,
                confidence=rag_response.confidence_score,
                sources=self._convert_sources_to_agent_format(rag_response.sources),
                execution_trace=self._convert_trace_to_agent_format(rag_response.execution_trace),
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in agent knowledge search: {e}")
            return AgentToolResult(
                success=False,
                content=f"Knowledge search failed: {str(e)}",
                confidence=0.0,
                sources=[],
                recommendations=["Retry with a different query", "Check system status"]
            )
    
    async def batch_knowledge_search_tool(
        self,
        queries: List[str],
        agent_context: AgentContext,
        tool_config: Optional[Dict[str, Any]] = None
    ) -> List[AgentToolResult]:
        """
        Batch knowledge search for agents that need multiple queries
        
        Args:
            queries: List of search queries
            agent_context: Agent context
            tool_config: Tool configuration
            
        Returns:
            List of AgentToolResult objects
        """
        
        if not queries:
            return []
        
        # Process queries with concurrency control
        config = tool_config or {}
        max_concurrent = config.get("max_concurrent", 2)
        
        search_context = SearchContext(
            user_id=agent_context.agent_id,
            domain=self._extract_domain_from_agent_role(agent_context.role)
        )
        
        rag_options = RAGOptions(
            max_iterations=2,  # Faster for batch
            include_execution_trace=False,  # Reduce overhead
            confidence_threshold=config.get("confidence_threshold", 0.6)
        )
        
        # Use RAG interface batch processing
        rag_responses = await self.rag_interface.batch_query(
            queries=queries,
            context=search_context,
            options=rag_options,
            max_concurrent=max_concurrent
        )
        
        # Convert to agent tool results
        agent_results = []
        for query, rag_response in zip(queries, rag_responses):
            success = rag_response.confidence_score >= rag_options.confidence_threshold
            
            agent_result = AgentToolResult(
                success=success,
                content=rag_response.content,
                confidence=rag_response.confidence_score,
                sources=self._convert_sources_to_agent_format(rag_response.sources),
                recommendations=self._generate_agent_recommendations(
                    rag_response, agent_context, success
                )
            )
            agent_results.append(agent_result)
        
        return agent_results
    
    def _extract_domain_from_agent_role(self, agent_role: str) -> str:
        """Extract domain context from agent role"""
        
        role_lower = agent_role.lower()
        
        if any(word in role_lower for word in ["compliance", "regulatory", "audit"]):
            return "compliance"
        elif any(word in role_lower for word in ["technical", "developer", "engineer"]):
            return "technical"
        elif any(word in role_lower for word in ["product", "sales", "marketing"]):
            return "product"
        elif any(word in role_lower for word in ["support", "customer", "service"]):
            return "support"
        elif any(word in role_lower for word in ["risk", "security"]):
            return "risk"
        
        return "general"
    
    def _convert_sources_to_agent_format(self, sources) -> List[Dict[str, Any]]:
        """Convert sources to agent-friendly format"""
        
        agent_sources = []
        for source in sources:
            agent_source = {
                "id": source.document_id,
                "collection": source.collection_name,
                "content": source.content,
                "relevance_score": source.score,
                "metadata": source.metadata
            }
            
            if source.page:
                agent_source["location"] = f"page {source.page}"
            if source.section:
                agent_source["section"] = source.section
            
            agent_sources.append(agent_source)
        
        return agent_sources
    
    def _convert_trace_to_agent_format(self, execution_trace) -> Optional[Dict[str, Any]]:
        """Convert execution trace to agent-friendly format"""
        
        if not execution_trace:
            return None
        
        return {
            "plan_id": execution_trace.plan_id,
            "total_time_ms": execution_trace.total_time_ms,
            "collections_searched": execution_trace.collections_searched,
            "strategy_used": execution_trace.final_strategy.value,
            "steps_executed": len(execution_trace.steps_executed),
            "query_refinements": execution_trace.query_refinements
        }
    
    def _generate_agent_recommendations(
        self,
        rag_response,
        agent_context: AgentContext,
        success: bool
    ) -> List[str]:
        """Generate recommendations for agents based on results"""
        
        recommendations = []
        
        if not success:
            recommendations.append("Consider refining your query with more specific terms")
            recommendations.append("Try breaking down complex queries into simpler parts")
            
            if not rag_response.sources:
                recommendations.append("No relevant sources found - verify the topic exists in knowledge base")
            elif rag_response.confidence_score < 0.3:
                recommendations.append("Low confidence results - consider alternative query approaches")
        
        else:
            if len(rag_response.sources) > 5:
                recommendations.append("Many sources found - consider filtering for most relevant")
            
            if rag_response.confidence_score > 0.9:
                recommendations.append("High confidence results - information is well-supported")
        
        # Agent-role specific recommendations
        if "research" in agent_context.role.lower():
            recommendations.append("Consider cross-referencing with additional collections")
        elif "compliance" in agent_context.role.lower():
            recommendations.append("Verify information with latest regulatory sources")
        
        return recommendations


# Utility functions for quick integration
def create_chat_adapter() -> StandardChatAdapter:
    """Create a chat adapter instance"""
    return StandardChatAdapter()


def create_agent_adapter() -> MultiAgentAdapter:
    """Create a multi-agent adapter instance"""
    return MultiAgentAdapter()


async def quick_chat_query(
    message: str,
    conversation_id: str = "default",
    user_id: Optional[str] = None
) -> ChatResponse:
    """Quick utility for chat queries"""
    
    adapter = StandardChatAdapter()
    return await adapter.process_chat_message(message, conversation_id, user_id)


async def quick_agent_search(
    query: str,
    agent_id: str = "default_agent",
    agent_role: str = "assistant"
) -> AgentToolResult:
    """Quick utility for agent searches"""
    
    adapter = MultiAgentAdapter()
    context = AgentContext(
        agent_id=agent_id,
        role=agent_role,
        current_task="knowledge_search",
        execution_context={}
    )
    
    return await adapter.knowledge_search_tool(query, context)