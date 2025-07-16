"""
LangGraph-based Multi-Agent System with intelligent collaboration
Implements proper LangGraph patterns for agent coordination and state management

## Optimal Configuration Ranges (Based on Production Testing)

### Token Limits:
- **Router/Lightweight agents**: 500-1000 tokens
- **Standard agents**: 1500-2000 tokens  
- **Analysis/Technical agents**: 2000-2500 tokens
- **Research/Complex agents**: 2500-3000 tokens (maximum safe limit)

### Timeout Settings:
- **Quick response agents**: 15-30 seconds
- **Standard agents**: 30-45 seconds
- **Complex analysis agents**: 45-90 seconds (maximum safe limit)

### Temperature Ranges:
- **Router/Classification**: 0.1-0.3 (deterministic)
- **Analysis/Technical**: 0.5-0.7 (balanced)
- **Creative/Communication**: 0.7-0.9 (creative)

### Performance Guidelines:
- Tokens > 3000: High risk of timeouts and hangs
- Timeout > 90s: May cause workflow bottlenecks
- Temperature > 1.0 or < 0.1: May produce poor quality responses

### Configuration Validation:
The system automatically validates and clamps configurations to safe limits:
- Max tokens capped at 3000
- Max timeout capped at 90 seconds
- Temperature clamped between 0.1-1.0
- Adaptive timeout scaling for large token requests

### Performance Monitoring:
- LLM call duration tracking
- Success/failure/timeout rates per agent
- Token usage statistics
- Agent performance rankings
"""
import json
import uuid
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_name
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.llm_settings_cache import get_llm_settings
from app.core.redis_client import get_redis_client_for_langgraph
from app.llm.ollama import JarvisLLM
import logging
import httpx
import os

logger = logging.getLogger(__name__)

class LLMPerformanceMetrics:
    """Track LLM call performance metrics"""
    
    def __init__(self):
        self.call_count = 0
        self.total_duration = 0.0
        self.success_count = 0
        self.failure_count = 0
        self.timeout_count = 0
        self.token_usage = {"total_requested": 0, "avg_requested": 0}
        self.agent_metrics = {}  # Per-agent metrics
        
    def record_call(self, agent_name: str, duration: float, success: bool, 
                   timeout: bool, tokens_requested: int, error: str = None):
        """Record metrics for an LLM call"""
        self.call_count += 1
        self.total_duration += duration
        self.token_usage["total_requested"] += tokens_requested
        self.token_usage["avg_requested"] = self.token_usage["total_requested"] / self.call_count
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        if timeout:
            self.timeout_count += 1
            
        # Per-agent metrics
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = {
                "calls": 0, "successes": 0, "failures": 0, "timeouts": 0, 
                "total_duration": 0.0, "avg_duration": 0.0
            }
            
        agent_stats = self.agent_metrics[agent_name]
        agent_stats["calls"] += 1
        agent_stats["total_duration"] += duration
        agent_stats["avg_duration"] = agent_stats["total_duration"] / agent_stats["calls"]
        
        if success:
            agent_stats["successes"] += 1
        else:
            agent_stats["failures"] += 1
            
        if timeout:
            agent_stats["timeouts"] += 1
            
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        avg_duration = self.total_duration / self.call_count if self.call_count > 0 else 0
        success_rate = (self.success_count / self.call_count * 100) if self.call_count > 0 else 0
        timeout_rate = (self.timeout_count / self.call_count * 100) if self.call_count > 0 else 0
        
        return {
            "total_calls": self.call_count,
            "success_rate": f"{success_rate:.1f}%",
            "timeout_rate": f"{timeout_rate:.1f}%",
            "avg_duration": f"{avg_duration:.2f}s",
            "avg_tokens_requested": self.token_usage["avg_requested"],
            "agent_count": len(self.agent_metrics),
            "top_performing_agents": self._get_top_agents()
        }
        
    def _get_top_agents(self) -> List[Dict[str, Any]]:
        """Get top performing agents by success rate"""
        agent_performance = []
        for agent_name, stats in self.agent_metrics.items():
            if stats["calls"] > 0:
                success_rate = (stats["successes"] / stats["calls"]) * 100
                agent_performance.append({
                    "agent": agent_name,
                    "success_rate": f"{success_rate:.1f}%",
                    "avg_duration": f"{stats['avg_duration']:.2f}s",
                    "calls": stats["calls"]
                })
        
        return sorted(agent_performance, key=lambda x: float(x["success_rate"].rstrip('%')), reverse=True)[:5]

class MultiAgentLangGraphState(TypedDict):
    """Unified state for LangGraph multi-agent system"""
    # Core query and conversation
    query: str
    conversation_id: str
    messages: List[BaseMessage]
    
    # Agent selection and routing
    selected_agents: List[str]
    agent_selection_reasoning: str
    execution_pattern: Literal["sequential", "parallel", "hierarchical", "consensus"]
    
    # Agent execution state
    current_agent: Optional[str]
    agent_outputs: Dict[str, Any]
    agent_contexts: Dict[str, str]
    agent_handoffs: List[Dict[str, Any]]
    
    # Inter-agent communication
    shared_context: str
    agent_decisions: Dict[str, Any]
    collaboration_metadata: Dict[str, Any]
    
    # Tool and document integration
    tool_calls: List[Dict[str, Any]]
    documents_retrieved: List[Dict[str, Any]]
    compressed_context: str
    
    # Final output
    final_response: str
    confidence_score: float
    metadata: Dict[str, Any]
    error: Optional[str]

class LangGraphMultiAgentSystem:
    """LangGraph-based multi-agent system with proper state management"""
    
    def _get_agent_model_config(self, agent_config: Dict[str, Any]) -> tuple[str, int, float]:
        """
        Get model name and parameters based on agent configuration flags.
        Returns: (model_name, max_tokens, temperature)
        """
        if agent_config.get('use_main_llm'):
            from app.core.llm_settings_cache import get_main_llm_full_config
            main_llm_config = get_main_llm_full_config()
            model_name = main_llm_config.get('model')
            # Use main_llm settings for params if not overridden in agent config
            max_tokens = agent_config.get('max_tokens', main_llm_config.get('max_tokens', 1000))
            temperature = agent_config.get('temperature', main_llm_config.get('temperature', 0.7))
        elif agent_config.get('use_second_llm'):
            from app.core.llm_settings_cache import get_second_llm_full_config
            second_llm_config = get_second_llm_full_config()
            model_name = second_llm_config.get('model')
            # Use second_llm settings for params if not overridden in agent config
            max_tokens = agent_config.get('max_tokens', second_llm_config.get('max_tokens', 1000))
            temperature = agent_config.get('temperature', second_llm_config.get('temperature', 0.7))
        else:
            # Use specific model or fall back to None (will use second_llm in _efficient_llm_call)
            model_name = agent_config.get('model')
            max_tokens = agent_config.get('max_tokens', 1000)
            temperature = agent_config.get('temperature', 0.7)
        
        return model_name, max_tokens, temperature

    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.redis_client = get_redis_client_for_langgraph()
        if not self.redis_client:
            logger.warning("Redis client not available for LangGraph multi-agent system")
            # Continue without Redis for conversation storage
            self.redis_client = None
        
        # Initialize checkpointer for state persistence
        try:
            from app.core.config import get_settings
            settings = get_settings()
            redis_host = settings.REDIS_HOST
            redis_port = settings.REDIS_PORT
            
            # In development, if Redis host is localhost but we're in Docker, try 'redis' first
            if redis_host == "localhost" and os.path.exists("/.dockerenv"):
                redis_host = "redis"
            
            # RedisSaver expects a Redis URL string
            redis_url = f"redis://{redis_host}:{redis_port}"
            self.checkpointer = RedisSaver.from_conn_string(redis_url)
            logger.info(f"LangGraph checkpointer initialized with Redis URL: {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to initialize RedisSaver: {e}")
            self.checkpointer = None
        
        # Load agents and settings
        self.agents = get_langgraph_agents()
        self.llm_settings = get_llm_settings()
        
        # Use agent configurations as-is from the database - no hardcoded limits
        logger.info(f"Loaded {len(self.agents)} agents with their configured settings")
        
        # Initialize performance metrics
        self.performance_metrics = LLMPerformanceMetrics()
        
        # Store LLM configuration for efficient per-request creation
        import os
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        logger.info(f"Using Ollama base_url: {self.ollama_base_url}")
        
        # Don't create LLM instance here - create per request for efficiency
        
        logger.info(f"LangGraph Multi-Agent System initialized with {len(self.agents)} agents")



    def _clean_llm_response(self, response_text: str) -> str:
        """PRESERVE ALL THINKING CONTENT - qwen3:30b-a3b is a thinking model"""
        import re
        
        # CRITICAL FIX: PRESERVE ALL THINKING CONTENT INCLUDING NATURAL LANGUAGE REASONING
        # Do NOT remove <think>...</think> blocks or natural reasoning patterns
        # This was the root cause - aggressive cleaning was removing thinking content
        
        if not response_text:
            return ""
            
        # MINIMAL cleaning - only remove excessive whitespace
        cleaned = response_text.strip()
        
        # Only clean up excessive newlines (more than 3 in a row)
        cleaned = re.sub(r'\n\s*\n\s*\n\s*\n+', '\n\n\n', cleaned)
        
        # If response is too short, return original
        if len(cleaned) < 10:
            return response_text
            
        return cleaned

    async def _efficient_llm_call(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.7, timeout: float = 45.0, agent_name: str = "unknown", model_name: str = None) -> str:
        """Efficient LLM call using direct OllamaLLM like the working codebase patterns"""
        from app.llm.ollama import OllamaLLM
        from app.llm.base import LLMConfig
        
        start_time = time.time()
        success = False
        is_timeout = False
        error_msg = None
        
        try:
            # Validate and clamp parameters with special handling for synthesizer
            is_synthesizer = agent_name.lower() == 'synthesizer'
            
            # Apply appropriate limits based on agent type
            if is_synthesizer:
                max_tokens = min(max_tokens, 3000)  # Higher limit for synthesizer
                timeout = min(timeout, 90)          # Longer timeout for synthesizer
            else:
                max_tokens = min(max_tokens, 2000)  # Standard limit for other agents
                timeout = min(timeout, 45)          # Standard timeout for other agents
                
            temperature = max(0.1, min(temperature, 1.0))  # Temperature bounds
            
            if max_tokens > 1500:
                logger.debug(f"Large token request ({max_tokens}), using extended timeout")
                max_timeout = 90 if is_synthesizer else 45
                timeout = min(timeout * 1.2, max_timeout)  # Scale timeout for large requests
            
            # CRITICAL FIX: Support agent-specific model configuration
            # Use provided model_name if available, otherwise fall back to second_llm
            if model_name:
                effective_model = model_name
            else:
                # Get second_llm config as default
                from app.core.llm_settings_cache import get_second_llm_full_config
                second_llm_config = get_second_llm_full_config()
                effective_model = second_llm_config.get('model', 'qwen3:1.7b')
            
            logger.debug(f"Agent {agent_name} using model: {effective_model} (agent-specific: {bool(model_name)})")
            
            # Create LLM config using codebase efficient pattern
            llm_config = LLMConfig(
                model_name=effective_model,
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens
            )
            
            # Create LLM instance per request (efficient pattern from service.py)
            llm = OllamaLLM(llm_config, base_url=self.ollama_base_url)
            
            # Make efficient call with timeout
            import asyncio
            response = await asyncio.wait_for(
                llm.generate(prompt),
                timeout=timeout
            )
            
            success = True
            # Clean the response text by removing <think> blocks and verbose reasoning
            cleaned_text = self._clean_llm_response(response.text)
            return cleaned_text
            
        except asyncio.TimeoutError:
            is_timeout = True
            error_msg = f"LLM call timed out after {timeout} seconds"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"LLM call failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        finally:
            # Record performance metrics
            duration = time.time() - start_time
            self.performance_metrics.record_call(
                agent_name=agent_name,
                duration=duration,
                success=success,
                timeout=is_timeout,
                tokens_requested=max_tokens,
                error=error_msg
            )

    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for multi-agent collaboration"""
        
        workflow = StateGraph(MultiAgentLangGraphState)
        
        # Define nodes
        workflow.add_node("analyze_query", self.analyze_query_node)
        workflow.add_node("select_agents", self.select_agents_node)
        workflow.add_node("determine_execution_pattern", self.determine_execution_pattern_node)
        workflow.add_node("execute_sequential", self.execute_sequential_node)
        workflow.add_node("execute_parallel", self.execute_parallel_node)
        workflow.add_node("execute_hierarchical", self.execute_hierarchical_node)
        workflow.add_node("execute_consensus", self.execute_consensus_node)
        workflow.add_node("synthesize_response", self.synthesize_response_node)
        workflow.add_node("finalize_output", self.finalize_output_node)
        
        # Define workflow
        workflow.set_entry_point("analyze_query")
        
        # Sequential flow
        workflow.add_edge("analyze_query", "select_agents")
        workflow.add_edge("select_agents", "determine_execution_pattern")
        
        # Conditional routing based on execution pattern
        workflow.add_conditional_edges(
            "determine_execution_pattern",
            self.route_by_execution_pattern,
            {
                "sequential": "execute_sequential",
                "parallel": "execute_parallel", 
                "hierarchical": "execute_hierarchical",
                "consensus": "execute_consensus"
            }
        )
        
        # All execution patterns lead to synthesis
        workflow.add_edge("execute_sequential", "synthesize_response")
        workflow.add_edge("execute_parallel", "synthesize_response")
        workflow.add_edge("execute_hierarchical", "synthesize_response")
        workflow.add_edge("execute_consensus", "synthesize_response")
        
        # Final steps
        workflow.add_edge("synthesize_response", "finalize_output")
        workflow.add_edge("finalize_output", END)
        
        # Compile with checkpointer if available, otherwise compile without
        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

    async def analyze_query_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Analyze the user query to understand requirements"""
        
        analysis_prompt = f"""
        Analyze this user query to understand what type of expertise and approach is needed:
        
        Query: "{state['query']}"
        
        Consider:
        1. What domains of expertise are needed?
        2. What type of collaboration would be most effective?
        3. Are there any specific tools or knowledge sources required?
        4. What is the complexity level of this query?
        
        Provide a brief analysis focusing on requirements.
        """
        
        try:
            # Use efficient LLM call with minimal tokens for analysis
            analysis = await self._efficient_llm_call(
                analysis_prompt, 
                max_tokens=500, 
                temperature=0.3,
                agent_name="query_analyzer"
            )
            state["metadata"]["query_analysis"] = analysis
            state["metadata"]["analysis_timestamp"] = datetime.utcnow().isoformat()
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback analysis based on keywords
            query_lower = state["query"].lower()
            if any(word in query_lower for word in ["corporate", "business", "company", "organization"]):
                analysis = "Business/corporate strategy query requiring expert analysis"
            elif any(word in query_lower for word in ["technical", "technology", "software", "system"]):
                analysis = "Technical query requiring specialized expertise"
            else:
                analysis = "General query requiring diverse expertise and collaboration"
            state["metadata"]["query_analysis"] = analysis
        
        return state

    async def select_agents_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Select appropriate agents based on query analysis"""
        
        available_agents = self.agents
        query = state["query"]
        
        # Create agent selection prompt
        agent_descriptions = []
        for agent_name, agent_data in available_agents.items():
            role = agent_data.get('role', 'General Assistant')
            description = agent_data.get('description', 'No description available')
            agent_descriptions.append(f"- {agent_name}: {role} - {description}")
        
        selection_prompt = f"""
        Select the most appropriate agents for this query: "{query}"
        
        Available agents:
        {chr(10).join(agent_descriptions)}
        
        Query analysis: {state["metadata"].get("query_analysis", "Not available")}
        
        Select 1-3 agents that would be most helpful. Consider:
        1. Relevance of their expertise to the query
        2. Potential for meaningful collaboration
        3. Avoiding redundancy
        
        Respond with only the agent names, one per line, followed by a brief reasoning.
        Format:
        AGENTS:
        agent_name_1
        agent_name_2
        
        REASONING:
        Brief explanation of why these agents were selected.
        """
        
        try:
            # Use efficient LLM call with minimal tokens for agent selection
            selection_response = await self._efficient_llm_call(
                selection_prompt,
                max_tokens=200,
                temperature=0.1,
                agent_name="agent_selector"
            )
            
            # Try to parse structured response, but also handle free-form responses
            lines = selection_response.split('\n')
            agents_section = False
            reasoning_section = False
            selected_agents = []
            reasoning_lines = []
            
            for line in lines:
                line = line.strip()
                if line.upper() == "AGENTS:":
                    agents_section = True
                    reasoning_section = False
                    continue
                elif line.upper() == "REASONING:":
                    agents_section = False
                    reasoning_section = True
                    continue
                elif agents_section and line and line in available_agents:
                    selected_agents.append(line)
                elif reasoning_section and line:
                    reasoning_lines.append(line)
                else:
                    # Also check if any line contains agent names
                    for agent_name in available_agents.keys():
                        if agent_name in line and agent_name not in selected_agents:
                            selected_agents.append(agent_name)
            
            # Fallback to simple selection if parsing fails
            if not selected_agents:
                selected_agents = self._fallback_agent_selection(query, available_agents)
                reasoning = "Used keyword-based selection - LLM response could not be parsed"
            else:
                reasoning = ' '.join(reasoning_lines) if reasoning_lines else "Agents selected from LLM response"
            
            state["selected_agents"] = selected_agents[:3]  # Limit to 3 agents
            state["agent_selection_reasoning"] = reasoning
            
            
        except Exception as e:
            logger.error(f"Agent selection failed: {e}")
            # Fallback to simple selection
            selected_agents = self._fallback_agent_selection(query, available_agents)
            reasoning = f"Used keyword-based selection due to LLM error. Selected agents based on role/description matching."
            
            state["selected_agents"] = selected_agents
            state["agent_selection_reasoning"] = reasoning
        
        return state

    def _fallback_agent_selection(self, query: str, available_agents: Dict[str, Any]) -> List[str]:
        """Fallback agent selection based on word overlap - fully dynamic from database"""
        query_words = set(query.lower().split())
        agent_scores = {}
        
        # Score agents based on their role and description from database
        for agent_name, agent_data in available_agents.items():
            role = agent_data.get('role', '')
            description = agent_data.get('description', '')
            agent_words = set((role + ' ' + description).lower().split())
            overlap = len(query_words.intersection(agent_words))
            
            if overlap > 0:
                agent_scores[agent_name] = overlap
        
        # Return top 2 agents or first 2 available if no matches
        if agent_scores:
            sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
            return [agent[0] for agent in sorted_agents[:2]]
        else:
            # Return first 2 active agents from database
            active_agents = [name for name, data in available_agents.items() 
                           if data.get('is_active', True)]
            return active_agents[:2] if len(active_agents) >= 2 else active_agents

    async def determine_execution_pattern_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Determine optimal execution pattern for selected agents"""
        
        selected_agents = state["selected_agents"]
        query = state["query"]
        
        pattern_prompt = f"""
        Determine the best execution pattern for these agents working on this query:
        
        Query: "{query}"
        Selected agents: {', '.join(selected_agents)}
        
        Available patterns:
        1. sequential - Agents work one after another, building on previous outputs
        2. parallel - Agents work simultaneously on different aspects
        3. hierarchical - One agent supervises and coordinates others
        4. consensus - Agents collaborate to reach agreement
        
        Consider:
        - Query complexity and requirements
        - Number of agents selected
        - Potential for agent collaboration
        - Efficiency vs thoroughness trade-offs
        
        Respond with only the pattern name: sequential, parallel, hierarchical, or consensus
        """
        
        try:
            # Use efficient LLM call with minimal tokens for pattern determination
            pattern_response = await self._efficient_llm_call(
                pattern_prompt,
                max_tokens=50,
                temperature=0.1,
                agent_name="pattern_selector"
            )
            pattern = pattern_response.strip().lower()
            
            # Validate pattern
            valid_patterns = ["sequential", "parallel", "hierarchical", "consensus"]
            if pattern not in valid_patterns:
                pattern = "sequential"  # Default fallback
            
            state["execution_pattern"] = pattern
            
            
        except Exception as e:
            logger.error(f"Execution pattern determination failed: {e}")
            # Smart fallback based on query and agent count
            if len(selected_agents) == 1:
                state["execution_pattern"] = "sequential"
            elif len(selected_agents) == 2:
                state["execution_pattern"] = "parallel" 
            else:
                # For 3+ agents, use hierarchical for complex queries
                if any(word in query.lower() for word in ["complex", "analyze", "strategy", "comprehensive"]):
                    state["execution_pattern"] = "hierarchical"
                else:
                    state["execution_pattern"] = "parallel"
        
        state["metadata"]["pattern_selection_timestamp"] = datetime.utcnow().isoformat()
        return state

    def route_by_execution_pattern(self, state: MultiAgentLangGraphState) -> str:
        """Route to appropriate execution node based on pattern"""
        return state['execution_pattern']

    async def execute_sequential_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Execute agents sequentially with handoffs and improved error handling"""
        
        selected_agents = state["selected_agents"]
        query = state["query"]
        shared_context = ""
        successful_agents = 0
        
        for i, agent_name in enumerate(selected_agents):
            agent_data = self.agents.get(agent_name, {})
            
            # Build context for this agent
            agent_prompt = self._build_agent_prompt(
                agent_name=agent_name,
                agent_data=agent_data,
                query=query,
                shared_context=shared_context,
                is_sequential=True,
                position=i + 1,
                total_agents=len(selected_agents)
            )
            
            try:
                # Use agent-specific config from database with reduced limits for faster execution
                agent_config = agent_data.get('config', {})
                timeout = min(agent_config.get('timeout', 30), 30)  # Cap at 30s for speed
                
                # Get model and parameters based on configuration flags
                model_name, max_tokens, temperature = self._get_agent_model_config(agent_config)
                max_tokens = min(max_tokens, 1500)  # Cap at 1500 for speed in sequential execution
                
                logger.info(f"Executing agent {agent_name} with {max_tokens} tokens, {timeout}s timeout, model: {model_name or 'global'}")
                
                
                # Use efficient LLM call for agent execution with agent-specific config including model
                agent_response = await self._efficient_llm_call(
                    agent_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    agent_name=agent_name,
                    model_name=model_name
                )
                
                # Store agent output
                state["agent_outputs"][agent_name] = agent_response
                state["agent_contexts"][agent_name] = agent_prompt
                successful_agents += 1
                
                # Add to shared context for next agent (keep it concise)
                role = agent_data.get('role', agent_name)
                # Limit shared context to prevent prompts from getting too long
                context_snippet = agent_response[:500] + "..." if len(agent_response) > 500 else agent_response
                shared_context += f"\n\n{role}: {context_snippet}"
                
                # Record handoff
                if i < len(selected_agents) - 1:
                    next_agent = selected_agents[i + 1]
                    state["agent_handoffs"].append({
                        "from_agent": agent_name,
                        "to_agent": next_agent,
                        "context_passed": context_snippet,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                logger.info(f"Agent {agent_name} completed successfully")
                
                
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {e}")
                state["agent_outputs"][agent_name] = f"Error: {str(e)}"
                
                # Continue with next agent even if this one fails
                # Add a brief error note to shared context
                role = agent_data.get('role', agent_name)
                shared_context += f"\n\n{role}: [Execution timeout - unable to provide analysis]"
        
        state["shared_context"] = shared_context
        state["metadata"]["execution_pattern_used"] = "sequential"
        state["metadata"]["successful_agents"] = successful_agents
        state["metadata"]["total_agents"] = len(selected_agents)
        
        logger.info(f"Sequential execution completed: {successful_agents}/{len(selected_agents)} agents succeeded")
        
        return state

    async def execute_parallel_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Execute agents in parallel on different aspects"""
        
        selected_agents = state["selected_agents"]
        query = state["query"]
        
        # Note: In a real implementation, you'd use asyncio for true parallelism
        # For now, we'll simulate parallel execution with sequential calls
        
        for agent_name in selected_agents:
            agent_data = self.agents.get(agent_name, {})
            
            agent_prompt = self._build_agent_prompt(
                agent_name=agent_name,
                agent_data=agent_data,
                query=query,
                shared_context="",
                is_parallel=True
            )
            
            try:
                # Use agent-specific config from database
                agent_config = agent_data.get('config', {})
                timeout = agent_config.get('timeout', 45)
                
                # Get model and parameters based on configuration flags
                model_name, max_tokens, temperature = self._get_agent_model_config(agent_config)
                
                # Use efficient LLM call for agent execution with agent-specific config including model
                agent_response = await self._efficient_llm_call(
                    agent_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    agent_name=agent_name,
                    model_name=model_name
                )
                
                state["agent_outputs"][agent_name] = agent_response
                state["agent_contexts"][agent_name] = agent_prompt
                
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {e}")
                state["agent_outputs"][agent_name] = f"Error: {str(e)}"
        
        # Combine outputs for shared context
        context_parts = []
        for agent_name, output in state["agent_outputs"].items():
            agent_role = self.agents.get(agent_name, {}).get('role', agent_name)
            context_parts.append(f"{agent_role} Perspective:\n{output}")
        
        state["shared_context"] = "\n\n".join(context_parts)
        state["metadata"]["execution_pattern_used"] = "parallel"
        
        return state

    async def execute_hierarchical_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Execute with hierarchical coordination"""
        
        selected_agents = state["selected_agents"]
        query = state["query"]
        
        # First agent acts as coordinator
        coordinator = selected_agents[0]
        workers = selected_agents[1:] if len(selected_agents) > 1 else []
        
        # Coordinator plans the approach
        coordinator_data = self.agents.get(coordinator, {})
        planning_prompt = f"""
        As the {coordinator_data.get('role', coordinator)}, you are coordinating the response to this query: "{query}"
        
        You have these team members available: {', '.join([self.agents.get(w, {}).get('role', w) for w in workers])}
        
        First, provide your analysis of the query and plan how to coordinate the team response.
        """
        
        try:
            # Use coordinator-specific config from database
            coordinator_config = coordinator_data.get('config', {})
            max_tokens = coordinator_config.get('max_tokens', 1000)
            timeout = coordinator_config.get('timeout', 45)
            temperature = coordinator_config.get('temperature', 0.7)
            
            coordinator_plan = await self._efficient_llm_call(
                planning_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            state["agent_outputs"][coordinator] = coordinator_plan
            
            # Workers provide input based on coordinator's plan
            for worker in workers:
                worker_data = self.agents.get(worker, {})
                worker_prompt = self._build_agent_prompt(
                    agent_name=worker,
                    agent_data=worker_data,
                    query=query,
                    shared_context=f"Coordinator Plan:\n{coordinator_plan}",
                    is_hierarchical=True
                )
                
                try:
                    # Use worker-specific config from database
                    worker_config = worker_data.get('config', {})
                    max_tokens = worker_config.get('max_tokens', 1000)
                    timeout = worker_config.get('timeout', 45)
                    temperature = worker_config.get('temperature', 0.7)
                    model_name = worker_config.get('model')  # Get worker-specific model
                    
                    worker_response = await self._efficient_llm_call(
                        worker_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=timeout,
                        agent_name=worker,
                        model_name=model_name
                    )
                    state["agent_outputs"][worker] = worker_response
                except Exception as e:
                    logger.error(f"Worker {worker} execution failed: {e}")
                    state["agent_outputs"][worker] = f"Error: {str(e)}"
            
            # Final coordination
            final_prompt = f"""
            As the coordinating {coordinator_data.get('role', coordinator)}, synthesize the team inputs:
            
            Original query: "{query}"
            Your initial plan: {coordinator_plan}
            
            Team inputs:
            {chr(10).join([f"{w}: {state['agent_outputs'].get(w, 'No input')}" for w in workers])}
            
            Provide a coordinated final response.
            """
            
            # Use coordinator config for final coordination
            coordinator_config = coordinator_data.get('config', {})
            max_tokens = coordinator_config.get('max_tokens', 1500)
            timeout = coordinator_config.get('timeout', 45)
            temperature = coordinator_config.get('temperature', 0.7)
            
            final_coordination = await self._efficient_llm_call(
                final_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            state["agent_outputs"][f"{coordinator}_final"] = final_coordination
            state["shared_context"] = final_coordination
            
        except Exception as e:
            logger.error(f"Hierarchical execution failed: {e}")
            # Fallback to parallel execution
            return self.execute_parallel_node(state)
        
        state["metadata"]["execution_pattern_used"] = "hierarchical"
        return state

    async def execute_consensus_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Execute with consensus building"""
        
        selected_agents = state["selected_agents"]
        query = state["query"]
        
        # Round 1: Initial responses
        initial_responses = {}
        for agent_name in selected_agents:
            agent_data = self.agents.get(agent_name, {})
            agent_prompt = self._build_agent_prompt(
                agent_name=agent_name,
                agent_data=agent_data,
                query=query,
                shared_context="",
                is_consensus=True,
                consensus_round=1
            )
            
            try:
                # Use agent-specific config from database for consensus round 1
                agent_config = agent_data.get('config', {})
                timeout = agent_config.get('timeout', 45)
                
                # Get model and parameters based on configuration flags
                model_name, max_tokens, temperature = self._get_agent_model_config(agent_config)
                
                response = await self._efficient_llm_call(
                    agent_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    agent_name=agent_name,
                    model_name=model_name
                )
                initial_responses[agent_name] = response
                state["agent_outputs"][f"{agent_name}_round1"] = response
            except Exception as e:
                logger.error(f"Agent {agent_name} round 1 failed: {e}")
                initial_responses[agent_name] = f"Error: {str(e)}"
        
        # Round 2: Consensus building
        other_responses = "\n\n".join([
            f"{name} ({self.agents.get(name, {}).get('role', name)}):\n{resp}"
            for name, resp in initial_responses.items()
        ])
        
        consensus_responses = {}
        for agent_name in selected_agents:
            agent_data = self.agents.get(agent_name, {})
            consensus_prompt = f"""
            As {agent_data.get('role', agent_name)}, review the other team members' responses and build consensus:
            
            Original query: "{query}"
            Your initial response: {initial_responses.get(agent_name, "Not available")}
            
            Other team members' responses:
            {other_responses}
            
            Provide a consensus-building response that acknowledges other perspectives and works toward agreement.
            """
            
            try:
                # Use agent-specific config from database for consensus round 2
                agent_config = agent_data.get('config', {})
                max_tokens = agent_config.get('max_tokens', 1000)
                timeout = agent_config.get('timeout', 45)
                temperature = agent_config.get('temperature', 0.7)
                
                consensus_response = await self._efficient_llm_call(
                    consensus_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout
                )
                consensus_responses[agent_name] = consensus_response
                state["agent_outputs"][f"{agent_name}_consensus"] = consensus_response
            except Exception as e:
                logger.error(f"Agent {agent_name} consensus failed: {e}")
                consensus_responses[agent_name] = initial_responses.get(agent_name, f"Error: {str(e)}")
        
        # Build shared context from consensus
        consensus_context = "\n\n".join([
            f"{name} Consensus:\n{resp}"
            for name, resp in consensus_responses.items()
        ])
        
        state["shared_context"] = consensus_context
        state["metadata"]["execution_pattern_used"] = "consensus"
        
        return state

    async def synthesize_response_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Synthesize final response from agent outputs"""
        
        query = state["query"]
        shared_context = state["shared_context"]
        agent_outputs = state["agent_outputs"]
        execution_pattern = state["execution_pattern"]
        
        # If shared_context is empty, build it from agent_outputs
        if not shared_context and agent_outputs:
            context_parts = []
            for agent_name, output in agent_outputs.items():
                if not output.startswith("Error:"):
                    agent_role = self.agents.get(agent_name, {}).get('role', agent_name)
                    context_parts.append(f"{agent_role} Analysis:\n{output}")
            shared_context = "\n\n".join(context_parts)
        
        # If we still have no useful context, provide a simple fallback
        if not shared_context or len(shared_context.strip()) < 50:
            logger.warning("No useful agent outputs for synthesis, using fallback")
            state["final_response"] = f"I apologize, but I encountered issues processing your question: '{query}'. Please try rephrasing your question or try again."
            state["confidence_score"] = 0.1
            state["error"] = "No useful agent outputs for synthesis"
            state["metadata"]["synthesis_timestamp"] = datetime.utcnow().isoformat()
            return state
        
        synthesis_prompt = f"""
        Provide a clear, direct answer to: "{query}"
        
        Analysis summary:
        {shared_context[:1000]}{'...' if len(shared_context) > 1000 else ''}
        
        Give a focused, actionable response without explaining your thinking process.
        """
        
        try:
            # Emit synthesis start event
            
            # Use efficient LLM call for response synthesis - prefer thinking model for better synthesis
            synthesized_response = await self._efficient_llm_call(
                synthesis_prompt,
                max_tokens=800,   # Further reduced for speed
                temperature=0.6,  # Slightly more focused
                timeout=20,       # Aggressive timeout for responsiveness
                agent_name="synthesizer",
                model_name="qwen3:30b-a3b"  # Use thinking model for synthesis if available
            )
            state["final_response"] = synthesized_response
            
            # Calculate confidence based on agent agreement and execution success
            confidence = self._calculate_confidence(state)
            state["confidence_score"] = confidence
            
        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            # Improved fallback that uses agent outputs directly
            if agent_outputs:
                successful_outputs = [output for output in agent_outputs.values() if not output.startswith("Error:")]
                if successful_outputs:
                    state["final_response"] = f"Based on expert analysis:\n\n{successful_outputs[0]}"
                    state["confidence_score"] = 0.6
                else:
                    state["final_response"] = f"I encountered issues processing your question: '{query}'. Please try again."
                    state["confidence_score"] = 0.2
            else:
                state["final_response"] = f"Multi-agent processing completed for: {query}"
                state["confidence_score"] = 0.3
            
            state["error"] = f"Synthesis failed: {str(e)}"
        
        state["metadata"]["synthesis_timestamp"] = datetime.utcnow().isoformat()
        return state

    async def finalize_output_node(self, state: MultiAgentLangGraphState) -> MultiAgentLangGraphState:
        """Finalize output and update conversation memory"""
        
        # Store conversation in Redis
        conversation_key = f"multi_agent_conversation:{state['conversation_id']}"
        conversation_data = {
            "query": state["query"],
            "response": state["final_response"],
            "agents_used": state["selected_agents"],
            "execution_pattern": state["execution_pattern"],
            "confidence_score": state["confidence_score"],
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": state["metadata"]
        }
        
        try:
            if self.redis_client:
                self.redis_client.lpush(conversation_key, json.dumps(conversation_data))
                self.redis_client.ltrim(conversation_key, 0, 49)  # Keep last 50 interactions
                self.redis_client.expire(conversation_key, 86400)  # 24 hour TTL
            else:
                logger.debug("Redis client not available, skipping conversation storage")
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
        
        state["metadata"]["finalization_timestamp"] = datetime.utcnow().isoformat()
        return state

    def _build_agent_prompt(self, agent_name: str, agent_data: Dict[str, Any], query: str, 
                          shared_context: str = "", **kwargs) -> str:
        """Build context-appropriate prompt for an agent"""
        
        # Use system_prompt from database if available, otherwise fallback to role/description
        system_prompt = agent_data.get('system_prompt', '')
        role = agent_data.get('role', 'Assistant')
        description = agent_data.get('description', 'General purpose assistant')
        
        if system_prompt:
            prompt_parts = [
                system_prompt,
                "",
                f"User query: \"{query}\"",
            ]
        else:
            prompt_parts = [
                f"You are acting as: {role}",
                f"Your expertise: {description}",
                "",
                f"User query: \"{query}\"",
            ]
        
        if shared_context:
            prompt_parts.extend([
                "",
                "Previous context/analysis:",
                shared_context
            ])
        
        # Add pattern-specific instructions
        if kwargs.get('is_sequential'):
            position = kwargs.get('position', 1)
            total = kwargs.get('total_agents', 1)
            prompt_parts.extend([
                "",
                f"You are agent {position} of {total} in a sequential workflow.",
                "Build upon previous analysis and prepare context for the next agent." if position < total else "Provide the final analysis for this workflow."
            ])
        
        elif kwargs.get('is_parallel'):
            prompt_parts.extend([
                "",
                "You are working in parallel with other agents.",
                "Focus on your area of expertise and provide your unique perspective."
            ])
        
        elif kwargs.get('is_hierarchical'):
            prompt_parts.extend([
                "",
                "You are working under hierarchical coordination.",
                "Follow the coordinator's plan and provide specialized input."
            ])
        
        elif kwargs.get('is_consensus'):
            round_num = kwargs.get('consensus_round', 1)
            if round_num == 1:
                prompt_parts.extend([
                    "",
                    "This is round 1 of consensus building.",
                    "Provide your initial expert perspective."
                ])
            else:
                prompt_parts.extend([
                    "",
                    "This is the consensus building round.",
                    "Consider other perspectives and work toward agreement."
                ])
        
        prompt_parts.extend([
            "",
            "Provide a clear, focused response based on your expertise."
        ])
        
        return "\n".join(prompt_parts)

    def _calculate_confidence(self, state: MultiAgentLangGraphState) -> float:
        """Calculate confidence score based on execution metrics"""
        
        base_confidence = 0.7
        
        # Boost for successful agent execution
        successful_agents = len([a for a in state["agent_outputs"].values() if not a.startswith("Error:")])
        total_agents = len(state["selected_agents"])
        success_ratio = successful_agents / total_agents if total_agents > 0 else 0
        
        # Boost for appropriate agent selection
        selection_boost = 0.1 if len(state["selected_agents"]) > 1 else 0
        
        # Pattern-specific adjustments
        pattern_boost = {
            "sequential": 0.05,
            "parallel": 0.1,
            "hierarchical": 0.15,
            "consensus": 0.2
        }.get(state["execution_pattern"], 0)
        
        confidence = base_confidence * success_ratio + selection_boost + pattern_boost
        return min(1.0, max(0.1, confidence))  # Clamp between 0.1 and 1.0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics summary"""
        return self.performance_metrics.get_summary()

    async def process_query(self, query: str, conversation_id: Optional[str] = None, trace=None) -> Dict[str, Any]:
        """Process a query through the LangGraph multi-agent workflow"""
        
        if not conversation_id:
            conversation_id = self.conversation_id
        
        # Initialize state
        initial_state = MultiAgentLangGraphState(
            query=query,
            conversation_id=conversation_id,
            messages=[],
            selected_agents=[],
            agent_selection_reasoning="",
            execution_pattern="sequential",
            current_agent=None,
            agent_outputs={},
            agent_contexts={},
            agent_handoffs=[],
            shared_context="",
            agent_decisions={},
            collaboration_metadata={},
            tool_calls=[],
            documents_retrieved=[],
            compressed_context="",
            final_response="",
            confidence_score=0.0,
            metadata={"start_time": datetime.utcnow().isoformat()},
            error=None
        )
        
        # Create and run workflow
        app = self.create_workflow()
        config = {"configurable": {"thread_id": conversation_id}}
        
        try:
            # Use ainvoke with timeout to prevent hanging
            import asyncio
            final_state = await asyncio.wait_for(
                app.ainvoke(initial_state, config),
                timeout=120.0  # 2 minute timeout
            )
            
            return {
                "answer": final_state.get("final_response", ""),
                "agents_used": final_state.get("selected_agents", []),
                "execution_pattern": final_state.get("execution_pattern", ""),
                "confidence_score": final_state.get("confidence_score", 0.0),
                "agent_selection_reasoning": final_state.get("agent_selection_reasoning", ""),
                "conversation_id": conversation_id,
                "metadata": final_state.get("metadata", {}),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Multi-agent workflow failed: {e}")
            return {
                "answer": f"Multi-agent processing failed: {str(e)}",
                "agents_used": [],
                "execution_pattern": "error",
                "confidence_score": 0.0,
                "agent_selection_reasoning": "Error occurred",
                "conversation_id": conversation_id,
                "metadata": {"error_timestamp": datetime.utcnow().isoformat()},
                "error": str(e)
            }

# Export main function for API integration
async def langgraph_multi_agent_answer(
    question: str,
    conversation_id: Optional[str] = None,
    stream: bool = False,
    trace=None
):
    """
    LangGraph-based multi-agent answer with streaming support using real workflow
    
    Args:
        question: User query
        conversation_id: Optional conversation ID for continuity
        stream: Whether to return streaming generator or final result
        
    Returns:
        Dict with answer and collaboration metadata OR async generator if stream=True
    """
    
    system = LangGraphMultiAgentSystem(conversation_id)
    
    if stream:
        # Use FIXED multi-agent streaming with proper thinking and tool usage
        from app.langchain.fixed_multi_agent_streaming import fixed_multi_agent_streaming
        return await fixed_multi_agent_streaming(question, conversation_id, trace=trace)
    else:
        # Return final result (non-streaming)
        return await system.process_query(question, conversation_id, trace=trace)