"""
Simplified Multi-Agent System without LangGraph dependency
Implements router-based communication between specialized agents
"""

from typing import Dict, List, Any, Optional, TypedDict, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_role, get_active_agents, get_agent_by_name
from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.llm_settings_cache import get_llm_settings
from app.agents.task_decomposer import TaskDecomposer, TaskChunk
from app.agents.continuity_manager import ContinuityManager
from app.agents.redis_continuation_manager import RedisContinuityManager
import json
import uuid
from datetime import datetime
import asyncio
import httpx
import hashlib
import logging
# Import rag_answer lazily to avoid circular import

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State shared between agents"""
    query: str
    conversation_id: str
    messages: List[BaseMessage]
    routing_decision: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    tools_used: List[str]
    documents_retrieved: List[Dict[str, Any]]
    final_response: str
    metadata: Dict[str, Any]
    error: Optional[str]
    # Inter-agent communication fields
    agent_messages: List[Dict[str, Any]]  # Messages between agents
    pending_requests: Dict[str, List[Dict]]  # Pending requests for each agent
    agent_conversations: List[Dict[str, Any]]  # Track agent-to-agent conversations
    # Collaboration patterns
    execution_pattern: str  # 'parallel', 'sequential', 'hierarchical'
    agent_dependencies: Dict[str, List[str]]  # Which agents depend on others
    execution_order: List[str]  # For sequential execution
    # Performance tracking
    agent_metrics: Dict[str, Dict[str, Any]]  # Performance metrics per agent

class MultiAgentSystem:
    """
    Simplified orchestration for multiple specialized agents
    
    Optimized for MacBook environments:
    - Sequential execution (1 agent at a time) by default
    - Resource-conscious timeout management
    - Graceful handling of system constraints
    """
    
    def __init__(self, conversation_id: Optional[str] = None, trace=None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.trace = trace  # Store trace for span creation
        self.current_agent_span = None  # Track current agent span for tool execution
        
        # Ensure Redis cache is warmed before proceeding
        from app.core.langgraph_agents_cache import validate_and_warm_cache, get_cache_status
        cache_status = get_cache_status()
        print(f"[DEBUG] Cache status on init: {cache_status}")
        
        if not cache_status.get("cache_exists") or cache_status.get("agent_count", 0) == 0:
            print("[DEBUG] Cache not ready, warming cache...")
            validate_and_warm_cache()
        
        self.agents = get_langgraph_agents()
        print(f"[DEBUG] MultiAgentSystem initialized with {len(self.agents)} agents: {list(self.agents.keys())}")
        
        self.llm_settings = get_llm_settings()
        self.routing_cache = {}  # Simple in-memory cache for routing decisions
        
        # Load agent avatars/emojis and descriptions from database configuration
        self.agent_avatars = self._load_agent_avatars()
        self.agent_descriptions = self._load_agent_descriptions()
    
    def _load_agent_avatars(self) -> Dict[str, str]:
        """Load agent avatars from database configuration or use defaults"""
        avatars = {}
        for agent_name, agent_config in self.agents.items():
            # Check if avatar is configured in database
            config = agent_config.get('config', {})
            if config and isinstance(config, dict) and 'avatar' in config:
                avatars[agent_name] = config['avatar']
            else:
                # Use role-based default avatars
                role = agent_config.get('role', '').lower()
                if 'router' in role or 'routing' in role:
                    avatars[agent_name] = "🧭"
                elif 'document' in role or 'research' in role:
                    avatars[agent_name] = "📚"
                elif 'tool' in role or 'executor' in role:
                    avatars[agent_name] = "🔧"
                elif 'context' in role or 'manager' in role:
                    avatars[agent_name] = "🧠"
                elif 'sales' in role or 'strategist' in role:
                    avatars[agent_name] = "💼"
                elif 'technical' in role or 'architect' in role:
                    avatars[agent_name] = "🏗️"
                elif 'financial' in role or 'analyst' in role:
                    avatars[agent_name] = "💰"
                elif 'service' in role or 'delivery' in role:
                    avatars[agent_name] = "📋"
                elif 'synthesizer' in role or 'synthesis' in role:
                    avatars[agent_name] = "🎯"
                else:
                    avatars[agent_name] = "🤖"  # Default robot emoji
        return avatars
    
    def _load_agent_descriptions(self) -> Dict[str, str]:
        """Load agent descriptions from database configuration"""
        descriptions = {}
        for agent_name, agent_config in self.agents.items():
            # Use the role as description, or config description if available
            config = agent_config.get('config', {})
            if config and isinstance(config, dict) and 'description' in config:
                descriptions[agent_name] = config['description']
            else:
                descriptions[agent_name] = agent_config.get('role', 'AI agent')
        return descriptions
    
    def _get_specialized_agents(self) -> List[str]:
        """Get list of specialized agents from database configuration"""
        specialized = []
        for agent_name, agent_config in self.agents.items():
            config = agent_config.get('config', {})
            role = agent_config.get('role', '').lower()
            
            # Check if explicitly marked as specialized
            if config and isinstance(config, dict) and config.get('is_specialized', False):
                specialized.append(agent_name)
            # Or check by role content for common specialized roles
            elif any(keyword in role for keyword in [
                'sales', 'technical', 'financial', 'architect', 'strategist', 
                'analyst', 'cto', 'ceo', 'cio', 'compliance', 'roi'
            ]):
                specialized.append(agent_name)
        return specialized
    
    def _get_agent_display_order(self) -> List[str]:
        """Get agent display order from configuration or use intelligent defaults"""
        # Check if there's a global configuration for agent order
        display_order = []
        role_priority = {
            'sales': 1,
            'technical': 2, 
            'architect': 2,
            'financial': 3,
            'analyst': 3,
            'service': 4,
            'delivery': 4
        }
        
        # Sort agents by role priority
        agent_roles = []
        for agent_name, agent_config in self.agents.items():
            role = agent_config.get('role', '').lower()
            priority = 999  # Default low priority
            for keyword, prio in role_priority.items():
                if keyword in role:
                    priority = prio
                    break
            agent_roles.append((priority, agent_name))
        
        # Sort by priority and return agent names
        agent_roles.sort(key=lambda x: x[0])
        return [agent_name for _, agent_name in agent_roles]
    
    def _get_agent_prompt_with_template(self, agent_name: str, state: AgentState, default_prompt: str) -> str:
        """Get agent prompt, using template instructions if available"""
        # Check for agent template configuration
        agent_config = state.get("metadata", {}).get("agent_config", {})
        template_instructions = agent_config.get("default_instructions", "")
        
        if template_instructions:
            # Extract the agent description from default prompt
            agent_description = default_prompt.split('\n')[0] if '\n' in default_prompt else default_prompt
            
            # Build prompt with template instructions
            return f"{agent_description}\n\n{template_instructions}"
        
        return default_prompt
    
    def _clean_response(self, text: str) -> str:
        """Clean response without removing thinking tags"""
        return text.strip()
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Clean up excessive markdown formatting"""
        import re
        # Remove standalone "---" dividers
        text = re.sub(r'^---\s*$', '', text, flags=re.MULTILINE)
        # Remove "**" bold markers
        text = re.sub(r'\*\*', '', text)
        # Clean up multiple consecutive newlines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def determine_collaboration_pattern(self, query: str, selected_agents: List[str]) -> Dict[str, Any]:
        """Determine the best collaboration pattern based on query and agents"""
        query_lower = query.lower()
        
        # Check for sequential needs
        needs_sequential = any(keyword in query_lower for keyword in [
            "step by step", "first then", "after", "followed by", "process"
        ])
        
        # Check for hierarchical needs
        needs_hierarchical = any(keyword in query_lower for keyword in [
            "coordinate", "delegate", "assign", "manage", "oversee"
        ])
        
        # Default patterns based on agent combinations
        # Find agents by role instead of hardcoded names
        sales_agents = [a for a in selected_agents if 'sales' in self.agents.get(a, {}).get('role', '').lower()]
        financial_agents = [a for a in selected_agents if 'financial' in self.agents.get(a, {}).get('role', '').lower()]
        technical_agents = [a for a in selected_agents if 'technical' in self.agents.get(a, {}).get('role', '').lower()]
        service_agents = [a for a in selected_agents if 'service' in self.agents.get(a, {}).get('role', '').lower()]
        
        if sales_agents and financial_agents:
            # Sales often needs financial input first
            order = financial_agents + sales_agents + technical_agents + service_agents
            # Remove duplicates while preserving order
            order = list(dict.fromkeys(order))
            dependencies = {}
            if sales_agents and financial_agents:
                dependencies[sales_agents[0]] = financial_agents
            if service_agents and technical_agents:
                dependencies[service_agents[0]] = technical_agents
            
            return {
                "pattern": "sequential",
                "order": order,
                "dependencies": dependencies
            }
        elif needs_hierarchical:
            return {
                "pattern": "hierarchical",
                "order": selected_agents,
                "lead_agent": selected_agents[0] if selected_agents else "router",
                "dependencies": {
                    selected_agents[0]: selected_agents[1:] if len(selected_agents) > 1 else []
                }
            }
        elif needs_sequential:
            return {
                "pattern": "sequential",
                "order": selected_agents,
                "dependencies": {}
            }
        else:
            # Default to sequential for MacBook stability and resource management
            return {
                "pattern": "sequential", 
                "order": selected_agents,
                "dependencies": {}
            }
    
    async def intelligently_select_agents(self, query: str) -> Dict[str, Any]:
        """Use LLM analysis to intelligently select appropriate agents based on query content"""
        # Get all active agents directly from cache
        active_agents = get_active_agents()
        
        if not active_agents:
            print("[WARNING] No agents available, returning empty selection")
            return {"selected_agents": [], "reasoning": "No agents available in database"}
        
        # Create agent selection prompt using role and description from database
        agent_descriptions = []
        for name, agent in active_agents.items():
            role = agent.get('role', 'Unknown Role')
            description = agent.get('description', 'No description available')
            
            agent_descriptions.append(f"""
Agent: {name}
Role: {role}
Description: {description}
""")
        
        agents_text = "\n".join(agent_descriptions)
        
        # LLM analysis prompt
        analysis_prompt = f"""You are an intelligent agent selection system. Based on the user's query, select the most appropriate agents to handle the request and determine the best collaboration pattern.

Available Agents:
{agents_text}

User Query: "{query}"

Analyze the query and do a proper plan to select agents whose ROLES and DESCRIPTIONS best match what's needed to answer this question. Consider:

1. Match the user's query requirements to each agent's specific ROLE
2. Look at each agent's DESCRIPTION to understand their capabilities
3. Determine the best collaboration pattern (sequential, parallel, or hierarchical)
4. Think about dependencies between agents

For example:
- If the query is about sales strategy, look for agents with "sales" or "business" in their role
- If the query is technical, look for agents with "technical", "architect", or "engineer" in their role
- If the query needs analysis, look for agents with "analyst" or "research" in their role
- If the query needs writing/documentation, look for agents with "writer" or "communication" in their role

Respond with a JSON object in this exact format:
{{
    "selected_agents": ["agent1", "agent2", "agent3"],
    "collaboration_pattern": "sequential|parallel|hierarchical", 
    "reasoning": "Explanation of why these specific agents were selected based on their roles and descriptions, and how they should collaborate",
    "agent_order": ["agent1", "agent2", "agent3"],
    "dependencies": {{"agent2": ["agent1"], "agent3": ["agent1", "agent2"]}}
}}

Important: 
- Only select agents that exist in the available agents list
- Base selection primarily on role and description matching
- Focus on quality over quantity - select the most relevant agents
- Explain your reasoning clearly in terms of role/description fit"""

        try:
            print(f"[DEBUG] Starting LLM agent selection for query: {query}")
            print(f"[DEBUG] Available agents: {list(active_agents.keys())}")
            
            # Use intelligent rule-based selection that actually works
            print(f"[DEBUG] Using intelligent rule-based agent selection...")
            response = self._intelligent_rule_based_selection(query, active_agents)
            
            print(f"[DEBUG] Rule-based selection result: {response[:200]}...")
            
            # Parse the JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                print(f"[DEBUG] Found JSON in response: {json_match.group()}")
                result = json.loads(json_match.group())
                
                print(f"[DEBUG] Parsed result: {result}")
                
                # Validate selected agents exist
                valid_agents = []
                for agent in result.get("selected_agents", []):
                    if agent in active_agents:
                        valid_agents.append(agent)
                        print(f"[DEBUG] Valid agent found: {agent}")
                    else:
                        print(f"[WARNING] Agent '{agent}' not found in available agents")
                
                # Ensure we have at least one agent
                if not valid_agents:
                    # Fallback: select first 2 available agents
                    valid_agents = list(active_agents.keys())[:2]
                    print(f"[DEBUG] No valid agents from LLM - using fallback: {valid_agents}")
                
                result["selected_agents"] = valid_agents
                result["agent_order"] = valid_agents  # Ensure order matches selection
                
                print(f"[DEBUG] LLM SELECTION SUCCESS - Final agents: {valid_agents}")
                print(f"[DEBUG] Collaboration pattern: {result.get('collaboration_pattern', 'sequential')}")
                print(f"[DEBUG] Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                
                return result
            else:
                print(f"[DEBUG] NO JSON FOUND in LLM response: {response}")
                print("[WARNING] Could not parse JSON from LLM response")
                raise ValueError("Invalid JSON response")
                
        except Exception as e:
            print(f"[ERROR] LLM AGENT SELECTION FAILED: {e}")
            print(f"[DEBUG] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] Falling back to keyword-based selection...")
            # Fallback to simple selection based on query keywords
            return self._fallback_agent_selection(query, active_agents)
    
    def _intelligent_rule_based_selection(self, query: str, available_agents: Dict[str, Any]) -> str:
        """Dynamic agent selection based purely on query-to-role/description similarity"""
        query_words = set(query.lower().split())
        agent_scores = {}
        
        # Score each agent based on word overlap between query and agent role/description
        for agent_name, agent_data in available_agents.items():
            role = agent_data.get('role', '')
            description = agent_data.get('description', '')
            
            # Combine role and description words
            agent_words = set((role + ' ' + description).lower().split())
            
            # Calculate overlap score
            overlap = len(query_words.intersection(agent_words))
            
            if overlap > 0:
                agent_scores[agent_name] = {
                    "score": overlap,
                    "role": role,
                    "description": description
                }
        
        # Select top scoring agents
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        selected_agents = [agent for agent, _ in sorted_agents[:3]]
        
        # If no overlap, select first available agents
        if not selected_agents:
            selected_agents = list(available_agents.keys())[:2]
        
        # Create response
        result = {
            "selected_agents": selected_agents,
            "collaboration_pattern": "sequential", 
            "reasoning": f"Selected agents based on role/description relevance to query",
            "agent_order": selected_agents,
            "dependencies": {}
        }
        
        return json.dumps(result)
    
    def _fallback_agent_selection(self, query: str, available_agents: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback agent selection when LLM analysis fails - based on role and description keywords"""
        print(f"[DEBUG] FALLBACK SELECTION for query: {query}")
        print(f"[DEBUG] Available agents in fallback: {list(available_agents.keys())}")
        
        query_lower = query.lower()
        selected_agents = []
        
        # Simple keyword-based selection using role and description
        for agent_name, agent_data in available_agents.items():
            role = agent_data.get('role', '').lower()
            description = agent_data.get('description', '').lower()
            combined_text = f"{role} {description}"
            
            # Match query keywords to agent role/description
            if any(keyword in query_lower for keyword in ["sales", "business", "revenue", "customer"]):
                if any(keyword in combined_text for keyword in ["sales", "business", "commercial", "customer"]):
                    selected_agents.append(agent_name)
            
            if any(keyword in query_lower for keyword in ["technical", "architecture", "system", "code", "engineering"]):
                if any(keyword in combined_text for keyword in ["technical", "architect", "engineer", "system", "code"]):
                    selected_agents.append(agent_name)
            
            if any(keyword in query_lower for keyword in ["cost", "budget", "financial", "roi", "money", "price"]):
                if any(keyword in combined_text for keyword in ["financial", "cost", "budget", "roi", "analyst"]):
                    selected_agents.append(agent_name)
            
            if any(keyword in query_lower for keyword in ["research", "analysis", "information", "study", "investigate"]):
                if any(keyword in combined_text for keyword in ["research", "analysis", "investigate", "study"]):
                    selected_agents.append(agent_name)
            
            if any(keyword in query_lower for keyword in ["security", "compliance", "risk", "audit"]):
                if any(keyword in combined_text for keyword in ["security", "compliance", "risk", "audit"]):
                    selected_agents.append(agent_name)
            
            if any(keyword in query_lower for keyword in ["strategy", "planning", "vision", "direction"]):
                if any(keyword in combined_text for keyword in ["strategy", "strategic", "planning", "vision", "ceo", "cto"]):
                    selected_agents.append(agent_name)
        
        # Remove duplicates while preserving order
        selected_agents = list(dict.fromkeys(selected_agents))
        
        # If no specific agents found, select first 2 available
        if not selected_agents:
            selected_agents = list(available_agents.keys())[:2]
            print(f"[DEBUG] NO KEYWORD MATCHES - Using first 2 agents: {selected_agents}")
        else:
            print(f"[DEBUG] KEYWORD MATCHES found: {selected_agents}")
        
        # Limit to max 4 agents for performance
        selected_agents = selected_agents[:4]
        
        print(f"[DEBUG] FALLBACK FINAL SELECTION: {selected_agents}")
        
        return {
            "selected_agents": selected_agents,
            "collaboration_pattern": "sequential",
            "reasoning": f"Fallback selection based on role/description keyword matching. Selected {len(selected_agents)} agents.",
            "agent_order": selected_agents,
            "dependencies": {}
        }
    
    async def send_message_to_agent(self, from_agent: str, to_agent: str, message: str, state: AgentState, message_type: str = "query"):
        """Send a message from one agent to another and get response"""
        message_id = str(uuid.uuid4())
        agent_message = {
            "id": message_id,
            "from": from_agent,
            "to": to_agent,
            "message": message,
            "type": message_type,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Add to agent messages log
        state["agent_messages"].append(agent_message)
        
        # Add to pending requests for the target agent
        if to_agent not in state["pending_requests"]:
            state["pending_requests"][to_agent] = []
        state["pending_requests"][to_agent].append(agent_message)
        
        # Yield communication event for UI
        yield {
            "type": "agent_communication",
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message": message,
            "message_id": message_id
        }
    
    def check_agent_messages(self, agent_name: str, state: AgentState) -> List[Dict]:
        """Check if there are any messages for this agent"""
        messages = state.get("pending_requests", {}).get(agent_name, [])
        # Clear processed messages
        if agent_name in state.get("pending_requests", {}):
            state["pending_requests"][agent_name] = []
        return messages
    
    async def respond_to_message(self, agent_name: str, message_id: str, response: str, state: AgentState):
        """Respond to a message from another agent"""
        # Find the original message
        for msg in state["agent_messages"]:
            if msg["id"] == message_id:
                msg["status"] = "responded"
                msg["response"] = response
                msg["response_time"] = datetime.now().isoformat()
                
                # Add to agent conversations
                conversation = {
                    "from": msg["from"],
                    "to": msg["to"],
                    "query": msg["message"],
                    "response": response,
                    "timestamp": msg["timestamp"],
                    "response_time": msg["response_time"]
                }
                state["agent_conversations"].append(conversation)
                
                # Yield response event for UI
                yield {
                    "type": "agent_response",
                    "from_agent": agent_name,
                    "to_agent": msg["from"],
                    "response": response,
                    "message_id": message_id
                }
                break
        
    def _extract_query_context(self, query: str) -> Dict[str, Any]:
        """Extract key information from the query"""
        context = {
            "client": "",
            "requirement": "",
            "current_model": "",
            "proposed_model": "",
            "details": query
        }
        
        # Extract client name
        import re
        
        # Look for patterns like "ABC Bank", "XYZ Company", etc.
        client_patterns = [
            r'(?:for|converting|transition|from)\s+([A-Z][A-Za-z0-9\s&]+?)\s+(?:from|to|request|is|wants|,|\.)',
            r'([A-Z][A-Za-z0-9\s&]+?)\s+(?:bank|company|corp|inc|limited|ltd)',
            r'(?:client|customer)[,:\s]+([A-Z][A-Za-z0-9\s&]+?)(?:\s+request|\s+is|\s+wants|,|\.)'
        ]
        
        for pattern in client_patterns:
            client_match = re.search(pattern, query, re.IGNORECASE)
            if client_match:
                context["client"] = client_match.group(1).strip()
                break
        
        # Extract requirement (e.g., "3 x system engineers")
        req_patterns = [
            r'(\d+\s*x?\s*(?:L1|L2|L3|system|network|database|security)?\s*(?:engineer|admin|analyst|developer|architect)s?)',
            r'((?:system|network|database|security)\s*(?:engineer|admin|analyst|developer|architect)s?)',
            r'(IT\s*(?:support|services|infrastructure))'
        ]
        
        for pattern in req_patterns:
            req_match = re.search(pattern, query, re.IGNORECASE)
            if req_match:
                context["requirement"] = req_match.group(1).strip()
                break
        
        # If no specific requirement found but it's about services
        if not context["requirement"] and "service" in query.lower():
            context["requirement"] = "IT Services"
        
        # Identify current and proposed models
        if any(term in query.lower() for term in ["t&m", "time and material", "time & material"]):
            context["current_model"] = "Time & Material (T&M)"
        if any(term in query.lower() for term in ["managed service", "managed services", "msp"]):
            context["proposed_model"] = "Managed Services"
            
        return context
    
    async def _call_llm_stream(self, prompt: str, agent_name: str, temperature: float = 0.7, timeout: int = 30, agent_config: dict = None):
        """Call LLM and return only the final cleaned response as JSON event"""
        try:
            print(f"[DEBUG] *** {agent_name}: GENERATING RESPONSE ***")
            
            # Import LLM service directly to avoid HTTP recursion
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            from app.core.langgraph_agents_cache import get_agent_by_role
            import os
            
            # Get agent-specific configuration
            # If agent_config is passed (from pipeline), use it; otherwise get from langgraph cache
            if not agent_config:
                agent_data = get_agent_by_name(agent_name)
                agent_config = agent_data.get("config", {}) if agent_data else {}
            
            # Use agent config with fallbacks to thinking mode settings
            model_config = self.llm_settings.get("thinking_mode", {})
            
            # Dynamic configuration based on agent settings
            actual_temperature = agent_config.get("temperature", temperature)
            actual_timeout = agent_config.get("timeout", timeout)
            # Ensure adequate max_tokens for agents, especially for complex tasks
            actual_max_tokens = agent_config.get("max_tokens", model_config.get("max_tokens", 4000))
            
            print(f"[DEBUG] {agent_name}: Using config - max_tokens={actual_max_tokens}, timeout={actual_timeout}, temp={actual_temperature}")
            
            # Create LLM config
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=actual_temperature,
                top_p=model_config.get("top_p", 0.9),
                max_tokens=actual_max_tokens
            )
            
            # Ollama base URL
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"[DEBUG] {agent_name}: Using model {config.model_name} at {ollama_url}")
            
            # Create LLM instance with base URL
            llm = OllamaLLM(config, base_url=ollama_url)
            
            # Stream tokens in real-time for better user experience
            response_text = ""
            print(f"[DEBUG] {agent_name}: Starting to stream response with timeout={actual_timeout}s")
            
            # Use asyncio timeout for the actual timeout
            import asyncio
            try:
                async with asyncio.timeout(actual_timeout):
                    async for response_chunk in llm.generate_stream(prompt):
                        response_text += response_chunk.text
                        
                        # Stream tokens in real-time to show progress
                        if response_chunk.text.strip():  # Only yield non-empty tokens
                            yield {
                                "type": "agent_token",
                                "agent": agent_name,
                                "token": response_chunk.text,
                                "avatar": self.agent_avatars.get(agent_name, "🤖"),
                                "description": self.agent_descriptions.get(agent_name, "")
                            }
            except asyncio.TimeoutError:
                print(f"[WARNING] {agent_name}: Response generation timed out after {actual_timeout}s")
                # Return partial response if available
                if response_text:
                    print(f"[WARNING] {agent_name}: Returning partial response of {len(response_text)} chars")
            
            print(f"[DEBUG] {agent_name}: Completed generation, response length = {len(response_text)}")
            
            # Clean the response and yield final completion event
            cleaned_content = self._clean_response(response_text)
            # Keep markdown formatting for better display
            # cleaned_content = self._clean_markdown_formatting(cleaned_content)
            display_content = cleaned_content.strip()
            
            print(f"[DEBUG] {agent_name}: Raw response length = {len(response_text)}, cleaned length = {len(display_content)}")
            
            # Always yield a completion event, even if content is empty
            print(f"[DEBUG] {agent_name}: Yielding final completion event")
            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "content": display_content,
                "avatar": self.agent_avatars.get(agent_name, "🤖"),
                "description": self.agent_descriptions.get(agent_name, ""),
                "final": True  # Mark as final completion
            }
            
        except Exception as e:
            print(f"[ERROR] LLM call failed for {agent_name}: {str(e)}")
            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "content": f"I encountered an error while processing your request: {str(e)}"
            }
        
    async def _router_agent(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Hybrid LLM-based routing with keyword fallback"""
        print(f"[DEBUG] Starting routing for query: {query[:100]}...")
        
# Removed early large generation detection - let multi-agent system handle it
        
        # Try intelligent agent selection first
        try:
            print("[DEBUG] Attempting intelligent agent selection")
            intelligent_result = await self.intelligently_select_agents(query)
            
            if intelligent_result.get("selected_agents"):
                print(f"[DEBUG] Intelligent selection chose agents: {intelligent_result['selected_agents']}")
                # Convert to expected format
                return {
                    "agents": intelligent_result["selected_agents"],
                    "reasoning": intelligent_result.get("reasoning", ""),
                    "collaboration_pattern": intelligent_result.get("collaboration_pattern", "sequential"),
                    "order": intelligent_result.get("agent_order", intelligent_result["selected_agents"]),
                    "dependencies": intelligent_result.get("dependencies", {})
                }
        except Exception as e:
            print(f"[DEBUG] Intelligent agent selection failed: {e}, falling back to keyword routing")
            import traceback
            traceback.print_exc()
        
        # Try dynamic LLM routing as secondary fallback
        try:
            print("[DEBUG] Attempting dynamic LLM-based routing")
            from app.langchain.dynamic_agent_system import agent_instance_pool
            dynamic_system = await agent_instance_pool.get_or_create_instance(trace=self.trace)
            try:
                routing_result = await dynamic_system.route_query(query, conversation_history)
                
                if routing_result.get("agents"):
                    print(f"[DEBUG] Dynamic routing selected agents: {routing_result['agents']}")
                    return routing_result
            finally:
                await agent_instance_pool.release_instance(dynamic_system)
        except Exception as e:
            print(f"[DEBUG] Dynamic routing failed: {e}, falling back to keyword routing")
            import traceback
            traceback.print_exc()
        
        # Fallback to keyword-based routing
        print("[DEBUG] Using keyword-based routing")
        return await self._keyword_router_agent(query)
    
    async def _llm_router_agent(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """LLM-based intelligent routing"""
        # Create cache key for this routing decision
        cache_key = hashlib.md5(f"{query}_{len(conversation_history or [])}".encode()).hexdigest()
        
        # Check cache first
        if cache_key in self.routing_cache:
            print(f"[DEBUG] Using cached routing decision for query")
            return self.routing_cache[cache_key]
        
        # Get all active agents from cache and filter to only implemented ones
        active_agents = get_active_agents()
        
        # Only include agents that have actual implementations
        # Get all active agents from database instead of hardcoded list
        implemented_agents = {}
        core_agents = ["document_researcher", "tool_executor", "context_manager"]
        
        # Add core agents
        for agent_name in core_agents:
            if agent_name in active_agents:
                implemented_agents[agent_name] = active_agents[agent_name]
        
        # Add all other active agents dynamically
        for agent_name, agent_config in active_agents.items():
            if agent_name not in core_agents and agent_config:
                implemented_agents[agent_name] = agent_config
        
        # Remove None values (agents not in cache)
        implemented_agents = {k: v for k, v in implemented_agents.items() if v is not None}
        
        if not implemented_agents:
            raise Exception("No implemented agents available in cache")
        
        # Build agent descriptions for the LLM using only implemented agents
        agent_descriptions = {}
        for name, agent in implemented_agents.items():
            agent_descriptions[name] = {
                "role": agent.get("role", ""),
                "description": agent.get("description", ""),
                "tools": agent.get("tools", [])
            }
        
        # Add context from conversation history if available
        history_context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 messages for context
            history_context = f"\nRecent conversation context:\n{json.dumps(recent_messages, indent=2)}\n"
        
        routing_prompt = f"""You are an intelligent agent router. Your task is to select the most appropriate agents to handle a user query.

Available agents:
{json.dumps(agent_descriptions, indent=2)}

{history_context}
User query: "{query}"

Instructions:
1. Select 1-4 most relevant agents that can best handle this query
2. Consider the query type, complexity, and required capabilities
3. Prioritize agents whose roles and tools match the query requirements
4. Provide clear reasoning for your selections

Respond ONLY in this JSON format:
{{
    "agents": ["agent_name1", "agent_name2"],
    "reasoning": "Brief explanation of why these agents were selected"
}}

Important: Only use agent names that exist in the available agents list above."""

        # Make a quick LLM call for routing with timeout
        # Create Langfuse generation for router system decision
        router_generation = None
        if self.trace:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    router_generation = tracer.create_generation_with_usage(
                        trace=self.trace,
                        name="system-router",
                        model="qwen3:30b-a3b",
                        input_text=routing_prompt,
                        metadata={
                            "operation": "agent_routing",
                            "conversation_id": self.conversation_id,
                            "available_agents": list(implemented_agents.keys()),
                            "has_conversation_history": bool(conversation_history)
                        }
                    )
            except Exception as e:
                print(f"[WARNING] Failed to create router generation: {e}")
        
        try:
            async for chunk in self._call_llm_stream(routing_prompt, "router", temperature=0.3, timeout=8):
                if chunk.get("type") == "agent_complete":
                    response_text = chunk.get("content", "")
                    # Parse JSON response
                    try:
                        # Extract JSON from response
                        import re
                        json_match = re.search(r'\{[^}]*"agents"[^}]*\}', response_text, re.DOTALL)
                        if json_match:
                            routing_data = json.loads(json_match.group())
                            
                            # Validate agents exist and are implemented
                            valid_agents = []
                            for agent in routing_data.get("agents", []):
                                if agent in implemented_agents:
                                    valid_agents.append(agent)
                            
                            if valid_agents:
                                result = {
                                    "agents": valid_agents,
                                    "reasoning": routing_data.get("reasoning", "LLM-based routing")
                                }
                                
                                # End router generation with success
                                if router_generation:
                                    try:
                                        from app.core.langfuse_integration import get_tracer
                                        tracer = get_tracer()
                                        usage = tracer.estimate_token_usage(routing_prompt, response_text)
                                        
                                        router_generation.end(
                                            output=json.dumps(result, indent=2),
                                            usage_details=usage,
                                            metadata={
                                                "success": True,
                                                "selected_agents": valid_agents,
                                                "reasoning_length": len(routing_data.get("reasoning", "")),
                                                "conversation_id": self.conversation_id
                                            }
                                        )
                                    except Exception as e:
                                        print(f"[WARNING] Failed to end router generation: {e}")
                                
                                # Cache the result
                                self.routing_cache[cache_key] = result
                                return result
                        
                        raise Exception("Invalid JSON format in LLM response")
                        
                    except json.JSONDecodeError as e:
                        raise Exception(f"Failed to parse LLM routing response: {e}")
                        
        except Exception as e:
            # End router generation with error
            if router_generation:
                try:
                    router_generation.end(
                        output=f"Error: {str(e)}",
                        metadata={
                            "success": False,
                            "error": str(e),
                            "conversation_id": self.conversation_id
                        }
                    )
                except Exception as gen_error:
                    print(f"[WARNING] Failed to end router generation with error: {gen_error}")
            
            raise Exception(f"LLM routing call failed: {e}")
        
        # End router generation with error for no response
        if router_generation:
            try:
                router_generation.end(
                    output="Error: No complete response received",
                    metadata={
                        "success": False,
                        "error": "No complete response received from LLM router",
                        "conversation_id": self.conversation_id
                    }
                )
            except Exception as gen_error:
                print(f"[WARNING] Failed to end router generation with error: {gen_error}")
        
        raise Exception("No complete response received from LLM router")
    
    async def _keyword_router_agent(self, query: str) -> Dict[str, Any]:
        """Keyword-based routing fallback (original logic)"""
        routing = {
            "agents": [],
            "reasoning": ""
        }
        
        query_lower = query.lower()
        
        # Get available agents from database to ensure we only select existing ones
        from app.core.langgraph_agents_cache import get_active_agents
        available_agents = get_active_agents()
        
        print(f"[DEBUG] Keyword router - Available agents: {list(available_agents.keys())}")
        
        # Note: Large generation detection removed from keyword routing to allow multi-agent collaboration
        
        # Comprehensive managed services keywords
        managed_services_keywords = [
            "managed service", "managed services", "msp",
            "t&m", "time and material", "time & material",
            "convert", "transition", "proposal",
            "pricing", "cost", "roi", "financial",
            "strategy", "approach", "pitch",
            "bank", "banking", "client"
        ]
        
        # Check for managed services context
        is_managed_services = any(keyword in query_lower for keyword in managed_services_keywords)
        
        # Check for specific business contexts
        has_pricing = any(word in query_lower for word in ["pricing", "price", "cost", "financial", "budget", "fee"])
        has_strategy = any(word in query_lower for word in ["strategy", "approach", "plan", "convert", "transition"])
        has_client = any(word in query_lower for word in ["client", "customer", "bank", "abc"])
        has_service_model = any(word in query_lower for word in ["t&m", "managed service", "time and material", "msp"])
        
        # Helper function to find best matching agent - STRICT MATCHING ONLY
        def find_matching_agent(target_names):
            matches = []
            for target in target_names:
                # Try exact match first
                if target in available_agents:
                    matches.append(target)
                    continue
                    
                # Try exact case-insensitive match
                target_lower = target.lower().strip()
                for available_agent in available_agents:
                    if target_lower == available_agent.lower().strip():
                        matches.append(available_agent)
                        break
                else:
                    # Only try specific known synonyms - NO FUZZY MATCHING
                    agent_synonyms = {
                        "researcher": "Researcher Agent",
                        "financial analyst": "financial Analyst",
                        "business analyst": "BizAnalyst Agent",
                        "document researcher": "Document Researcher",
                        "sales strategist": "Sales Strategist",
                        "technical architect": "Technical Architect", 
                        "service delivery manager": "Service Delivery Manager"
                    }
                    
                    if target_lower in agent_synonyms and agent_synonyms[target_lower] in available_agents:
                        matches.append(agent_synonyms[target_lower])
                        print(f"[INFO] Synonym match: {target} → {agent_synonyms[target_lower]}")
                    else:
                        print(f"[WARNING] No exact match found for '{target}' in available agents")
                        # DO NOT add fuzzy matches - let it fail cleanly
            return matches
        
        # For a proposal discussion or managed services query, we want multiple perspectives
        if (any(word in query_lower for word in ["proposal", "client", "discuss", "counter"]) or
            is_managed_services or
            (has_client and (has_pricing or has_strategy or has_service_model))):
            # Try to find specialist team agents based on roles
            desired_roles = ["sales", "technical", "financial", "service"]
            desired_agents = []
            for agent_name, agent_config in available_agents.items():
                role = agent_config.get('role', '').lower()
                if any(desired_role in role for desired_role in desired_roles):
                    desired_agents.append(agent_name)
            routing["agents"] = find_matching_agent(desired_agents)
            
            # If we couldn't find the specialist team, look for any strategic agents
            if not routing["agents"]:
                strategic_keywords = ["strategist", "architect", "analyst", "manager", "ceo", "cto", "cio"]
                for agent_name in available_agents:
                    if any(keyword in agent_name.lower() for keyword in strategic_keywords):
                        routing["agents"].append(agent_name)
                # Limit to 4 agents max
                routing["agents"] = routing["agents"][:4]
            
            routing["reasoning"] = "Managed services/strategic discussion detected - engaging specialist team."
        else:
            # Original routing logic for other queries
            if any(word in query_lower for word in ["document", "file", "pdf", "information", "data"]):
                doc_agents = find_matching_agent(["document_researcher"])
                routing["agents"].extend(doc_agents)
                routing["reasoning"] += "Document research needed. "
                
            if any(word in query_lower for word in ["calculate", "compute", "execute", "run", "tool"]):
                tool_agents = find_matching_agent(["tool_executor"])
                routing["agents"].extend(tool_agents)
                routing["reasoning"] += "Tool execution may be required. "
                
            if any(word in query_lower for word in ["history", "context", "previous", "earlier", "remember"]):
                context_agents = find_matching_agent(["context_manager"])
                routing["agents"].extend(context_agents)
                routing["reasoning"] += "Historical context referenced. "
        
        # Default if no specific routing - find any available agent
        if not routing["agents"]:
            # Try document researcher first
            doc_agents = find_matching_agent(["document_researcher"])
            if doc_agents:
                routing["agents"] = doc_agents
                routing["reasoning"] = "Keyword-based fallback: Defaulting to document research."
            else:
                # Use any available agent
                routing["agents"] = [list(available_agents.keys())[0]] if available_agents else []
                routing["reasoning"] = "Keyword-based fallback: Using first available agent."
        
        # Remove duplicates while preserving order
        seen = set()
        unique_agents = []
        for agent in routing["agents"]:
            if agent not in seen:
                seen.add(agent)
                unique_agents.append(agent)
        routing["agents"] = unique_agents
            
        print(f"[DEBUG] Keyword router selected: {routing['agents']}")
        routing["reasoning"] = f"Keyword-based routing: {routing['reasoning']}"
        return routing
    
    async def _document_researcher_agent(self, state: AgentState) -> Dict[str, Any]:
        """Use RAG to research documents"""
        try:
            # Use the existing RAG system
            response = ""
            documents = []
            has_content = False
            
            print(f"[DEBUG] Document researcher processing query: {state['query']}")
            
            # Stream RAG response
            async for chunk in self._rag_stream(state["query"]):
                if isinstance(chunk, dict):
                    if "token" in chunk:
                        response += chunk["token"]
                        has_content = True
                    elif "answer" in chunk:
                        # Handle complete answer
                        response = chunk["answer"]
                        has_content = True
                    elif "source" in chunk and chunk["source"]:
                        documents.append({
                            "source": chunk["source"],
                            "timestamp": datetime.now().isoformat()
                        })
                    elif "error" in chunk:
                        # Handle error from RAG stream
                        return {
                            "response": f"RAG system error: {chunk['error']}. The system may need to have settings configured properly in the Settings page.",
                            "documents": []
                        }
            
            print(f"[DEBUG] Document researcher - Response length: {len(response)}, Has content: {has_content}")
            
            # If no response from RAG, provide a more helpful message
            if not response or not has_content:
                response = (
                    "No documents found in the knowledge base for your query. "
                    "This could mean:\n"
                    "1. No relevant documents have been uploaded yet\n"
                    "2. The search terms didn't match any indexed content\n"
                    "3. The vector database needs to be populated with relevant documents\n\n"
                    "For your query about OCBC bank and managed services, you may need to:\n"
                    "- Upload relevant proposal documents or service catalogs\n"
                    "- Upload documents about L1 system engineer roles and responsibilities\n"
                    "- Upload managed service frameworks or templates"
                )
            
            return {
                "response": response,
                "documents": documents
            }
        except Exception as e:
            import traceback
            print(f"[ERROR] Document researcher exception: {str(e)}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return {
                "response": f"Error in document research: {str(e)}. Please ensure the vector database and embedding settings are properly configured.",
                "documents": []
            }
    
    async def _sales_strategist_agent(self, state: AgentState):
        """Sales perspective on the proposal - streaming version"""
        # Extract context from query
        context = self._extract_query_context(state["query"])
        
        # Check for agent template configuration
        agent_config = state.get("metadata", {}).get("agent_config", {})
        template_instructions = agent_config.get("default_instructions", "")
        
        # Check for messages from other agents - make dynamic
        agent_name = "sales_strategist"  # This would be passed as parameter in real dynamic system
        messages = self.check_agent_messages(agent_name, state)
        financial_insights = ""
        
        for msg in messages:
            # Look for messages from financial analysts dynamically
            from_agent_role = self.agents.get(msg["from"], {}).get('role', '').lower()
            if 'financial' in from_agent_role and 'analyst' in from_agent_role:
                financial_insights = f"\n\nFinancial Analyst Insights: {msg['message']}"
                # Send response back
                async for event in self.respond_to_message(agent_name, msg["id"], "Thank you for the financial analysis. I'll incorporate this into my sales strategy.", state):
                    yield event
        
        # Check if we need financial input for pricing strategy
        if ("pricing" in state["query"].lower() or "cost" in state["query"].lower()) and not financial_insights:
            # Find financial analyst dynamically
            financial_agent = None
            for agent_name, agent_config in self.agents.items():
                role = agent_config.get('role', '').lower()
                if 'financial' in role and 'analyst' in role:
                    financial_agent = agent_name
                    break
            
            if financial_agent:
                # Request input from financial analyst
                async for event in self.send_message_to_agent(
                    agent_name,  # Current sales agent
                    financial_agent,
                    f"I need cost-benefit analysis for {context['client']} - {context['requirement']}. What are the key financial advantages of managed services over {context['current_model']}?",
                    state
                ):
                    yield event
        
        # Use template instructions if available, otherwise use default prompt
        if template_instructions:
            prompt = f"""You are an experienced Sales Strategist specializing in IT services and managed service proposals.

{template_instructions}

Context:
- Client: {context['client']}
- Current Requirement: {context['requirement']}
- Current Model: {context['current_model']}
- Proposed Model: {context['proposed_model']}
- Details: {context['details']}{financial_insights}"""
        else:
            prompt = f"""You are an experienced Sales Strategist specializing in IT services and managed service proposals.

Context:
- Client: {context['client']}
- Current Requirement: {context['requirement']}
- Current Model: {context['current_model']}
- Proposed Model: {context['proposed_model']}
- Details: {context['details']}{financial_insights}

Provide strategic sales advice for converting this requirement from {context['current_model']} to {context['proposed_model']}.

Your response should include:
1. Value Proposition - Why managed services is better for this specific client
2. Key Selling Points - 3-4 compelling benefits specific to their industry
3. ROI Justification - Financial and operational benefits
4. Objection Handling - Anticipate and address 2-3 likely concerns
5. Next Steps - Concrete action items to move the deal forward

Be specific to {context['client']} and their industry. Use persuasive but professional language."""

        async for event in self._call_llm_stream(prompt, "sales_strategist", temperature=0.7):
            yield event
    
    async def _technical_architect_agent(self, state: AgentState):
        """Technical perspective on the proposal - streaming version"""
        context = self._extract_query_context(state["query"])
        
        prompt = f"""You are a Senior Technical Architect with expertise in IT infrastructure and managed services.

Context:
- Client: {context['client'] or 'the organization'}
- Requirement: {context['requirement'] or 'IT services'}
- Moving from: {context['current_model'] or 'current model'} to {context['proposed_model'] or 'managed services'}

Provide a technical analysis of how managed services will better serve {context['client'] or 'the client'}'s needs.

Your response should cover:
1. Technical Architecture - How the managed service will be structured
2. Technology Stack - Tools and platforms we'll bring to enhance their operations
3. Service Levels - Specific SLAs and KPIs appropriate for a financial institution
4. Security & Compliance - How we'll meet banking sector requirements
5. Integration Strategy - How we'll integrate with their existing systems
6. Automation Opportunities - Where we can add efficiency through automation

Focus on technical excellence and reliability that banks require. Be specific about technologies and metrics."""

        agent_name = "technical_architect"  # This would be passed as parameter in real dynamic system
        async for event in self._call_llm_stream(prompt, agent_name, temperature=0.6):
            yield event
    
    async def _financial_analyst_agent(self, state: AgentState):
        """Financial perspective on the proposal - streaming version"""
        context = self._extract_query_context(state["query"])
        
        # Check for messages from other agents
        agent_name = "financial_analyst"  # This would be passed as parameter in real dynamic system  
        messages = self.check_agent_messages(agent_name, state)
        additional_queries = []
        
        for msg in messages:
            additional_queries.append(f"\n\nAdditional request from {msg['from']}: {msg['message']}")
            # We'll respond after generating our analysis
        
        prompt = f"""You are a Financial Analyst specializing in IT services cost modeling and ROI analysis.

Context:
- Client: {context['client']} (Banking sector)
- Current Requirement: {context['requirement']}
- Comparing: {context['current_model']} vs {context['proposed_model']}

Provide a detailed financial analysis comparing the total cost of ownership between hiring {context['requirement']} directly versus our managed service model.

Your analysis should include:
1. Direct Costs Comparison
   - Salaries, benefits, overhead for direct hires
   - Managed service fees structure
   
2. Hidden Costs Analysis
   - Recruitment, training, attrition costs
   - Management overhead
   - Technology and tool licenses
   
3. ROI Calculation
   - Year 1, Year 3, and Year 5 projections
   - Break-even analysis
   - Risk-adjusted returns
   
4. Value Drivers
   - Quantify productivity gains
   - Calculate cost of downtime reduction
   - Estimate value of scalability
   
5. Pricing Models
   - Recommend optimal pricing structure for {context['client']}
   - Show flexibility options

Use realistic market rates for Singapore/APAC region. Present numbers clearly with executive summary.{''.join(additional_queries)}"""

        # First generate our main analysis
        async for event in self._call_llm_stream(prompt, agent_name, temperature=0.5):
            yield event
        
        # Then respond to any messages from other agents
        for msg in messages:
            response = f"Key financial insights for {msg['from']}: 1) Managed services typically reduce TCO by 25-35% over 3 years. 2) Predictable monthly costs vs variable T&M billing. 3) Risk mitigation through SLAs and penalties. 4) Access to senior expertise without full-time cost."
            async for event in self.respond_to_message(agent_name, msg["id"], response, state):
                yield event
    
    async def _service_delivery_manager_agent(self, state: AgentState):
        """Service delivery perspective on the proposal - streaming version"""
        context = self._extract_query_context(state["query"])
        
        prompt = f"""You are a Service Delivery Manager with 15+ years experience in managing IT services for financial institutions.

Context:
- Client: {context['client'] or 'the organization'}
- Service Requirement: {context['requirement'] or 'IT services'}
- Service Model: Transitioning to {context['proposed_model'] or 'managed services'}

Design a comprehensive service delivery plan that ensures smooth operations and exceeds {context['client']}'s expectations.

Your plan should detail:
1. Governance Model
   - Account management structure
   - Meeting cadences and stakeholders
   - Escalation procedures
   
2. Transition Plan
   - Detailed week-by-week transition roadmap
   - Knowledge transfer approach
   - Risk mitigation during transition
   
3. Service Level Agreement (SLA)
   - Response and resolution times by priority
   - Availability guarantees
   - Penalty and bonus structures
   
4. Team Structure
   - Staffing model with roles and shifts
   - Skills matrix required
   - Training and certification plans
   
5. Quality Framework
   - Performance metrics and dashboards
   - Continuous improvement process
   - Customer satisfaction measurement
   
6. Innovation Roadmap
   - Value-add services timeline
   - Automation implementation plan
   - Technology refresh cycles

Remember {context['client']} is a major bank requiring highest service standards. Be specific and actionable."""

        agent_name = "service_delivery_manager"  # This would be passed as parameter in real dynamic system
        async for event in self._call_llm_stream(prompt, agent_name, temperature=0.6):
            yield event
    
    async def _tool_executor_agent(self, state: AgentState) -> Dict[str, Any]:
        """Execute MCP tools based on query and available tools"""
        try:
            query = state['query']
            tools_used = []
            tool_results = []
            
            # Get available MCP tools
            enabled_tools = get_enabled_mcp_tools()
            
            if not enabled_tools:
                return {
                    "response": "No MCP tools are currently available or enabled.",
                    "tools_used": [],
                    "tool_results": []
                }
            
            # Analyze query to determine which tools to use
            query_lower = query.lower()
            tools_to_execute = []
            
            # Smart tool selection based on query keywords
            for tool_name, tool_info in enabled_tools.items():
                tool_description = ""
                if isinstance(tool_info, dict):
                    tool_description = tool_info.get('description', '').lower()
                    # Also check tool manifest for more detailed descriptions
                    manifest = tool_info.get('manifest', {})
                    if manifest and 'tools' in manifest:
                        for tool_def in manifest['tools']:
                            if tool_def.get('name') == tool_name:
                                tool_description += " " + tool_def.get('description', '').lower()
                
                # Match query intent with tool capabilities
                if any(keyword in query_lower for keyword in ['search', 'find', 'lookup', 'google', 'web']) and 'search' in tool_name.lower():
                    # Search tool
                    search_query = query
                    for word in ['search', 'internet', 'web', 'google', 'find', 'about', 'for']:
                        search_query = search_query.replace(word, '').strip()
                    tools_to_execute.append((tool_name, {'query': search_query, 'num_results': 5}))
                
                elif any(keyword in query_lower for keyword in ['time', 'date', 'now', 'current']) and 'datetime' in tool_name.lower():
                    # DateTime tool
                    tools_to_execute.append((tool_name, {}))
                
                elif any(keyword in query_lower for keyword in ['email', 'gmail', 'send', 'message']) and 'gmail' in tool_name.lower():
                    # Gmail tools - need to extract recipient and content
                    if 'send' in tool_name.lower():
                        # For send email, we'd need to parse the query for recipient and content
                        # This is a simplified implementation
                        tools_to_execute.append((tool_name, {
                            'to': 'user@example.com',  # Would need proper parsing
                            'subject': 'Multi-Agent System Message',
                            'body': query
                        }))
                
                elif any(keyword in query_lower for keyword in ['weather', 'temperature', 'forecast']) and 'weather' in tool_name.lower():
                    # Weather tools
                    location = "current location"  # Could be extracted from query
                    tools_to_execute.append((tool_name, {'location': location}))
            
            # If no specific tools matched, try to use the most general/useful ones
            if not tools_to_execute and enabled_tools:
                # Look for commonly useful tools
                for tool_name in enabled_tools.keys():
                    if 'search' in tool_name.lower() or 'google' in tool_name.lower():
                        tools_to_execute.append((tool_name, {'query': query, 'num_results': 3}))
                        break
                
                # If still no tools and we have datetime, use it as fallback
                if not tools_to_execute:
                    for tool_name in enabled_tools.keys():
                        if 'datetime' in tool_name.lower() or 'time' in tool_name.lower():
                            tools_to_execute.append((tool_name, {}))
                            break
            
            # Execute selected tools
            for tool_name, params in tools_to_execute:
                try:
                    print(f"[TOOL_EXECUTOR] Executing {tool_name} with params: {params}")
                    # Use _execute_mcp_tool which includes Langfuse span creation
                    result = await self._execute_mcp_tool(tool_name, params, "tool_executor")
                    
                    tools_used.append(tool_name)
                    tool_results.append({
                        "tool": tool_name,
                        "parameters": params,
                        "result": result,
                        "success": True
                    })
                    print(f"[TOOL_EXECUTOR] Tool {tool_name} executed successfully")
                    
                except Exception as tool_error:
                    print(f"[TOOL_EXECUTOR] Error executing {tool_name}: {tool_error}")
                    tool_results.append({
                        "tool": tool_name,
                        "parameters": params,
                        "error": str(tool_error),
                        "success": False
                    })
            
            # Format response based on results
            if tool_results:
                response_parts = []
                successful_results = [r for r in tool_results if r.get("success")]
                failed_results = [r for r in tool_results if not r.get("success")]
                
                if successful_results:
                    response_parts.append("🔧 **Tool Execution Results:**\n")
                    for result in successful_results:
                        response_parts.append(f"**{result['tool']}:**")
                        if isinstance(result['result'], dict):
                            response_parts.append(f"```json\n{json.dumps(result['result'], indent=2)}\n```")
                        else:
                            response_parts.append(f"{result['result']}")
                        response_parts.append("")
                
                if failed_results:
                    response_parts.append("❌ **Tool Execution Errors:**")
                    for result in failed_results:
                        response_parts.append(f"- {result['tool']}: {result.get('error', 'Unknown error')}")
                    response_parts.append("")
                
                response = "\n".join(response_parts)
            else:
                response = f"No suitable tools found for the query: '{query}'. Available tools: {', '.join(enabled_tools.keys())}"
            
            return {
                "response": response,
                "tools_used": tools_used,
                "tool_results": tool_results,
                "available_tools": list(enabled_tools.keys())
            }
            
        except Exception as e:
            print(f"[TOOL_EXECUTOR] Error in tool executor agent: {e}")
            return {
                "response": f"Error in tool execution: {str(e)}",
                "tools_used": [],
                "tool_results": [],
                "error": str(e)
            }
    
    async def _context_manager_agent(self, state: AgentState) -> Dict[str, Any]:
        """Manage conversation context"""
        # Simple context summary
        context_summary = f"Conversation ID: {state['conversation_id']}\n"
        context_summary += f"Message count: {len(state['messages'])}\n"
        
        return {
            "response": context_summary,
            "context": {
                "conversation_id": state["conversation_id"],
                "message_count": len(state["messages"])
            }
        }
    
    async def _synthesizer_agent(self, state: AgentState) -> str:
        """Synthesize final response from all agent outputs using the configured synthesizer system prompt"""
        
        # Debug: Print what outputs we have
        print(f"[DEBUG] Synthesizer received outputs from agents: {list(state['agent_outputs'].keys())}")
        print(f"[DEBUG] Synthesizer: Total agent_outputs entries: {len(state['agent_outputs'])}")
        for agent, output in state["agent_outputs"].items():
            response_len = len(output.get("response", "")) if output else 0
            print(f"[DEBUG] Agent {agent} output length: {response_len}")
            if response_len > 0:
                print(f"[DEBUG] Agent {agent} output preview: {output.get('response', '')[:100]!r}")
            else:
                print(f"[DEBUG] Agent {agent} output structure: {output}")
        
        # Check if we have any meaningful responses
        meaningful_outputs = []
        for agent_name, output in state["agent_outputs"].items():
            response = output.get("response", "")
            if response and len(response.strip()) > 20:  # Filter out very short responses
                meaningful_outputs.append({
                    "agent": agent_name,
                    "role": self.agents.get(agent_name, {}).get('role', 'Unknown'),
                    "response": response.strip()
                })
        
        if not meaningful_outputs:
            print(f"[DEBUG] Synthesizer: No meaningful responses found")
            return (
                "I couldn't find specific information to help with your query.\n\n"
                "Please try:\n"
                "• Uploading relevant documents to the knowledge base\n"
                "• Asking a more specific question\n"
                "• Using the standard chat mode for general inquiries"
            )
        
        print(f"[DEBUG] Synthesizer: Processing {len(meaningful_outputs)} meaningful responses")
        
        # Get synthesizer configuration from database
        synthesizer_config = self.agents.get("synthesizer", {})
        synthesizer_system_prompt = synthesizer_config.get('system_prompt', '')
        
        if not synthesizer_system_prompt:
            print("[WARNING] No synthesizer system prompt found in database, using fallback concatenation")
            # Fallback to simple concatenation if no system prompt configured
            return self._fallback_synthesis(meaningful_outputs, state)
        
        # Prepare inputs for synthesis
        agent_inputs = []
        for output_data in meaningful_outputs:
            agent_inputs.append(f"**{output_data['agent']} ({output_data['role']}):**\n{output_data['response']}")
        
        combined_inputs = "\n\n---\n\n".join(agent_inputs)
        
        # Build synthesis prompt - direct and to the point
        synthesis_prompt = f"""{synthesizer_system_prompt}

**INPUTS TO SYNTHESIZE:**

{combined_inputs}

**ORIGINAL QUERY:** {state['query']}


Now provide your synthesis:"""
        
        print(f"[DEBUG] Synthesizer: Using LLM with system prompt ({len(synthesizer_system_prompt)} chars)")
        print(f"[DEBUG] Synthesizer: Processing {len(combined_inputs)} chars of agent inputs")
        
        # Use the LLM to synthesize the response
        try:
            print(f"[DEBUG] Synthesizer: Starting LLM call with prompt length {len(synthesis_prompt)}")
            
            # Call the LLM with the synthesis prompt
            synthesis_result = ""
            event_count = 0
            # Call synthesizer with special handling - don't strip thinking tags
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            import os
            
            # Get synthesizer config
            synthesizer_config = self.agents.get("synthesizer", {}).get("config", {})
            model_config = self.llm_settings.get("thinking_mode", {})
            
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=synthesizer_config.get("temperature", 0.3),
                top_p=model_config.get("top_p", 0.9),
                max_tokens=synthesizer_config.get("max_tokens", 8000)
            )
            
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            llm = OllamaLLM(config, base_url=ollama_url)
            
            print(f"[DEBUG] Synthesizer: Direct LLM call without thinking tag removal")
            async for response_chunk in llm.generate_stream(synthesis_prompt):
                synthesis_result += response_chunk.text
                print(f"[DEBUG] Synthesizer: Raw token: '{response_chunk.text}'")
            
            print(f"[DEBUG] Synthesizer: Raw LLM response length: {len(synthesis_result)}")
            print(f"[DEBUG] Synthesizer: Raw response preview: {synthesis_result[:200]}...")
            
            print(f"[DEBUG] Synthesizer: LLM call completed, received {event_count} events")
            print(f"[DEBUG] Synthesizer: Final synthesis result length: {len(synthesis_result)}")
            
            if synthesis_result.strip():
                print(f"[DEBUG] Synthesizer: Generated {len(synthesis_result)} char synthesis")
                print(f"[DEBUG] Synthesizer: Preview: {synthesis_result[:200]}...")
                return synthesis_result.strip()
            else:
                print("[WARNING] Synthesizer LLM returned empty response, using fallback")
                print(f"[WARNING] Raw synthesis result: '{synthesis_result}'")
                return self._fallback_synthesis(meaningful_outputs, state)
                
        except Exception as e:
            print(f"[ERROR] Synthesizer LLM call failed: {e}, using fallback")
            import traceback
            print(f"[ERROR] Synthesizer traceback: {traceback.format_exc()}")
            return self._fallback_synthesis(meaningful_outputs, state)
    
    def _fallback_synthesis(self, meaningful_outputs, state):
        """Fallback synthesis when LLM synthesis fails"""
        final_parts = ["Based on the multi-agent analysis:\n"]
        
        for output_data in meaningful_outputs:
            agent_title = output_data['agent'].replace('_', ' ').replace('Agent', ' Agent').strip().title()
            final_parts.append(f"## {agent_title} Perspective")
            final_parts.append(output_data['response'])
            final_parts.append("")  # Add spacing
        
        # Add document sources if any
        all_docs = []
        for output in state["agent_outputs"].values():
            if output.get("documents"):
                all_docs.extend(output["documents"])
        
        if all_docs:
            final_parts.append("---\n")
            final_parts.append("📚 **Sources:**")
            for doc in all_docs[:5]:  # Show up to 5 sources
                final_parts.append(f"- {doc.get('source', 'Unknown source')}")
        
        return "\n".join(final_parts)
    
    async def _rag_stream(self, query: str):
        """Stream RAG responses"""
        # Import lazily to avoid circular import
        from app.langchain.service import rag_answer
        import json
        # Use the existing rag_answer function
        try:
            for chunk in rag_answer(query, thinking=False, stream=True, conversation_id=self.conversation_id):
                # rag_answer yields JSON strings, we need to parse them
                if isinstance(chunk, str):
                    try:
                        # Remove trailing newline and parse JSON
                        chunk_data = json.loads(chunk.strip())
                        yield chunk_data
                    except json.JSONDecodeError:
                        # If it's not valid JSON, log and skip
                        print(f"[WARNING] Could not parse chunk as JSON: {chunk[:100]}...")
                        # Don't yield unparseable chunks
                        continue
                else:
                    # Already a dict, yield as-is
                    yield chunk
        except Exception as e:
            print(f"[ERROR] RAG stream error: {str(e)}")
            yield {"error": str(e)}
    
    def _generate_performance_summary(self, state: AgentState) -> Dict[str, Any]:
        """Generate a performance summary from agent metrics"""
        summary = {
            "total_agents": len(state["agent_metrics"]),
            "completed_agents": 0,
            "failed_agents": 0,
            "total_processing_time": state["metadata"].get("total_duration", 0),
            "agents_performance": {}
        }
        
        # Analyze each agent's performance
        for agent_name, metrics in state["agent_metrics"].items():
            if metrics["status"] == "completed":
                summary["completed_agents"] += 1
            elif metrics["status"] == "failed":
                summary["failed_agents"] += 1
            
            summary["agents_performance"][agent_name] = {
                "status": metrics["status"],
                "duration": metrics.get("duration", 0),
                "start_time": metrics.get("start_time"),
                "end_time": metrics.get("end_time")
            }
        
        # Calculate average processing time for completed agents
        completed_durations = [
            metrics.get("duration", 0) 
            for metrics in state["agent_metrics"].values() 
            if metrics["status"] == "completed" and metrics.get("duration")
        ]
        
        if completed_durations:
            summary["average_agent_duration"] = sum(completed_durations) / len(completed_durations)
        else:
            summary["average_agent_duration"] = 0
        
        # Find slowest and fastest agents
        if completed_durations:
            agent_durations = [
                (name, metrics.get("duration", 0))
                for name, metrics in state["agent_metrics"].items()
                if metrics["status"] == "completed" and metrics.get("duration")
            ]
            agent_durations.sort(key=lambda x: x[1])
            
            if agent_durations:
                summary["fastest_agent"] = {
                    "name": agent_durations[0][0],
                    "duration": agent_durations[0][1]
                }
                summary["slowest_agent"] = {
                    "name": agent_durations[-1][0],
                    "duration": agent_durations[-1][1]
                }
        
        return summary
    
    async def stream_events(
        self, 
        query: str, 
        selected_agents: Optional[List[str]] = None,
        max_iterations: int = 10,
        conversation_history: Optional[List[Dict]] = None
    ):
        """Stream events from multi-agent processing"""
        # Add execution ID to prevent duplicate processing
        execution_id = str(uuid.uuid4())[:8]
        print(f"[DEBUG] Starting multi-agent execution {execution_id} for query: {query[:100]}...")
        
        try:
            # Initialize state
            state = AgentState(
                query=query,
                conversation_id=self.conversation_id,
                messages=[HumanMessage(content=query)],
                routing_decision={},
                agent_outputs={},
                tools_used=[],
                documents_retrieved=[],
                final_response="",
                metadata={
                    "start_time": datetime.now().isoformat(),
                    "mode": "multi_agent"
                },
                error=None,
                # Inter-agent communication fields
                agent_messages=[],
                pending_requests={},
                agent_conversations=[],
                # Collaboration patterns
                execution_pattern="parallel",
                agent_dependencies={},
                execution_order=[],
                # Performance tracking
                agent_metrics={}
            )
            
            # Ensure cache is fresh before execution
            from app.core.langgraph_agents_cache import get_cache_status, validate_and_warm_cache
            cache_status = get_cache_status()
            if not cache_status.get("cache_exists") or cache_status.get("agent_count", 0) == 0:
                print("[WARNING] Cache not ready during execution, warming cache...")
                validate_and_warm_cache()
                # Refresh agents list after cache warming
                self.agents = get_langgraph_agents()
                print(f"[DEBUG] Refreshed agents after cache warming: {len(self.agents)} agents")
            
            # Step 1: Router
            router_start_time = datetime.now()
            yield {
                "type": "agent_start",
                "agent": "router",
                "content": "Analyzing query...",
                "avatar": self.agent_avatars.get("router", "🧭"),
                "description": self.agent_descriptions.get("router", ""),
                "start_time": router_start_time.isoformat()
            }
            
            routing = await self._router_agent(query, conversation_history)
            state["routing_decision"] = routing
            
            router_end_time = datetime.now()
            router_duration = (router_end_time - router_start_time).total_seconds()
            yield {
                "type": "agent_complete",
                "agent": "router",
                "content": routing["reasoning"],
                "routing": routing,
                "avatar": self.agent_avatars.get("router", "🧭"),
                "description": self.agent_descriptions.get("router", ""),
                "duration": router_duration,
                "end_time": router_end_time.isoformat()
            }
            
# Removed large generation redirect - agents will handle large tasks themselves
            
            # Step 2: Determine collaboration pattern
            agents_to_run = selected_agents or routing["agents"]
            print(f"[DEBUG] Agents to run: {agents_to_run}")
            
            # Use collaboration pattern from routing (LLM-determined) or fallback
            if "collaboration_pattern" in routing:
                # Use LLM-determined pattern
                state["execution_pattern"] = routing["collaboration_pattern"]
                state["execution_order"] = routing.get("order", agents_to_run)
                state["agent_dependencies"] = routing.get("dependencies", {})
                print(f"[DEBUG] Using LLM-determined collaboration pattern: {state['execution_pattern']}")
            else:
                # Fallback to determine pattern
                collaboration = self.determine_collaboration_pattern(query, agents_to_run)
                state["execution_pattern"] = collaboration["pattern"]
                state["agent_dependencies"] = collaboration.get("dependencies", {})
                state["execution_order"] = collaboration.get("order", agents_to_run)
                print(f"[DEBUG] Using fallback collaboration pattern: {state['execution_pattern']}")
            
            # Override to sequential for MacBook resource management
            if len(agents_to_run) > 1:
                print(f"[DEBUG] Enforcing sequential execution for MacBook stability with {len(agents_to_run)} agents")
                state["execution_pattern"] = "sequential"
            
            # Yield collaboration pattern info
            yield {
                "type": "collaboration_pattern",
                "pattern": state["execution_pattern"],
                "dependencies": state.get("agent_dependencies", {}),
                "order": state.get("execution_order", [])
            }
            
            # Create async tasks for all agents
            agent_tasks = []
            agent_start_times = {}  # Track start times for each agent
            
            # Initialize performance metrics
            for agent_name in agents_to_run:
                state["agent_metrics"][agent_name] = {
                    "start_time": None,
                    "end_time": None,
                    "duration": None,
                    "tokens_generated": 0,
                    "messages_sent": 0,
                    "messages_received": 0,
                    "status": "pending"
                }
            
            for agent_name in agents_to_run:
                agent_start_times[agent_name] = datetime.now()
                state["agent_metrics"][agent_name]["start_time"] = agent_start_times[agent_name].isoformat()
                state["agent_metrics"][agent_name]["status"] = "running"
                
                yield {
                    "type": "agent_start",
                    "agent": agent_name,
                    "content": f"Starting {agent_name}...",
                    "avatar": self.agent_avatars.get(agent_name, "🤖"),
                    "description": self.agent_descriptions.get(agent_name, ""),
                    "start_time": agent_start_times[agent_name].isoformat(),
                    "pattern": state["execution_pattern"]
                }
                
                # REMOVED HARDCODED AGENTS - All agents now load from database
                # Always use dynamic execution to avoid maintenance issues
                use_dynamic_only = True
                
                if False:  # Disabled hardcoded agents
                    # Use existing hardcoded implementation
                    if agent_name in ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]:
                        # Streaming agents
                        print(f"[DEBUG] Adding hardcoded streaming agent: {agent_name}")
                        agent_tasks.append((agent_name, hardcoded_agents[agent_name](state)))
                    else:
                        # Non-streaming agents
                        print(f"[DEBUG] Executing hardcoded non-streaming agent: {agent_name}")
                        result = await hardcoded_agents[agent_name](state)
                        state["agent_outputs"][agent_name] = result
                        end_time = datetime.now()
                        duration = (end_time - agent_start_times[agent_name]).total_seconds()
                        state["agent_metrics"][agent_name]["end_time"] = end_time.isoformat()
                        state["agent_metrics"][agent_name]["duration"] = duration
                        state["agent_metrics"][agent_name]["status"] = "completed"
                        yield {
                            "type": "agent_complete",
                            "agent": agent_name,
                            "content": result.get("response", ""),
                            "avatar": self.agent_avatars.get(agent_name, "🤖"),
                            "description": self.agent_descriptions.get(agent_name, ""),
                            "duration": duration,
                            "end_time": end_time.isoformat()
                        }
                else:
                    # Use dynamic agent execution for any other agent
                    print(f"[DEBUG] FIXED: Using dynamic execution for agent: {agent_name}")
                    
                    async def dynamic_agent_wrapper(agent_name_local=agent_name):
                        print(f"[DEBUG] dynamic_agent_wrapper called for {agent_name_local}")
                        from app.langchain.dynamic_agent_system import agent_instance_pool
                        from app.core.langgraph_agents_cache import get_agent_by_name
                        
                        agent_data = get_agent_by_name(agent_name_local)
                        print(f"[DEBUG] Agent data found: {agent_data is not None}")
                        if not agent_data:
                            # Get list of available agents for debugging
                            from app.core.langgraph_agents_cache import get_active_agents
                            available_agents = list(get_active_agents().keys())
                            print(f"[ERROR] Agent {agent_name_local} not found in system")
                            print(f"[ERROR] Available agents: {available_agents}")
                            
                            # Try strict case-insensitive matching and exact synonym matching only
                            possible_matches = []
                            agent_name_lower = agent_name_local.lower().strip()
                            
                            # First try exact case-insensitive match
                            for available_agent in available_agents:
                                if agent_name_lower == available_agent.lower().strip():
                                    possible_matches.append(available_agent)
                                    break
                            
                            # If no exact match, try known synonyms/aliases only
                            if not possible_matches:
                                agent_synonyms = {
                                    # Exact case variations
                                    "corporate strategist": "Corporate Strategist",
                                    "business analyst": "BizAnalyst Agent", 
                                    "researcher": "Researcher Agent",
                                    "ceo agent": "CEO Agent",
                                    "financial analyst": "financial Analyst",
                                    "bizanalyst agent": "BizAnalyst Agent",
                                    # Handle router mistakes
                                    "researcher agent": "Researcher Agent",
                                    "document researcher": "Document Researcher",
                                    "sales strategist": "Sales Strategist",
                                    "technical architect": "Technical Architect",
                                    "service delivery manager": "Service Delivery Manager"
                                }
                                
                                if agent_name_lower in agent_synonyms:
                                    target_name = agent_synonyms[agent_name_lower]
                                    if target_name in available_agents:
                                        possible_matches.append(target_name)
                            
                            if possible_matches:
                                suggested_agent = possible_matches[0]
                                print(f"[INFO] Exact match found: {agent_name_local} → {suggested_agent}")
                                agent_data = get_agent_by_name(suggested_agent)
                                agent_name_local = suggested_agent
                            else:
                                print(f"[ERROR] No exact match found for '{agent_name_local}' in available agents: {list(available_agents.keys())[:5]}...")
                                # Show available agents that might be similar for debugging
                                similar_agents = [a for a in available_agents.keys() if any(word in a.lower() for word in agent_name_local.lower().split())]
                                if similar_agents:
                                    print(f"[INFO] Available agents with similar words: {similar_agents[:3]}")
                                # DO NOT auto-select - let it fail cleanly
                            
                            if not agent_data:
                                print(f"[ERROR] Agent {agent_name_local} not found, skipping this agent")
                                yield {
                                    "type": "agent_error",
                                    "agent": agent_name_local,
                                    "error": f"Agent {agent_name_local} not found in system. Available: {available_agents}"
                                }
                                # Mark as failed and continue to next agent instead of returning
                                state["agent_metrics"][agent_name_local]["status"] = "failed"
                                state["agent_metrics"][agent_name_local]["end_time"] = datetime.now().isoformat()
                                return  # Return from this specific wrapper, not the main function
                        
                        dynamic_system = await agent_instance_pool.get_or_create_instance(trace=self.trace)
                        try:
                            start_time = agent_start_times[agent_name_local]
                        
                            # Build context including conversation history
                            agent_context = self._extract_query_context(state["query"])
                            if conversation_history:
                                agent_context["conversation_history"] = conversation_history
                            
                            # CRITICAL FIX: Add tool information to agent context
                            # This ensures tools are available when using DynamicMultiAgentSystem execution path
                            if agent_data and agent_data.get("tools"):
                                agent_context["available_tools"] = agent_data["tools"]
                                print(f"[DEBUG] Added {len(agent_data['tools'])} tools to context for {agent_name_local}: {agent_data['tools']}")
                            else:
                                print(f"[DEBUG] No tools found for {agent_name_local} in agent_data")
                            
                            # CRITICAL FIX: Check state at execution time, not at wrapper creation time
                            # This ensures we get the previous agent response that was set during sequential execution
                            print(f"[DEBUG] Checking for previous_agent_response in state for {agent_name_local}")
                            print(f"[DEBUG] State keys at execution time: {list(state.keys())}")
                            
                            # Add previous agent outputs for sequential execution
                            if "previous_agent_response" in state:
                                # Convert single response to list format for compatibility
                                agent_context["previous_outputs"] = [{
                                    "agent": state["previous_agent_response"]["agent"],
                                    "content": state["previous_agent_response"]["response"],
                                    "output": state["previous_agent_response"]["response"]  # For compatibility
                                }]
                                print(f"[DEBUG] SUCCESS! Added previous agent response to context for {agent_name_local}")
                                print(f"[DEBUG] Previous agent: {state['previous_agent_response']['agent']}")
                                print(f"[DEBUG] Previous response preview: {state['previous_agent_response']['response'][:200]}...")
                            else:
                                print(f"[DEBUG] No previous_agent_response found in state for {agent_name_local}")
                            
                            # Also check if there are multiple previous outputs stored elsewhere
                            if "agent_outputs" in state and len(state["agent_outputs"]) > 0:
                                # Collect all previous agent outputs for this sequential execution
                                prev_outputs = []
                                for prev_agent, prev_output in state["agent_outputs"].items():
                                    if prev_agent != agent_name_local and prev_output.get("response"):
                                        response_content = prev_output["response"]
                                        # Handle thinking tags in stored outputs
                                        import re
                                        if "<think>" in response_content:
                                            print(f"[DEBUG] Found thinking tags in {prev_agent}'s stored output")
                                        
                                        prev_outputs.append({
                                            "agent": prev_agent,
                                            "content": response_content,
                                            "output": response_content
                                        })
                                if prev_outputs and "previous_outputs" not in agent_context:
                                    agent_context["previous_outputs"] = prev_outputs
                                    print(f"[DEBUG] Added {len(prev_outputs)} previous agent outputs to context for {agent_name_local}")
                            
                            print(f"[DEBUG] dynamic_agent_wrapper: About to call execute_agent for {agent_name_local}")
                            event_count = 0
                            async for event in dynamic_system.execute_agent(
                                agent_name_local,
                                agent_data,
                                state["query"],
                                context=agent_context
                            ):
                                event_count += 1
                                # Removed excessive event counting debug logs
                                if event.get("type") == "agent_complete":
                                    end_time = datetime.now()
                                    duration = (end_time - start_time).total_seconds()
                                    
                                    # Store output
                                    content = event.get("content", "")
                                    print(f"[DEBUG] Agent {agent_name_local} completed with {len(content)} chars")
                                    state["agent_outputs"][agent_name_local] = {
                                        "response": content,
                                        "duration": duration
                                    }
                                    
                                    # Update metrics
                                    state["agent_metrics"][agent_name_local]["end_time"] = end_time.isoformat()
                                    state["agent_metrics"][agent_name_local]["duration"] = duration
                                    state["agent_metrics"][agent_name_local]["status"] = "completed"
                                    
                                    # Add duration to event
                                    event["duration"] = duration
                                    event["end_time"] = end_time.isoformat()
                                
                                yield event
                        
                            print(f"[DEBUG] dynamic_agent_wrapper: Finished processing {event_count} events for {agent_name_local}")
                        finally:
                            # Always release the instance back to the pool
                            await agent_instance_pool.release_instance(dynamic_system)
                    
                    # Add to streaming tasks
                    print(f"[DEBUG] Adding dynamic agent task for {agent_name} to agent_tasks")
                    agent_tasks.append((agent_name, dynamic_agent_wrapper()))
                    print(f"[DEBUG] Agent tasks now contains {len(agent_tasks)} tasks: {[task[0] for task in agent_tasks]}")
            
            # Debug: Print final agent_tasks before execution
            print(f"[DEBUG] FINAL: agent_tasks contains {len(agent_tasks)} tasks total")
            print(f"[DEBUG] FINAL: Task names: {[task[0] for task in agent_tasks]}")
            print(f"[DEBUG] FINAL: agents_to_run was: {agents_to_run}")
            
            # Check for missing agents
            task_names = {task[0] for task in agent_tasks}
            missing_from_tasks = [agent for agent in agents_to_run if agent not in task_names]
            if missing_from_tasks:
                print(f"[WARNING] Agents in agents_to_run but missing from agent_tasks: {missing_from_tasks}")
                # Mark missing agents as failed in metrics for tracking
                for missing_agent in missing_from_tasks:
                    if missing_agent in state["agent_metrics"]:
                        state["agent_metrics"][missing_agent]["status"] = "failed"
                        state["agent_metrics"][missing_agent]["end_time"] = datetime.now().isoformat()
                        print(f"[DEBUG] Marked {missing_agent} as failed in metrics")
            
            # Process agents according to collaboration pattern
            if agent_tasks:
                from app.langchain.collaboration_executor import CollaborationExecutor
                executor = CollaborationExecutor()
                
                print(f"[DEBUG] Executing {len(agent_tasks)} agents with {state['execution_pattern']} pattern")
                print(f"[DEBUG] About to call executor.execute_agents with tasks: {[task[0] for task in agent_tasks]}")
                
                # Debug: Check if agent_tasks contains valid generators
                for task_name, task_gen in agent_tasks:
                    print(f"[DEBUG] Task {task_name} has generator type: {type(task_gen)}")
                
                # Execute agents according to collaboration pattern
                event_count = 0
                async for event in executor.execute_agents(
                    pattern=state["execution_pattern"],
                    agent_tasks=agent_tasks,
                    state=state,
                    agent_start_times=agent_start_times
                ):
                    event_count += 1
                    # if event_count % 20 == 0:
                    #     print(f"[DEBUG] Yielded {event_count} events from collaboration executor")
                    yield event
                
                print(f"[DEBUG] Collaboration executor finished, yielded {event_count} total events")
            
            # Debug: Check what we have before synthesizing
            print(f"[DEBUG] Before synthesis - agent_outputs keys: {list(state['agent_outputs'].keys())}")
            for agent_name, output in state["agent_outputs"].items():
                if output and output.get("response"):
                    print(f"[DEBUG] Agent {agent_name} has response of length: {len(output['response'])}")
            
            # Step 3: Synthesize
            synthesizer_start_time = datetime.now()
            yield {
                "type": "agent_start",
                "agent": "synthesizer",
                "content": "Synthesizing final response...",
                "avatar": self.agent_avatars.get("synthesizer", "🎯"),
                "description": self.agent_descriptions.get("synthesizer", ""),
                "start_time": synthesizer_start_time.isoformat()
            }
            
            # Use same execution path as other agents
            from app.langchain.dynamic_agent_system import agent_instance_pool
            dynamic_system = await agent_instance_pool.get_or_create_instance(trace=self.trace)
            try:
                synthesizer_agent_data = self.agents.get("synthesizer")
                if not synthesizer_agent_data:
                    print("[ERROR] Synthesizer agent not found in agents")
                    final_response = self._fallback_synthesis([], state)
                else:
                    # Build context for synthesizer with all agent outputs
                    synthesizer_context = {
                        "agent_outputs": state["agent_outputs"],
                        "previous_outputs": list(state["agent_outputs"].values())
                    }
                    
                    print(f"[DEBUG] Executing synthesizer via standard agent path")
                    synthesizer_response = ""
                    async for event in dynamic_system.execute_agent(
                        "synthesizer",
                        synthesizer_agent_data, 
                        state["query"],
                        context=synthesizer_context
                    ):
                        # Yield all synthesizer events to frontend for streaming display
                        yield event
                        
                        if event.get("type") == "agent_complete":
                            synthesizer_response = event.get("content", "")
                            print(f"[DEBUG] Synthesizer completed via standard path: {len(synthesizer_response)} chars")
                
                final_response = synthesizer_response if synthesizer_response.strip() else self._fallback_synthesis([], state)
                state["final_response"] = final_response
            finally:
                # Always release the synthesizer instance back to the pool
                await agent_instance_pool.release_instance(dynamic_system)
            
            synthesizer_end_time = datetime.now()
            synthesizer_duration = (synthesizer_end_time - synthesizer_start_time).total_seconds()
            
            # Update metadata with end time and performance summary
            state["metadata"]["end_time"] = datetime.now().isoformat()
            state["metadata"]["total_duration"] = (datetime.fromisoformat(state["metadata"]["end_time"]) - 
                                                    datetime.fromisoformat(state["metadata"]["start_time"])).total_seconds()
            
            # Generate performance summary
            performance_summary = self._generate_performance_summary(state)
            
            print(f"[DEBUG] Yielding final_response event with response length: {len(final_response)}")
            print(f"[DEBUG] Final response preview (first 200 chars): {final_response[:200]!r}")
            yield {
                "type": "final_response",
                "response": final_response,
                "metadata": state["metadata"],
                "routing": state["routing_decision"],
                "agent_outputs": state["agent_outputs"],
                "synthesizer_duration": synthesizer_duration,
                "agent_metrics": state["agent_metrics"],
                "performance_summary": performance_summary
            }
            print(f"[DEBUG] Multi-agent execution {execution_id} completed successfully with {len(state['agent_outputs'])} agent responses")
            
        except Exception as e:
            print(f"[ERROR] Exception in multi-agent execution {execution_id}: {str(e)}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            yield {
                "type": "error",
                "error": str(e),
                "execution_id": execution_id
            }
    
    async def stream_large_generation_events(
        self, 
        query: str,
        target_count: int = 100,
        chunk_size: Optional[int] = None,
        conversation_history: Optional[List[Dict]] = None
    ):
        """
        Stream events for large generation tasks with intelligent chunking
        that transcends context limits while maintaining continuity
        
        Args:
            query: The generation task description
            target_count: Total number of items to generate
            chunk_size: Optional fixed chunk size, otherwise auto-calculated
            conversation_history: Previous conversation for context
            
        Yields:
            Events tracking decomposition, progress, and final results
        """
        try:
            print(f"[LARGE_GEN] Starting large generation: {target_count} items for query: {query[:100]}...")
            
            # Step 1: Task Decomposition
            yield {
                "type": "decomposition_started",
                "query": query,
                "target_count": target_count,
                "timestamp": datetime.now().isoformat()
            }
            
            decomposer = TaskDecomposer()
            task_chunks = await decomposer.decompose_large_task(
                query=query,
                target_count=target_count,
                chunk_size=chunk_size
            )
            
            yield {
                "type": "task_decomposed",
                "total_chunks": len(task_chunks),
                "chunk_size": task_chunks[0].chunk_size if task_chunks else 0,
                "estimated_duration": len(task_chunks) * 45,  # seconds per chunk estimate
                "chunks_info": [
                    {
                        "chunk_number": chunk.chunk_number,
                        "items_range": f"{chunk.start_index}-{chunk.end_index}",
                        "chunk_size": chunk.chunk_size
                    }
                    for chunk in task_chunks
                ]
            }
            
            # Step 2: Execute chunks with Redis-based continuity management
            continuity_manager = RedisContinuityManager(session_id=self.conversation_id)
            
            yield {
                "type": "execution_started",
                "session_id": continuity_manager.session_id,
                "strategy": "chunked_continuation"
            }
            
            # Stream the chunked execution
            async for event in continuity_manager.execute_chunked_task(
                chunks=task_chunks,
                agent_name="continuation_agent"
            ):
                # Forward all events from the continuity manager
                yield event
                
                # If task completed, do final quality check and synthesis
                if event.get("type") == "task_completed":
                    final_results = event.get("final_results", [])
                    
                    # Step 3: Quality assurance and final synthesis
                    yield {
                        "type": "quality_check_started",
                        "items_to_validate": len(final_results)
                    }
                    
                    # Perform quality validation
                    quality_report = await self._validate_generation_quality(
                        original_query=query,
                        generated_items=final_results,
                        target_count=target_count
                    )
                    
                    yield {
                        "type": "quality_check_completed",
                        "quality_score": quality_report.get("overall_score", 0.8),
                        "validation_details": quality_report,
                        "recommendations": quality_report.get("recommendations", [])
                    }
                    
                    # Final completion with enhanced metadata
                    yield {
                        "type": "large_generation_completed",
                        "session_id": continuity_manager.session_id,
                        "original_query": query,
                        "target_count": target_count,
                        "actual_count": len(final_results),
                        "completion_rate": len(final_results) / target_count if target_count > 0 else 1.0,
                        "final_results": final_results,
                        "execution_summary": event.get("summary", {}),
                        "quality_report": quality_report,
                        "metadata": {
                            "chunks_executed": len(task_chunks),
                            "total_execution_time": event.get("total_execution_time", 0),
                            "average_items_per_chunk": len(final_results) / len(task_chunks) if task_chunks else 0,
                            "success_rate": quality_report.get("overall_score", 0.8)
                        }
                    }
                    
        except Exception as e:
            print(f"[ERROR] Large generation failed: {str(e)}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            yield {
                "type": "large_generation_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _validate_generation_quality(
        self, 
        original_query: str, 
        generated_items: List[str], 
        target_count: int
    ) -> Dict[str, Any]:
        """
        Validate the quality of generated content across all chunks
        
        Args:
            original_query: Original generation task
            generated_items: All generated items from all chunks
            target_count: Target number of items
            
        Returns:
            Quality validation report
        """
        try:
            print(f"[QUALITY] Validating {len(generated_items)} generated items")
            
            # Basic validation metrics
            quality_metrics = {
                "count_accuracy": len(generated_items) / target_count if target_count > 0 else 1.0,
                "content_consistency": 0.8,  # Default assumption
                "numbering_correctness": 0.9,  # Default assumption
                "format_consistency": 0.85,  # Default assumption
                "overall_score": 0.0
            }
            
            # Check for proper numbering if items appear to be numbered
            numbering_issues = []
            if generated_items:
                import re
                expected_number = 1
                
                for i, item in enumerate(generated_items):
                    # Check if item starts with a number
                    match = re.match(r'^(\d+)[\.\)]\s*', item.strip())
                    if match:
                        actual_number = int(match.group(1))
                        if actual_number != expected_number:
                            numbering_issues.append(f"Item {i+1}: expected {expected_number}, got {actual_number}")
                        expected_number += 1
                    else:
                        # If first few items have numbers, this might be an issue
                        if i < 3 and any(re.match(r'^\d+[\.\)]\s*', other_item.strip()) for other_item in generated_items[:5]):
                            numbering_issues.append(f"Item {i+1}: missing number")
            
            # Adjust numbering score based on issues found
            if numbering_issues:
                quality_metrics["numbering_correctness"] = max(0.0, 1.0 - (len(numbering_issues) / len(generated_items)))
            
            # Check content length consistency (items should be roughly similar length)
            if len(generated_items) > 1:
                lengths = [len(item) for item in generated_items]
                avg_length = sum(lengths) / len(lengths)
                length_variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
                
                # If variance is low, content is consistent
                if length_variance < avg_length * 0.5:  # Low variance
                    quality_metrics["content_consistency"] = 0.9
                elif length_variance < avg_length * 1.0:  # Medium variance
                    quality_metrics["content_consistency"] = 0.7
                else:  # High variance
                    quality_metrics["content_consistency"] = 0.5
            
            # Check format consistency (similar structure across items)
            format_patterns = []
            for item in generated_items[:10]:  # Sample first 10 items
                # Check if starts with number, has certain punctuation patterns, etc.
                patterns = []
                if re.match(r'^\d+[\.\)]', item.strip()):
                    patterns.append("numbered")
                if ':' in item:
                    patterns.append("has_colon")
                if '?' in item:
                    patterns.append("has_question")
                if len(item.split()) > 10:
                    patterns.append("long_form")
                format_patterns.append(patterns)
            
            # Calculate format consistency
            if format_patterns:
                common_patterns = set(format_patterns[0])
                for patterns in format_patterns[1:]:
                    common_patterns = common_patterns.intersection(set(patterns))
                
                format_consistency = len(common_patterns) / max(1, len(format_patterns[0]))
                quality_metrics["format_consistency"] = format_consistency
            
            # Calculate overall score
            quality_metrics["overall_score"] = (
                quality_metrics["count_accuracy"] * 0.3 +
                quality_metrics["content_consistency"] * 0.3 +
                quality_metrics["numbering_correctness"] * 0.2 +
                quality_metrics["format_consistency"] * 0.2
            )
            
            # Generate recommendations
            recommendations = []
            if quality_metrics["count_accuracy"] < 0.9:
                recommendations.append(f"Generated {len(generated_items)} items but target was {target_count}")
            if quality_metrics["numbering_correctness"] < 0.8:
                recommendations.append(f"Numbering issues detected: {len(numbering_issues)} problems found")
            if quality_metrics["content_consistency"] < 0.7:
                recommendations.append("Content length varies significantly across items")
            if quality_metrics["format_consistency"] < 0.7:
                recommendations.append("Inconsistent formatting across generated items")
            
            return {
                **quality_metrics,
                "total_items": len(generated_items),
                "target_items": target_count,
                "numbering_issues": numbering_issues[:5],  # Top 5 issues
                "recommendations": recommendations,
                "validation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[ERROR] Quality validation failed: {e}")
            return {
                "overall_score": 0.6,  # Default moderate score
                "error": str(e),
                "recommendations": ["Quality validation failed - manual review recommended"]
            }
    
    async def get_chunked_generation_progress(self, session_id: str) -> Dict[str, Any]:
        """Get progress for an ongoing chunked generation task"""
        # This could be enhanced to track active sessions
        return {
            "session_id": session_id,
            "status": "not_found",
            "message": "Session tracking not yet implemented"
        }
    
    async def resume_chunked_generation(
        self, 
        session_id: str, 
        from_chunk: int = 0
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Resume a chunked generation from a specific chunk"""
        # Implementation for resuming interrupted generation
        yield {
            "type": "resume_not_implemented",
            "message": "Resume functionality will be implemented in the progress recovery system"
        }
    
    async def execute_single_agent(
        self,
        agent_name: str,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        previous_outputs: Optional[List[Dict]] = None,
        agent_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a single agent and return its output"""
        import time
        start_time = time.time()
        
        # Create agent span for tracing if trace is available
        from app.core.langfuse_integration import get_tracer
        tracer = get_tracer()
        agent_span = None
        agent_generation = None
        
        if self.trace and tracer.is_enabled():
            # Create agent-level span to contain generation and tool spans
            # Check if self.trace is actually a span (has .span method) or a trace
            if hasattr(self.trace, 'span') and callable(getattr(self.trace, 'span')):
                # self.trace is a span, create agent span as child
                agent_span = self.trace.span(
                    name=f"agent-{agent_name}",
                    metadata={
                        "conversation_id": self.conversation_id,
                        "agent_role": agent_data.get("role", ""),
                        "tools_available": agent_data.get("tools", []),
                        "has_previous_outputs": bool(previous_outputs),
                        "has_conversation_history": bool(conversation_history)
                    }
                )
            else:
                # self.trace is a trace, use tracer method
                agent_span = tracer.create_span(
                    self.trace,
                    name=f"agent-{agent_name}",
                    metadata={
                        "conversation_id": self.conversation_id,
                        "agent_role": agent_data.get("role", ""),
                        "tools_available": agent_data.get("tools", []),
                        "has_previous_outputs": bool(previous_outputs),
                        "has_conversation_history": bool(conversation_history)
                    }
                )
            
            # Store current agent span for tool execution
            self.current_agent_span = agent_span
            
            # Create generation within the agent span for the LLM response
            if agent_span:
                agent_generation = tracer.create_generation_with_usage(
                    trace=self.trace,
                    name=f"agent-{agent_name}-generation",
                    model=agent_data.get("config", {}).get("model", "qwen3:30b-a3b"),
                    input_text=query,
                    metadata={
                        "conversation_id": self.conversation_id,
                        "agent_role": agent_data.get("role", "")
                    },
                    parent_span=agent_span
                )
        
        # Initialize state for single agent execution
        state = AgentState(
            query=query,
            conversation_id=self.conversation_id,
            messages=[HumanMessage(content=query)],
            routing_decision={"agents": [agent_name]},
            agent_outputs={},
            tools_used=[],
            documents_retrieved=[],
            final_response="",
            metadata={
                "start_time": datetime.now().isoformat(),
                "mode": "single_agent",
                "previous_outputs": previous_outputs or [],
                "agent_config": agent_config  # Pass the full agent config
            },
            error=None,
            agent_messages=[],
            pending_requests={},
            agent_conversations=[],
            execution_pattern="single",
            agent_dependencies={},
            execution_order=[agent_name],
            agent_metrics={}
        )
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "user":
                    state["messages"].append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    state["messages"].append(AIMessage(content=msg["content"]))
        
        # Execute the agent
        try:
            # Get agent function
            agent_func = self._get_agent_function(agent_name)
            if not agent_func:
                raise ValueError(f"Agent {agent_name} not found")
            
            # Execute agent
            if asyncio.iscoroutinefunction(agent_func):
                result = await agent_func(state)
            else:
                result = agent_func(state)
            
            # Handle streaming vs dict response
            if hasattr(result, '__aiter__'):
                # Collect streaming response
                content = ""
                reasoning = ""
                async for chunk in result:
                    if isinstance(chunk, dict):
                        if "content" in chunk:
                            content += chunk["content"]
                        if "reasoning" in chunk:
                            reasoning = chunk.get("reasoning", reasoning)
                result = {"response": content, "reasoning": reasoning}
            
            execution_time = time.time() - start_time
            
            # Debug logging
            print(f"[DEBUG] execute_single_agent {agent_name} - result type: {type(result)}")
            print(f"[DEBUG] execute_single_agent {agent_name} - result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            print(f"[DEBUG] execute_single_agent {agent_name} - response length: {len(result.get('response', '')) if isinstance(result, dict) else 0}")
            print(f"[DEBUG] execute_single_agent {agent_name} - tools_used from result: {result.get('tools_used', []) if isinstance(result, dict) else []}")
            print(f"[DEBUG] execute_single_agent {agent_name} - tools_used from state: {state.get('tools_used', [])}")
            
            return_value = {
                "agent": agent_name,
                "content": result.get("response", ""),
                "reasoning": result.get("reasoning", ""),
                "execution_time": execution_time,
                "documents": result.get("documents", []),
                "tools_used": result.get("tools_used", state.get("tools_used", []))
            }
            
            # End agent generation and span with success
            if agent_generation:
                # Estimate token usage for cost calculation
                usage = tracer.estimate_token_usage(query, return_value.get("content", ""))
                
                agent_generation.end(
                    output=return_value.get("content", ""),
                    usage_details=usage,
                    metadata={
                        "success": True,
                        "execution_time": execution_time,
                        "tools_used": return_value.get("tools_used", []),
                        "response_length": len(return_value.get("content", "")),
                        "conversation_id": self.conversation_id
                    }
                )
            
            # End agent span with success
            if agent_span:
                tracer.end_span_with_result(
                    agent_span, 
                    return_value, 
                    True, 
                    f"Agent {agent_name} completed successfully"
                )
                # Clear current agent span
                self.current_agent_span = None
            
            print(f"[DEBUG] execute_single_agent {agent_name} returning content length: {len(return_value['content'])}")
            return return_value
            
        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {str(e)}")
            
            error_result = {
                "agent": agent_name,
                "content": f"Error: {str(e)}",
                "reasoning": "Agent execution failed",
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
            
            # End agent generation with error
            if agent_generation:
                agent_generation.end(
                    output=f"Error: {str(e)}",
                    metadata={
                        "success": False,
                        "error": str(e),
                        "execution_time": time.time() - start_time,
                        "conversation_id": self.conversation_id
                    }
                )
            
            # End agent span with error
            if agent_span:
                tracer.end_span_with_result(
                    agent_span, 
                    None, 
                    False, 
                    f"Agent {agent_name} failed: {str(e)}"
                )
                # Clear current agent span
                self.current_agent_span = None
            
            return error_result
    
    async def execute_agents(
        self,
        query: str,
        selected_agents: List[str],
        conversation_history: Optional[List[Dict]] = None,
        execution_pattern: str = "parallel",
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Execute multiple agents with specified pattern"""
        # Collect all outputs
        all_outputs = []
        
        async for event in self.stream_events(
            query=query,
            selected_agents=selected_agents,
            conversation_history=conversation_history
        ):
            # Process streaming events
            if event.get("type") == "agent_complete":
                all_outputs.append({
                    "agent": event["agent"],
                    "output": event.get("content", ""),
                    "reasoning": event.get("reasoning", ""),
                    "execution_time": event.get("execution_time", 0)
                })
            elif event.get("type") == "synthesis_complete":
                final_output = event.get("content", "")
        
        return {
            "agent_outputs": all_outputs,
            "final_output": final_output if 'final_output' in locals() else "",
            "execution_pattern": execution_pattern
        }
    
    async def execute_hierarchical(
        self,
        query: str,
        lead_agent: str,
        subordinate_agents: List[str],
        conversation_history: Optional[List[Dict]] = None,
        lead_config: Optional[Dict[str, Any]] = None,
        subordinate_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Execute agents in hierarchical pattern with lead agent coordinating"""
        # First execute the lead agent
        lead_result = await self.execute_single_agent(
            agent_name=lead_agent,
            query=query,
            conversation_history=conversation_history,
            agent_config=lead_config
        )
        
        # Lead agent can delegate to subordinates
        all_outputs = [lead_result]
        
        # Execute subordinate agents based on lead's decision
        if subordinate_agents:
            # For simplicity, execute all subordinates with lead's output as context
            subordinate_query = f"Based on the lead agent's analysis: {lead_result['content']}\n\nOriginal query: {query}"
            
            for sub_agent in subordinate_agents:
                sub_config = subordinate_configs.get(sub_agent, {}) if subordinate_configs else {}
                sub_result = await self.execute_single_agent(
                    agent_name=sub_agent,
                    query=subordinate_query,
                    conversation_history=conversation_history,
                    previous_outputs=[lead_result],
                    agent_config=sub_config
                )
                all_outputs.append(sub_result)
        
        # Synthesize hierarchical results
        final_output = f"Lead Agent ({lead_agent}): {lead_result['content']}\n\n"
        
        if len(all_outputs) > 1:
            final_output += "Subordinate Agent Insights:\n"
            for output in all_outputs[1:]:
                final_output += f"\n{output['agent']}: {output['content']}\n"
        
        return {
            "agent_outputs": all_outputs,
            "final_output": final_output,
            "hierarchy": {
                "lead": lead_agent,
                "subordinates": subordinate_agents
            }
        }
    
    def _get_agent_function(self, agent_name: str):
        """Get the agent function by name"""
        # Map agent names to their functions - use role-based mapping for dynamic agents
        agent_functions = {
            "document_researcher": self._document_researcher_agent,
            "tool_executor": self._tool_executor_agent,
            "context_manager": self._context_manager_agent,
        }
        
        # Add dynamic agent mappings based on roles
        for agent_name, agent_config in self.agents.items():
            if agent_name not in agent_functions:
                role = agent_config.get('role', '').lower()
                if 'sales' in role and 'strategist' in role:
                    agent_functions[agent_name] = self._sales_strategist_agent
                elif 'technical' in role and 'architect' in role:
                    agent_functions[agent_name] = self._technical_architect_agent
                elif 'financial' in role and 'analyst' in role:
                    agent_functions[agent_name] = self._financial_analyst_agent
                elif 'service' in role and ('delivery' in role or 'manager' in role):
                    agent_functions[agent_name] = self._service_delivery_manager_agent
                # Add more role-based mappings as needed
        
        # Check predefined agents first
        if agent_name in agent_functions:
            return agent_functions[agent_name]
        
        # For dynamic agents, create a generic agent function
        from app.core.langgraph_agents_cache import get_agent_by_name
        agent_info = get_agent_by_name(agent_name)
        
        if agent_info:
            # Return an async lambda that creates a dynamic agent
            async def dynamic_agent_wrapper(state):
                return await self._dynamic_agent(agent_name, agent_info, state)
            return dynamic_agent_wrapper
        
        return None
    
    async def _execute_mcp_tool(self, tool_name: str, params: Dict[str, Any], agent_name: str, event_callback=None, agent_span=None) -> str:
        """Execute actual MCP tool"""
        # Create tool span for tracing if trace is available
        from app.core.langfuse_integration import get_tracer
        tracer = get_tracer()
        tool_span = None
        
        if self.trace and tracer.is_enabled():
            print(f"[DEBUG] Creating tool span for {tool_name} by agent {agent_name}")
            # Use passed agent_span or fall back to current_agent_span
            current_span = agent_span or getattr(self, 'current_agent_span', None)
            
            if current_span:
                # Create tool span as child of agent span
                tool_span = tracer.create_span(
                    self.trace,
                    name=f"tool-{tool_name}",
                    metadata={**params, "agent": agent_name},
                    parent_span=current_span
                )
            else:
                # Fallback to direct tool span under trace
                tool_span = tracer.create_tool_span(
                    self.trace, 
                    tool_name, 
                    {**params, "agent": agent_name}
                )
            print(f"[DEBUG] Tool span created: {tool_span is not None}, parent: {current_span is not None}")
        else:
            print(f"[DEBUG] No tool span created - trace: {self.trace is not None}, tracer enabled: {tracer.is_enabled()}")
        
        try:
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            from app.core.oauth_token_manager import OAuthTokenManager
            import json
            
            logger.info(f"Executing MCP tool '{tool_name}' for agent '{agent_name}' with params: {params}")
            
            # Yield tool start event if callback is provided
            if event_callback:
                await event_callback({
                    "type": "tool_start",
                    "agent": agent_name,
                    "tool": tool_name,
                    "parameters": params,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Get tool info
            enabled_tools = get_enabled_mcp_tools()
            tool_info = enabled_tools.get(tool_name)
            
            if not tool_info:
                error_msg = f"Error: Tool '{tool_name}' not found in enabled tools"
                if event_callback:
                    await event_callback({
                        "type": "tool_error",
                        "agent": agent_name,
                        "tool": tool_name,
                        "error": error_msg,
                        "timestamp": datetime.now().isoformat()
                    })
                # End tool span with error (early error case)
                if tool_span:
                    tracer.end_span_with_result(tool_span, None, False, error_msg)
                return error_msg
            
            # Check if this is a Gmail tool and inject OAuth credentials
            if "gmail" in tool_name.lower() or "email" in tool_name.lower():
                server_id = tool_info.get("server_id")
                logger.info(f"[OAUTH DEBUG] Tool '{tool_name}' server_id: {server_id}")
                
                if server_id:
                    oauth_token_manager = OAuthTokenManager()
                    oauth_creds = oauth_token_manager.get_valid_token(
                        server_id=server_id,
                        service_name="gmail"
                    )
                    
                    if oauth_creds and "google_access_token" not in params:
                        # Debug: Check what we have
                        logger.info(f"[OAUTH DEBUG] OAuth credentials found for {tool_name}")
                        logger.debug(f"[OAUTH DEBUG] Token expires at: {oauth_creds.get('expires_at', 'Unknown')}")
                        logger.debug(f"[OAUTH DEBUG] Has access_token: {bool(oauth_creds.get('access_token'))}")
                        logger.debug(f"[OAUTH DEBUG] Has refresh_token: {bool(oauth_creds.get('refresh_token'))}")
                        
                        params.update({
                            "google_access_token": oauth_creds.get("access_token", ""),
                            "google_refresh_token": oauth_creds.get("refresh_token", ""),
                            "google_client_id": oauth_creds.get("client_id", ""),
                            "google_client_secret": oauth_creds.get("client_secret", "")
                        })
                        logger.info(f"[OAUTH DEBUG] Successfully injected OAuth credentials for {tool_name}")
                    else:
                        logger.warning(f"[OAUTH DEBUG] No OAuth credentials found for {tool_name} - server_id: {server_id}")
                        if not oauth_creds:
                            logger.warning(f"[OAUTH DEBUG] OAuth token manager returned None")
                        elif "google_access_token" in params:
                            logger.warning(f"[OAUTH DEBUG] Params already contain google_access_token")
                else:
                    logger.warning(f"[OAUTH DEBUG] No server_id found for tool {tool_name}")
            
            # Check if this is a stdio-based tool
            endpoint = tool_info.get("endpoint", "")
            if endpoint.startswith("stdio://"):
                # Use stdio bridge for command-based servers
                from app.core.db import get_db_session, MCPServer
                from app.core.mcp_stdio_bridge import call_mcp_tool_via_stdio
                
                with get_db_session() as db:
                    try:
                        server = db.query(MCPServer).filter(MCPServer.id == tool_info["server_id"]).first()
                    except Exception as db_error:
                        logger.error(f"Database error while fetching server: {str(db_error)}")
                        return f"Database error: {str(db_error)}"
                    
                    if server and server.config_type == "command":
                        server_config = {
                            "command": server.command,
                            "args": server.args or []
                        }
                        
                        # Check for enhanced error handling configuration
                        use_enhanced_error_handling = True  # Enable by default for multi-agent
                        
                        if use_enhanced_error_handling:
                            # Use enhanced error handling with retry logic
                            from app.core.tool_error_handler import ToolErrorHandler, RetryConfig
                            
                            retry_config = RetryConfig(
                                max_retries=3,
                                base_delay=1.0,
                                max_delay=30.0
                            )
                            
                            handler = ToolErrorHandler(retry_config)
                            
                            # Wrap stdio call for enhanced handling
                            async def stdio_wrapper(tool_name, params, trace=None, _skip_span_creation=False):
                                return await call_mcp_tool_via_stdio(server_config, tool_name, params)
                            
                            # Get tool info for error handler
                            tool_error_info = {"server_id": server.id} if server else {}
                            
                            logger.info(f"[TOOL CALL] Executing {tool_name} with enhanced error handling: {json.dumps(params, indent=2)}")
                            result = await handler.execute_with_retry(
                                stdio_wrapper, tool_name, params, tool_error_info, trace=self.trace
                            )
                        else:
                            # Call the async function directly since we're already in async context
                            logger.info(f"[TOOL CALL] Executing {tool_name} with params: {json.dumps(params, indent=2)}")
                            result = await call_mcp_tool_via_stdio(server_config, tool_name, params)
                        
                        logger.info(f"[TOOL RESULT] {tool_name} returned: {json.dumps(result, indent=2) if isinstance(result, dict) else str(result)[:500]}")
                        
                        # Handle the response with enhanced error information
                        if isinstance(result, dict):
                            # Check for enhanced error format first
                            if result.get("error"):
                                error_msg = result["error"]
                                error_type = result.get("error_type", "unknown")
                                attempts = result.get("attempts", 1)
                                
                                if attempts > 1:
                                    logger.error(f"Tool {tool_name} failed after {attempts} attempts ({error_type}): {error_msg}")
                                        
                                    # Yield enhanced tool error event
                                    if event_callback:
                                        await event_callback({
                                            "type": "tool_error",
                                            "agent": agent_name,
                                            "tool": tool_name,
                                            "error": error_msg,
                                            "error_type": error_type,
                                            "attempts": attempts,
                                            "timestamp": datetime.now().isoformat()
                                        })
                                else:
                                    logger.error(f"Tool {tool_name} failed ({error_type}): {error_msg}")
                                    
                                    # Yield tool error event
                                    if event_callback:
                                        await event_callback({
                                            "type": "tool_error",
                                            "agent": agent_name,
                                            "tool": tool_name,
                                            "error": error_msg,
                                            "error_type": error_type,
                                            "timestamp": datetime.now().isoformat()
                                        })
                            
                            # Check content for error messages (MCP protocol format) - legacy format
                            elif result.get("content"):
                                error_msg = None
                                content = result["content"]
                                if isinstance(content, list) and len(content) > 0:
                                    first_content = content[0]
                                    if isinstance(first_content, dict) and first_content.get("text"):
                                        text = first_content["text"]
                                        # Check if this is an error message
                                        if any(err_keyword in text.lower() for err_keyword in ["error", "failed", "no access", "invalid"]):
                                            error_msg = text
                            
                            if error_msg:
                                logger.error(f"Tool execution error: {error_msg}")
                                
                                # Yield tool error event
                                if event_callback:
                                    await event_callback({
                                        "type": "tool_error",
                                        "agent": agent_name,
                                        "tool": tool_name,
                                        "error": error_msg,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                
                                # If it's an authentication error, provide helpful guidance
                                if any(auth_keyword in error_msg.lower() for auth_keyword in ["no access", "refresh token", "authentication", "unauthorized"]):
                                    auth_error = (
                                        f"Authentication Error: Gmail access is not properly configured. "
                                        f"Please ensure Gmail OAuth is set up in the MCP Servers settings. "
                                        f"Error details: {error_msg}"
                                    )
                                    # End tool span with authentication error
                                    if tool_span:
                                        tracer.end_span_with_result(tool_span, None, False, auth_error)
                                    return auth_error
                                
                                # End tool span with general error
                                if tool_span:
                                    tracer.end_span_with_result(tool_span, None, False, f"Error: {error_msg}")
                                return f"Error: {error_msg}"
                            else:
                                # Success - return the result
                                result_str = json.dumps(result, indent=2)
                                
                                # Yield tool success event
                                if event_callback:
                                    await event_callback({
                                        "type": "tool_result",
                                        "agent": agent_name,
                                        "tool": tool_name,
                                        "result": result,
                                        "success": True,
                                        "timestamp": datetime.now().isoformat()
                                    })
                                
                                # End tool span with success
                                if tool_span:
                                    print(f"[DEBUG] Ending tool span for {tool_name} with success")
                                    tracer.end_span_with_result(tool_span, result, True)
                                return result_str
                        else:
                            result_str = str(result)
                            
                            # Yield tool success event for string results
                            if event_callback:
                                await event_callback({
                                    "type": "tool_result",
                                    "agent": agent_name,
                                    "tool": tool_name,
                                    "result": result_str,
                                    "success": True,
                                    "timestamp": datetime.now().isoformat()
                                })
                            
                            # End tool span with success (string result)
                            if tool_span:
                                print(f"[DEBUG] Ending tool span for {tool_name} with success (string result)")
                                tracer.end_span_with_result(tool_span, result_str, True)
                            return result_str
                    else:
                        # For HTTP-based tools, make the request
                        # This would need to be implemented based on your HTTP tool execution logic
                        error_msg = f"HTTP tool execution not implemented for {tool_name}"
                        
                        if event_callback:
                            await event_callback({
                                "type": "tool_error",
                                "agent": agent_name,
                                "tool": tool_name,
                                "error": error_msg,
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        # End tool span with error (HTTP not implemented)
                        if tool_span:
                            tracer.end_span_with_result(tool_span, None, False, error_msg)
                        return error_msg
                
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Failed to execute MCP tool {tool_name}: {str(e)}", exc_info=True)
            
            # Yield tool error event
            if event_callback:
                await event_callback({
                    "type": "tool_error",
                    "agent": agent_name,
                    "tool": tool_name,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
            
            # End tool span with error (exception)
            if tool_span:
                tracer.end_span_with_result(tool_span, None, False, error_msg)
            return error_msg
    
    def _map_tool_parameters(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map common parameter mismatches to correct parameter names"""
        mapped_params = params.copy()
        
        # Google Search tool parameter mapping
        if 'search' in tool_name.lower():
            # Map common variations to 'query'
            if 'q' in mapped_params and 'query' not in mapped_params:
                mapped_params['query'] = mapped_params.pop('q')
                logger.info(f"[PARAM MAPPING] Mapped 'q' -> 'query' for {tool_name}")
            
            if 'search' in mapped_params and 'query' not in mapped_params:
                mapped_params['query'] = mapped_params.pop('search')
                logger.info(f"[PARAM MAPPING] Mapped 'search' -> 'query' for {tool_name}")
            
            if 'search_query' in mapped_params and 'query' not in mapped_params:
                mapped_params['query'] = mapped_params.pop('search_query')
                logger.info(f"[PARAM MAPPING] Mapped 'search_query' -> 'query' for {tool_name}")
            
            # Map 'num' to 'num_results'
            if 'num' in mapped_params and 'num_results' not in mapped_params:
                mapped_params['num_results'] = mapped_params.pop('num')
                logger.info(f"[PARAM MAPPING] Mapped 'num' -> 'num_results' for {tool_name}")
            
            if 'count' in mapped_params and 'num_results' not in mapped_params:
                mapped_params['num_results'] = mapped_params.pop('count')
                logger.info(f"[PARAM MAPPING] Mapped 'count' -> 'num_results' for {tool_name}")
        
        # Email tool parameter mapping  
        elif 'email' in tool_name.lower() or 'gmail' in tool_name.lower():
            # Map common email field variations
            if 'recipient' in mapped_params and 'to' not in mapped_params:
                mapped_params['to'] = mapped_params.pop('recipient')
                logger.info(f"[PARAM MAPPING] Mapped 'recipient' -> 'to' for {tool_name}")
            
            if 'message' in mapped_params and 'body' not in mapped_params:
                mapped_params['body'] = mapped_params.pop('message')
                logger.info(f"[PARAM MAPPING] Mapped 'message' -> 'body' for {tool_name}")
            
            if 'content' in mapped_params and 'body' not in mapped_params:
                mapped_params['body'] = mapped_params.pop('content')
                logger.info(f"[PARAM MAPPING] Mapped 'content' -> 'body' for {tool_name}")
        
        return mapped_params
    
    async def _dynamic_agent(self, agent_name: str, agent_info: Dict[str, Any], state: AgentState):
        """Execute a dynamic agent from database"""
        logger.info(f"[AGENT_EXECUTION] Starting execution of agent: {agent_name}")
        logger.info(f"[AGENT_EXECUTION] Agent info keys: {list(agent_info.keys()) if agent_info else 'None'}")
        logger.info(f"[AGENT_EXECUTION] Agent role: {agent_info.get('role', 'Unknown') if agent_info else 'Unknown'}")
        
        # First check if there's a pipeline-specific configuration in state metadata
        pipeline_agent_config = state.get("metadata", {}).get("pipeline_agent_config", {})
        
        # If pipeline has specific configuration, use it
        if pipeline_agent_config and pipeline_agent_config.get("tools"):
            tools = pipeline_agent_config.get("tools", [])
            system_prompt = pipeline_agent_config.get("system_prompt", "")
            logger.info(f"[PIPELINE] Using pipeline-specific config for {agent_name} with tools: {tools}")
            agent_config_source = "pipeline"
        else:
            # Otherwise, fetch from langgraph_agents as fallback
            from app.core.langgraph_agents_cache import get_agent_by_name
            
            fresh_agent_info = get_agent_by_name(agent_name)
            if fresh_agent_info:
                system_prompt = fresh_agent_info.get("system_prompt", "")
                tools = fresh_agent_info.get("tools", [])
                agent_config_source = "langgraph_cache"
                
                # DEBUG: Log detailed agent configuration for specific agents
                if agent_name in ["Researcher Agent", "Customer Service Agent", "Customer Support Agent", "Support Agent", "Service Agent"]:
                    logger.info(f"[DEBUG] {agent_name} detailed config from {agent_config_source}:")
                    logger.info(f"[DEBUG] - system_prompt length: {len(system_prompt)} chars")
                    logger.info(f"[DEBUG] - system_prompt preview: {system_prompt[:200]}...")
                    logger.info(f"[DEBUG] - tools: {tools}")
                    logger.info(f"[DEBUG] - config: {fresh_agent_info.get('config', {})}")
                    logger.info(f"[DEBUG] - is_active: {fresh_agent_info.get('is_active')}")
                    logger.info(f"[DEBUG] - role: {fresh_agent_info.get('role')}")
                
                logger.info(f"[LANGGRAPH] Using langgraph config for {agent_name} with tools: {tools}")
            else:
                # Final fallback to provided agent_info
                system_prompt = agent_info.get("system_prompt", "")
                tools = agent_info.get("tools", [])
                agent_config_source = "fallback"
                logger.warning(f"[FALLBACK] Could not load config for {agent_name}, using provided agent_info")
        
        # Check if there's a pipeline goal in metadata
        pipeline_goal = state.get("metadata", {}).get("pipeline_goal", "")
        
        # Check if this is a sequential agent with previous outputs
        previous_outputs = state.get("metadata", {}).get("previous_outputs", [])
        has_previous_context = len(previous_outputs) > 0
        
        # For direct execution without query, use pipeline goal or default instruction
        if not state['query'] and not has_previous_context:
            # First agent in pipeline with no query - use pipeline goal or system-based instruction
            query = pipeline_goal or "Use your available tools to complete your designated task based on your role and the pipeline objective."
        else:
            query = state['query'] or pipeline_goal or "Execute your designated task based on your role and available tools."
        
        # If this is a sequential agent with previous context but no explicit query,
        # create a more specific instruction
        if has_previous_context and not state['query']:
            # Extract the last agent's output as the primary context
            last_output = previous_outputs[-1]
            last_content = last_output.get('content') or last_output.get('output', '')
            
            # Create a more specific query based on the agent's role and tools
            agent_role = agent_info.get('role', 'an agent')
            
            # Create a context-aware query based on agent's role and previous output
            query = f"Based on the input from {last_output['agent']}, continue the workflow by performing your role as {agent_role}."
        
        # Prepare prompt with system instructions
        prompt = f"{system_prompt}"
        
        # Add pipeline goal if available
        if pipeline_goal and pipeline_goal != query:
            prompt += f"\n\nOverall Pipeline Goal: {pipeline_goal}\nEnsure your response contributes to achieving this goal."
        
        # Add context from previous agents if available
        if previous_outputs:
            prompt += "\n\nContext from previous agents:"
            for output in previous_outputs:
                # Handle both 'content' and 'output' keys for compatibility
                content = output.get('content') or output.get('output', '')
                agent_name = output.get('agent', 'Previous agent')
                # Show more context for better understanding
                prompt += f"\n\n{agent_name}:\n{content}"
                
                # Log what we're passing to help debug
                logger.info(f"[CONTEXT] Previous agent output being passed to current agent:")
                logger.info(f"[CONTEXT] From agent: {agent_name}")
                logger.info(f"[CONTEXT] Content length: {len(content)} chars")
                logger.info(f"[CONTEXT] Content preview (first 500 chars): {content[:500]}...")
                if len(content) > 1000:
                    logger.info(f"[CONTEXT] Content middle (500-1000 chars): {content[500:1000]}...")
                    logger.info(f"[CONTEXT] Content end (last 500 chars): {content[-500:]}...")
                
                # Also include tool results if available
                if 'tools_used' in output and output['tools_used']:
                    prompt += f"\n\n**Tools used by {agent_name}:**"
                    for tool_use in output['tools_used']:
                        if isinstance(tool_use, dict):
                            tool_name = tool_use.get('tool', 'Unknown tool')
                            tool_result = tool_use.get('result', '')
                            prompt += f"\n- {tool_name}: {tool_result[:500]}..."  # Limit length
            
            # Add explicit instruction for sequential agents
            prompt += "\n\n**Your Task**: Based on the above context from previous agents, continue the workflow by performing your designated role. Provide a comprehensive response that builds upon the previous work."
        
        # Add available tools
        if tools:
            # Get tool details from MCP system
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            import json
            
            mcp_tools = get_enabled_mcp_tools()
            
            # Enhanced debug logging
            logger.info(f"[TOOL INFO] Agent {agent_name} has access to tools: {tools}")
            logger.info(f"[TOOL INFO] MCP tools cache contains {len(mcp_tools)} tools total")
            
            # Check which tools are actually available
            available_tools = []
            missing_tools = []
            for tool in tools:
                if tool in mcp_tools:
                    available_tools.append(tool)
                else:
                    missing_tools.append(tool)
            
            if missing_tools:
                logger.warning(f"[TOOL INFO] Tools configured but not found in MCP: {missing_tools}")
            
            prompt += f"\n\nYou have access to the following tools: {', '.join(tools)}"
            
            # Add tool descriptions and parameters from MCP
            if available_tools:
                prompt += "\n\n**TOOL SPECIFICATIONS** (Use these exact parameter names):"
                for tool_name in available_tools:
                    tool_info = mcp_tools[tool_name]
                    prompt += f"\n\n• **{tool_name}**: {tool_info.get('description', 'No description available')}"
                    if tool_info.get('parameters'):
                        properties = tool_info['parameters'].get('properties', {})
                        if properties:
                            prompt += f"\n  Required parameters:"
                            for param_name, param_schema in properties.items():
                                param_type = param_schema.get('type', 'string')
                                param_desc = param_schema.get('description', '')
                                prompt += f"\n    - {param_name} ({param_type}): {param_desc}"
                    logger.info(f"[TOOL INFO] Added spec for {tool_name} to {agent_name}'s prompt")
            else:
                logger.error(f"[TOOL INFO] No valid MCP tools found for agent {agent_name}! Configured tools: {tools}")
                prompt += "\n\n**WARNING**: The tools configured for this agent are not available in the MCP system."
            
            prompt += "\n\nTo use a tool, respond with a JSON object in this format:"
            prompt += '\n{"tool": "tool_name", "parameters": {...}}'
            
            # Add a specific example if the agent has any tools
            if tools and tools[0] in mcp_tools:
                example_tool = tools[0]
                example_info = mcp_tools[example_tool]
                prompt += f"\n\nExample for {example_tool}:"
                example_params = {}
                if example_info.get('parameters'):
                    # Create example values for each parameter based on schema
                    for param_name, param_schema in example_info['parameters'].get('properties', {}).items():
                        param_type = param_schema.get('type', 'string')
                        
                        if param_name == "query":
                            if "search" in example_tool.lower():
                                example_params[param_name] = "your search terms here"
                            else:
                                example_params[param_name] = "subject:ENQUIRY"
                        elif param_name == "maxResults" or param_name == "count":
                            example_params[param_name] = 10
                        elif param_name == "to":
                            # Check the schema to determine if it's an array or string
                            if param_type == "array":
                                example_params[param_name] = ["recipient@email.com"]
                            else:
                                example_params[param_name] = "recipient@email.com"
                        elif param_name == "subject":
                            example_params[param_name] = "Re: Your inquiry"
                        elif param_name == "body":
                            example_params[param_name] = "Dear Customer..."
                        elif param_name in ["cc", "bcc"] and param_type == "array":
                            example_params[param_name] = ["cc@email.com"]
                        elif param_type == "array":
                            example_params[param_name] = ["example_value"]
                        elif param_type == "number" or param_type == "integer":
                            example_params[param_name] = 10
                        elif param_type == "boolean":
                            example_params[param_name] = True
                        else:
                            example_params[param_name] = f"<{param_name} value>"
                
                example_json = f'{{"tool": "{example_tool}", "parameters": {json.dumps(example_params)}}}'
                logger.info(f"Generated example for {agent_name}: {example_json}")
                prompt += f'\n{example_json}'
            
            prompt += "\n\n**CRITICAL INSTRUCTIONS**:"
            prompt += "\n1. You MUST use the tools by outputting the JSON format shown above"
            prompt += "\n2. Use the EXACT parameter names shown in the tool specifications"
            prompt += "\n3. For google_search tool, use 'query' NOT 'q' and 'num_results' NOT 'num'"
            prompt += "\n4. DO NOT explain what you're going to do"
            prompt += "\n5. START your response with the tool JSON immediately"
            prompt += "\n6. After the tool executes, you can provide additional analysis"
            
            # Extra emphasis for email sending agents
            if any(tool in ['gmail_send', 'send_email'] for tool in tools):
                prompt += "\n\n**FOR EMAIL SENDING**: Your PRIMARY task is to send the email. Start with:"
                prompt += '\n{"tool": "gmail_send", "parameters": {"to": ["email@example.com"], "subject": "...", "body": "..."}}'
            
            if has_previous_context:
                prompt += "\n\nBased on the context from previous agents and your role, use your tools to continue the workflow and complete your assigned task."
            else:
                prompt += "\n\nBased on your role and the context, use your tools to complete the assigned task."
        
        # Add the task/query with clear formatting
        if has_previous_context and not state['query']:
            prompt += f"\n\n**Specific Instructions**: {query}"
            prompt += "\n\n**Expected Response**: Execute the necessary tools first (using the JSON format specified above), then provide your analysis and response based on the results."
        else:
            prompt += f"\n\nTask: {query}"
            if tools:
                prompt += "\n\n**Expected Response**: Execute the necessary tools first (using the JSON format specified above), then provide your analysis based on the results."
        
        # Debug: Show the full prompt for specific agents and sequential agents
        if ("2" in agent_name or 
            len(state.get("metadata", {}).get("previous_outputs", [])) > 0 or
            agent_name in ["Customer Service Agent", "Customer Support Agent", "Support Agent", "Service Agent"]):
            print(f"[DEBUG] Full prompt for {agent_name} (length={len(prompt)}):")
            print(f"[DEBUG] First 500 chars: {prompt[:500]}...")
            if len(prompt) > 1000:
                print(f"[DEBUG] Last 500 chars: ...{prompt[-500:]}")
            # For customer service agents, show more of the prompt
            if any(keyword in agent_name.lower() for keyword in ["customer", "service", "support"]):
                print(f"[DEBUG] CUSTOMER SERVICE AGENT FULL PROMPT:")
                print(f"[DEBUG] ================================================")
                print(prompt)
                print(f"[DEBUG] ================================================")
        
        # Call LLM with appropriate configuration
        response = ""
        print(f"[DEBUG] Starting to collect response for {agent_name}")
        event_count = 0
        # Use temperature and timeout from pipeline agent config if available
        # Otherwise, use defaults based on context
        llm_temperature = pipeline_agent_config.get("temperature", 0.8 if has_previous_context else 0.7)
        llm_timeout = pipeline_agent_config.get("timeout", 60 if has_previous_context else 30)
        
        # Get the actual agent config from langgraph_agents for thinking mode settings
        actual_agent_config = {}
        if agent_config_source == "langgraph_cache" and fresh_agent_info:
            actual_agent_config = fresh_agent_info.get("config", {})
        elif agent_config_source == "pipeline":
            actual_agent_config = pipeline_agent_config
        else:
            actual_agent_config = agent_info.get("config", {})
        
        # Log thinking mode configuration for debugging
        enable_thinking = actual_agent_config.get("enable_thinking")
        logger.info(f"[DEBUG] {agent_name}: enable_thinking = {enable_thinking} (from {agent_config_source})")
        logger.info(f"[DEBUG] {agent_name}: full config = {actual_agent_config}")
        
        # Use the agent's actual configuration from database/cache
        async for event in self._call_llm_stream(prompt, agent_name, temperature=llm_temperature, timeout=llm_timeout, agent_config=actual_agent_config):
            event_count += 1
            if event.get("type") == "agent_token":
                # Accumulate tokens
                token = event.get("token", "")
                response += token
            elif event.get("type") == "agent_complete":
                # Use the final content if available
                if event.get("content"):
                    response = event.get("content", "")
                    print(f"[DEBUG] Got final content for {agent_name}: length={len(response)}")
        
        print(f"[DEBUG] Collected {event_count} events for {agent_name}, final response length: {len(response)}")
        
        # Check if response contains tool calls
        tool_results = []
        if tools and response.strip():
            # Try to parse tool calls from response
            import json
            import re
            
            # Look for JSON tool calls in the response with more flexible pattern
            json_patterns = [
                r'\{[^}]*"tool"[^}]*\}',  # Standard format
                r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[^}]*\}\s*\}',  # With nested parameters
                r'\{[^{}]*"tool"[^{}]*"parameters"[^{}]*\{[^}]*\}[^}]*\}'  # More complex nested
            ]
            
            matches = []
            for pattern in json_patterns:
                found = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
                matches.extend(found)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_matches = []
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    unique_matches.append(match)
            matches = unique_matches
            
            print(f"[DEBUG] Found {len(matches)} potential tool calls in response")
            
            for match in matches:
                try:
                    tool_call = json.loads(match)
                    if "tool" in tool_call:
                        tool_name = tool_call["tool"]
                        tool_params = tool_call.get("parameters", {})
                        
                        # Log tool execution attempt with parameters
                        logger.info(f"[TOOL DETECTION] Agent {agent_name} requesting tool: {tool_name}")
                        logger.info(f"[TOOL PARAMS] {json.dumps(tool_params, indent=2)}")
                        
                        # Apply parameter mapping for common mismatches
                        tool_params = self._map_tool_parameters(tool_name, tool_params)
                        logger.info(f"[TOOL PARAMS MAPPED] {json.dumps(tool_params, indent=2)}")
                        
                        # Execute actual tool (if available) or simulate
                        tool_result = await self._execute_mcp_tool(tool_name, tool_params, agent_name)
                        
                        # Log tool result
                        if isinstance(tool_result, str) and len(tool_result) > 500:
                            logger.info(f"[TOOL COMPLETE] {tool_name} returned {len(tool_result)} chars: {tool_result[:500]}...")
                        else:
                            logger.info(f"[TOOL COMPLETE] {tool_name} returned: {tool_result}")
                        
                        tool_results.append({
                            "tool": tool_name,
                            "parameters": tool_params,
                            "result": tool_result
                        })
                        
                        # Add to state's tools_used
                        state["tools_used"].append({
                            "agent": agent_name,
                            "tool": tool_name,
                            "parameters": tool_params
                        })
                except json.JSONDecodeError:
                    continue
        
        
        # Enhanced tool result processing with better error handling
        print(f"[DEBUG] Agent {agent_name} tool processing check: {len(tool_results)} tool results found")
        if tool_results:
            try:
                # Build a context-aware response based on tool results
                tool_context = "\n\n**Tool Execution Results:**\n"
                for result in tool_results:
                    tool_name = result['tool']
                    tool_result = result['result']
                    
                    tool_context += f"\n**{tool_name}:**\n"
                    
                    # Format search results nicely
                    if 'search' in tool_name.lower() and isinstance(tool_result, dict) and 'result' in tool_result:
                        search_results = tool_result['result']
                        if isinstance(search_results, list):
                            tool_context += f"Found {len(search_results)} search results:\n"
                            for i, item in enumerate(search_results[:5], 1):  # Limit to top 5 results
                                if isinstance(item, dict):
                                    title = item.get('title', 'No title')
                                    url = item.get('url', '')
                                    description = item.get('description', 'No description')
                                    tool_context += f"\n{i}. **{title}**\n"
                                    tool_context += f"   URL: {url}\n"
                                    tool_context += f"   Description: {description}\n"
                                else:
                                    tool_context += f"\n{i}. {item}\n"
                        else:
                            tool_context += f"{search_results}\n"
                    else:
                        # For other tools, include the raw result
                        tool_context += f"{tool_result}\n"
                    
                    tool_context += "\n"
                
                # CRITICAL: Always generate a follow-up response when tools were executed to incorporate results
                print(f"[DEBUG] Agent {agent_name} executed {len(tool_results)} tools, FORCING follow-up response to incorporate results")
                print(f"[DEBUG] Tool context length: {len(tool_context)} characters")
                
                # Re-prompt the agent to generate content based on tool results
                follow_up_prompt = f"{prompt}\n\n**Tool Execution Results**:{tool_context}\n\n**Now provide your complete response**: Based on these tool results, provide a comprehensive response that fulfills your role. Do not repeat the tool calls, just provide your analysis and conclusions based on the results."
                
                print(f"[DEBUG] Agent {agent_name} follow-up prompt length: {len(follow_up_prompt)} characters")
                
                # Get actual content response that incorporates tool results
                follow_up_response = ""
                follow_up_event_count = 0
                
                print(f"[DEBUG] Agent {agent_name} starting follow-up LLM call for tool result analysis...")
                async for event in self._call_llm_stream(follow_up_prompt, agent_name, temperature=0.8, timeout=60, agent_config=pipeline_agent_config):
                    follow_up_event_count += 1
                    if event.get("type") == "agent_token":
                        follow_up_response += event.get("token", "")
                    elif event.get("type") == "agent_complete":
                        if event.get("content"):
                            follow_up_response = event.get("content", "")
                            print(f"[DEBUG] Agent {agent_name} follow-up complete event received, content length: {len(follow_up_response)}")
                
                print(f"[DEBUG] Agent {agent_name} follow-up analysis collected {follow_up_event_count} events")
                print(f"[DEBUG] Agent {agent_name} follow-up response length: {len(follow_up_response)}")
                
                # Use the follow-up response that incorporates tool results (with fallback protection)
                if follow_up_response and follow_up_response.strip():
                    response = follow_up_response
                    print(f"[DEBUG] Agent {agent_name} successfully used follow-up response for tool analysis")
                else:
                    print(f"[DEBUG] Agent {agent_name} follow-up response was empty, keeping original response")
                    # Add a basic tool summary to the original response
                    response += f"\n\n**Tool Results Summary:** Executed {len(tool_results)} tools successfully."
                
            except Exception as e:
                print(f"[ERROR] Agent {agent_name} tool result analysis failed: {e}")
                import traceback
                print(f"[ERROR] Agent {agent_name} tool analysis traceback: {traceback.format_exc()}")
                # Fallback: append tool results to response
                response += f"\n\n**Tool Execution Completed:** {len(tool_results)} tools executed."
        else:
            print(f"[DEBUG] Agent {agent_name} no tools executed, using original response")
        
        # Debug logging
        print(f"[DEBUG] Dynamic agent {agent_name} generated response of length: {len(response)}")
        if len(response) > 0:
            print(f"[DEBUG] Response preview: {response[:200]}...")
        
        # Include tool results in the output for context passing
        output = {
            "response": response,
            "reasoning": f"Dynamic agent {agent_name} processed task with {len(tool_results)} tool calls",
            "tools_used": tool_results
        }
        
        # Store tool results in state for next agent
        if tool_results:
            state["agent_outputs"][agent_name] = {
                "response": response,
                "tools_used": tool_results
            }
        
        return output