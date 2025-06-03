"""
Simplified Multi-Agent System without LangGraph dependency
Implements router-based communication between specialized agents
"""

from typing import Dict, List, Any, Optional, TypedDict, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_role, get_active_agents
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
# Import rag_answer lazily to avoid circular import

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
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        
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
        
        # Agent avatars/emojis for better UI
        self.agent_avatars = {
            "router": "🧭",
            "document_researcher": "📚",
            "tool_executor": "🔧",
            "context_manager": "🧠",
            "sales_strategist": "💼",
            "technical_architect": "🏗️",
            "financial_analyst": "💰",
            "service_delivery_manager": "📋",
            "synthesizer": "🎯"
        }
        
        # Agent descriptions for hover tooltips
        self.agent_descriptions = {
            "router": "Analyzes your query and selects the most appropriate agents to handle it",
            "document_researcher": "Searches through uploaded documents and knowledge base for relevant information",
            "tool_executor": "Executes tools and calculations to provide computational results",
            "context_manager": "Manages conversation history and maintains context across interactions",
            "sales_strategist": "Provides strategic sales perspectives, value propositions, and client engagement strategies",
            "technical_architect": "Offers technical architecture insights, system design, and implementation recommendations",
            "financial_analyst": "Analyzes costs, ROI, pricing models, and financial implications",
            "service_delivery_manager": "Designs service delivery plans, SLAs, and operational frameworks",
            "synthesizer": "Combines insights from all agents into a comprehensive, coherent response"
        }
    
    def _remove_thinking_tags(self, text: str) -> str:
        """Remove thinking tags from LLM response"""
        import re
        # Remove <think>...</think> tags and their content
        pattern = r'<think>.*?</think>'
        cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
        # Also remove any standalone tags
        cleaned = re.sub(r'</?think>', '', cleaned)
        return cleaned.strip()
    
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
        if "sales_strategist" in selected_agents and "financial_analyst" in selected_agents:
            # Sales often needs financial input first
            return {
                "pattern": "sequential",
                "order": ["financial_analyst", "sales_strategist", "technical_architect", "service_delivery_manager"],
                "dependencies": {
                    "sales_strategist": ["financial_analyst"],
                    "service_delivery_manager": ["technical_architect"]
                }
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
    
    async def _call_llm_stream(self, prompt: str, agent_name: str, temperature: float = 0.7, timeout: int = 30):
        """Call LLM and return only the final cleaned response as JSON event"""
        try:
            print(f"[DEBUG] *** {agent_name}: GENERATING RESPONSE ***")
            
            # Import LLM service directly to avoid HTTP recursion
            from app.llm.ollama import OllamaLLM
            from app.llm.base import LLMConfig
            from app.core.langgraph_agents_cache import get_agent_by_role
            import os
            
            # Get agent-specific configuration
            agent_data = get_agent_by_role(agent_name)
            agent_config = agent_data.get("config", {}) if agent_data else {}
            
            # Use agent config with fallbacks to thinking mode settings
            model_config = self.llm_settings.get("thinking_mode", {})
            
            # Dynamic configuration based on agent settings
            actual_temperature = agent_config.get("temperature", temperature)
            actual_timeout = agent_config.get("timeout", timeout)
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
            cleaned_content = self._remove_thinking_tags(response_text)
            cleaned_content = self._clean_markdown_formatting(cleaned_content)
            display_content = cleaned_content.strip()
            
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
        
        # Try dynamic LLM routing first
        try:
            print("[DEBUG] Attempting dynamic LLM-based routing")
            from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
            dynamic_system = DynamicMultiAgentSystem()
            routing_result = await dynamic_system.route_query(query, conversation_history)
            
            if routing_result.get("agents"):
                print(f"[DEBUG] Dynamic routing selected agents: {routing_result['agents']}")
                return routing_result
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
        implemented_agents = {
            "document_researcher": active_agents.get("document_researcher"),
            "tool_executor": active_agents.get("tool_executor"),
            "context_manager": active_agents.get("context_manager"),
            "sales_strategist": active_agents.get("sales_strategist"),
            "technical_architect": active_agents.get("technical_architect"),
            "financial_analyst": active_agents.get("financial_analyst"),
            "service_delivery_manager": active_agents.get("service_delivery_manager")
        }
        
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
                                # Cache the result
                                self.routing_cache[cache_key] = result
                                return result
                        
                        raise Exception("Invalid JSON format in LLM response")
                        
                    except json.JSONDecodeError as e:
                        raise Exception(f"Failed to parse LLM routing response: {e}")
                        
        except Exception as e:
            raise Exception(f"LLM routing call failed: {e}")
        
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
        
        # Helper function to find best matching agent
        def find_matching_agent(target_names):
            matches = []
            for target in target_names:
                # First try exact match
                if target in available_agents:
                    matches.append(target)
                else:
                    # Try case-insensitive and partial matching
                    target_lower = target.lower()
                    for available_agent in available_agents:
                        if (target_lower in available_agent.lower() or 
                            available_agent.lower() in target_lower or
                            any(word in available_agent.lower() for word in target_lower.split('_'))):
                            matches.append(available_agent)
                            break
            return matches
        
        # For a proposal discussion or managed services query, we want multiple perspectives
        if (any(word in query_lower for word in ["proposal", "client", "discuss", "counter"]) or
            is_managed_services or
            (has_client and (has_pricing or has_strategy or has_service_model))):
            # Try to find specialist team agents
            desired_agents = ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]
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
        
        # Check for messages from other agents
        messages = self.check_agent_messages("sales_strategist", state)
        financial_insights = ""
        
        for msg in messages:
            if msg["from"] == "financial_analyst":
                financial_insights = f"\n\nFinancial Analyst Insights: {msg['message']}"
                # Send response back
                async for event in self.respond_to_message("sales_strategist", msg["id"], "Thank you for the financial analysis. I'll incorporate this into my sales strategy.", state):
                    yield event
        
        # Check if we need financial input for pricing strategy
        if ("pricing" in state["query"].lower() or "cost" in state["query"].lower()) and not financial_insights:
            # Request input from financial analyst
            async for event in self.send_message_to_agent(
                "sales_strategist", 
                "financial_analyst",
                f"I need cost-benefit analysis for {context['client']} - {context['requirement']}. What are the key financial advantages of managed services over {context['current_model']}?",
                state
            ):
                yield event
        
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

        async for event in self._call_llm_stream(prompt, "technical_architect", temperature=0.6):
            yield event
    
    async def _financial_analyst_agent(self, state: AgentState):
        """Financial perspective on the proposal - streaming version"""
        context = self._extract_query_context(state["query"])
        
        # Check for messages from other agents
        messages = self.check_agent_messages("financial_analyst", state)
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
        async for event in self._call_llm_stream(prompt, "financial_analyst", temperature=0.5):
            yield event
        
        # Then respond to any messages from other agents
        for msg in messages:
            response = f"Key financial insights for {msg['from']}: 1) Managed services typically reduce TCO by 25-35% over 3 years. 2) Predictable monthly costs vs variable T&M billing. 3) Risk mitigation through SLAs and penalties. 4) Access to senior expertise without full-time cost."
            async for event in self.respond_to_message("financial_analyst", msg["id"], response, state):
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

        async for event in self._call_llm_stream(prompt, "service_delivery_manager", temperature=0.6):
            yield event
    
    async def _tool_executor_agent(self, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on query"""
        # For now, return a placeholder
        return {
            "response": "Tool execution capability is being developed.",
            "tools_used": []
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
        """Synthesize final response from all agent outputs"""
        final_parts = []
        
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
        
        # Check if we have any meaningful responses (including timeout responses)
        has_meaningful_response = False
        timeout_responses = 0
        successful_responses = 0
        
        for output in state["agent_outputs"].values():
            response = output.get("response", "")
            if response and len(response) > 20:
                has_meaningful_response = True
                if "⏰" in response or "timed out" in response.lower():
                    timeout_responses += 1
                else:
                    successful_responses += 1
        
        print(f"[DEBUG] Synthesizer: {successful_responses} successful responses, {timeout_responses} timeout responses")
        
        # Skip the fallback message if we have specialized agents responding
        # Check both hardcoded and dynamic agents
        specialized_agents = [
            "sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager",
            "PreSalesArchitect", "CTO_Agent", "ROI_Analyst", "ComplianceAgent",
            "CEO_Agent", "CIO_Agent", "BizAnalystAgent", "Corporate_Strategist"
        ]
        has_specialized_agents = any(
            agent in state["agent_outputs"] 
            for agent in specialized_agents
        ) or len(state["agent_outputs"]) > 1  # Multiple agents usually means specialized response
        
        if not has_meaningful_response and not has_specialized_agents:
            # No meaningful responses, provide a helpful default
            print(f"[DEBUG] Synthesizer: No meaningful responses found. has_meaningful_response={has_meaningful_response}, has_specialized_agents={has_specialized_agents}")
            return (
                "I couldn't find specific information in the knowledge base to help with your query.\n\n"
                "Please try:\n"
                "• Uploading relevant documents to the knowledge base\n"
                "• Asking a more specific question\n"
                "• Using the standard chat mode for general inquiries"
            )
        
        # For specialized agents, create a well-formatted comprehensive response
        if has_specialized_agents:
            if timeout_responses > 0:
                final_parts.append(f"Based on the multi-agent analysis (⚠️ {timeout_responses} agents experienced delays), here's a comprehensive response:\n")
            else:
                final_parts.append("Based on the multi-agent analysis, here's a comprehensive response:\n")
            
            # Add each specialized agent's response with proper formatting
            # First try hardcoded agent order for consistency
            agent_order = ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]
            displayed_agents = set()
            
            for agent_name in agent_order:
                if agent_name in state["agent_outputs"]:
                    output = state["agent_outputs"][agent_name]
                    if output.get("response"):
                        agent_title = agent_name.replace('_', ' ').title()
                        # Use markdown headers for better structure
                        final_parts.append(f"## {agent_title} Perspective\n")
                        final_parts.append(output["response"])
                        final_parts.append("")  # Add blank line for spacing
                        displayed_agents.add(agent_name)
            
            # Then add any dynamic agents that weren't in the predefined order
            for agent_name, output in state["agent_outputs"].items():
                if agent_name not in displayed_agents and output.get("response"):
                    # Format agent name nicely
                    agent_title = agent_name.replace('_', ' ').replace('Agent', ' Agent').strip().title()
                    final_parts.append(f"## {agent_title} Perspective\n")
                    final_parts.append(output["response"])
                    final_parts.append("")  # Add blank line for spacing
        else:
            # Standard routing with icons
            if state["routing_decision"] and state["routing_decision"].get("reasoning"):
                final_parts.append(f"📊 **Analysis**: {state['routing_decision']['reasoning']}\n")
            
            # Add agent responses
            for agent_name, output in state["agent_outputs"].items():
                if output.get("response"):
                    agent_title = agent_name.replace('_', ' ').title()
                    icon = "📄" if "document" in agent_name else "🔧" if "tool" in agent_name else "🧠"
                    final_parts.append(f"{icon} **{agent_title}**:\n")
                    final_parts.append(output['response'])
                    final_parts.append("")  # Add blank line for spacing
        
        # Add document sources if any
        all_docs = []
        for output in state["agent_outputs"].values():
            if output.get("documents"):
                all_docs.extend(output["documents"])
        
        if all_docs:
            final_parts.append("\n---\n")  # Horizontal rule
            final_parts.append("📚 **Sources**:")
            for doc in all_docs[:5]:  # Show up to 5 sources
                final_parts.append(f"- {doc.get('source', 'Unknown source')}")
        
        return "\n".join(final_parts) if final_parts else "No response generated."
    
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
                        from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
                        from app.core.langgraph_agents_cache import get_agent_by_name
                        
                        agent_data = get_agent_by_name(agent_name_local)
                        print(f"[DEBUG] Agent data found: {agent_data is not None}")
                        if not agent_data:
                            # Get list of available agents for debugging
                            from app.core.langgraph_agents_cache import get_active_agents
                            available_agents = list(get_active_agents().keys())
                            print(f"[ERROR] Agent {agent_name_local} not found in system")
                            print(f"[ERROR] Available agents: {available_agents}")
                            
                            # Try case-insensitive and partial matching
                            possible_matches = []
                            agent_name_lower = agent_name_local.lower()
                            for available_agent in available_agents:
                                if agent_name_lower in available_agent.lower() or available_agent.lower() in agent_name_lower:
                                    possible_matches.append(available_agent)
                            
                            if possible_matches:
                                print(f"[INFO] Possible matches for {agent_name_local}: {possible_matches}")
                                # Use the first match
                                suggested_agent = possible_matches[0]
                                print(f"[INFO] Using {suggested_agent} instead of {agent_name_local}")
                                agent_data = get_agent_by_name(suggested_agent)
                                agent_name_local = suggested_agent  # Update the local agent name
                            
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
                        
                        dynamic_system = DynamicMultiAgentSystem()
                        start_time = agent_start_times[agent_name_local]
                        
                        # Build context including conversation history
                        agent_context = self._extract_query_context(state["query"])
                        if conversation_history:
                            agent_context["conversation_history"] = conversation_history
                        
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
            
            final_response = await self._synthesizer_agent(state)
            state["final_response"] = final_response
            
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