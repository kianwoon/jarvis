"""
Dynamic Multi-Agent System that can use any agent from the database
"""
import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
import hashlib

from app.core.langgraph_agents_cache import get_langgraph_agents, get_active_agents, get_agent_by_role, get_agent_by_name
from app.core.llm_settings_cache import get_llm_settings
from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
import os


class DynamicMultiAgentSystem:
    """Orchestrates ANY agents dynamically based on their capabilities"""
    
    def __init__(self):
        self.llm_settings = get_llm_settings()
        self.routing_cache = {}
        
        # Agent avatars mapping
        self.agent_avatars = {
            "router": "🧭",
            "document_researcher": "📚", 
            "tool_executor": "🔧",
            "context_manager": "🧠",
            "sales_strategist": "💼",
            "technical_architect": "🏗️",
            "financial_analyst": "💰",
            "service_delivery_manager": "📋",
            "synthesizer": "🎯",
            # Add more generic mappings
            "analyst": "📊",
            "strategist": "🎯",
            "manager": "👔",
            "architect": "🏛️",
            "engineer": "⚙️",
            "researcher": "🔬",
            "writer": "✍️",
            "compliance": "⚖️",
            "ceo": "👑",
            "cio": "💻",
            "cto": "🚀",
            "director": "📈"
        }
    
    def get_agent_avatar(self, agent_name: str, agent_role: str = "") -> str:
        """Get avatar for agent based on name or role"""
        # Direct match
        if agent_name.lower() in self.agent_avatars:
            return self.agent_avatars[agent_name.lower()]
        
        # Try role-based match
        role_lower = agent_role.lower()
        for key, avatar in self.agent_avatars.items():
            if key in role_lower or key in agent_name.lower():
                return avatar
        
        # Default
        return "🤖"
    
    async def route_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Intelligently route query to appropriate agents using LLM"""
        
        # Get ALL active agents from database
        all_agents = get_active_agents()
        
        if not all_agents:
            return {
                "agents": [],
                "reasoning": "No active agents available in the system"
            }
        
        # Build comprehensive agent catalog
        agent_catalog = []
        for agent_name, agent_data in all_agents.items():
            agent_catalog.append({
                "name": agent_name,
                "role": agent_data.get("role", ""),
                "description": agent_data.get("description", ""),
                "capabilities": self._extract_capabilities(agent_data),
                "tools": agent_data.get("tools", [])
            })
        
        # Create routing prompt with all available agents
        routing_prompt = f"""You are an intelligent agent router for a comprehensive business system.

AVAILABLE AGENTS ({len(agent_catalog)} total):
{json.dumps(agent_catalog, indent=2)}

USER QUERY: "{query}"

INSTRUCTIONS:
1. Analyze the query to understand what capabilities are needed
2. Select 1-5 most relevant agents based on their roles and descriptions
3. Consider agent specializations:
   - C-Suite agents (CEO, CIO, CTO) for strategic/high-level queries
   - Analyst agents for detailed analysis
   - Technical agents for implementation details
   - Compliance/Risk agents for regulatory matters
   - Operational agents for day-to-day tasks
4. Prefer specialized agents over generic ones when applicable

IMPORTANT SELECTION CRITERIA:
- Match query intent with agent capabilities
- Consider multi-perspective needs (e.g., technical + financial + strategic)
- Balance between focused expertise and comprehensive coverage

Respond in JSON:
{{
    "agents": ["agent_name1", "agent_name2", ...],
    "reasoning": "Clear explanation of why these agents were selected",
    "collaboration_pattern": "parallel|sequential|hierarchical"
}}"""

        try:
            # Use LLM to select agents
            response_text = await self._call_llm(routing_prompt, temperature=0.3, max_tokens=1000)
            
            print(f"[DEBUG] LLM routing response: {response_text[:200]}...")
            
            # Parse response
            routing_decision = self._parse_json_response(response_text)
            
            # Validate selected agents exist with improved matching
            valid_agents = []
            for agent in routing_decision.get("agents", []):
                if agent in all_agents:
                    valid_agents.append(agent)
                else:
                    # Try various matching strategies
                    found = False
                    agent_lower = agent.lower()
                    
                    # Strategy 1: Case-insensitive exact match
                    for existing_agent in all_agents:
                        if agent_lower == existing_agent.lower():
                            valid_agents.append(existing_agent)
                            found = True
                            break
                    
                    # Strategy 2: Partial matching (contains)
                    if not found:
                        for existing_agent in all_agents:
                            if (agent_lower in existing_agent.lower() or 
                                existing_agent.lower() in agent_lower):
                                valid_agents.append(existing_agent)
                                print(f"[INFO] Matched '{agent}' to '{existing_agent}' (partial match)")
                                found = True
                                break
                    
                    # Strategy 3: Word-based matching
                    if not found:
                        agent_words = agent_lower.replace('_', ' ').split()
                        for existing_agent in all_agents:
                            existing_words = existing_agent.lower().replace('_', ' ').split()
                            if any(word in existing_words for word in agent_words):
                                valid_agents.append(existing_agent)
                                print(f"[INFO] Matched '{agent}' to '{existing_agent}' (word match)")
                                found = True
                                break
                    
                    if not found:
                        print(f"[WARNING] Router selected non-existent agent: {agent}")
                        print(f"[DEBUG] Available agents: {list(all_agents.keys())}")
                        print(f"[DEBUG] Trying to find similar agents...")
                        
                        # Show possible alternatives
                        suggestions = []
                        for existing_agent in all_agents:
                            # Calculate simple similarity
                            if any(word in existing_agent.lower() for word in agent_lower.split('_')):
                                suggestions.append(existing_agent)
                        
                        if suggestions:
                            print(f"[DEBUG] Possible alternatives for '{agent}': {suggestions}")
                            # Auto-select the first suggestion
                            valid_agents.append(suggestions[0])
                            print(f"[INFO] Auto-selected '{suggestions[0]}' as replacement for '{agent}'")
            
            routing_decision["agents"] = valid_agents
            
            # Ensure collaboration pattern has required fields
            if "collaboration_pattern" not in routing_decision:
                routing_decision["collaboration_pattern"] = "parallel"
            
            # Add order field for UI flow diagram
            if "order" not in routing_decision:
                routing_decision["order"] = valid_agents
                
            print(f"[DEBUG] Final routing decision: {routing_decision}")
            return routing_decision
            
        except Exception as e:
            print(f"[ERROR] LLM routing failed: {e}")
            # Fallback to simple keyword matching
            return self._fallback_routing(query, all_agents)
    
    def _extract_capabilities(self, agent_data: Dict) -> List[str]:
        """Extract key capabilities from agent description and prompt"""
        capabilities = []
        
        # Parse from description
        description = agent_data.get("description", "").lower()
        if "strategic" in description:
            capabilities.append("strategic planning")
        if "financial" in description or "cost" in description:
            capabilities.append("financial analysis")
        if "technical" in description or "architect" in description:
            capabilities.append("technical design")
        if "compliance" in description or "risk" in description:
            capabilities.append("risk management")
        if "operational" in description:
            capabilities.append("operations")
        
        # Parse from system prompt
        prompt = agent_data.get("system_prompt", "").lower()
        if "analyze" in prompt:
            capabilities.append("analysis")
        if "design" in prompt:
            capabilities.append("solution design")
        if "implement" in prompt:
            capabilities.append("implementation")
        
        return list(set(capabilities))  # Remove duplicates
    
    def _fallback_routing(self, query: str, all_agents: Dict) -> Dict[str, Any]:
        """Simple keyword-based fallback routing"""
        query_lower = query.lower()
        selected_agents = []
        
        # Keywords to agent mapping
        keyword_map = {
            "strategic": ["ceo_agent", "corporate_strategist", "cio_agent"],
            "financial": ["financial_analyst", "roi_analyst", "cfo_agent"],
            "technical": ["technical_architect", "cto_agent", "infrastructure_agent"],
            "compliance": ["compliance_agent", "risk_agent"],
            "proposal": ["proposal_writer", "sales_strategist", "presales_architect"],
            "database": ["dba", "infrastructure_agent"],
            "security": ["secops_agent", "compliance_agent"],
            "document": ["document_researcher", "context_manager"]
        }
        
        # Check keywords
        for keyword, agents in keyword_map.items():
            if keyword in query_lower:
                for agent in agents:
                    if agent in all_agents and agent not in selected_agents:
                        selected_agents.append(agent)
        
        # Default to document researcher if no matches
        if not selected_agents and "document_researcher" in all_agents:
            selected_agents = ["document_researcher"]
        
        return {
            "agents": selected_agents[:5],  # Limit to 5 agents
            "reasoning": "Selected based on keyword matching",
            "collaboration_pattern": "parallel"
        }
    
    async def execute_agent(self, agent_name: str, agent_data: Dict, query: str, context: Dict = None) -> AsyncGenerator[Dict, None]:
        """Execute any agent dynamically using its system prompt"""
        
        print(f"[DEBUG] DynamicMultiAgentSystem.execute_agent called for {agent_name}")
        print(f"[DEBUG] Agent data keys: {list(agent_data.keys()) if agent_data else 'None'}")
        print(f"[DEBUG] execute_agent: Starting async generator for {agent_name}")
        
        # Get agent configuration
        agent_config = agent_data.get("config", {})
        max_tokens = agent_config.get("max_tokens", 4000)
        temperature = agent_config.get("temperature", 0.7)
        
        # Dynamic timeout based on query complexity and agent type
        base_timeout = agent_config.get("timeout", 60)
        query_length = len(query)
        
        # Increase timeout for complex queries and strategic agents
        if query_length > 100 or "strategy" in query.lower() or "discuss" in query.lower():
            timeout = max(base_timeout, 90)  # At least 90 seconds for complex queries
        else:
            timeout = base_timeout
            
        # Strategic agents get extra time
        if any(keyword in agent_name.lower() for keyword in ["strategist", "analyst", "architect", "ceo", "cto", "cio"]):
            timeout = max(timeout, 120)  # Strategic agents get at least 2 minutes
            
        print(f"[DEBUG] {agent_name}: Dynamic timeout set to {timeout}s (base={base_timeout}s, query_len={query_length})")
        
        # Build prompt with agent's system prompt and query
        system_prompt = agent_data.get("system_prompt", "You are a helpful assistant.")
        
        # Add context if available
        context_str = ""
        if context:
            # Special handling for conversation history
            if "conversation_history" in context:
                context_str = f"\n\nCONTEXT:\nPrevious conversation:\n"
                for msg in context.get("conversation_history", [])[-6:]:  # Last 6 messages
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    content = msg.get("content", "")[:200]  # Truncate long messages
                    context_str += f"{role}: {content}\n"
                
                # Add other context fields
                other_context = {k: v for k, v in context.items() if k != "conversation_history"}
                if other_context:
                    context_str += f"\nAdditional context:\n{json.dumps(other_context, indent=2)}"
            else:
                context_str = f"\n\nCONTEXT:\n{json.dumps(context, indent=2)}"
        
        # Enhance prompt for completion tasks
        task_enhancement = ""
        if any(keyword in query.lower() for keyword in ["50", "interview", "questions", "generate", "create"]):
            task_enhancement = "\n\nIMPORTANT: If the user is asking you to generate a specific number of items (like 50 interview questions), you MUST generate the complete requested amount. Do not stop early or provide fewer items than requested."
        
        full_prompt = f"""{system_prompt}

USER QUERY: {query}{context_str}{task_enhancement}

Please provide a comprehensive response based on your role and expertise."""

        # Execute agent
        try:
            print(f"[DEBUG] Executing {agent_name} with config: max_tokens={max_tokens}, temp={temperature}, timeout={timeout}")
            print(f"[DEBUG] Prompt length: {len(full_prompt)} chars")
            print(f"[DEBUG] {agent_name} prompt preview (last 200 chars): {full_prompt[-200:]!r}")
            if task_enhancement:
                print(f"[DEBUG] {agent_name}: Task enhancement applied for completion task")
            
            response_text = ""
            error_occurred = False
            
            async for chunk in self._call_llm_stream(
                full_prompt, 
                agent_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            ):
                # Removed excessive debug logging for chunk processing
                
                if chunk.get("type") == "agent_complete":
                    response_text = chunk.get("content", "")
                    # Enhance the completion event with avatar and description
                    enhanced_chunk = {
                        **chunk,
                        "avatar": self.get_agent_avatar(agent_name, agent_data.get("role", "")),
                        "description": agent_data.get("description", "")
                    }
                    print(f"[DEBUG] execute_agent: Agent {agent_name} completed")
                    yield enhanced_chunk
                elif chunk.get("type") == "agent_error":
                    error_occurred = True
                    print(f"[DEBUG] execute_agent: Agent {agent_name} error")
                    yield chunk
                elif chunk.get("type") == "agent_token":
                    # Forward tokens without logging each one
                    yield chunk
                else:
                    # Forward any other event types
                    yield chunk
                    yield chunk
            
        except Exception as e:
            print(f"[ERROR] Agent {agent_name} execution failed: {e}")
            import traceback
            print(f"[ERROR] Agent {agent_name} traceback: {traceback.format_exc()}")
            yield {
                "type": "agent_error",
                "agent": agent_name,
                "error": str(e)
            }
        finally:
            print(f"[DEBUG] execute_agent: Finished (finally block) for {agent_name}")
    
    async def _call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Simple LLM call that returns complete response"""
        model_config = self.llm_settings.get("thinking_mode", {})
        
        config = LLMConfig(
            model_name=model_config.get("model", "qwen3:30b-a3b"),
            temperature=temperature,
            top_p=model_config.get("top_p", 0.9),
            max_tokens=max_tokens
        )
        
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = OllamaLLM(config, base_url=ollama_url)
        
        response_text = ""
        async for response_chunk in llm.generate_stream(prompt):
            response_text += response_chunk.text
            
        return response_text
    
    async def _call_llm_stream(self, prompt: str, agent_name: str, temperature: float = 0.7, 
                               max_tokens: int = 4000, timeout: int = 60):
        """Streaming LLM call for agent execution with real-time token streaming"""
        try:
            response_text = ""
            token_count = 0
            
            # Create LLM instance for streaming
            model_config = self.llm_settings.get("thinking_mode", {})
            config = LLMConfig(
                model_name=model_config.get("model", "qwen3:30b-a3b"),
                temperature=temperature,
                top_p=model_config.get("top_p", 0.9),
                max_tokens=max_tokens
            )
            
            ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            llm = OllamaLLM(config, base_url=ollama_url)
            
            print(f"[DEBUG] {agent_name}: Starting FIXED real-time streaming with timeout={timeout}s")
            print(f"[DEBUG] {agent_name}: Using model {config.model_name} with max_tokens={max_tokens}")
            
            # Use asyncio timeout and stream tokens in real-time
            async with asyncio.timeout(timeout):
                async for response_chunk in llm.generate_stream(prompt):
                    token_count += 1
                    response_text += response_chunk.text
                    
                    # Log every 100th token to track progress (reduced verbosity)
                    # if token_count % 100 == 0:
                    #     print(f"[DEBUG] {agent_name}: Received {token_count} tokens, total length: {len(response_text)}")
                    
                    # Stream tokens in real-time for better UX
                    if response_chunk.text:  # Don't filter out whitespace tokens
                        yield {
                            "type": "agent_token",
                            "agent": agent_name,
                            "token": response_chunk.text
                        }
            
            print(f"[DEBUG] {agent_name}: *** STREAMING COMPLETE ***")
            print(f"[DEBUG] {agent_name}: Total tokens received: {token_count}")
            print(f"[DEBUG] {agent_name}: Total response length: {len(response_text)}")
            print(f"[DEBUG] {agent_name}: Response preview (first 200 chars): {response_text[:200]!r}")
            
            # Clean response and yield final completion
            print(f"[DEBUG] {agent_name}: Calling _clean_response on {len(response_text)} chars")
            cleaned_content = self._clean_response(response_text)
            print(f"[DEBUG] {agent_name}: After cleaning: {len(cleaned_content)} chars")
            print(f"[DEBUG] {agent_name}: Cleaned preview (first 200 chars): {cleaned_content[:200]!r}")
            
            # ALWAYS yield completion event, even if content is empty
            print(f"[DEBUG] {agent_name}: Yielding completion event with content_length={len(cleaned_content)}")
            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "content": cleaned_content
            }
                
        except asyncio.TimeoutError:
            print(f"[WARNING] Agent {agent_name} timed out after {timeout}s")
            # Yield a completion event with timeout message instead of error
            # This allows the multi-agent system to continue with other agents
            
            # Generate a context-aware timeout message
            agent_role = agent_name.replace('_', ' ').title()
            timeout_content = f"⏰ **{agent_role} Response**\n\n"
            timeout_content += "I apologize, but my analysis is taking longer than expected. "
            timeout_content += "Due to the complexity of your request, I need more time to provide a comprehensive response.\n\n"
            
            # Provide a brief, generic acknowledgment based on the agent type
            if "researcher" in agent_name.lower() or "document" in agent_name.lower():
                timeout_content += "I'm currently searching through the knowledge base for relevant information. "
                timeout_content += "The search process may take additional time depending on the volume of documents to analyze."
            elif "analyst" in agent_name.lower() or "financial" in agent_name.lower():
                timeout_content += "I'm performing detailed analysis to ensure accuracy and completeness. "
                timeout_content += "Complex calculations and data processing require additional time."
            elif "strategist" in agent_name.lower() or "architect" in agent_name.lower():
                timeout_content += "I'm developing comprehensive strategies and recommendations. "
                timeout_content += "Strategic planning requires careful consideration of multiple factors."
            elif "manager" in agent_name.lower() or "delivery" in agent_name.lower():
                timeout_content += "I'm creating detailed plans and frameworks for your requirements. "
                timeout_content += "Service delivery planning involves multiple considerations and dependencies."
            else:
                timeout_content += "I'm processing your request and gathering the necessary information. "
                timeout_content += "Complex queries may require additional processing time."
            
            timeout_content += "\n\nThe system will continue with other agents to provide you with available insights."
            
            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "content": timeout_content,
                "timeout": True,
                "timeout_duration": timeout
            }
        except Exception as e:
            print(f"[ERROR] Agent {agent_name} failed: {e}")
            yield {
                "type": "agent_error", 
                "agent": agent_name,
                "error": str(e)
            }
    
    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        import re
        
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                # Ensure order field exists
                if "agents" in result and "order" not in result:
                    result["order"] = result["agents"]
                return result
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return {
            "agents": [],
            "reasoning": "Failed to parse routing decision",
            "collaboration_pattern": "parallel",
            "order": []
        }
    
    def _clean_response(self, text: str) -> str:
        """Clean LLM response - preserve thinking tags for frontend processing"""
        import re
        
        print(f"[DEBUG] _clean_response: Input length: {len(text)}")
        print(f"[DEBUG] _clean_response: Input preview: {text[:100]!r}")
        
        original_text = text
        
        # DON'T remove thinking tags - let frontend handle them for "Reasoning:" display
        # text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # text = re.sub(r'</?think>', '', text)
        # print(f"[DEBUG] _clean_response: After removing thinking tags: {len(text)} chars")
        
        # Only clean excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        print(f"[DEBUG] _clean_response: After cleaning whitespace: {len(text)} chars")
        
        cleaned = text.strip()
        print(f"[DEBUG] _clean_response: After strip(): {len(cleaned)} chars")
        print(f"[DEBUG] _clean_response: Final preview: {cleaned[:100]!r}")
        
        # Check if we have thinking content to preserve
        if '<think>' in cleaned:
            print(f"[DEBUG] _clean_response: Preserving thinking content for frontend extraction")
        
        if len(cleaned) == 0 and len(original_text) > 0:
            print(f"[WARNING] _clean_response: Content was completely cleaned away!")
            print(f"[WARNING] _clean_response: Original had {len(original_text)} chars: {original_text[:200]!r}")
        
        return cleaned
    
    async def synthesize_responses(self, agent_responses: Dict[str, str], original_query: str) -> str:
        """Synthesize multiple agent responses into cohesive answer"""
        
        if not agent_responses:
            return "No agent responses to synthesize."
        
        synthesis_prompt = f"""You are a master synthesizer combining insights from multiple expert agents.

ORIGINAL QUERY: {original_query}

AGENT RESPONSES:
{json.dumps(agent_responses, indent=2)}

INSTRUCTIONS:
1. Combine all agent insights into a comprehensive, cohesive response
2. Eliminate redundancy while preserving unique insights
3. Organize information logically
4. Highlight key recommendations and action items
5. Ensure the response directly addresses the original query

Provide a well-structured synthesis that leverages all agent contributions."""

        try:
            return await self._call_llm(synthesis_prompt, temperature=0.6, max_tokens=6000)
        except Exception as e:
            print(f"[ERROR] Synthesis failed: {e}")
            # Fallback to simple concatenation
            return "\n\n---\n\n".join([
                f"**{agent}**:\n{response}" 
                for agent, response in agent_responses.items()
            ])