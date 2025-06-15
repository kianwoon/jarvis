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
        # Check if there's a pipeline-specific system prompt in the config
        if agent_config.get("system_prompt"):
            system_prompt = agent_config["system_prompt"]
            print(f"[DEBUG] Using pipeline-specific system prompt for {agent_name}")
        else:
            system_prompt = agent_data.get("system_prompt", "You are a helpful assistant.")
            print(f"[DEBUG] Using default system prompt for {agent_name}")
        
        # Add context if available
        context_str = ""
        if context:
            # Special handling for previous agent outputs in sequential execution
            if "previous_outputs" in context:
                context_str = f"\n\nCONTEXT FROM PREVIOUS AGENTS:\n"
                for prev_output in context.get("previous_outputs", []):
                    agent_name_prev = prev_output.get("agent", "Previous Agent")
                    content = prev_output.get("content") or prev_output.get("output", "")
                    
                    # Extract content from thinking tags if present
                    import re
                    thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
                    if thinking_match and len(thinking_match.group(1).strip()) > len(content.replace(thinking_match.group(0), '').strip()):
                        # Most content is in thinking tags, extract it
                        thinking_content = thinking_match.group(1).strip()
                        other_content = content.replace(thinking_match.group(0), '').strip()
                        if other_content:
                            # There's content outside thinking tags, show both
                            content = f"{other_content}\n\n[Agent's Analysis Process:]\n{thinking_content}"
                        else:
                            # Only thinking content exists, use it directly
                            content = thinking_content
                        print(f"[DEBUG] Extracted content from thinking tags for {agent_name_prev}")
                    
                    # Show substantial context from previous agents
                    context_str += f"\n{agent_name_prev}:\n{content}\n"
                    context_str += "-" * 50 + "\n"
                
                # Add instruction to build upon previous work
                agent_position = len(context.get("previous_outputs", [])) + 1
                context_str += "\n\n🚨 CRITICAL WORKFLOW INSTRUCTION 🚨\n"
                context_str += f"You are agent #{agent_position} in a SEQUENTIAL workflow. DO NOT start by analyzing what the user wants - that's already been done!\n\n"
                context_str += "YOUR SPECIFIC TASKS:\n"
                context_str += "1. READ the previous agent's work above carefully\n"
                context_str += "2. CONTINUE building on their analysis - don't start over\n"
                context_str += "3. ADD your unique expertise without repeating their points\n"
                context_str += "4. FOCUS on your specific role: provide insights the previous agent couldn't\n\n"
                context_str += "❌ DO NOT say 'The user is asking...' or 'Let me understand the request...'\n"
                context_str += "✅ DO say 'Building on the previous analysis...' or 'Adding to what was identified...'\n\n"
                context_str += "IMPORTANT: Provide your response directly without <think> tags. Your actual analysis should be outside any thinking tags.\n"
            
            # Special handling for conversation history
            if "conversation_history" in context:
                if not context_str:  # Only add header if we haven't already
                    context_str = f"\n\nCONTEXT:\n"
                context_str += "Previous conversation:\n"
                for msg in context.get("conversation_history", [])[-6:]:  # Last 6 messages
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    content = msg.get("content", "")[:200]  # Truncate long messages
                    context_str += f"{role}: {content}\n"
            
            # Add other context fields (excluding already processed ones)
            other_context = {k: v for k, v in context.items() if k not in ["conversation_history", "previous_outputs"]}
            if other_context:
                if not context_str:
                    context_str = f"\n\nCONTEXT:\n"
                else:
                    context_str += f"\nAdditional context:\n"
                context_str += json.dumps(other_context, indent=2)
                
        # Add available tools to context
        available_tools = []
        if context and "available_tools" in context:
            available_tools = context["available_tools"]
        elif agent_config.get("tools"):
            available_tools = agent_config["tools"]
        elif agent_data.get("tools"):
            available_tools = agent_data["tools"]
            
        if available_tools:
            if not context_str:
                context_str = f"\n\nCONTEXT:\n"
            context_str += f"\n\nAVAILABLE TOOLS:\n"
            context_str += f"You have access to the following tools: {available_tools}\n"
            context_str += "To use a tool, output JSON in this format:\n"
            context_str += '{"tool": "tool_name", "parameters": {"param1": "value1"}}'
        
        # Enhance prompt for completion tasks
        task_enhancement = ""
        if any(keyword in query.lower() for keyword in ["50", "interview", "questions", "generate", "create"]):
            task_enhancement = "\n\nIMPORTANT: If the user is asking you to generate a specific number of items (like 50 interview questions), you MUST generate the complete requested amount. Do not stop early or provide fewer items than requested."
        
        # Adjust prompt based on whether this is part of a sequential workflow
        if context and "previous_outputs" in context and context["previous_outputs"]:
            # This is a continuation in a sequential workflow
            agent_position = len(context["previous_outputs"]) + 1
            full_prompt = f"""{system_prompt}

YOU ARE AGENT #{agent_position} IN A SEQUENTIAL WORKFLOW. 

The previous agent(s) have already analyzed the user's request and provided their insights above. DO NOT re-analyze what the user wants.
{context_str}

ORIGINAL USER REQUEST (for reference only): {query}

YOUR SPECIFIC TASK: 
- Build directly on the previous agent's work
- Add your unique expertise and insights
- DO NOT repeat their analysis or findings
- Start with phrases like "Building on the previous analysis..." or "To add to what was identified..."
- Provide your response WITHOUT thinking tags - give your analysis directly{task_enhancement}"""
        else:
            # This is a standalone agent or first in sequence
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
            
            # Check for and execute any tool calls
            # Debug: Verify all variables are defined before the call
            print(f"[DEBUG] {agent_name}: About to call _process_tool_calls")
            print(f"[DEBUG] {agent_name}: Variables check - agent_name: {agent_name is not None}, cleaned_content: {len(cleaned_content)}")
            try:
                print(f"[DEBUG] {agent_name}: query length: {len(query)}")
                print(f"[DEBUG] {agent_name}: context is None: {context is None}")
                print(f"[DEBUG] {agent_name}: agent_data is None: {agent_data is None}")
                # If we get here, all variables are properly defined
                query_to_use = query
                context_to_use = context
                agent_data_to_use = agent_data
            except NameError as e:
                print(f"[ERROR] {agent_name}: NameError in variable check: {e}")
                print(f"[ERROR] {agent_name}: Available locals: {list(locals().keys())}")
                # Use fallbacks to avoid breaking the system
                query_to_use = "fallback query due to scoping issue"
                context_to_use = None
                agent_data_to_use = agent_data if 'agent_data' in locals() else {}
                print(f"[ERROR] {agent_name}: Using fallback variables")
            
            enhanced_content, tool_results = await self._process_tool_calls(agent_name, cleaned_content, query_to_use, context_to_use, agent_data_to_use)
            
            # ALWAYS yield completion event, even if content is empty
            print(f"[DEBUG] {agent_name}: Yielding completion event with content_length={len(enhanced_content)}")
            completion_data = {
                "type": "agent_complete",
                "agent": agent_name,
                "content": enhanced_content
            }
            
            # Add tool results if any were executed
            if tool_results:
                completion_data["tools_used"] = tool_results
            
            yield completion_data
                
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
    
    async def _process_tool_calls(self, agent_name: str, agent_response: str, query: str, context: Dict = None, agent_data: Dict = None) -> tuple[str, list]:
        """Process agent response for tool calls and execute them"""
        try:
            print(f"[DEBUG] {agent_name}: _process_tool_calls started with query: {query[:50]}...")
            print(f"[DEBUG] {agent_name}: Method parameters - query type: {type(query)}, context: {context is not None}, agent_data: {agent_data is not None}")
            # Import tool executor at runtime to avoid circular imports
            from app.langchain.tool_executor import tool_executor
            
            print(f"[DEBUG] {agent_name}: _process_tool_calls called with response length: {len(agent_response)}")
            print(f"[DEBUG] {agent_name}: Response preview: {agent_response[:200]}")
            
            # Check if response contains potential tool calls (generic patterns)
            tool_patterns = [
                r'\{\s*"tool"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{.*?\}\s*\}',  # JSON format - improved
                r'"tool"\s*:\s*"[^"]+"',  # Simple tool mention
                r'<tool>[^<]+</tool>',  # XML format
                r'\w+\([^)]*\)'  # Function call format
            ]
            
            import re
            import json
            has_tool_mention = any(re.search(pattern, agent_response, re.DOTALL) for pattern in tool_patterns)
            
            print(f"[DEBUG] {agent_name}: Tool patterns found: {has_tool_mention}")
            if has_tool_mention:
                for i, pattern in enumerate(tool_patterns):
                    matches = re.findall(pattern, agent_response)
                    if matches:
                        print(f"[DEBUG] {agent_name}: Pattern {i} matches: {matches}")
            
            if not has_tool_mention:
                print(f"[DEBUG] {agent_name}: No tool patterns found, returning original response")
                return agent_response, []
            
            print(f"[DEBUG] {agent_name}: Tool patterns detected, extracting tool calls")
            
            # Extract tool calls first to debug
            tool_calls = tool_executor.extract_tool_calls(agent_response)
            print(f"[DEBUG] {agent_name}: Extracted {len(tool_calls)} tool calls: {tool_calls}")
            
            if not tool_calls:
                print(f"[DEBUG] {agent_name}: No valid tool calls extracted")
                return agent_response, []
            
            # Execute tool calls - call synchronously since call_mcp_tool is sync
            print(f"[DEBUG] {agent_name}: Executing {len(tool_calls)} tool calls")
            
            # Execute tools in a separate thread to avoid event loop conflicts
            tool_results = []
            
            # Import for thread execution
            import concurrent.futures
            import threading
            
            def execute_tool_sync(tool_name, parameters):
                """Execute tool in a separate thread to avoid event loop conflicts"""
                try:
                    print(f"[DEBUG] Thread execution: {tool_name} with parameters: {parameters}")
                    
                    # Import and call MCP tool in thread context
                    from app.langchain.service import call_mcp_tool
                    result = call_mcp_tool(tool_name, parameters)
                    
                    return {
                        "tool": tool_name,
                        "parameters": parameters,
                        "result": result,
                        "success": "error" not in result if isinstance(result, dict) else True
                    }
                    
                except Exception as e:
                    print(f"[ERROR] Thread execution failed for {tool_name}: {e}")
                    return {
                        "tool": tool_name,
                        "parameters": parameters,
                        "error": str(e),
                        "success": False
                    }
            
            # Execute tools using thread pool to avoid event loop conflicts
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                # Submit all tool calls
                future_to_tool = {
                    executor.submit(execute_tool_sync, tool_call.get("tool"), tool_call.get("parameters", {})): tool_call
                    for tool_call in tool_calls if tool_call.get("tool")
                }
                
                # Wait for all results
                for future in concurrent.futures.as_completed(future_to_tool):
                    tool_call = future_to_tool[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per tool
                        tool_results.append(result)
                        print(f"[DEBUG] {agent_name}: Tool {result['tool']} -> {'Success' if result['success'] else 'Failed'}")
                    except concurrent.futures.TimeoutError:
                        print(f"[ERROR] {agent_name}: Tool {tool_call.get('tool')} timed out")
                        tool_results.append({
                            "tool": tool_call.get("tool"),
                            "parameters": tool_call.get("parameters", {}),
                            "error": "Tool execution timeout (30s)",
                            "success": False
                        })
                    except Exception as e:
                        print(f"[ERROR] {agent_name}: Tool {tool_call.get('tool')} execution error: {e}")
                        tool_results.append({
                            "tool": tool_call.get("tool"),
                            "parameters": tool_call.get("parameters", {}),
                            "error": str(e),
                            "success": False
                        })
            
            print(f"[DEBUG] {agent_name}: All tools executed. Results: {len(tool_results)} tools")
            
            # If tools were executed successfully, generate a follow-up response that incorporates the results
            if tool_results and any(result.get("success") for result in tool_results):
                print(f"[DEBUG] {agent_name}: Tools executed successfully, generating follow-up response to incorporate results")
                
                # Build tool context for the follow-up prompt
                tool_context = "\n\n**Tool Execution Results:**\n"
                for result in tool_results:
                    if result.get("success"):
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
                
                # Create follow-up prompt to generate a response that incorporates tool results
                agent_config = context.get("agent_config", {}) if context else {}
                system_prompt = agent_config.get("system_prompt") or agent_data.get("system_prompt", "You are a helpful assistant.")
                
                print(f"[DEBUG] {agent_name}: Building follow-up prompt with query length: {len(query)}")
                
                # Build the follow-up prompt carefully to avoid scoping issues
                user_query_part = f"USER QUERY: {query}"
                
                follow_up_prompt = f"""{system_prompt}

{user_query_part}

You have executed tools and received the following results:
{tool_context}

Based on these tool results, provide a comprehensive response that:
1. Directly answers the user's question
2. Incorporates the relevant information from the tool results
3. Synthesizes the findings into a coherent response
4. Does not repeat the raw tool output but interprets and explains it

Provide your complete response based on the tool results above."""

                # Generate follow-up response that incorporates tool results
                follow_up_response = ""
                async for chunk in self._call_llm_stream(
                    follow_up_prompt,
                    agent_name, 
                    temperature=0.8,
                    max_tokens=2000,
                    timeout=60
                ):
                    if chunk.get("type") == "agent_complete":
                        follow_up_response = chunk.get("content", "")
                        break
                
                # Use the follow-up response that incorporates tool results, fallback to original if needed
                enhanced_response = follow_up_response if follow_up_response.strip() else agent_response
                print(f"[DEBUG] {agent_name}: Generated follow-up response incorporating tool results (length: {len(follow_up_response)})")
                
            else:
                # No successful tools or no tools executed
                enhanced_response = agent_response
                if tool_results:
                    enhanced_response += "\n\n**Tool Execution Results:**\n"
                    for result in tool_results:
                        if not result.get("success"):
                            enhanced_response += f"❌ {result['tool']}: {result.get('error', 'Unknown error')}\n"
                
                print(f"[DEBUG] {agent_name}: No successful tools executed, using original response")
            
            return enhanced_response, tool_results
            
        except Exception as e:
            print(f"[ERROR] {agent_name}: Tool processing failed: {e}")
            import traceback
            print(f"[ERROR] {agent_name}: Tool processing traceback: {traceback.format_exc()}")
            return agent_response, []
    
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