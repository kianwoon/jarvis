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


def _is_tool_result_successful(result) -> bool:
    """
    Determine if a tool execution result indicates success
    
    Args:
        result: The tool execution result
        
    Returns:
        bool: True if successful, False if failed
    """
    if not isinstance(result, dict):
        return True  # Non-dict results are considered successful
    
    # Check for explicit error indicators
    if "error" in result and isinstance(result["error"], str) and result["error"].strip():
        return False
    
    # Check for successful content patterns
    if "content" in result:
        content = result["content"]
        if isinstance(content, list) and len(content) > 0:
            # Check if first content item indicates success
            first_item = content[0]
            if isinstance(first_item, dict) and "text" in first_item:
                text = first_item["text"].lower()
                # Look for explicit success indicators
                if any(indicator in text for indicator in ["✅", "success", "sent successfully", "completed successfully"]):
                    return True
                # Look for explicit error indicators
                if any(indicator in text for indicator in ["❌", "failed", "error:", "exception:"]):
                    return False
        return True  # Content exists, assume success
    
    # Check for result field
    if "result" in result:
        return True  # Has result field, assume success
    
    # Default to success for unrecognized formats
    return True


class DynamicMultiAgentSystem:
    """Orchestrates ANY agents dynamically based on their capabilities"""
    
    def __init__(self, trace=None):
        self.llm_settings = get_llm_settings()
        self.routing_cache = {}
        self.trace = trace  # Store trace for span creation
        
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
        
        # Build comprehensive agent catalog with dynamic capabilities
        agent_catalog = []
        for agent_name, agent_data in all_agents.items():
            capabilities = self._extract_capabilities(agent_data)
            agent_catalog.append({
                "name": agent_name,
                "role": agent_data.get("role", ""),
                "description": agent_data.get("description", ""),
                "capabilities": capabilities,
                "tools": agent_data.get("tools", []),
                "config_metadata": agent_data.get("config", {})
            })
        
        # Create routing prompt with all available agents
        routing_prompt = f"""You are an intelligent agent router for a comprehensive business system.

AVAILABLE AGENTS ({len(agent_catalog)} total):
{json.dumps(agent_catalog, indent=2)}

USER QUERY: "{query}"

INSTRUCTIONS:
1. Analyze the query to understand what specific capabilities are needed
2. Match query requirements with agent capabilities (focus on the "capabilities" field)
3. Consider agent tools for direct functionality requirements
4. Select 1-5 most relevant agents based on capability scores
5. Consider agent specializations:
   - Strategic agents for high-level planning and vision
   - Technical agents for implementation and architecture
   - Research agents for investigation and analysis
   - Operational agents for process and workflow management
   - Specialized agents for domain-specific expertise

INTELLIGENT SELECTION CRITERIA:
- PRIORITIZE agents whose capabilities directly match query requirements
- Look for specific tools that can execute the requested tasks
- Consider role descriptions for domain expertise
- Balance comprehensive coverage with focused specialization
- Use config_metadata for additional context when available
- Prefer agents with proven track record in similar tasks

CRITICAL: Respond ONLY with valid JSON. Do not use thinking tags or explanations. Output must start with {{.

Required JSON format:
{{
    "agents": ["agent_name1", "agent_name2", ...],
    "reasoning": "Clear explanation of why these agents were selected",
    "collaboration_pattern": "parallel|sequential|hierarchical"
}}"""

        try:
            # Use LLM to select agents with enhanced agent information
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
            # Fallback to intelligent capability-based routing
            print("[INFO] Using intelligent capability-based fallback routing")
            return self._intelligent_capability_routing(query, all_agents)
    
    def _extract_capabilities(self, agent_data: Dict) -> List[str]:
        """Dynamically extract capabilities from agent profile data"""
        capabilities = []
        
        # Extract from tools (direct capability indicators)
        tools = agent_data.get("tools", [])
        for tool in tools:
            if isinstance(tool, str):
                # Map tools to capabilities
                tool_lower = tool.lower()
                if any(keyword in tool_lower for keyword in ["search", "google", "web"]):
                    capabilities.append("web_research")
                if any(keyword in tool_lower for keyword in ["pdf", "document", "file"]):
                    capabilities.append("document_analysis")
                if any(keyword in tool_lower for keyword in ["email", "gmail", "mail"]):
                    capabilities.append("email_management")
                if any(keyword in tool_lower for keyword in ["database", "sql", "query"]):
                    capabilities.append("data_analysis")
                if any(keyword in tool_lower for keyword in ["api", "integration", "webhook"]):
                    capabilities.append("system_integration")
                if any(keyword in tool_lower for keyword in ["calculate", "math", "compute"]):
                    capabilities.append("computational_analysis")
        
        # Extract from role (primary function indicator)
        role = agent_data.get("role", "").lower()
        if any(keyword in role for keyword in ["strategic", "ceo", "executive", "director"]):
            capabilities.append("strategic_planning")
        if any(keyword in role for keyword in ["financial", "cfo", "analyst", "budget"]):
            capabilities.append("financial_analysis")
        if any(keyword in role for keyword in ["technical", "cto", "architect", "engineer"]):
            capabilities.append("technical_design")
        if any(keyword in role for keyword in ["compliance", "risk", "legal", "audit"]):
            capabilities.append("risk_management")
        if any(keyword in role for keyword in ["operation", "delivery", "project", "manager"]):
            capabilities.append("operations_management")
        if any(keyword in role for keyword in ["research", "analyst", "investigator"]):
            capabilities.append("research_analysis")
        if any(keyword in role for keyword in ["sales", "business", "commercial"]):
            capabilities.append("business_development")
        if any(keyword in role for keyword in ["writer", "content", "communication"]):
            capabilities.append("content_creation")
        
        # Extract from description (detailed capabilities)
        description = agent_data.get("description", "").lower()
        if any(keyword in description for keyword in ["analyze", "analysis", "examine"]):
            capabilities.append("analytical_thinking")
        if any(keyword in description for keyword in ["design", "architect", "build", "create"]):
            capabilities.append("solution_design")
        if any(keyword in description for keyword in ["implement", "execute", "deploy"]):
            capabilities.append("implementation")
        if any(keyword in description for keyword in ["optimize", "improve", "enhance"]):
            capabilities.append("optimization")
        if any(keyword in description for keyword in ["security", "secure", "protect"]):
            capabilities.append("security_analysis")
        if any(keyword in description for keyword in ["customer", "client", "service"]):
            capabilities.append("customer_focus")
        
        # Extract from system prompt (behavioral patterns)
        prompt = agent_data.get("system_prompt", "").lower()
        if any(keyword in prompt for keyword in ["strategic", "high-level", "vision"]):
            capabilities.append("strategic_thinking")
        if any(keyword in prompt for keyword in ["detail", "thorough", "comprehensive"]):
            capabilities.append("detailed_analysis")
        if any(keyword in prompt for keyword in ["creative", "innovative", "brainstorm"]):
            capabilities.append("creative_thinking")
        if any(keyword in prompt for keyword in ["collaborative", "team", "coordinate"]):
            capabilities.append("collaboration")
        
        # Extract from config metadata (custom capabilities)
        config = agent_data.get("config", {})
        if isinstance(config, dict):
            # Check for custom capability tags
            custom_capabilities = config.get("capabilities", [])
            if isinstance(custom_capabilities, list):
                capabilities.extend(custom_capabilities)
            
            # Check for domain specializations
            domains = config.get("domains", [])
            if isinstance(domains, list):
                capabilities.extend([f"{domain}_expertise" for domain in domains])
            
            # Check for industry focus
            industries = config.get("industries", [])
            if isinstance(industries, list):
                capabilities.extend([f"{industry}_knowledge" for industry in industries])
        
        return list(set(capabilities))  # Remove duplicates
    
    def _intelligent_capability_routing(self, query: str, all_agents: Dict) -> Dict[str, Any]:
        """Intelligent routing based on dynamic capability analysis"""
        query_lower = query.lower()
        
        # Analyze query to identify required capabilities
        required_capabilities = self._analyze_query_requirements(query_lower)
        
        # Score each agent based on capability match
        agent_scores = {}
        capability_matches = {}
        
        for agent_name, agent_data in all_agents.items():
            agent_capabilities = self._extract_capabilities(agent_data)
            
            # Calculate capability match score
            matches = set(required_capabilities).intersection(set(agent_capabilities))
            match_score = len(matches)
            
            # Bonus for exact role matches
            role_bonus = self._calculate_role_bonus(query_lower, agent_data.get("role", ""))
            
            # Tool relevance bonus
            tool_bonus = self._calculate_tool_relevance(query_lower, agent_data.get("tools", []))
            
            # Final score
            total_score = match_score + (role_bonus * 0.5) + (tool_bonus * 0.3)
            
            if total_score > 0:
                agent_scores[agent_name] = total_score
                capability_matches[agent_name] = list(matches)
        
        # Select top agents based on scores
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        selected_agents = [agent for agent, score in sorted_agents[:5]]
        
        # Determine collaboration pattern based on query complexity
        collaboration_pattern = self._determine_collaboration_pattern(query_lower, len(selected_agents))
        
        # Generate reasoning
        reasoning_parts = []
        for agent in selected_agents[:3]:  # Explain top 3 selections
            if agent in capability_matches and capability_matches[agent]:
                reasoning_parts.append(f"{agent}: {', '.join(capability_matches[agent])}")
        
        reasoning = f"Selected based on capability analysis. Required: {', '.join(required_capabilities[:3])}. " + \
                   f"Top matches: {'; '.join(reasoning_parts)}"
        
        # Fallback if no matches found
        if not selected_agents:
            # Find agents with the most tools or general capabilities
            fallback_agents = []
            for agent_name, agent_data in all_agents.items():
                tools_count = len(agent_data.get("tools", []))
                if tools_count > 0:
                    fallback_agents.append((agent_name, tools_count))
            
            fallback_agents.sort(key=lambda x: x[1], reverse=True)
            selected_agents = [agent for agent, _ in fallback_agents[:2]]
            reasoning = "No direct capability matches found. Selected agents with most tools available."
            collaboration_pattern = "parallel"
        
        return {
            "agents": selected_agents,
            "reasoning": reasoning,
            "collaboration_pattern": collaboration_pattern,
            "capability_analysis": {
                "required_capabilities": required_capabilities,
                "agent_scores": dict(sorted_agents[:5]) if sorted_agents else {},
                "capability_matches": capability_matches
            }
        }
    
    def _analyze_query_requirements(self, query_lower: str) -> List[str]:
        """Analyze query to identify required capabilities"""
        requirements = []
        
        # Intent analysis
        if any(word in query_lower for word in ["strategy", "strategic", "plan", "roadmap", "vision"]):
            requirements.append("strategic_planning")
        
        if any(word in query_lower for word in ["cost", "budget", "financial", "roi", "revenue", "profit"]):
            requirements.append("financial_analysis")
        
        if any(word in query_lower for word in ["technical", "architecture", "system", "technology", "implement"]):
            requirements.append("technical_design")
        
        if any(word in query_lower for word in ["risk", "compliance", "security", "audit", "legal"]):
            requirements.append("risk_management")
        
        if any(word in query_lower for word in ["research", "analyze", "investigate", "study", "examine"]):
            requirements.append("research_analysis")
        
        if any(word in query_lower for word in ["document", "pdf", "file", "content", "text"]):
            requirements.append("document_analysis")
        
        if any(word in query_lower for word in ["search", "google", "web", "internet", "online"]):
            requirements.append("web_research")
        
        if any(word in query_lower for word in ["email", "mail", "communication", "message"]):
            requirements.append("email_management")
        
        if any(word in query_lower for word in ["data", "database", "analytics", "metrics", "statistics"]):
            requirements.append("data_analysis")
        
        if any(word in query_lower for word in ["sales", "business", "client", "customer", "market"]):
            requirements.append("business_development")
        
        if any(word in query_lower for word in ["write", "content", "create", "generate", "compose"]):
            requirements.append("content_creation")
        
        if any(word in query_lower for word in ["operations", "process", "workflow", "management"]):
            requirements.append("operations_management")
        
        # If no specific requirements found, default to general capabilities
        if not requirements:
            requirements = ["analytical_thinking", "research_analysis"]
        
        return requirements
    
    def _calculate_role_bonus(self, query_lower: str, role: str) -> float:
        """Calculate bonus score for role relevance"""
        if not role:
            return 0.0
        
        role_lower = role.lower()
        bonus = 0.0
        
        # Direct role mentions in query
        role_keywords = role_lower.split()
        for keyword in role_keywords:
            if len(keyword) > 3 and keyword in query_lower:
                bonus += 1.0
        
        return bonus
    
    def _calculate_tool_relevance(self, query_lower: str, tools: List[str]) -> float:
        """Calculate bonus score for tool relevance"""
        if not tools:
            return 0.0
        
        relevance_score = 0.0
        
        for tool in tools:
            if isinstance(tool, str):
                tool_lower = tool.lower()
                # Check if tool keywords appear in query
                if any(keyword in query_lower for keyword in tool_lower.split('_')):
                    relevance_score += 0.5
        
        return min(relevance_score, 2.0)  # Cap at 2.0
    
    def _determine_collaboration_pattern(self, query_lower: str, agent_count: int) -> str:
        """Determine the best collaboration pattern for the query"""
        
        # Sequential for complex, multi-step processes
        if any(word in query_lower for word in ["step", "process", "workflow", "phase", "sequence"]):
            return "sequential"
        
        # Hierarchical for strategic/executive queries
        if any(word in query_lower for word in ["strategy", "decision", "approve", "executive", "leadership"]):
            return "hierarchical"
        
        # Parallel for research/analysis queries or when multiple perspectives needed
        if agent_count > 2 or any(word in query_lower for word in ["analyze", "research", "compare", "evaluate"]):
            return "parallel"
        
        # Default to parallel for general queries
        return "parallel"
    
    async def execute_agent(self, agent_name: str, agent_data: Dict, query: str, context: Dict = None) -> AsyncGenerator[Dict, None]:
        """Execute any agent dynamically using its system prompt"""
        
        # Import json at the top to ensure it's always available
        import json
        
        print(f"[DEBUG] DynamicMultiAgentSystem.execute_agent called for {agent_name}")
        print(f"[DEBUG] Agent data keys: {list(agent_data.keys()) if agent_data else 'None'}")
        print(f"[DEBUG] execute_agent: Starting async generator for {agent_name}")
        print(f"[DEBUG] Parameters: query_type={type(query)}, query_length={len(query) if query else 0}")
        
        # Initialize variables for Langfuse generation (will be created after prompt is built)
        agent_generation = None
        tracer = None
        if self.trace:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
            except Exception as e:
                print(f"[WARNING] Failed to get tracer for {agent_name}: {e}")
        
        generation_ended = False
        
        def _end_agent_generation(output_text: str = None, error: str = None, success: bool = True, input_prompt: str = None):
            """Safely end agent generation with proper error handling"""
            nonlocal generation_ended
            if agent_generation and tracer and tracer.is_enabled() and not generation_ended:
                try:
                    # Estimate token usage for cost calculation
                    actual_input = input_prompt or query
                    if output_text:
                        usage = tracer.estimate_token_usage(actual_input, output_text)
                    else:
                        usage = tracer.estimate_token_usage(actual_input, "")
                    
                    print(f"[DEBUG] Ending generation for {agent_name} with usage: {usage}")
                    agent_generation.end(
                        output=error if error else (output_text or "Agent completed successfully"),
                        usage_details=usage,
                        metadata={
                            "success": success,
                            "agent_name": agent_name,
                            "error": error if error else None,
                            "output_length": len(output_text) if output_text else 0
                        }
                    )
                    generation_ended = True
                    print(f"[DEBUG] Ended Langfuse generation for agent {agent_name}")
                except Exception as e:
                    print(f"[WARNING] Failed to end agent generation for {agent_name}: {e}")
        
        # Ensure all parameters are valid to prevent scope issues
        if not query:
            query = "No query provided"
        if not agent_data:
            agent_data = {}
        if context is None:
            context = {}
            
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
                # Remove conflicting thinking instruction for tool-using agents
                if agent_data.get("tools"):
                    context_str += "IMPORTANT: Use tools first if needed, then provide your response.\n"
                else:
                    context_str += "IMPORTANT: Provide your response directly. Your actual analysis should be outside any thinking tags. /NO_THINK\n"
            
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
            # Get detailed tool specifications from MCP system
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            mcp_tools = get_enabled_mcp_tools()
            
            if not context_str:
                context_str = f"\n\nCONTEXT:\n"
            context_str += f"\n\nAVAILABLE TOOLS:\n"
            context_str += f"You have access to the following tools: {available_tools}\n"
            
            # Add detailed tool specifications like _dynamic_agent does
            verified_tools = []
            for tool in available_tools:
                if tool in mcp_tools:
                    verified_tools.append(tool)
            
            if verified_tools:
                context_str += "\n**TOOL SPECIFICATIONS** (Use these exact parameter names):"
                for tool_name in verified_tools:
                    tool_info = mcp_tools[tool_name]
                    context_str += f"\n\n• **{tool_name}**: {tool_info.get('description', 'No description available')}"
                    if tool_info.get('parameters'):
                        properties = tool_info['parameters'].get('properties', {})
                        if properties:
                            context_str += f"\n  Required parameters:"
                            for param_name, param_schema in properties.items():
                                param_type = param_schema.get('type', 'string')
                                param_desc = param_schema.get('description', '')
                                context_str += f"\n    - {param_name} ({param_type}): {param_desc}"
                
                # Add example usage
                context_str += f"\n\n**TOOL USAGE EXAMPLE:**"
                example_tool = verified_tools[0]  # Use first available tool as example
                example_params = {}
                example_info = mcp_tools.get(example_tool, {})
                if example_info.get('parameters'):
                    for param_name, param_schema in example_info['parameters'].get('properties', {}).items():
                        param_type = param_schema.get('type', 'string')
                        if param_name == "query":
                            example_params[param_name] = "your search terms here"
                        elif param_name in ["maxResults", "count", "num_results"]:
                            example_params[param_name] = 10
                        elif param_name == "to":
                            example_params[param_name] = ["recipient@email.com"]
                        else:
                            example_params[param_name] = f"<{param_name} value>"
                
                example_json = f'{{"tool": "{example_tool}", "parameters": {json.dumps(example_params)}}}'
                context_str += f'\n{example_json}'
                
                # Add critical instructions - SIMPLIFIED for better model reliability
                context_str += "\n\n**HOW TO USE TOOLS:**"
                context_str += f"\n\nTo use {example_tool}, output this JSON format first:"
                context_str += f"\n{example_json}"
                context_str += "\n\nThen provide your analysis after the tool executes."
            else:
                # Fallback to basic format if tool specs aren't available
                context_str += "\nTo use a tool, output JSON in this format:\n"
                context_str += '{"tool": "tool_name", "parameters": {"param1": "value1"}}'
        
        # Store agent data and sequential workflow flag for _call_llm_stream
        self._current_agent_data = agent_data
        self._is_sequential_workflow = context and "previous_outputs" in context and context["previous_outputs"]
        
        # Enhance prompt for completion tasks
        task_enhancement = ""
        if any(keyword in query.lower() for keyword in ["50", "interview", "questions", "generate", "create"]):
            task_enhancement = "\n\nIMPORTANT: If the user is asking you to generate a specific number of items (like 50 interview questions), you MUST generate the complete requested amount. Do not stop early or provide fewer items than requested."
        
        # Adjust prompt based on whether this is part of a sequential workflow
        if self._is_sequential_workflow:
            # This is a continuation in a sequential workflow
            agent_position = len(context["previous_outputs"]) + 1
            full_prompt = f"""{system_prompt}

YOU ARE AGENT #{agent_position} IN A SEQUENTIAL WORKFLOW. 

The previous agent(s) have already analyzed the user's request and provided their insights above. DO NOT re-analyze what the user wants.
{context_str}

ORIGINAL USER REQUEST (for reference only): {query}

TASK: If you need current information, start by using the available tools. Then provide your analysis based on the results.{task_enhancement}"""
        else:
            # This is a standalone agent or first in sequence  
            if available_tools:
                # More explicit instructions for tool-enabled agents
                full_prompt = f"""{system_prompt}

USER QUERY: {query}{context_str}{task_enhancement}

TASK: Analyze the query above. If you need current information to answer properly, use the available tools first, then provide your strategic analysis."""
            else:
                full_prompt = f"""{system_prompt}

USER QUERY: {query}{context_str}{task_enhancement}

TASK: Provide your analysis of the query above."""

        # Execute agent
        try:
            print(f"[DEBUG] Executing {agent_name} with config: max_tokens={max_tokens}, temp={temperature}, timeout={timeout}")
            print(f"[DEBUG] Prompt length: {len(full_prompt)} chars")
            print(f"[DEBUG] {agent_name} prompt preview (last 200 chars): {full_prompt[-200:]!r}")
            if task_enhancement:
                print(f"[DEBUG] {agent_name}: Task enhancement applied for completion task")
            
            # Create Langfuse generation now that we have the actual input prompt
            if self.trace and tracer and tracer.is_enabled():
                try:
                    model_name = agent_data.get("config", {}).get("model", "qwen3:30b-a3b")
                    print(f"[DEBUG] Creating Langfuse generation for {agent_name} with model: '{model_name}'")
                    agent_generation = tracer.create_generation_with_usage(
                        trace=self.trace,
                        name=f"agent-{agent_name}",
                        model=model_name,
                        input_text=full_prompt,
                        metadata={
                            "agent_role": agent_data.get("role", ""),
                            "tools_available": agent_data.get("tools", []),
                            "context_keys": list(context.keys()) if context else [],
                            "agent_name": agent_name,
                            "prompt_length": len(full_prompt),
                            "has_context": bool(context and context.get("previous_outputs")),
                            "agent_position": len(context.get("previous_outputs", [])) + 1 if context else 1
                        }
                    )
                    print(f"[DEBUG] Created Langfuse generation for agent {agent_name} with actual prompt")
                except Exception as e:
                    print(f"[WARNING] Failed to create agent generation for {agent_name}: {e}")
            
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
                    print(f"[DEBUG] execute_agent: Agent {agent_name} completed, processing tools...")
                    
                    # Process tool calls with proper variable access
                    print(f"[DEBUG] {agent_name}: About to call _process_tool_calls")
                    print(f"[DEBUG] {agent_name}: Variables check - agent_name: {agent_name is not None}, content: {len(response_text)}")
                    
                    enhanced_content, tool_results = await self._process_tool_calls(agent_name, response_text, query, context, agent_data)
                    
                    # Enhance the completion event with avatar, description, and tool results
                    enhanced_chunk = {
                        **chunk,
                        "content": enhanced_content,  # Use enhanced content with tool results
                        "avatar": self.get_agent_avatar(agent_name, agent_data.get("role", "")),
                        "description": agent_data.get("description", "")
                    }
                    
                    # Add tool results if any were executed
                    if tool_results:
                        enhanced_chunk["tools_used"] = tool_results
                    
                    print(f"[DEBUG] execute_agent: Agent {agent_name} yielding enhanced completion")
                    _end_agent_generation(output_text=enhanced_content, success=True, input_prompt=full_prompt)
                    yield enhanced_chunk
                elif chunk.get("type") == "agent_error":
                    error_occurred = True
                    print(f"[DEBUG] execute_agent: Agent {agent_name} error")
                    _end_agent_generation(error=chunk.get("error", "Agent error occurred"), success=False, input_prompt=full_prompt)
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
            _end_agent_generation(error=str(e), success=False, input_prompt=full_prompt if 'full_prompt' in locals() else query)
            yield {
                "type": "agent_error",
                "agent": agent_name,
                "error": str(e)
            }
        finally:
            print(f"[DEBUG] execute_agent: Finished (finally block) for {agent_name}")
            # Safety cleanup: end generation if it hasn't been ended yet
            if not generation_ended and agent_generation and tracer and tracer.is_enabled():
                try:
                    # This will only end if not already ended
                    _end_agent_generation(output_text="Agent execution completed", success=True, input_prompt=full_prompt if 'full_prompt' in locals() else query)
                except Exception:
                    pass  # Ignore if already ended
    
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
            
            # Clean response and return it for tool processing in the calling method
            print(f"[DEBUG] {agent_name}: Calling _clean_response on {len(response_text)} chars")
            cleaned_content = self._clean_response(response_text)
            print(f"[DEBUG] {agent_name}: After cleaning: {len(cleaned_content)} chars")
            print(f"[DEBUG] {agent_name}: Cleaned preview (first 200 chars): {cleaned_content[:200]!r}")
            
            # Check for incomplete responses (common with tool-enabled agents)
            if len(cleaned_content.strip()) < 10 and cleaned_content.strip() in ['<think>', '<think', 'think>', '']:
                print(f"[WARNING] {agent_name}: Generated incomplete response, likely model generation issue")
                cleaned_content = f"I need to analyze this request. Let me search for current information to provide accurate insights."
                print(f"[DEBUG] {agent_name}: Using fallback response for incomplete generation")
            
            # Return the cleaned content - tool processing will be handled by execute_agent
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
            
            # Note: _end_agent_generation will be handled in the calling method
            yield {
                "type": "agent_complete",
                "agent": agent_name,
                "content": timeout_content,
                "timeout": True,
                "timeout_duration": timeout
            }
        except Exception as e:
            print(f"[ERROR] Agent {agent_name} failed: {e}")
            import traceback
            print(f"[ERROR] Full traceback: {traceback.format_exc()}")
            
            # Note: _end_agent_generation will be handled in the calling method
            yield {
                "type": "agent_error", 
                "agent": agent_name,
                "error": str(e)
            }
    
    async def _process_tool_calls(self, agent_name: str, agent_response: str, query: str, context: Dict = None, agent_data: Dict = None) -> tuple[str, list]:
        """Process agent response for tool calls and execute them using intelligent tool system"""
        try:
            print(f"[DEBUG] {agent_name}: _process_tool_calls started with query: {query[:50]}...")
            print(f"[DEBUG] {agent_name}: Method parameters - query type: {type(query)}, context: {context is not None}, agent_data: {agent_data is not None}")
            
            # Use legacy tool executor to parse agent JSON responses (like standard chat)
            try:
                from app.langchain.tool_executor import tool_executor
                use_intelligent_tools = False
                use_intelligent_tool_results = False
                print(f"[DEBUG] {agent_name}: Using legacy tool executor to parse agent JSON responses")
            except ImportError:
                print(f"[DEBUG] {agent_name}: No tool executor available")
                use_intelligent_tools = False
                use_intelligent_tool_results = False
            
            print(f"[DEBUG] {agent_name}: _process_tool_calls called with response length: {len(agent_response)}")
            print(f"[DEBUG] {agent_name}: Response preview: {agent_response[:200]}")
            
            if use_intelligent_tools:
                # Use intelligent tool system for multi-agent mode
                print(f"[DEBUG] {agent_name}: Analyzing agent response for intelligent tool execution")
                
                # Create agent execution span if trace is available
                agent_span = None
                if self.trace:
                    try:
                        from app.core.langfuse_integration import get_tracer
                        tracer = get_tracer()
                        if tracer.is_enabled():
                            agent_span = tracer.create_agent_execution_span(
                                self.trace, agent_name, query, {"agent_response_length": len(agent_response)}
                            )
                    except Exception as e:
                        print(f"[DEBUG] {agent_name}: Failed to create agent span: {e}")
                
                # Execute tools using intelligent system - detect pipeline mode for proper constraints
                try:
                    # Check if we're running in pipeline mode to enforce proper tool constraints
                    pipeline_id = None
                    
                    # Method 1: Check conversation_context for pipeline info
                    pipeline_context = context.get("conversation_context", {}) if context else {}
                    pipeline_info = pipeline_context.get("pipeline", {}) if pipeline_context else {}
                    pipeline_id = pipeline_info.get("id") or pipeline_context.get("pipeline_id")
                    
                    # Method 1.5: Check pipeline_context directly for pipeline_id
                    if not pipeline_id and context:
                        direct_pipeline_context = context.get("pipeline_context", {})
                        pipeline_id = direct_pipeline_context.get("pipeline_id")
                    
                    # Method 2: Check direct context keys
                    if not pipeline_id and context:
                        for key in ["pipeline_id", "execution_id"]:
                            if key in context:
                                pipeline_id = context[key]
                                break
                    
                    # Method 3: Check if self.trace has pipeline info (from multi-agent system config)
                    if not pipeline_id and hasattr(self, 'current_config') and self.current_config:
                        config_pipeline = self.current_config.get("pipeline", {})
                        pipeline_id = config_pipeline.get("id")
                    
                    # Method 4: Check agent_data for pipeline context
                    if not pipeline_id and agent_data:
                        agent_pipeline_ctx = agent_data.get("pipeline_context", {})
                        pipeline_id = agent_pipeline_ctx.get("pipeline_id")
                    
                    # Method 5: Check if trace metadata contains pipeline info
                    if not pipeline_id and self.trace and hasattr(self.trace, 'metadata'):
                        trace_meta = getattr(self.trace, 'metadata', {}) or {}
                        pipeline_id = trace_meta.get("pipeline_id")
                    
                    # Method 6: Simple heuristic - if we have a multi-agent system and conversation_id looks like execution ID
                    if not pipeline_id and hasattr(self, 'conversation_id') and self.conversation_id:
                        # Pipeline execution IDs are typically numeric strings
                        if self.conversation_id.isdigit():
                            print(f"[DEBUG] {agent_name}: Detected potential pipeline execution based on numeric conversation_id")
                            pipeline_id = self.conversation_id
                    
                    # Convert pipeline_id to integer if it's a string
                    if pipeline_id and isinstance(pipeline_id, str) and pipeline_id.isdigit():
                        pipeline_id = int(pipeline_id)
                    elif pipeline_id and not isinstance(pipeline_id, int):
                        pipeline_id = None  # Invalid format
                    
                    print(f"[DEBUG] {agent_name}: Pipeline detection - pipeline_id: {pipeline_id} (type: {type(pipeline_id)})")
                    
                    if pipeline_id:
                        # Use pipeline-specific tool execution with proper constraints
                        print(f"[DEBUG] {agent_name}: Using pipeline tool execution with constraints")
                        from app.langchain.intelligent_tool_integration import execute_pipeline_agent_tools
                        execution_events = await execute_pipeline_agent_tools(
                            task=query,
                            agent_name=agent_name,
                            pipeline_id=str(pipeline_id),
                            context={
                                "agent_response": agent_response,
                                "conversation_context": context,
                                "agent_data": agent_data
                            },
                            trace=agent_span if agent_span else self.trace
                        )
                    else:
                        # Use multi-agent tool execution (original behavior)
                        print(f"[DEBUG] {agent_name}: Using multi-agent tool execution")
                        from app.langchain.intelligent_tool_integration import execute_multi_agent_tools
                        execution_events = await execute_multi_agent_tools(
                            task=query,
                            agent_name=agent_name,
                            context={
                                "agent_response": agent_response,
                                "conversation_context": context,
                                "agent_data": agent_data
                            },
                            trace=agent_span if agent_span else self.trace
                        )
                    
                    # Process execution events and extract results
                    tool_results = []
                    for event in execution_events:
                        if event.get("type") == "tool_complete" and event.get("success"):
                            tool_results.append({
                                "tool": event.get("tool_name"),
                                "success": True,
                                "result": event.get("result"),
                                "execution_time": event.get("execution_time")
                            })
                    
                    # End agent span
                    if agent_span:
                        try:
                            from app.core.langfuse_integration import get_tracer
                            tracer = get_tracer()
                            tracer.end_span_with_result(
                                agent_span,
                                {
                                    "tools_executed": len(tool_results),
                                    "successful_tools": sum(1 for r in tool_results if r.get("success")),
                                    "execution_events": len(execution_events)
                                },
                                success=len(tool_results) > 0
                            )
                        except Exception as e:
                            print(f"[DEBUG] {agent_name}: Failed to end agent span: {e}")
                    
                    # Return tool results for unified processing below
                    print(f"[DEBUG] {agent_name}: Intelligent tools executed, will process results in unified flow")
                    # Set flag to use intelligent tool results in unified processing
                    use_intelligent_tool_results = True
                        
                except Exception as e:
                    print(f"[DEBUG] {agent_name}: Intelligent tool execution failed: {e}")
                    # Fall through to legacy tool processing
                    use_intelligent_tools = False
                    use_intelligent_tool_results = False
            
            if not use_intelligent_tools:
                # Legacy tool processing (fallback)
                print(f"[DEBUG] {agent_name}: Using legacy tool processing")
                
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
                    # Create tool span for tracing if trace is available
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tool_span = None
                    
                    if self.trace and tracer.is_enabled():
                        try:
                            print(f"[DEBUG] Creating tool span for {tool_name} by agent {agent_name}")
                            # Add agent info to parameters
                            params_with_agent = dict(parameters) if isinstance(parameters, dict) else {}
                            params_with_agent["agent"] = agent_name
                            
                            tool_span = tracer.create_tool_span(
                                self.trace, 
                                tool_name, 
                                params_with_agent
                            )
                            print(f"[DEBUG] Tool span created: {tool_span is not None}")
                        except Exception as e:
                            print(f"[ERROR] Failed to create tool span: {e}")
                            tool_span = None
                    else:
                        print(f"[DEBUG] No tool span created - trace: {self.trace is not None}, tracer enabled: {tracer.is_enabled() if tracer else False}")
                    
                    try:
                        print(f"[DEBUG] Thread execution: {tool_name} with parameters: {parameters}")
                        
                        # DISABLE enhanced error handling in threads to prevent duplicate executions
                        # The enhanced error handling is already being triggered at a higher level
                        # and parallel thread execution + retry logic causes conflicts
                        use_enhanced_error_handling = False
                        
                        if use_enhanced_error_handling:
                            # Use enhanced error handling with retry logic
                            from app.core.tool_error_handler import call_mcp_tool_with_retry, RetryConfig
                            import asyncio
                            
                            retry_config = RetryConfig(
                                max_retries=agent_data.get("max_tool_retries", 3),
                                base_delay=agent_data.get("retry_base_delay", 1.0),
                                max_delay=agent_data.get("retry_max_delay", 30.0)
                            )
                            
                            # Execute with retry logic in a new event loop (since we're in a thread)
                            def run_async():
                                return asyncio.run(call_mcp_tool_with_retry(tool_name, parameters, trace=self.trace, retry_config=retry_config))
                            
                            result = run_async()
                        else:
                            # Import and call MCP tool in thread context (legacy)
                            from app.langchain.service import call_mcp_tool
                            # Skip span creation in call_mcp_tool since we already created it
                            result = call_mcp_tool(tool_name, parameters, trace=self.trace, _skip_span_creation=True)
                        
                        # Check result and handle enhanced error information
                        if isinstance(result, dict) and "error" in result:
                            error_msg = result["error"]
                            error_type = result.get("error_type", "unknown")
                            attempts = result.get("attempts", 1)
                            
                            # End tool span with error
                            if tool_span:
                                try:
                                    print(f"[DEBUG] Ending tool span for {tool_name} with error: {error_msg}")
                                    tracer.end_span_with_result(tool_span, result, False, error_msg)
                                except Exception as e:
                                    print(f"[ERROR] Failed to end tool span: {e}")
                            
                            if attempts > 1:
                                print(f"[ERROR] Tool {tool_name} failed after {attempts} attempts ({error_type}): {error_msg}")
                            else:
                                print(f"[ERROR] Tool {tool_name} failed ({error_type}): {error_msg}")
                            
                            return {
                                "tool": tool_name,
                                "parameters": parameters,
                                "result": result,
                                "success": False,
                                "error_type": error_type,
                                "attempts": attempts
                            }
                        else:
                            # Success
                            # End tool span with success
                            if tool_span:
                                try:
                                    print(f"[DEBUG] Ending tool span for {tool_name} with success")
                                    tracer.end_span_with_result(tool_span, result, True)
                                except Exception as e:
                                    print(f"[ERROR] Failed to end tool span: {e}")
                            
                            return {
                                "tool": tool_name,
                                "parameters": parameters,
                                "result": result,
                                "success": _is_tool_result_successful(result)
                            }
                        
                    except Exception as e:
                        print(f"[ERROR] Thread execution failed for {tool_name}: {e}")
                        
                        # End tool span with error
                        if tool_span:
                            try:
                                print(f"[DEBUG] Ending tool span for {tool_name} with error")
                                tracer.end_span_with_result(tool_span, None, False, str(e)[:500])
                            except Exception as span_error:
                                print(f"[ERROR] Failed to end tool span with error: {span_error}")
                        
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
            else:
                # Intelligent tools path - no legacy tool execution needed
                tool_results = []
            
            # If tools were executed successfully, generate a follow-up response that incorporates the results
            # UNIFIED TOOL RESULT PROCESSING - consolidate both intelligent and legacy results
            if (tool_results and any(result.get("success") for result in tool_results)) or use_intelligent_tool_results:
                print(f"[DEBUG] {agent_name}: Tools executed successfully, generating unified follow-up response")
                
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
                
                print(f"[DEBUG] {agent_name}: Building unified follow-up prompt with query length: {len(query)}")
                
                # Build the follow-up prompt carefully to avoid scoping issues
                user_query_part = f"USER QUERY: {query}"
                
                follow_up_prompt = f"""{system_prompt}

{user_query_part}

You have already executed the necessary tools and received the following results:
{tool_context}

IMPORTANT: Do NOT generate any new tool calls. The tools have already been executed.

Based on these tool results, provide your final comprehensive response that:
1. Directly answers the user's question
2. Incorporates the relevant information from the tool results
3. Synthesizes the findings into a coherent response
4. Does not repeat the raw tool output but interprets and explains it
5. NEVER includes tool JSON - only provide your analysis and response

Provide your complete response based on the tool results above. Do not generate any tool calls."""

                # Generate SINGLE follow-up response that incorporates tool results (UNIFIED PROCESSING)
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
                print(f"[DEBUG] {agent_name}: Generated unified follow-up response incorporating tool results (length: {len(follow_up_response)})")
                
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
        """Parse JSON from LLM response with enhanced error handling"""
        import re
        
        print(f"[DEBUG] Full routing response: {response}")
        
        # Remove thinking tags if present
        cleaned_response = response
        if "<think>" in response and "</think>" in response:
            # Extract content after thinking tags
            think_end = response.find("</think>")
            if think_end != -1:
                cleaned_response = response[think_end + 8:].strip()
                print(f"[DEBUG] Extracted content after thinking tags: {cleaned_response[:200]}...")
        
        # Try multiple JSON extraction strategies
        json_patterns = [
            r'\{[^{}]*"agents"[^{}]*\}',  # Simple JSON with agents field
            r'\{.*?"agents".*?\}',        # More flexible JSON with agents
            r'\{.*\}',                    # Any JSON-like structure
        ]
        
        for pattern in json_patterns:
            json_matches = re.findall(pattern, cleaned_response, re.DOTALL)
            for json_match in json_matches:
                try:
                    print(f"[DEBUG] Trying to parse JSON: {json_match[:200]}...")
                    result = json.loads(json_match)
                    if "agents" in result:
                        # Ensure order field exists
                        if "order" not in result:
                            result["order"] = result["agents"]
                        print(f"[DEBUG] Successfully parsed JSON: {result}")
                        return result
                except json.JSONDecodeError as e:
                    print(f"[DEBUG] JSON parse failed: {e}")
                    continue
        
        # Try to extract agents list even if full JSON fails
        agent_match = re.search(r'"agents"\s*:\s*\[(.*?)\]', cleaned_response, re.DOTALL)
        if agent_match:
            try:
                agents_str = agent_match.group(1)
                # Extract agent names from the list
                agent_names = re.findall(r'"([^"]+)"', agents_str)
                if agent_names:
                    print(f"[DEBUG] Extracted agents from partial parse: {agent_names}")
                    return {
                        "agents": agent_names,
                        "reasoning": "Extracted from partial JSON parse",
                        "collaboration_pattern": "parallel",
                        "order": agent_names
                    }
            except Exception as e:
                print(f"[DEBUG] Partial agent extraction failed: {e}")
        
        print(f"[DEBUG] All JSON parsing strategies failed for response: {cleaned_response[:500]}...")
        
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