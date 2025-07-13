"""
Intelligent Tool Planning System

Replaces regex-based tool extraction with intelligent task analysis and tool planning.
Uses LLM to understand tasks and select appropriate tools based on capabilities.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ToolPlan:
    """Represents a planned tool execution"""
    tool_name: str
    purpose: str
    parameters: Dict[str, Any]
    depends_on: Optional[List[str]] = None  # Dependencies on previous tool results
    context_keys: Optional[List[str]] = None  # What context this tool needs

@dataclass
class ExecutionPlan:
    """Complete execution plan for a task"""
    task: str
    tools: List[ToolPlan]
    reasoning: str
    estimated_duration: Optional[int] = None
    fallback_plan: Optional[List[ToolPlan]] = None

class IntelligentToolPlanner:
    """
    Intelligent tool planning system that uses LLM reasoning to select and sequence tools
    based on task requirements and available tool capabilities.
    """
    
    def __init__(self):
        self.planning_cache = {}  # Cache successful plans for similar tasks
        
    def get_enhanced_tool_metadata(self, mode: str = "standard", agent_name: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get enhanced tool metadata with capabilities and use cases, 
        respecting tool constraints based on mode
        
        Args:
            mode: "standard", "multi_agent", or "pipeline"
            agent_name: Required for multi_agent and pipeline modes
            pipeline_id: Required for pipeline mode
        """
        try:
            from app.core.mcp_tools_cache import get_enabled_mcp_tools
            raw_tools = get_enabled_mcp_tools()
            
            # Get available tools based on mode
            available_tool_names = self._get_available_tools_for_mode(mode, agent_name)
            
            enhanced_tools = {}
            for tool_name, tool_data in raw_tools.items():
                # Skip tools not available for this mode/agent
                if available_tool_names is not None and tool_name not in available_tool_names:
                    continue
                    
                # Extract parameters schema for better understanding
                parameters_schema = tool_data.get("parameters", {})
                
                # Analyze tool capabilities from description and parameters
                capabilities = self._analyze_tool_capabilities(
                    tool_name, 
                    tool_data.get("description", ""),
                    parameters_schema
                )
                
                enhanced_tools[tool_name] = {
                    "name": tool_name,
                    "description": tool_data.get("description", ""),
                    "parameters": parameters_schema,
                    "capabilities": capabilities,
                    "category": self._categorize_tool(tool_name, tool_data.get("description", "")),
                    "complexity": self._estimate_complexity(parameters_schema),
                    "dependencies": self._identify_dependencies(tool_name, parameters_schema),
                    "access_mode": mode,
                    "assigned_agent": agent_name if mode != "standard" else None
                }
                
            logger.info(f"[TOOL PLANNER] Enhanced metadata for {len(enhanced_tools)} tools (mode: {mode}, agent: {agent_name})")
            return enhanced_tools
            
        except Exception as e:
            logger.error(f"[TOOL PLANNER] Failed to get tool metadata: {e}")
            return {}
    
    def _get_available_tools_for_mode(self, mode: str, agent_name: str = None) -> Optional[List[str]]:
        """
        Get list of available tool names based on mode and constraints
        
        Returns:
            None for standard mode (all tools available)
            List[str] for multi_agent/pipeline modes (only assigned tools)
        """
        if mode == "standard":
            # Standard chat: all MCP tools available
            return None
            
        elif mode == "multi_agent":
            # MULTI-AGENT MODE: Use ONLY langgraph_agents table
            if not agent_name:
                logger.warning("[TOOL PLANNER] Agent name required for multi_agent mode")
                return []
                
            try:
                from app.core.langgraph_agents_cache import get_agent_by_name
                logger.info(f"[TOOL PLANNER] MULTI-AGENT MODE: Getting tools for {agent_name} from langgraph_agents table")
                agent_data = get_agent_by_name(agent_name)
                if agent_data and "tools" in agent_data:
                    tools = agent_data["tools"]
                    if isinstance(tools, list):
                        logger.info(f"[TOOL PLANNER] Multi-agent {agent_name} has {len(tools)} assigned tools: {tools}")
                        return tools
                    else:
                        logger.warning(f"[TOOL PLANNER] Multi-agent {agent_name} tools not in list format: {type(tools)}")
                        return []
                else:
                    logger.error(f"[TOOL PLANNER] Multi-agent {agent_name} not found in langgraph_agents table or has no tools")
                    return []
                    
            except Exception as e:
                logger.error(f"[TOOL PLANNER] Failed to get tools for multi-agent {agent_name}: {e}")
                return []
                
        
        else:
            logger.warning(f"[TOOL PLANNER] Unknown mode: {mode}")
            return []
    
    def _analyze_tool_capabilities(self, tool_name: str, description: str, parameters: Dict) -> List[str]:
        """Analyze what capabilities a tool provides based on name, description, and parameters"""
        capabilities = []
        
        # Name-based capabilities
        name_lower = tool_name.lower()
        if "search" in name_lower:
            capabilities.extend(["search", "find", "lookup", "discover"])
        if "email" in name_lower:
            capabilities.extend(["email", "communication", "messaging"])
        if "read" in name_lower:
            capabilities.extend(["read", "retrieve", "access", "get"])
        if "write" in name_lower or "create" in name_lower:
            capabilities.extend(["write", "create", "generate", "compose"])
        if "analyze" in name_lower or "summary" in name_lower:
            capabilities.extend(["analyze", "summarize", "process", "understand"])
        if "calculate" in name_lower or "math" in name_lower:
            capabilities.extend(["calculate", "compute", "math", "numbers"])
        if "web" in name_lower or "http" in name_lower:
            capabilities.extend(["web", "internet", "online", "fetch"])
        if "file" in name_lower or "document" in name_lower:
            capabilities.extend(["file", "document", "storage", "save"])
        
        # Description-based capabilities
        desc_lower = description.lower()
        if "download" in desc_lower:
            capabilities.append("download")
        if "upload" in desc_lower:
            capabilities.append("upload")
        if "delete" in desc_lower:
            capabilities.append("delete")
        if "update" in desc_lower:
            capabilities.append("update")
        if "translate" in desc_lower:
            capabilities.append("translate")
        if "convert" in desc_lower:
            capabilities.append("convert")
        
        # Parameter-based capabilities
        if parameters:
            param_keys = [key.lower() for key in parameters.keys()]
            if "query" in param_keys:
                capabilities.append("query-based")
            if "url" in param_keys:
                capabilities.append("url-based")
            if "file" in param_keys or "path" in param_keys:
                capabilities.append("file-based")
        
        return list(set(capabilities))  # Remove duplicates
    
    def _categorize_tool(self, tool_name: str, description: str) -> str:
        """Categorize tool into functional categories"""
        name_lower = tool_name.lower()
        desc_lower = description.lower()
        
        if "email" in name_lower or "email" in desc_lower:
            return "communication"
        elif "search" in name_lower or "find" in desc_lower:
            return "search"
        elif "web" in name_lower or "http" in name_lower:
            return "web"
        elif "file" in name_lower or "document" in desc_lower:
            return "file_management"
        elif "calculate" in name_lower or "math" in desc_lower:
            return "computation"
        elif "analyze" in desc_lower or "summary" in desc_lower:
            return "analysis"
        elif "create" in desc_lower or "generate" in desc_lower:
            return "creation"
        else:
            return "utility"
    
    def _estimate_complexity(self, parameters: Dict) -> str:
        """Estimate tool complexity based on parameters"""
        if not parameters:
            return "simple"
        
        param_count = len(parameters)
        if param_count <= 2:
            return "simple"
        elif param_count <= 5:
            return "medium"
        else:
            return "complex"
    
    def _identify_dependencies(self, tool_name: str, parameters: Dict) -> List[str]:
        """Identify what this tool might depend on"""
        dependencies = []
        
        if parameters:
            for param_key in parameters.keys():
                param_lower = param_key.lower()
                if "id" in param_lower or "reference" in param_lower:
                    dependencies.append("requires_id")
                if "url" in param_lower:
                    dependencies.append("requires_url")
                if "content" in param_lower or "text" in param_lower:
                    dependencies.append("requires_content")
        
        return dependencies

    async def plan_tool_execution(
        self, 
        task: str, 
        context: Dict[str, Any] = None, 
        mode: str = "standard",
        agent_name: str = None,
    ) -> ExecutionPlan:
        """
        Create an intelligent execution plan for a task using available tools
        
        Args:
            task: The task to accomplish
            context: Additional context (conversation history, previous results, etc.)
            mode: "standard", "multi_agent", or "pipeline" 
            agent_name: Required for multi_agent and pipeline modes
            pipeline_id: Required for pipeline mode
            
        Returns:
            ExecutionPlan with tools, reasoning, and fallback options
        """
        try:
            # Create cache key that includes mode constraints
            cache_key = self._generate_cache_key(task, mode, agent_name)
            if cache_key in self.planning_cache:
                cached_plan = self.planning_cache[cache_key]
                logger.info(f"[TOOL PLANNER] Using cached plan for similar task (mode: {mode})")
                return cached_plan
            
            # Get available tools with enhanced metadata respecting mode constraints
            available_tools = self.get_enhanced_tool_metadata(mode, agent_name)
            if not available_tools:
                logger.warning(f"[TOOL PLANNER] No tools available for planning (mode: {mode}, agent: {agent_name})")
                return ExecutionPlan(
                    task=task, 
                    tools=[], 
                    reasoning=f"No tools available for {mode} mode" + (f" agent {agent_name}" if agent_name else "")
                )
            
            # Create planning prompt with mode-specific context
            planning_prompt = self._create_planning_prompt(task, available_tools, context, mode, agent_name)
            
            # Get LLM to create execution plan
            plan_response = await self._get_llm_plan(planning_prompt)
            
            # Parse and validate the plan
            execution_plan = self._parse_execution_plan(plan_response, available_tools)
            
            # Add mode information to execution plan
            execution_plan.reasoning += f"\n\nMode: {mode}" + (f", Agent: {agent_name}" if agent_name else "")
            
            # Add to cache
            self.planning_cache[cache_key] = execution_plan
            
            logger.info(f"[TOOL PLANNER] Created plan with {len(execution_plan.tools)} tools for {mode} mode: {[t.tool_name for t in execution_plan.tools]}")
            return execution_plan
            
        except Exception as e:
            logger.error(f"[TOOL PLANNER] Planning failed: {e}")
            # Return minimal plan for fallback
            return ExecutionPlan(
                task=task, 
                tools=[], 
                reasoning=f"Planning failed: {str(e)}"
            )
    
    def _generate_cache_key(self, task: str, mode: str = "standard", agent_name: str = None) -> str:
        """Generate cache key for task (simplified for similar tasks)"""
        # Normalize task for caching similar requests
        normalized = task.lower().strip()
        # Remove common variations to group similar tasks
        normalized = normalized.replace("please", "").replace("can you", "").replace("could you", "")
        
        # Include mode constraints in cache key
        cache_components = [normalized, mode]
        if agent_name:
            cache_components.append(agent_name)
            
        return str(hash(tuple(cache_components)))
    
    def _create_planning_prompt(self, task: str, tools: Dict[str, Any], context: Dict[str, Any] = None, mode: str = "standard", agent_name: str = None) -> str:
        """Create detailed planning prompt for LLM"""
        
        # Organize tools by category for better LLM understanding
        tools_by_category = {}
        for tool_name, tool_info in tools.items():
            category = tool_info.get("category", "utility")
            if category not in tools_by_category:
                tools_by_category[category] = []
            tools_by_category[category].append(tool_info)
        
        tools_description = ""
        for category, category_tools in tools_by_category.items():
            tools_description += f"\n## {category.upper()} TOOLS:\n"
            for tool in category_tools:
                tools_description += f"- **{tool['name']}**: {tool['description']}\n"
                tools_description += f"  Parameters: {json.dumps(tool['parameters'], indent=2)}\n"
                tools_description += f"  Capabilities: {', '.join(tool['capabilities'])}\n"
                tools_description += f"  Complexity: {tool['complexity']}\n\n"
        
        context_info = ""
        if context:
            context_info = f"\nCONTEXT:\n{json.dumps(context, indent=2)}\n"
        
        # Add mode-specific constraints and information
        mode_info = ""
        if mode == "multi_agent":
            mode_info = f"""
EXECUTION MODE: Multi-Agent System
AGENT: {agent_name}
CONSTRAINT: You can ONLY use tools that have been assigned to this specific agent.
The tools listed below are the ONLY tools available to agent '{agent_name}'.
"""
        elif mode == "pipeline":
            mode_info = f"""
EXECUTION MODE: Agentic Pipeline
AGENT: {agent_name}
CONSTRAINT: You can ONLY use tools that have been assigned to this agent within the pipeline context.
The tools listed below are the ONLY tools available to this pipeline agent.
"""
        elif mode == "standard":
            mode_info = """
EXECUTION MODE: Standard Chat
CONSTRAINT: Use ONLY tools that are DIRECTLY NECESSARY to accomplish the user's specific request.
- Be conservative and precise - avoid over-executing tools
- Do NOT perform actions the user didn't explicitly request (don't send emails, create tickets, or delete data unless explicitly asked)
- Focus on information gathering and retrieval unless the user specifically asks for actions
- If the user asks to "find" something, only use search/find tools, not creation or modification tools
"""
        
        prompt = f"""You are an intelligent task planner. Analyze the task and create an optimal execution plan using available tools.

{mode_info}

TASK TO ACCOMPLISH:
{task}

{context_info}

AVAILABLE TOOLS:
{tools_description}

PLANNING INSTRUCTIONS:
1. Understand what the task is asking for
2. Identify which tools can help accomplish this task
3. Consider dependencies between tools (some tools need output from others)
4. Plan the optimal sequence of tool executions
5. For each tool, explain why it's needed and what parameters to use
6. Consider error scenarios and provide fallback options

RESPONSE FORMAT (JSON):
{{
    "analysis": "Brief analysis of what the task requires",
    "plan": [
        {{
            "tool_name": "exact_tool_name",
            "purpose": "why this tool is needed for the task",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "depends_on": ["previous_tool_name"] or null,
            "context_keys": ["context_key_needed"] or null
        }}
    ],
    "reasoning": "detailed explanation of the plan and tool sequence",
    "fallback_plan": [
        {{
            "tool_name": "alternative_tool",
            "purpose": "fallback purpose",
            "parameters": {{"param": "value"}}
        }}
    ],
    "estimated_duration": estimated_seconds_integer
}}

IMPORTANT:
- Only use tools that exist in the available tools list
- Parameters must match the tool's parameter schema
- Consider that tool outputs can be used as inputs for subsequent tools
- If no tools are suitable, return empty plan array
- Be specific about parameter values, don't use placeholders like "search_term"
- CRITICAL: Only plan tools that are DIRECTLY needed for the user's request
- DO NOT plan tools for actions the user didn't explicitly ask for
- For "find" requests, use ONLY search/find tools, not creation, modification, or deletion tools
- Minimize tool usage - prefer 1-3 relevant tools over many tools
"""
        
        return prompt
    
    async def _get_llm_plan(self, planning_prompt: str) -> str:
        """Get LLM response for planning"""
        try:
            import asyncio
            
            # Import LLM function from existing system
            from app.langchain.service import make_llm_call
            
            # Use existing LLM settings
            from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config
            llm_settings = get_llm_settings()
            
            # Validate required settings using helper function
            try:
                main_llm_config = get_main_llm_full_config(llm_settings)
                if not main_llm_config or not main_llm_config.get('model'):
                    logger.error("[TOOL PLANNER] LLM settings not properly configured")
                    raise ValueError("LLM settings not available")
            except Exception as e:
                logger.error(f"[TOOL PLANNER] LLM settings validation failed: {e}")
                raise ValueError("LLM settings not available")
            
            # Run synchronous LLM call in executor to make it async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,  # Use default executor
                lambda: make_llm_call(
                    prompt=planning_prompt,
                    thinking=False,  # Use non-thinking mode for structured planning
                    context="",  # No additional context needed
                    llm_cfg=llm_settings
                )
            )
            
            return response
            
        except Exception as e:
            logger.error(f"[TOOL PLANNER] LLM call failed: {e}")
            raise
    
    def _parse_execution_plan(self, llm_response: str, available_tools: Dict[str, Any]) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan"""
        try:
            # Extract JSON from LLM response
            response_text = llm_response.strip()
            
            # Find JSON block
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in LLM response")
            
            json_text = response_text[json_start:json_end]
            plan_data = json.loads(json_text)
            
            # Validate and create ToolPlan objects
            tools = []
            for tool_spec in plan_data.get("plan", []):
                tool_name = tool_spec.get("tool_name")
                
                # Validate tool exists
                if tool_name not in available_tools:
                    logger.warning(f"[TOOL PLANNER] Tool '{tool_name}' not available, skipping")
                    continue
                
                # Validate parameters against tool schema
                parameters = self._validate_parameters(
                    tool_spec.get("parameters", {}),
                    available_tools[tool_name].get("parameters", {})
                )
                
                tool_plan = ToolPlan(
                    tool_name=tool_name,
                    purpose=tool_spec.get("purpose", ""),
                    parameters=parameters,
                    depends_on=tool_spec.get("depends_on"),
                    context_keys=tool_spec.get("context_keys")
                )
                tools.append(tool_plan)
            
            # Create fallback plan
            fallback_tools = []
            for fallback_spec in plan_data.get("fallback_plan", []):
                tool_name = fallback_spec.get("tool_name")
                if tool_name in available_tools:
                    parameters = self._validate_parameters(
                        fallback_spec.get("parameters", {}),
                        available_tools[tool_name].get("parameters", {})
                    )
                    fallback_tools.append(ToolPlan(
                        tool_name=tool_name,
                        purpose=fallback_spec.get("purpose", ""),
                        parameters=parameters
                    ))
            
            return ExecutionPlan(
                task=plan_data.get("analysis", ""),
                tools=tools,
                reasoning=plan_data.get("reasoning", ""),
                estimated_duration=plan_data.get("estimated_duration"),
                fallback_plan=fallback_tools if fallback_tools else None
            )
            
        except Exception as e:
            logger.error(f"[TOOL PLANNER] Failed to parse execution plan: {e}")
            logger.debug(f"[TOOL PLANNER] LLM response: {llm_response}")
            
            # Return empty plan as fallback
            return ExecutionPlan(
                task="Parse failed",
                tools=[],
                reasoning=f"Failed to parse LLM response: {str(e)}"
            )
    
    def _validate_parameters(self, provided_params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean parameters against tool schema"""
        validated = {}
        
        if not schema or not isinstance(schema, dict):
            return provided_params
        
        schema_properties = schema.get("properties", {})
        required_params = schema.get("required", [])
        
        # Add required parameters
        for param_name in required_params:
            if param_name in provided_params:
                validated[param_name] = provided_params[param_name]
            else:
                logger.warning(f"[TOOL PLANNER] Missing required parameter: {param_name}")
        
        # Add optional parameters
        for param_name, param_value in provided_params.items():
            if param_name in schema_properties:
                validated[param_name] = param_value
        
        return validated

# Global planner instance
_planner_instance = None

def get_tool_planner() -> IntelligentToolPlanner:
    """Get global tool planner instance"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = IntelligentToolPlanner()
    return _planner_instance