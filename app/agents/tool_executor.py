"""
Tool Executor for Agent Contracts

Connects agent contracts with MCP tool execution to enable
agents to perform actual actions based on their defined capabilities.
"""

import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from app.core.mcp_tools_cache import get_mcp_tool_by_name, get_enabled_mcp_tools
from app.agents.agent_contracts import AgentContract, AgentCapability

logger = logging.getLogger(__name__)


class AgentToolExecutor:
    """Executes MCP tools based on agent contracts and capabilities"""
    
    def __init__(self):
        self.tool_cache = {}
        self.execution_history = []
        
    async def validate_tool_access(self, agent_contract: AgentContract, tool_name: str) -> bool:
        """Validate if agent has access to a specific tool based on contract"""
        
        # Check if tool is in agent's allowed tools
        if tool_name not in agent_contract.tools:
            logger.warning(f"Agent {agent_contract.name} attempted to use unauthorized tool: {tool_name}")
            return False
            
        # Check if tool exists and is enabled
        tool_info = get_mcp_tool_by_name(tool_name)
        if not tool_info or not tool_info.get("is_active", False):
            logger.warning(f"Tool {tool_name} is not available or not active")
            return False
            
        return True
    
    async def execute_tool(self, agent_name: str, tool_name: str, 
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single MCP tool with parameters"""
        
        try:
            # Get tool information
            tool_info = get_mcp_tool_by_name(tool_name)
            if not tool_info:
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Log execution attempt
            execution_record = {
                "agent": agent_name,
                "tool": tool_name,
                "parameters": parameters,
                "timestamp": datetime.now().isoformat()
            }
            self.execution_history.append(execution_record)
            
            # Import MCP tool execution dynamically
            from app.api.v1.endpoints.mcp_tools import execute_mcp_tool
            
            # Execute the tool
            logger.info(f"Agent {agent_name} executing tool {tool_name} with params: {parameters}")
            
            # Create a mock request object for the tool execution
            class MockRequest:
                def __init__(self, data):
                    self.data = data
                    
                async def json(self):
                    return self.data
            
            # Prepare the request data
            request_data = {
                "server_name": tool_info.get("server_name"),
                "tool_name": tool_name,
                "arguments": parameters
            }
            
            mock_request = MockRequest(request_data)
            
            # Execute tool and get result
            result = await execute_mcp_tool(mock_request)
            
            # Parse result
            if hasattr(result, 'body'):
                result_data = json.loads(result.body)
            else:
                result_data = result
                
            return {
                "success": True,
                "result": result_data,
                "tool": tool_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "timestamp": datetime.now().isoformat()
            }
    
    async def execute_tool_sequence(self, agent_name: str, agent_contract: AgentContract,
                                   tool_sequence: List[Dict[str, Any]]) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a sequence of tools based on agent contract"""
        
        for idx, tool_call in enumerate(tool_sequence):
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            
            # Validate access
            if not await self.validate_tool_access(agent_contract, tool_name):
                yield {
                    "type": "tool_error",
                    "index": idx,
                    "tool": tool_name,
                    "error": "Unauthorized tool access"
                }
                continue
            
            # Execute tool
            result = await self.execute_tool(agent_name, tool_name, parameters)
            
            yield {
                "type": "tool_result",
                "index": idx,
                "tool": tool_name,
                "success": result.get("success", False),
                "result": result
            }
    
    def parse_tool_calls_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from agent response"""
        
        tool_calls = []
        
        # Try to find JSON-formatted tool calls
        try:
            # Look for tool call patterns
            import re
            
            # Pattern 1: Direct JSON tool calls
            json_pattern = r'\{[^{}]*"tool"[^{}]*"parameters"[^{}]*\}'
            matches = re.findall(json_pattern, response)
            
            for match in matches:
                try:
                    tool_call = json.loads(match)
                    if "tool" in tool_call:
                        tool_calls.append(tool_call)
                except:
                    pass
            
            # Pattern 2: Structured tool calls in response
            if "TOOL_CALLS:" in response:
                tool_section = response.split("TOOL_CALLS:")[1].split("\n\n")[0]
                # Try to parse as JSON array
                try:
                    calls = json.loads(tool_section.strip())
                    if isinstance(calls, list):
                        tool_calls.extend(calls)
                except:
                    # Try line-by-line parsing
                    for line in tool_section.strip().split("\n"):
                        if line.strip().startswith("{"):
                            try:
                                tool_calls.append(json.loads(line.strip()))
                            except:
                                pass
            
        except Exception as e:
            logger.warning(f"Failed to parse tool calls: {str(e)}")
        
        return tool_calls
    
    async def execute_from_agent_response(self, agent_name: str, agent_contract: AgentContract,
                                         response: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute tools parsed from agent response"""
        
        # Parse tool calls from response
        tool_calls = self.parse_tool_calls_from_response(response)
        
        if not tool_calls:
            yield {
                "type": "no_tools",
                "message": "No tool calls found in agent response"
            }
            return
        
        # Execute the tool sequence
        async for result in self.execute_tool_sequence(agent_name, agent_contract, tool_calls):
            yield result


class ToolAwareAgentExecutor:
    """Enhanced agent executor that integrates tool execution"""
    
    def __init__(self):
        self.tool_executor = AgentToolExecutor()
        
    async def execute_agent_with_tools(self, agent_name: str, agent_contract: AgentContract,
                                      query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agent with automatic tool execution based on contract"""
        
        # Import dynamic agent system
        from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
        
        dynamic_system = DynamicMultiAgentSystem()
        
        # Enhance agent data with tool information
        agent_data = {
            "name": agent_contract.name,
            "system_prompt": agent_contract.instructions,
            "tools": agent_contract.tools,
            "config": {
                "max_tokens": 4000,
                "temperature": 0.7
            }
        }
        
        # Add tool descriptions to context
        tool_descriptions = []
        for tool_name in agent_contract.tools:
            tool_info = get_mcp_tool_by_name(tool_name)
            if tool_info:
                tool_descriptions.append({
                    "name": tool_name,
                    "description": tool_info.get("description", ""),
                    "parameters": tool_info.get("input_schema", {})
                })
        
        enhanced_context = {
            **(context or {}),
            "available_tools": tool_descriptions,
            "tool_execution_enabled": True
        }
        
        # Execute agent
        response_complete = False
        agent_response = ""
        
        async for event in dynamic_system.execute_agent(
            agent_name,
            agent_data,
            query,
            context=enhanced_context
        ):
            # Forward agent events
            yield event
            
            # Capture complete response for tool parsing
            if event.get("type") == "agent_complete":
                response_complete = True
                agent_response = event.get("content", "")
        
        # If agent completed, check for tool calls
        if response_complete and agent_response:
            # Execute any tools mentioned in the response
            tool_executed = False
            async for tool_result in self.tool_executor.execute_from_agent_response(
                agent_name, agent_contract, agent_response
            ):
                tool_executed = True
                yield {
                    "type": "tool_execution",
                    "data": tool_result
                }
            
            # If tools were executed, we might want to run the agent again with results
            if tool_executed:
                # This is where we could implement a feedback loop
                # For now, just indicate tools were executed
                yield {
                    "type": "tools_complete",
                    "message": "Tool execution completed"
                }


# Helper function to create tool-aware agents
def create_tool_aware_agent(agent_contract: AgentContract) -> ToolAwareAgentExecutor:
    """Create a tool-aware agent executor from a contract"""
    
    executor = ToolAwareAgentExecutor()
    return executor