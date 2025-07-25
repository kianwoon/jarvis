"""
MCP Tools Bridge for Langflow Integration
Connects Langflow workflows to your existing MCP tools infrastructure
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from app.core.mcp_tools_cache import get_enabled_mcp_tools
from app.core.langfuse_integration import get_tracer

logger = logging.getLogger(__name__)

class MCPToolsBridge:
    """Bridge between Langflow and MCP tools infrastructure"""
    
    def __init__(self):
        self.tracer = get_tracer()
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get all available MCP tools for Langflow node configuration"""
        try:
            tools = get_enabled_mcp_tools()
            logger.info(f"[MCP BRIDGE] Retrieved {len(tools)} available MCP tools")
            
            # Format for Langflow consumption
            formatted_tools = {}
            for tool_name, tool_info in tools.items():
                formatted_tools[tool_name] = {
                    "name": tool_name,
                    "description": tool_info.get("description", "No description available"),
                    "parameters": tool_info.get("parameters", {}),
                    "endpoint": tool_info.get("endpoint", ""),
                    "method": tool_info.get("method", "POST"),
                    "server_id": tool_info.get("server_id")
                }
            
            return formatted_tools
        except Exception as e:
            logger.error(f"[MCP BRIDGE] Failed to get available tools: {e}")
            return {}
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get JSON schema for a specific MCP tool"""
        tools = get_enabled_mcp_tools()
        tool_info = tools.get(tool_name)
        
        if not tool_info:
            logger.warning(f"[MCP BRIDGE] Tool {tool_name} not found")
            return None
        
        return tool_info.get("parameters", {})
    
    def execute_tool_sync(self, tool_name: str, parameters: Dict[str, Any], trace=None) -> Dict[str, Any]:
        """Execute MCP tool synchronously (for Langflow nodes)"""
        try:
            logger.info(f"[MCP BRIDGE] Executing tool {tool_name} with parameters: {parameters}")
            
            # Use existing call_mcp_tool infrastructure (local import to avoid circular dependency)
            from app.langchain.service import call_mcp_tool
            result = call_mcp_tool(tool_name, parameters, trace=trace)
            
            # Format result for Langflow
            if isinstance(result, dict) and "error" in result:
                return {
                    "success": False,
                    "error": result["error"],
                    "tool": tool_name,
                    "parameters": parameters
                }
            else:
                return {
                    "success": True,
                    "result": result,
                    "tool": tool_name,
                    "parameters": parameters
                }
        except Exception as e:
            logger.error(f"[MCP BRIDGE] Tool execution failed for {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name,
                "parameters": parameters
            }
    
    async def execute_tool_async(self, tool_name: str, parameters: Dict[str, Any], trace=None) -> Dict[str, Any]:
        """Execute MCP tool asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Use ThreadPoolExecutor to run sync call_mcp_tool in async context
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor, 
                self.execute_tool_sync, 
                tool_name, 
                parameters, 
                trace
            )
            return await future
    
    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against tool schema"""
        schema = self.get_tool_schema(tool_name)
        if not schema:
            return {"valid": False, "error": f"Tool {tool_name} not found"}
        
        try:
            # Basic validation against JSON schema
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            
            # Check required parameters
            missing_params = []
            for req_param in required:
                if req_param not in parameters:
                    missing_params.append(req_param)
            
            if missing_params:
                return {
                    "valid": False, 
                    "error": f"Missing required parameters: {missing_params}"
                }
            
            # Check parameter types (basic validation)
            type_errors = []
            for param_name, param_value in parameters.items():
                if param_name in properties:
                    expected_type = properties[param_name].get("type")
                    if expected_type == "integer" and not isinstance(param_value, int):
                        type_errors.append(f"{param_name} should be integer")
                    elif expected_type == "boolean" and not isinstance(param_value, bool):
                        type_errors.append(f"{param_name} should be boolean")
                    elif expected_type == "string" and not isinstance(param_value, str):
                        type_errors.append(f"{param_name} should be string")
            
            if type_errors:
                return {"valid": False, "error": f"Type errors: {type_errors}"}
            
            return {"valid": True}
            
        except Exception as e:
            logger.error(f"[MCP BRIDGE] Parameter validation failed: {e}")
            return {"valid": False, "error": f"Validation error: {e}"}

# Global instance for use in Langflow nodes
mcp_bridge = MCPToolsBridge()