"""
APINode MCP Tool Integration Bridge
Dynamically registers APINodes as MCP tools when connected to AgentNodes in workflows
"""
import logging
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from app.automation.core.api_executor import APIExecutor
from app.core.mcp_tools_cache import cache, MCP_TOOLS_KEY

logger = logging.getLogger(__name__)

class APINodeMCPBridge:
    """Bridge that registers APINodes as MCP tools for connected AgentNodes"""
    
    def __init__(self):
        self.api_executor = None  # Will be initialized lazily
        self._workflow_tools: Dict[str, Dict[str, Any]] = {}  # workflow_id -> {tool_name: tool_info}
        self._registered_tools: Set[str] = set()  # Track registered tool names
        
    def discover_apinode_tools(self, workflow_config: Dict[str, Any], workflow_id: str = None) -> Dict[str, Any]:
        """
        Discover APINodes connected to AgentNodes and create MCP tool definitions
        
        Args:
            workflow_config: Complete workflow configuration with nodes and edges
            workflow_id: Workflow identifier for context
            
        Returns:
            Dict of tool definitions keyed by tool name
        """
        nodes = workflow_config.get("nodes", [])
        edges = workflow_config.get("edges", [])
        
        # Find all APINodes
        api_nodes = {}
        agent_nodes = {}
        
        for node in nodes:
            node_type = node.get("data", {}).get("type")
            node_id = node.get("id")
            
            if node_type == "APINode":
                api_nodes[node_id] = node
            elif node_type == "AgentNode":
                agent_nodes[node_id] = node
        
        # Find connections between APINodes and AgentNodes
        apinode_to_agent_connections = {}
        
        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            
            # Check if APINode is connected to AgentNode (either direction)
            if source_id in api_nodes and target_id in agent_nodes:
                apinode_to_agent_connections[source_id] = target_id
            elif source_id in agent_nodes and target_id in api_nodes:
                apinode_to_agent_connections[target_id] = source_id
        
        # Create MCP tool definitions for connected APINodes
        mcp_tools = {}
        
        for api_node_id, agent_node_id in apinode_to_agent_connections.items():
            api_node = api_nodes[api_node_id]
            agent_node = agent_nodes[agent_node_id]
            
            tool_definition = self._create_mcp_tool_from_apinode(
                api_node, 
                agent_node, 
                api_node_id,
                workflow_id
            )
            
            if tool_definition:
                tool_name = tool_definition["name"]
                mcp_tools[tool_name] = tool_definition
                logger.info(f"[APINODE MCP] Created tool '{tool_name}' for APINode {api_node_id} -> AgentNode {agent_node_id}")
        
        return mcp_tools
    
    def _create_mcp_tool_from_apinode(
        self, 
        api_node: Dict[str, Any], 
        agent_node: Dict[str, Any], 
        api_node_id: str,
        workflow_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create MCP tool definition from APINode configuration
        
        Args:
            api_node: APINode configuration
            agent_node: Connected AgentNode configuration
            api_node_id: APINode ID
            
        Returns:
            MCP tool definition or None if not enabled
        """
        api_config = api_node.get("data", {}).get("node", {})
        
        # Check if MCP tool is enabled
        if not api_config.get("enable_mcp_tool", True):
            return None
        
        # Validate required API configuration
        if not api_config.get("base_url") or not api_config.get("endpoint_path"):
            logger.warning(f"[APINODE MCP] APINode {api_node_id} missing required base_url or endpoint_path")
            return None
        
        # Generate tool name
        tool_name = f"workflow_api_{api_node_id}"
        
        # Get tool description
        tool_description = api_config.get("tool_description", "")
        if not tool_description:
            api_label = api_config.get("label", "API Adapter")
            base_url = api_config.get("base_url", "")
            endpoint_path = api_config.get("endpoint_path", "")
            tool_description = f"{api_label}: {base_url}{endpoint_path}"
        
        # Extract parameters from request schema
        request_schema = api_config.get("request_schema", {})
        parameters = request_schema.get("properties", {})
        required = request_schema.get("required", [])
        
        # Validate schema structure
        if not isinstance(parameters, dict):
            logger.warning(f"[APINODE MCP] APINode {api_node_id} has invalid parameters schema")
            parameters = {}
        
        if not isinstance(required, list):
            logger.warning(f"[APINODE MCP] APINode {api_node_id} has invalid required fields")
            required = []
        
        # Create MCP tool definition
        tool_definition = {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required
            },
            "endpoint": f"/internal/workflow/apinode/{api_node_id}",
            "method": "POST",
            "headers": {},
            "server_id": "workflow_internal",
            "workflow_context": {
                "workflow_id": workflow_id,
                "api_node_id": api_node_id,
                "agent_node_id": agent_node.get("id"),
                "api_config": api_config,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
        return tool_definition
    
    def register_workflow_tools(self, workflow_id: str, workflow_config: Dict[str, Any]) -> int:
        """
        Register APINode tools for a specific workflow
        
        Args:
            workflow_id: Workflow identifier
            workflow_config: Complete workflow configuration
            
        Returns:
            Number of tools registered
        """
        # Discover tools
        workflow_tools = self.discover_apinode_tools(workflow_config, workflow_id)
        
        if not workflow_tools:
            logger.info(f"[APINODE MCP] No APINode tools found for workflow {workflow_id}")
            return 0
        
        # Store workflow tools
        self._workflow_tools[workflow_id] = workflow_tools
        
        # Get current MCP tools cache
        current_tools = cache.get(MCP_TOOLS_KEY) or {}
        
        # Add workflow tools to cache
        tools_added = 0
        for tool_name, tool_info in workflow_tools.items():
            if tool_name not in current_tools:
                current_tools[tool_name] = tool_info
                self._registered_tools.add(tool_name)
                tools_added += 1
                logger.info(f"[APINODE MCP] Registered tool: {tool_name}")
        
        # Update cache
        cache.set(MCP_TOOLS_KEY, current_tools)
        
        logger.info(f"[APINODE MCP] Registered {tools_added} APINode tools for workflow {workflow_id}")
        return tools_added
    
    def unregister_workflow_tools(self, workflow_id: str) -> int:
        """
        Unregister APINode tools for a specific workflow
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Number of tools unregistered
        """
        if workflow_id not in self._workflow_tools:
            return 0
        
        # Get workflow tools
        workflow_tools = self._workflow_tools[workflow_id]
        
        # Get current MCP tools cache
        current_tools = cache.get(MCP_TOOLS_KEY) or {}
        
        # Remove workflow tools from cache
        tools_removed = 0
        for tool_name in workflow_tools.keys():
            if tool_name in current_tools:
                del current_tools[tool_name]
                self._registered_tools.discard(tool_name)
                tools_removed += 1
                logger.info(f"[APINODE MCP] Unregistered tool: {tool_name}")
        
        # Update cache
        cache.set(MCP_TOOLS_KEY, current_tools)
        
        # Remove from local storage
        del self._workflow_tools[workflow_id]
        
        logger.info(f"[APINODE MCP] Unregistered {tools_removed} APINode tools for workflow {workflow_id}")
        return tools_removed
    
    async def execute_apinode_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        workflow_id: str,
        execution_id: str
    ) -> Dict[str, Any]:
        """
        Execute APINode tool call
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            workflow_id: Workflow ID
            execution_id: Execution ID
            
        Returns:
            Tool execution result
        """
        # Find the tool definition
        tool_info = None
        for wf_id, tools in self._workflow_tools.items():
            if tool_name in tools:
                tool_info = tools[tool_name]
                break
        
        if not tool_info:
            raise ValueError(f"APINode tool '{tool_name}' not found")
        
        # Get APINode configuration
        workflow_context = tool_info.get("workflow_context", {})
        api_node_id = workflow_context.get("api_node_id")
        api_config = workflow_context.get("api_config", {})
        
        if not api_node_id or not api_config:
            raise ValueError(f"Invalid APINode configuration for tool '{tool_name}'")
        
        # Validate parameters against tool schema
        tool_schema = tool_info.get("parameters", {})
        if tool_schema:
            required_params = tool_schema.get("required", [])
            for param in required_params:
                if param not in parameters:
                    raise ValueError(f"Required parameter '{param}' missing for tool '{tool_name}'")
        
        # Validate parameter types (basic validation)
        schema_properties = tool_schema.get("properties", {})
        for param_name, param_value in parameters.items():
            if param_name in schema_properties:
                expected_type = schema_properties[param_name].get("type")
                if expected_type == "string" and not isinstance(param_value, str):
                    raise ValueError(f"Parameter '{param_name}' must be a string")
                elif expected_type == "integer" and not isinstance(param_value, int):
                    raise ValueError(f"Parameter '{param_name}' must be an integer")
                elif expected_type == "boolean" and not isinstance(param_value, bool):
                    raise ValueError(f"Parameter '{param_name}' must be a boolean")
        
        # Initialize API executor if needed
        if self.api_executor is None:
            from app.automation.core.api_executor import get_api_executor
            self.api_executor = await get_api_executor()
        
        # Execute API call using APIExecutor
        try:
            result = await self.api_executor.execute_api_call(
                node_config=api_config,
                parameters=parameters,
                workflow_id=int(workflow_id) if workflow_id and workflow_id.isdigit() else 0,
                execution_id=execution_id,
                node_id=api_node_id
            )
            
            # Format result for MCP tool response
            return {
                "success": True,
                "result": result.get("response"),
                "metadata": result.get("metadata", {}),
                "tool_name": tool_name,
                "api_node_id": api_node_id
            }
            
        except Exception as e:
            logger.error(f"[APINODE MCP] Tool execution failed for {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "api_node_id": api_node_id
            }
    
    def get_workflow_tools(self, workflow_id: str) -> Dict[str, Any]:
        """Get tools registered for a specific workflow"""
        return self._workflow_tools.get(workflow_id, {})
    
    def get_all_registered_tools(self) -> Set[str]:
        """Get all registered tool names"""
        return self._registered_tools.copy()
    
    def cleanup_all_workflow_tools(self):
        """Clean up all registered workflow tools"""
        if not self._workflow_tools:
            return
        
        # Get current MCP tools cache
        current_tools = cache.get(MCP_TOOLS_KEY) or {}
        
        # Remove all workflow tools
        for tool_name in self._registered_tools:
            if tool_name in current_tools:
                del current_tools[tool_name]
        
        # Update cache
        cache.set(MCP_TOOLS_KEY, current_tools)
        
        # Clear local storage
        self._workflow_tools.clear()
        self._registered_tools.clear()
        
        logger.info("[APINODE MCP] Cleaned up all workflow tools")

# Global instance
apinode_mcp_bridge = APINodeMCPBridge()