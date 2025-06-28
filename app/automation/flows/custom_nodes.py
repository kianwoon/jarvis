"""
Custom Langflow Nodes for Jarvis Infrastructure Integration
Creates custom nodes that connect to your existing MCP tools, agents, and infrastructure
"""
import logging
from typing import Dict, List, Any, Optional
import asyncio
import uuid

# Langflow imports
try:
    from langflow.custom import CustomComponent
    from langflow.field_typing import Data, Text
    from langflow.schema import dotdict
except ImportError:
    # Fallback for when Langflow is not installed
    class CustomComponent:
        def __init__(self):
            pass
    
    class Data:
        pass
    
    class Text:
        pass
    
    def dotdict(d):
        return d

from app.automation.integrations.mcp_bridge import mcp_bridge
from app.automation.integrations.agent_bridge import agent_bridge
from app.automation.integrations.redis_bridge import workflow_redis, shared_redis
from app.automation.integrations.postgres_bridge import postgres_bridge

logger = logging.getLogger(__name__)

class JarvisMCPToolNode(CustomComponent):
    """Custom Langflow node for executing MCP tools"""
    
    display_name = "Jarvis MCP Tool"
    description = "Execute MCP tools from your Jarvis infrastructure"
    icon = "tool"
    
    def build_config(self):
        # Get available tools for dropdown
        available_tools = mcp_bridge.get_available_tools()
        tool_options = list(available_tools.keys()) if available_tools else ["No tools available"]
        
        return {
            "tool_name": {
                "display_name": "Tool Name",
                "info": "Select MCP tool to execute",
                "options": tool_options,
                "required": True
            },
            "parameters": {
                "display_name": "Parameters",
                "info": "Tool parameters as JSON",
                "field_type": "code",
                "required": False
            },
            "query": {
                "display_name": "Query",
                "info": "Query parameter for search tools",
                "required": False
            },
            "num_results": {
                "display_name": "Number of Results",
                "info": "Number of results for search tools",
                "field_type": "int",
                "value": 10,
                "required": False
            }
        }
    
    def build(
        self,
        tool_name: str,
        parameters: str = "",
        query: str = "",
        num_results: int = 10
    ) -> Data:
        """Execute MCP tool and return results"""
        try:
            # Parse parameters
            import json
            if parameters:
                try:
                    tool_params = json.loads(parameters)
                except json.JSONDecodeError:
                    tool_params = {}
            else:
                tool_params = {}
            
            # Add common parameters
            if query:
                tool_params["query"] = query
            if num_results and "num_results" not in tool_params:
                tool_params["num_results"] = num_results
            
            # Execute tool
            result = mcp_bridge.execute_tool_sync(tool_name, tool_params)
            
            logger.info(f"[JARVIS MCP NODE] Executed {tool_name}: {result.get('success', False)}")
            
            # Return data in Langflow format
            return Data(
                value=result,
                text=str(result.get("result", result.get("error", ""))),
                data={
                    "tool_name": tool_name,
                    "parameters": tool_params,
                    "success": result.get("success", False),
                    "result": result
                }
            )
        except Exception as e:
            logger.error(f"[JARVIS MCP NODE] Error executing {tool_name}: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
            return Data(
                value=error_result,
                text=str(e),
                data=error_result
            )

class JarvisAgentNode(CustomComponent):
    """Custom Langflow node for executing LangGraph agents"""
    
    display_name = "Jarvis Agent"
    description = "Execute LangGraph agents from your Jarvis infrastructure"
    icon = "user"
    
    def build_config(self):
        # Get available agents for dropdown
        available_agents = agent_bridge.get_available_agents()
        agent_options = list(available_agents.keys()) if available_agents else ["No agents available"]
        
        return {
            "agent_name": {
                "display_name": "Agent Name",
                "info": "Select agent to execute",
                "options": agent_options,
                "required": True
            },
            "query": {
                "display_name": "Query",
                "info": "Query/prompt for the agent",
                "field_type": "text",
                "required": True
            },
            "context": {
                "display_name": "Context",
                "info": "Additional context for the agent",
                "field_type": "text",
                "required": False
            }
        }
    
    def build(
        self,
        agent_name: str,
        query: str,
        context: str = ""
    ) -> Data:
        """Execute agent and return response"""
        try:
            # Execute agent
            result = agent_bridge.execute_agent_sync(agent_name, query, context)
            
            logger.info(f"[JARVIS AGENT NODE] Executed {agent_name}: {result.get('success', False)}")
            
            # Return data in Langflow format
            response_text = result.get("response", result.get("error", ""))
            return Data(
                value=result,
                text=response_text,
                data={
                    "agent_name": agent_name,
                    "query": query,
                    "context": context,
                    "success": result.get("success", False),
                    "response": response_text,
                    "result": result
                }
            )
        except Exception as e:
            logger.error(f"[JARVIS AGENT NODE] Error executing {agent_name}: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "agent": agent_name
            }
            return Data(
                value=error_result,
                text=str(e),
                data=error_result
            )

class JarvisRedisNode(CustomComponent):
    """Custom Langflow node for Redis operations"""
    
    display_name = "Jarvis Redis"
    description = "Read/write data to Redis cache"
    icon = "database"
    
    def build_config(self):
        return {
            "operation": {
                "display_name": "Operation",
                "info": "Redis operation to perform",
                "options": ["get", "set", "delete", "exists"],
                "value": "get",
                "required": True
            },
            "key": {
                "display_name": "Key",
                "info": "Redis key",
                "required": True
            },
            "value": {
                "display_name": "Value",
                "info": "Value to set (for set operation)",
                "field_type": "text",
                "required": False
            },
            "expire": {
                "display_name": "Expire (seconds)",
                "info": "TTL for the key",
                "field_type": "int",
                "required": False
            },
            "namespace": {
                "display_name": "Namespace",
                "info": "Redis namespace (workflow, shared, general)",
                "options": ["workflow", "shared", "general"],
                "value": "workflow",
                "required": True
            }
        }
    
    def build(
        self,
        operation: str,
        key: str,
        value: str = "",
        expire: int = None,
        namespace: str = "workflow"
    ) -> Data:
        """Perform Redis operation"""
        try:
            # Select Redis bridge based on namespace
            if namespace == "workflow":
                redis_client = workflow_redis
            elif namespace == "shared":
                redis_client = shared_redis
            else:
                redis_client = workflow_redis  # Default
            
            result = None
            success = True
            
            if operation == "get":
                result = redis_client.get_value(key)
            elif operation == "set":
                result = redis_client.set_value(key, value, expire=expire)
            elif operation == "delete":
                result = redis_client.delete_value(key)
            elif operation == "exists":
                result = redis_client.exists(key)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            logger.info(f"[JARVIS REDIS NODE] {operation} on {key}: {result}")
            
            return Data(
                value=result,
                text=str(result),
                data={
                    "operation": operation,
                    "key": key,
                    "value": value,
                    "namespace": namespace,
                    "result": result,
                    "success": success
                }
            )
        except Exception as e:
            logger.error(f"[JARVIS REDIS NODE] Error in {operation} on {key}: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "operation": operation,
                "key": key
            }
            return Data(
                value=None,
                text=str(e),
                data=error_result
            )

class JarvisWorkflowStateNode(CustomComponent):
    """Custom Langflow node for workflow state management"""
    
    display_name = "Jarvis Workflow State"
    description = "Manage workflow execution state"
    icon = "settings"
    
    def build_config(self):
        return {
            "operation": {
                "display_name": "Operation",
                "info": "State operation to perform",
                "options": ["get_state", "set_state", "update_state"],
                "value": "get_state",
                "required": True
            },
            "workflow_id": {
                "display_name": "Workflow ID",
                "info": "Workflow identifier",
                "required": True
            },
            "execution_id": {
                "display_name": "Execution ID",
                "info": "Execution identifier (auto-generated if empty)",
                "required": False
            },
            "state_data": {
                "display_name": "State Data",
                "info": "State data as JSON (for set/update operations)",
                "field_type": "code",
                "required": False
            }
        }
    
    def build(
        self,
        operation: str,
        workflow_id: str,
        execution_id: str = "",
        state_data: str = ""
    ) -> Data:
        """Manage workflow state"""
        try:
            # Generate execution ID if not provided
            if not execution_id:
                execution_id = str(uuid.uuid4())
            
            result = None
            success = True
            
            if operation == "get_state":
                result = workflow_redis.get_workflow_state(workflow_id, execution_id)
            elif operation in ["set_state", "update_state"]:
                if state_data:
                    import json
                    try:
                        state_dict = json.loads(state_data)
                    except json.JSONDecodeError:
                        state_dict = {"data": state_data}
                else:
                    state_dict = {}
                
                if operation == "set_state":
                    success = workflow_redis.set_workflow_state(workflow_id, execution_id, state_dict)
                else:  # update_state
                    success = workflow_redis.update_workflow_state(workflow_id, execution_id, state_dict)
                
                result = state_dict if success else None
            
            logger.info(f"[JARVIS WORKFLOW STATE NODE] {operation} for {workflow_id}:{execution_id}")
            
            return Data(
                value=result,
                text=str(result),
                data={
                    "operation": operation,
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "result": result,
                    "success": success
                }
            )
        except Exception as e:
            logger.error(f"[JARVIS WORKFLOW STATE NODE] Error in {operation}: {e}")
            error_result = {
                "success": False,
                "error": str(e),
                "operation": operation,
                "workflow_id": workflow_id
            }
            return Data(
                value=None,
                text=str(e),
                data=error_result
            )

# Export custom nodes for Langflow registration
CUSTOM_NODES = [
    JarvisMCPToolNode,
    JarvisAgentNode,
    JarvisRedisNode,
    JarvisWorkflowStateNode
]