"""
Agent Node Definitions for Visual Workflow Editor
Simplified node system focused on agent-based workflows
"""
from typing import Dict, List, Any, Optional
from app.core.langgraph_agents_cache import get_langgraph_agents
from app.core.mcp_tools_cache import get_enabled_mcp_tools

def get_agent_node_schema() -> Dict[str, Any]:
    """Get schema for Agent Node in visual editor"""
    
    # Get available agents for dropdown
    cached_agents = get_langgraph_agents()
    agent_options = []
    
    for agent_name, agent_data in cached_agents.items():
        agent_options.append({
            "value": agent_data.get("name", ""),
            "label": f"{agent_data.get('name', '')} - {agent_data.get('role', '')}",
            "description": agent_data.get("description", "")
        })
    
    # Get available tools for multi-select
    cached_tools = get_enabled_mcp_tools()
    tool_options = []
    
    for tool_name, tool_data in cached_tools.items():
        tool_options.append({
            "value": tool_name,
            "label": tool_name,
            "description": tool_data.get("description", "")
        })
    
    return {
        "type": "AgentNode",
        "name": "AI Agent",
        "description": "Execute tasks using AI agents with tool access",
        "category": "Agent",
        "icon": "🤖",
        "color": "#4F46E5",
        "inputs": [
            {
                "name": "input",
                "type": "string",
                "label": "Input",
                "description": "Input data or context for the agent"
            },
            {
                "name": "query",
                "type": "string", 
                "label": "Query",
                "description": "Specific query or task for the agent"
            }
        ],
        "outputs": [
            {
                "name": "output",
                "type": "string",
                "label": "Agent Response",
                "description": "Agent's response and analysis"
            },
            {
                "name": "tools_used",
                "type": "array",
                "label": "Tools Used",
                "description": "List of tools called by the agent"
            },
            {
                "name": "context",
                "type": "object",
                "label": "Context",
                "description": "Generated context for next agents"
            }
        ],
        "properties": {
            "agent_name": {
                "type": "select",
                "label": "Select Agent",
                "description": "Choose which AI agent to use",
                "required": True,
                "options": agent_options,
                "default": agent_options[0]["value"] if agent_options else ""
            },
            "custom_prompt": {
                "type": "textarea",
                "label": "Custom Instructions",
                "description": "Optional custom instructions for this specific task",
                "required": False,
                "placeholder": "Enter custom instructions to override default agent behavior..."
            },
            "tools": {
                "type": "multiselect",
                "label": "Available Tools",
                "description": "Select which tools this agent can use",
                "required": False,
                "options": tool_options,
                "default": []
            },
            "timeout": {
                "type": "number",
                "label": "Timeout (seconds)",
                "description": "Maximum execution time for this agent",
                "required": False,
                "default": 45,
                "min": 10,
                "max": 300
            },
            "temperature": {
                "type": "number",
                "label": "Temperature",
                "description": "Control randomness in agent responses (0.1-1.0)",
                "required": False,
                "default": 0.7,
                "min": 0.1,
                "max": 1.0,
                "step": 0.1
            }
        }
    }

def get_input_node_schema() -> Dict[str, Any]:
    """Get schema for Input Node in visual editor"""
    return {
        "type": "InputNode",
        "name": "Workflow Input",
        "description": "Define input data and parameters for the workflow",
        "category": "Input/Output",
        "icon": "📥",
        "color": "#059669",
        "inputs": [],
        "outputs": [
            {
                "name": "data",
                "type": "object",
                "label": "Input Data",
                "description": "Data provided to start the workflow"
            },
            {
                "name": "message",
                "type": "string",
                "label": "User Message",
                "description": "Message from user if workflow triggered by chat"
            }
        ],
        "properties": {
            "input_schema": {
                "type": "json",
                "label": "Input Schema",
                "description": "Define expected input data structure",
                "required": False,
                "default": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User query or question"},
                        "data": {"type": "object", "description": "Additional data"}
                    }
                }
            },
            "default_values": {
                "type": "json",
                "label": "Default Values",
                "description": "Default values when no input provided",
                "required": False,
                "default": {}
            }
        }
    }

def get_output_node_schema() -> Dict[str, Any]:
    """Get schema for Output Node in visual editor"""
    return {
        "type": "OutputNode",
        "name": "Workflow Output",
        "description": "Define output format and final response",
        "category": "Input/Output",
        "icon": "📤",
        "color": "#DC2626",
        "inputs": [
            {
                "name": "result",
                "type": "any",
                "label": "Final Result",
                "description": "Final result from workflow execution"
            },
            {
                "name": "summary",
                "type": "string",
                "label": "Summary",
                "description": "Summary of workflow execution"
            }
        ],
        "outputs": [],
        "properties": {
            "output_format": {
                "type": "select",
                "label": "Output Format",
                "description": "How to format the final output",
                "required": True,
                "options": [
                    {"value": "text", "label": "Plain Text"},
                    {"value": "json", "label": "JSON Object"},
                    {"value": "markdown", "label": "Markdown"},
                    {"value": "html", "label": "HTML"}
                ],
                "default": "text"
            },
            "include_metadata": {
                "type": "boolean",
                "label": "Include Metadata",
                "description": "Include execution metadata in output",
                "default": False
            },
            "include_tool_calls": {
                "type": "boolean",
                "label": "Include Tool Calls",
                "description": "Include information about tools used",
                "default": False
            }
        }
    }

def get_condition_node_schema() -> Dict[str, Any]:
    """Get schema for Condition Node in visual editor"""
    return {
        "type": "ConditionNode",
        "name": "Condition",
        "description": "Branch workflow based on conditions",
        "category": "Control",
        "icon": "🔀",
        "color": "#7C3AED",
        "inputs": [
            {
                "name": "input",
                "type": "any",
                "label": "Input Data",
                "description": "Data to evaluate condition against"
            }
        ],
        "outputs": [
            {
                "name": "true",
                "type": "any",
                "label": "True Branch",
                "description": "Output when condition is true"
            },
            {
                "name": "false",
                "type": "any",
                "label": "False Branch", 
                "description": "Output when condition is false"
            }
        ],
        "properties": {
            "condition_type": {
                "type": "select",
                "label": "Condition Type",
                "description": "Type of condition to evaluate",
                "required": True,
                "options": [
                    {"value": "simple", "label": "Simple Comparison"},
                    {"value": "ai_decision", "label": "AI-Based Decision"},
                    {"value": "custom", "label": "Custom Logic"}
                ],
                "default": "simple"
            },
            "operator": {
                "type": "select",
                "label": "Operator",
                "description": "Comparison operator (for simple conditions)",
                "required": False,
                "options": [
                    {"value": "equals", "label": "Equals"},
                    {"value": "not_equals", "label": "Not Equals"},
                    {"value": "contains", "label": "Contains"},
                    {"value": "greater_than", "label": "Greater Than"},
                    {"value": "less_than", "label": "Less Than"}
                ],
                "default": "equals",
                "show_when": {"condition_type": "simple"}
            },
            "compare_value": {
                "type": "string",
                "label": "Compare Value",
                "description": "Value to compare against",
                "required": False,
                "show_when": {"condition_type": "simple"}
            },
            "ai_criteria": {
                "type": "textarea",
                "label": "Decision Criteria",
                "description": "Criteria for AI to make decision",
                "required": False,
                "placeholder": "Describe what the AI should evaluate...",
                "show_when": {"condition_type": "ai_decision"}
            }
        }
    }

def get_parallel_node_schema() -> Dict[str, Any]:
    """Get schema for Parallel Execution Node in visual editor"""
    return {
        "type": "ParallelNode",
        "name": "Parallel Execution",
        "description": "Execute multiple agents simultaneously",
        "category": "Control",
        "icon": "⚡",
        "color": "#F59E0B",
        "inputs": [
            {
                "name": "input",
                "type": "any",
                "label": "Shared Input",
                "description": "Input data shared across all parallel agents"
            }
        ],
        "outputs": [
            {
                "name": "results",
                "type": "array",
                "label": "All Results",
                "description": "Combined results from all parallel executions"
            },
            {
                "name": "summary",
                "type": "string",
                "label": "Summary",
                "description": "AI-generated summary of all results"
            }
        ],
        "properties": {
            "max_parallel": {
                "type": "number",
                "label": "Max Parallel Agents",
                "description": "Maximum number of agents to run simultaneously",
                "required": False,
                "default": 3,
                "min": 2,
                "max": 8
            },
            "wait_for_all": {
                "type": "boolean",
                "label": "Wait for All",
                "description": "Wait for all agents to complete before proceeding",
                "default": True
            },
            "combine_strategy": {
                "type": "select",
                "label": "Combine Strategy",
                "description": "How to combine results from parallel agents",
                "required": True,
                "options": [
                    {"value": "merge", "label": "Merge All Results"},
                    {"value": "best", "label": "Select Best Result"},
                    {"value": "summary", "label": "AI Summary of All"},
                    {"value": "vote", "label": "Majority Vote"}
                ],
                "default": "merge"
            }
        }
    }

def get_all_node_schemas() -> List[Dict[str, Any]]:
    """Get all available node schemas for the visual editor"""
    return [
        get_agent_node_schema(),
        get_input_node_schema(),
        get_output_node_schema(),
        get_condition_node_schema(),
        get_parallel_node_schema()
    ]

def get_node_categories() -> List[Dict[str, Any]]:
    """Get node categories for organizing the visual editor palette"""
    return [
        {
            "name": "Agent",
            "description": "AI Agents for task execution",
            "color": "#4F46E5",
            "icon": "🤖"
        },
        {
            "name": "Input/Output",
            "description": "Workflow input and output nodes",
            "color": "#059669", 
            "icon": "📋"
        },
        {
            "name": "Control",
            "description": "Flow control and logic nodes",
            "color": "#7C3AED",
            "icon": "⚙️"
        }
    ]

def validate_agent_workflow(workflow_config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate agent workflow configuration"""
    errors = []
    warnings = []
    
    nodes = workflow_config.get("nodes", [])
    edges = workflow_config.get("edges", [])
    
    # Check for required nodes
    has_input = any(node.get("data", {}).get("type") == "InputNode" for node in nodes)
    has_agent = any(node.get("data", {}).get("type") == "AgentNode" for node in nodes)
    has_output = any(node.get("data", {}).get("type") == "OutputNode" for node in nodes)
    
    if not has_input:
        warnings.append("Workflow has no input node - will use default input handling")
    
    if not has_agent:
        errors.append("Workflow must have at least one Agent node")
    
    if not has_output:
        warnings.append("Workflow has no output node - will use default output format")
    
    # Check agent configurations
    agent_nodes = [node for node in nodes if node.get("data", {}).get("type") == "AgentNode"]
    
    for node in agent_nodes:
        node_data = node.get("data", {}).get("node", {})
        agent_name = node_data.get("agent_name")
        
        if not agent_name:
            errors.append(f"Agent node {node.get('id', 'unknown')} has no agent selected")
    
    # Check for orphaned nodes
    connected_nodes = set()
    for edge in edges:
        connected_nodes.add(edge.get("source"))
        connected_nodes.add(edge.get("target"))
    
    all_node_ids = {node.get("id") for node in nodes}
    orphaned = all_node_ids - connected_nodes
    
    if orphaned and len(nodes) > 1:
        warnings.append(f"Found {len(orphaned)} orphaned nodes that are not connected")
    
    # Check for cycles (simplified check)
    if len(edges) >= len(nodes):
        warnings.append("Workflow may contain cycles - review connections")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }