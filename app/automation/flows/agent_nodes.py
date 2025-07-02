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
            },
            "state_enabled": {
                "type": "boolean",
                "label": "Enable State Management",
                "description": "Enable built-in state management for direct agent chaining",
                "required": False,
                "default": False
            },
            "state_operation": {
                "type": "select",
                "label": "State Operation",
                "description": "How to handle agent output for next agent",
                "required": False,
                "options": [
                    {"value": "merge", "label": "Merge with Previous"},
                    {"value": "replace", "label": "Replace Previous"},
                    {"value": "append", "label": "Append to Previous"},
                    {"value": "passthrough", "label": "Pass Output Directly"}
                ],
                "default": "passthrough",
                "show_when": {"state_enabled": True}
            },
            "output_format": {
                "type": "select",
                "label": "Output Format",
                "description": "How to format output for next agent",
                "required": False,
                "options": [
                    {"value": "text", "label": "Plain Text"},
                    {"value": "structured", "label": "Structured Data"},
                    {"value": "context", "label": "Context Object"},
                    {"value": "full", "label": "Full Agent Response"}
                ],
                "default": "text",
                "show_when": {"state_enabled": True}
            },
            "chain_key": {
                "type": "string",
                "label": "Chain Key",
                "description": "State key for agent chaining (optional)",
                "required": False,
                "placeholder": "e.g., analysis_result, processed_data",
                "show_when": {"state_enabled": True}
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
            },
            "auto_display": {
                "type": "boolean",
                "label": "Auto Display Result",
                "description": "Automatically display workflow result when complete",
                "default": True
            },
            "auto_save": {
                "type": "boolean",
                "label": "Auto Save to File",
                "description": "Automatically save workflow result to Downloads folder",
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

def get_state_node_schema() -> Dict[str, Any]:
    """Get schema for State Management Node in visual editor"""
    return {
        "type": "StateNode",
        "name": "State Manager",
        "description": "Manage and persist workflow state across execution steps",
        "category": "Control",
        "icon": "🔄",
        "color": "#9C27B0",
        "inputs": [
            {
                "name": "state_update",
                "type": "object",
                "label": "State Update",
                "description": "New state data to merge with current state"
            },
            {
                "name": "state_key",
                "type": "string", 
                "label": "State Key",
                "description": "Specific state key to update"
            }
        ],
        "outputs": [
            {
                "name": "current_state",
                "type": "object",
                "label": "Current State",
                "description": "Current workflow state"
            },
            {
                "name": "state_value",
                "type": "any",
                "label": "State Value",
                "description": "Value of specific state key"
            }
        ],
        "properties": {
            "state_schema": {
                "type": "json",
                "label": "State Schema",
                "description": "Define the structure of the workflow state",
                "required": False,
                "default": {
                    "type": "object",
                    "properties": {
                        "current_step": {"type": "string"},
                        "user_input": {"type": "string"},
                        "agent_outputs": {"type": "object"},
                        "execution_context": {"type": "object"}
                    }
                }
            },
            "persistence": {
                "type": "boolean",
                "label": "Persist State",
                "description": "Whether to persist state across workflow executions",
                "default": True
            },
            "state_operation": {
                "type": "select",
                "label": "State Operation",
                "description": "How to handle state updates",
                "required": True,
                "options": [
                    {"value": "merge", "label": "Merge with Current"},
                    {"value": "replace", "label": "Replace Current"},
                    {"value": "get", "label": "Get Current State"},
                    {"value": "set_key", "label": "Set Specific Key"}
                ],
                "default": "merge"
            },
            "checkpoint_name": {
                "type": "string",
                "label": "Checkpoint Name",
                "description": "Name for state checkpoint (for resumable workflows)",
                "required": False,
                "placeholder": "Enter checkpoint name..."
            }
        }
    }

def get_router_node_schema() -> Dict[str, Any]:
    """Get schema for Router Node in visual editor"""
    return {
        "type": "RouterNode",
        "name": "Multi-Route Router",
        "description": "Route workflow to one or more groups based on agent output",
        "category": "Control",
        "icon": "🔀",
        "color": "#EC4899",
        "inputs": [
            {
                "name": "agent_output",
                "type": "any",
                "label": "Agent Output",
                "description": "Output from previous agent to evaluate for routing"
            }
        ],
        "outputs": [
            {
                "name": "routes",
                "type": "array",
                "label": "Active Routes",
                "description": "List of activated route IDs"
            },
            {
                "name": "matched_groups",
                "type": "array",
                "label": "Matched Groups",
                "description": "List of group IDs to execute"
            }
        ],
        "properties": {
            "routing_mode": {
                "type": "select",
                "label": "Routing Mode",
                "description": "How to handle multiple matches",
                "required": True,
                "options": [
                    {"value": "multi-select", "label": "Multi-Select (Execute All Matches)"},
                    {"value": "single-select", "label": "Single-Select (First Match Only)"}
                ],
                "default": "multi-select"
            },
            "match_type": {
                "type": "select",
                "label": "Match Type",
                "description": "How to match agent output against route values",
                "required": True,
                "options": [
                    {"value": "exact", "label": "Exact Match"},
                    {"value": "contains", "label": "Contains"},
                    {"value": "regex", "label": "Regular Expression"},
                    {"value": "in_array", "label": "In Array (for array outputs)"}
                ],
                "default": "exact"
            },
            "routes": {
                "type": "json",
                "label": "Route Definitions", 
                "description": "Define routing rules and target nodes (auto-populated from connections)",
                "required": True,
                "default": [
                    {
                        "id": "route_1",
                        "match_values": ["Customer_Support", "customer", "support"],
                        "target_nodes": ["agentnode-1", "agentnode-2"],
                        "label": "Customer Support Route"
                    },
                    {
                        "id": "route_2",
                        "match_values": ["Sales_Inquiry", "sales", "pricing"],
                        "target_nodes": ["agentnode-3"],
                        "label": "Sales Route"
                    },
                    {
                        "id": "route_3",
                        "match_values": ["Technical_Issue", "technical", "bug"],
                        "target_nodes": ["agentnode-4", "agentnode-5"],
                        "label": "Technical Route"
                    }
                ]
            },
            "fallback_route": {
                "type": "string", 
                "label": "Fallback Route",
                "description": "Node ID to execute when no matches found",
                "required": False,
                "placeholder": "agentnode-fallback"
            },
            "case_sensitive": {
                "type": "boolean",
                "label": "Case Sensitive",
                "description": "Whether matching should be case-sensitive",
                "default": False
            },
            "output_field": {
                "type": "string",
                "label": "Output Field",
                "description": "Field to extract from agent output (leave empty for full output)",
                "required": False,
                "placeholder": "e.g., 'category' or 'labels'"
            }
        }
    }

def get_transform_node_schema() -> Dict[str, Any]:
    """Get schema for Transform Node in visual editor"""
    return {
        "type": "TransformNode",
        "name": "Data Transform",
        "description": "Transform data using JSONPath, JQ, JavaScript, or Python expressions",
        "category": "Data",
        "icon": "🔄",
        "color": "#10B981",
        "inputs": [
            {
                "name": "input",
                "type": "any",
                "label": "Input Data",
                "description": "Data to transform"
            }
        ],
        "outputs": [
            {
                "name": "output",
                "type": "any",
                "label": "Transformed Data",
                "description": "Result of transformation"
            },
            {
                "name": "error",
                "type": "string",
                "label": "Error",
                "description": "Error message if transformation fails"
            }
        ],
        "properties": {
            "transform_type": {
                "type": "select",
                "label": "Transform Type",
                "description": "Type of transformation to apply",
                "required": True,
                "options": [
                    {"value": "jsonpath", "label": "JSONPath"},
                    {"value": "jq", "label": "JQ Query"},
                    {"value": "javascript", "label": "JavaScript"},
                    {"value": "python", "label": "Python Expression"}
                ],
                "default": "jsonpath"
            },
            "expression": {
                "type": "textarea",
                "label": "Expression",
                "description": "Transformation expression",
                "required": True,
                "placeholder": "e.g., $.data.items[*].name",
                "default": "$"
            },
            "error_handling": {
                "type": "select",
                "label": "Error Handling",
                "description": "How to handle transformation errors",
                "required": True,
                "options": [
                    {"value": "fail", "label": "Fail Workflow"},
                    {"value": "continue", "label": "Continue with Error"},
                    {"value": "default", "label": "Use Default Value"}
                ],
                "default": "continue"
            },
            "default_value": {
                "type": "json",
                "label": "Default Value",
                "description": "Value to use if transformation fails",
                "required": False,
                "default": None,
                "show_when": {"error_handling": "default"}
            },
            "input_validation": {
                "type": "json",
                "label": "Input Schema (Optional)",
                "description": "JSON Schema to validate input data",
                "required": False,
                "default": {}
            },
            "output_validation": {
                "type": "json",
                "label": "Output Schema (Optional)",
                "description": "JSON Schema to validate output data",
                "required": False,
                "default": {}
            },
            "cache_results": {
                "type": "boolean",
                "label": "Cache Results",
                "description": "Cache transformation results for identical inputs",
                "default": False
            },
            "test_data": {
                "type": "json",
                "label": "Test Data",
                "description": "Sample data for testing the expression",
                "required": False,
                "default": {"example": "data"}
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
        get_parallel_node_schema(),
        get_state_node_schema(),
        get_router_node_schema(),
        get_transform_node_schema()
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
        },
        {
            "name": "Data",
            "description": "Data transformation and processing nodes",
            "color": "#10B981",
            "icon": "🔄"
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