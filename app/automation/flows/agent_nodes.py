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
                "description": "Specific query or task for the agent",
                "fieldType": "textarea",
                "minHeight": 4,
                "maxHeight": 12,
                "resizable": True
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
            "label": {
                "type": "string",
                "label": "Node Title",
                "description": "Descriptive title for this agent node",
                "required": False,
                "placeholder": "Describe what this agent does (e.g., 'Data Analyzer', 'Report Generator')",
                "default": ""
            },
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
                "placeholder": "Enter custom instructions to override default agent behavior...",
                "minHeight": 6,
                "maxHeight": 20,
                "resizable": True
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
                "description": "Define routing rules and target nodes (manually configured by user)",
                "required": True,
                "default": []
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

def get_trigger_node_schema() -> Dict[str, Any]:
    """Get schema for Trigger Node in visual editor"""
    return {
        "type": "TriggerNode",
        "name": "External Trigger",
        "description": "Allow external systems to trigger workflow execution via API endpoints",
        "category": "Input/Output",
        "icon": "🔗",
        "color": "#059669",
        "inputs": [],
        "outputs": [
            {
                "name": "trigger_data",
                "type": "object",
                "label": "Trigger Data",
                "description": "Data received from external trigger request"
            },
            {
                "name": "query_params",
                "type": "object",
                "label": "Query Parameters",
                "description": "URL query parameters from trigger request"
            },
            {
                "name": "headers",
                "type": "object",
                "label": "Request Headers",
                "description": "HTTP headers from trigger request"
            },
            {
                "name": "message",
                "type": "string",
                "label": "Extracted Message",
                "description": "User instruction/query extracted from request data"
            },
            {
                "name": "formatted_query",
                "type": "string",
                "label": "Formatted Query",
                "description": "Agent-ready query constructed from request data"
            }
        ],
        "properties": {
            "trigger_name": {
                "type": "string",
                "label": "Trigger Name",
                "description": "Unique name for this trigger endpoint",
                "required": True,
                "placeholder": "my-workflow-trigger"
            },
            "http_methods": {
                "type": "multiselect",
                "label": "HTTP Methods",
                "description": "Allowed HTTP methods for this trigger",
                "required": True,
                "options": [
                    {"value": "GET", "label": "GET"},
                    {"value": "POST", "label": "POST"},
                    {"value": "PUT", "label": "PUT"},
                    {"value": "DELETE", "label": "DELETE"},
                    {"value": "PATCH", "label": "PATCH"}
                ],
                "default": ["POST"]
            },
            "authentication_type": {
                "type": "select",
                "label": "Authentication Type",
                "description": "Authentication method for external access",
                "required": True,
                "options": [
                    {"value": "none", "label": "No Authentication"},
                    {"value": "api_key", "label": "API Key (Header)"},
                    {"value": "bearer_token", "label": "Bearer Token"},
                    {"value": "basic_auth", "label": "Basic Authentication"},
                    {"value": "custom_header", "label": "Custom Header"}
                ],
                "default": "api_key"
            },
            "auth_header_name": {
                "type": "string",
                "label": "Auth Header Name",
                "description": "Name of the authentication header",
                "required": False,
                "default": "X-API-Key",
                "show_when": {"authentication_type": ["api_key", "custom_header"]}
            },
            "auth_token": {
                "type": "string",
                "label": "Authentication Token",
                "description": "Token/key for authentication (auto-generated if empty)",
                "required": False,
                "placeholder": "Auto-generated secure token",
                "show_when": {"authentication_type": ["api_key", "bearer_token", "custom_header"]}
            },
            "basic_auth_username": {
                "type": "string",
                "label": "Username",
                "description": "Basic authentication username",
                "required": False,
                "show_when": {"authentication_type": "basic_auth"}
            },
            "basic_auth_password": {
                "type": "string",
                "label": "Password",
                "description": "Basic authentication password",
                "required": False,
                "show_when": {"authentication_type": "basic_auth"}
            },
            "rate_limit": {
                "type": "number",
                "label": "Rate Limit (requests/minute)",
                "description": "Maximum requests per minute (0 = unlimited)",
                "required": False,
                "default": 60,
                "min": 0,
                "max": 1000
            },
            "timeout": {
                "type": "number",
                "label": "Response Timeout (seconds)",
                "description": "Maximum time to wait for workflow completion",
                "required": False,
                "default": 300,
                "min": 30,
                "max": 3600
            },
            "response_format": {
                "type": "select",
                "label": "Response Format",
                "description": "Format of the response returned to external caller",
                "required": True,
                "options": [
                    {"value": "workflow_output", "label": "Workflow Output Only"},
                    {"value": "detailed", "label": "Detailed (with metadata)"},
                    {"value": "status_only", "label": "Status Only"},
                    {"value": "custom", "label": "Custom JSON Template"}
                ],
                "default": "workflow_output"
            },
            "custom_response_template": {
                "type": "textarea",
                "label": "Custom Response Template",
                "description": "JSON template for custom response format (use {{output}}, {{status}}, {{execution_time}})",
                "required": False,
                "placeholder": '{"result": "{{output}}", "status": "{{status}}", "timestamp": "{{timestamp}}"}',
                "show_when": {"response_format": "custom"}
            },
            "cors_enabled": {
                "type": "boolean",
                "label": "Enable CORS",
                "description": "Allow cross-origin requests from web browsers",
                "default": True
            },
            "cors_origins": {
                "type": "textarea",
                "label": "Allowed Origins",
                "description": "Comma-separated list of allowed origins (* for all)",
                "required": False,
                "default": "*",
                "show_when": {"cors_enabled": True}
            },
            "log_requests": {
                "type": "boolean",
                "label": "Log Requests",
                "description": "Log all incoming trigger requests for debugging",
                "default": True
            },
            "message_extraction_strategy": {
                "type": "select",
                "label": "Message Extraction Strategy",
                "description": "How to extract user message/instruction from request",
                "required": False,
                "options": [
                    {"value": "auto", "label": "Auto-detect (smart extraction)"},
                    {"value": "body_text", "label": "Request Body as Text"},
                    {"value": "query_param", "label": "Specific Query Parameter"},
                    {"value": "json_field", "label": "JSON Field from Body"},
                    {"value": "combined", "label": "Combined Sources"}
                ],
                "default": "auto"
            },
            "message_source_field": {
                "type": "string",
                "label": "Message Source Field",
                "description": "Field name for message extraction (query param or JSON field)",
                "required": False,
                "placeholder": "message, query, instruction, etc.",
                "show_when": {"message_extraction_strategy": ["query_param", "json_field"]}
            },
            "enable_parameter_extraction": {
                "type": "boolean",
                "label": "Enable Parameter Extraction",
                "description": "Extract and format parameters for agent consumption",
                "default": True
            },
            "parameter_sources": {
                "type": "multiselect",
                "label": "Parameter Sources",
                "description": "Sources to extract parameters from",
                "required": False,
                "options": [
                    {"value": "query_params", "label": "Query Parameters"},
                    {"value": "body_json", "label": "JSON Body Fields"},
                    {"value": "headers", "label": "Request Headers"}
                ],
                "default": ["query_params", "body_json"],
                "show_when": {"enable_parameter_extraction": True}
            },
            "webhook_url": {
                "type": "string",
                "label": "Webhook URL",
                "description": "Auto-generated URL for external systems (read-only)",
                "required": False,
                "readonly": True
            }
        }
    }

def get_cache_node_schema() -> Dict[str, Any]:
    """Get schema for Cache Node in visual editor"""
    return {
        "type": "CacheNode",
        "name": "Cache",
        "description": "Cache agent outputs to skip re-execution on subsequent runs",
        "category": "Data",
        "icon": "💾",
        "color": "#06B6D4",
        "inputs": [
            {
                "name": "input",
                "type": "any",
                "label": "Input Data",
                "description": "Data to use for cache key generation"
            }
        ],
        "outputs": [
            {
                "name": "output",
                "type": "any",
                "label": "Cached Output",
                "description": "Cached data or fresh execution result"
            },
            {
                "name": "cache_info",
                "type": "object",
                "label": "Cache Information",
                "description": "Cache hit/miss status and metadata"
            }
        ],
        "properties": {
            "cache_key": {
                "type": "string",
                "label": "Cache Key",
                "description": "Custom cache key (leave empty for auto-generation)",
                "required": False,
                "placeholder": "workflow_step_1"
            },
            "cache_key_pattern": {
                "type": "select",
                "label": "Cache Key Pattern",
                "description": "How to generate cache keys automatically",
                "required": True,
                "options": [
                    {"value": "auto", "label": "Auto (workflow + node + inputs)"},
                    {"value": "node_only", "label": "Node ID Only"},
                    {"value": "input_hash", "label": "Input Data Hash"},
                    {"value": "custom", "label": "Custom Key"}
                ],
                "default": "auto"
            },
            "ttl": {
                "type": "number",
                "label": "TTL (Time To Live)",
                "description": "Cache expiration time in seconds (0 = never expires)",
                "required": False,
                "default": 3600,
                "min": 0,
                "max": 86400
            },
            "cache_policy": {
                "type": "select",
                "label": "Cache Policy",
                "description": "When to use cached data",
                "required": True,
                "options": [
                    {"value": "always", "label": "Always Use Cache"},
                    {"value": "input_match", "label": "Cache on Input Match"},
                    {"value": "conditional", "label": "Conditional Caching"}
                ],
                "default": "always"
            },
            "invalidate_on": {
                "type": "multiselect",
                "label": "Invalidate Cache On",
                "description": "Events that should clear the cache",
                "required": False,
                "options": [
                    {"value": "input_change", "label": "Input Change"},
                    {"value": "workflow_change", "label": "Workflow Change"},
                    {"value": "manual", "label": "Manual Only"},
                    {"value": "upstream_change", "label": "Upstream Node Change"}
                ],
                "default": ["input_change"]
            },
            "cache_condition": {
                "type": "textarea",
                "label": "Cache Condition",
                "description": "Only cache if this condition is met (JavaScript expression)",
                "required": False,
                "placeholder": "output.length > 100",
                "show_when": {"cache_policy": "conditional"}
            },
            "enable_warming": {
                "type": "boolean",
                "label": "Enable Cache Warming",
                "description": "Pre-populate cache with common scenarios",
                "default": False
            },
            "max_cache_size": {
                "type": "number",
                "label": "Max Cache Size (MB)",
                "description": "Maximum size for cached data (0 = unlimited)",
                "required": False,
                "default": 10,
                "min": 0,
                "max": 100
            },
            "cache_namespace": {
                "type": "string",
                "label": "Cache Namespace",
                "description": "Namespace to group related cache entries",
                "required": False,
                "placeholder": "workflow_v1",
                "default": "default"
            },
            "show_statistics": {
                "type": "boolean",
                "label": "Show Cache Statistics",
                "description": "Display hit/miss rates and performance metrics",
                "default": True
            }
        }
    }

def get_aggregator_node_schema() -> Dict[str, Any]:
    """Get schema for Aggregator Node - Advanced multi-agent output combination"""
    return {
        "type": "AggregatorNode",
        "name": "Output Aggregator",
        "description": "Intelligently combine multiple agent outputs using advanced aggregation strategies",
        "category": "Control",
        "icon": "🔄",
        "color": "#8B5CF6",
        "inputs": [
            {
                "name": "input-top",
                "type": "any",
                "label": "Input 1",
                "description": "First input from AgentNode, OutputNode, or ParallelNode"
            },
            {
                "name": "input-left",
                "type": "any", 
                "label": "Input 2",
                "description": "Second input from AgentNode, OutputNode, or ParallelNode"
            },
            {
                "name": "input-right",
                "type": "any",
                "label": "Input 3", 
                "description": "Third input from AgentNode, OutputNode, or ParallelNode"
            },
            {
                "name": "input-top-2",
                "type": "any",
                "label": "Input 4",
                "description": "Fourth input from AgentNode, OutputNode, or ParallelNode"
            }
        ],
        "outputs": [
            {
                "name": "aggregated_result",
                "type": "any",
                "label": "Aggregated Result",
                "description": "Final combined result from all inputs"
            },
            {
                "name": "confidence_score",
                "type": "number",
                "label": "Confidence Score",
                "description": "Confidence level of the aggregated result (0-1)"
            },
            {
                "name": "source_analysis",
                "type": "object",
                "label": "Source Analysis",
                "description": "Analysis of input sources and their contributions"
            },
            {
                "name": "metadata",
                "type": "object",
                "label": "Aggregation Metadata",
                "description": "Details about the aggregation process"
            }
        ],
        "properties": {
            "aggregation_strategy": {
                "type": "select",
                "label": "Aggregation Strategy",
                "description": "Method for combining multiple agent outputs",
                "required": True,
                "options": [
                    {"value": "semantic_merge", "label": "Semantic Merge (AI-powered content fusion)"},
                    {"value": "weighted_vote", "label": "Weighted Vote (Quality-based voting)"},
                    {"value": "consensus_ranking", "label": "Consensus Ranking (Cross-validation)"},
                    {"value": "relevance_weighted", "label": "Relevance Weighted (Score-based ranking)"},
                    {"value": "confidence_filter", "label": "Confidence Filter (High-confidence only)"},
                    {"value": "diversity_preservation", "label": "Diversity Preservation (Balanced perspectives)"},
                    {"value": "temporal_priority", "label": "Temporal Priority (Recent results favored)"},
                    {"value": "simple_concatenate", "label": "Simple Concatenate (Basic text merge)"},
                    {"value": "best_selection", "label": "Best Selection (Highest quality single result)"},
                    {"value": "structured_fusion", "label": "Structured Fusion (JSON/object merging)"}
                ],
                "default": "semantic_merge"
            },
            "confidence_threshold": {
                "type": "number",
                "label": "Confidence Threshold",
                "description": "Minimum confidence score to include in aggregation (0.0-1.0)",
                "required": False,
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1
            },
            "max_inputs": {
                "type": "number",
                "label": "Maximum Inputs",
                "description": "Maximum number of inputs to process (0 = unlimited)",
                "required": False,
                "default": 0,
                "min": 0,
                "max": 20
            },
            "deduplication_enabled": {
                "type": "boolean",
                "label": "Enable Deduplication",
                "description": "Remove duplicate or highly similar content",
                "default": True
            },
            "similarity_threshold": {
                "type": "number",
                "label": "Similarity Threshold",
                "description": "Threshold for considering content as duplicate (0.0-1.0)",
                "required": False,
                "default": 0.85,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "show_when": {"deduplication_enabled": True}
            },
            "quality_weights": {
                "type": "json",
                "label": "Quality Weights",
                "description": "Weights for different quality factors in scoring",
                "required": False,
                "default": {
                    "length": 0.2,
                    "coherence": 0.3,
                    "relevance": 0.3,
                    "completeness": 0.2
                }
            },
            "output_format": {
                "type": "select",
                "label": "Output Format",
                "description": "Format of the aggregated result",
                "required": True,
                "options": [
                    {"value": "comprehensive", "label": "Comprehensive Report"},
                    {"value": "summary", "label": "Executive Summary"},
                    {"value": "structured", "label": "Structured Data"},
                    {"value": "ranked_list", "label": "Ranked List"},
                    {"value": "consensus", "label": "Consensus Statement"},
                    {"value": "raw_merge", "label": "Raw Merged Content"}
                ],
                "default": "comprehensive"
            },
            "include_source_attribution": {
                "type": "boolean",
                "label": "Include Source Attribution",
                "description": "Include references to which agents contributed which parts",
                "default": True
            },
            "conflict_resolution": {
                "type": "select",
                "label": "Conflict Resolution",
                "description": "How to handle contradictory information",
                "required": True,
                "options": [
                    {"value": "highlight_conflicts", "label": "Highlight Conflicts"},
                    {"value": "majority_wins", "label": "Majority Wins"},
                    {"value": "quality_weighted", "label": "Quality Weighted Decision"},
                    {"value": "include_all_perspectives", "label": "Include All Perspectives"},
                    {"value": "ask_human", "label": "Flag for Human Review"}
                ],
                "default": "highlight_conflicts"
            },
            "semantic_analysis": {
                "type": "boolean",
                "label": "Enable Semantic Analysis",
                "description": "Use AI for deeper semantic understanding of content",
                "default": True
            },
            "preserve_structure": {
                "type": "boolean",
                "label": "Preserve Input Structure",
                "description": "Maintain original structure of input data when possible",
                "default": False
            },
            "validation_rules": {
                "type": "json",
                "label": "Validation Rules",
                "description": "Custom rules for validating aggregated output",
                "required": False,
                "default": {}
            },
            "fallback_strategy": {
                "type": "select",
                "label": "Fallback Strategy",
                "description": "What to do if aggregation fails",
                "required": True,
                "options": [
                    {"value": "return_best", "label": "Return Best Single Input"},
                    {"value": "simple_merge", "label": "Fall Back to Simple Merge"},
                    {"value": "error", "label": "Throw Error"},
                    {"value": "empty_result", "label": "Return Empty Result"}
                ],
                "default": "return_best"
            }
        }
    }

def get_api_node_schema() -> Dict[str, Any]:
    """Get schema for API Node - Universal REST API adapter"""
    return {
        "type": "APINode",
        "name": "APINode",
        "description": "Universal REST API adapter that bridges LLM reasoning with any standard REST API",
        "category": "Integration",
        "icon": "🌐",
        "color": "#10B981",
        "inputs": [
            {
                "name": "parameters",
                "type": "object",
                "label": "API Parameters",
                "description": "Parameters from LLM or previous nodes to pass to the API"
            },
            {
                "name": "context",
                "type": "any",
                "label": "Context",
                "description": "Additional context data for the API call"
            }
        ],
        "outputs": [
            {
                "name": "response",
                "type": "object",
                "label": "API Response",
                "description": "Clean JSON response from the API"
            },
            {
                "name": "status",
                "type": "number",
                "label": "Status Code",
                "description": "HTTP status code of the response"
            },
            {
                "name": "headers",
                "type": "object",
                "label": "Response Headers",
                "description": "HTTP response headers"
            },
            {
                "name": "metadata",
                "type": "object",
                "label": "Request Metadata",
                "description": "Metadata about the API request execution"
            }
        ],
        "properties": {
            "label": {
                "type": "string",
                "label": "API Name",
                "description": "Descriptive name for this API endpoint",
                "required": True,
                "placeholder": "Weather API",
                "default": "API Adapter"
            },
            "base_url": {
                "type": "string",
                "label": "Base URL",
                "description": "Root URL of the API service",
                "required": True,
                "placeholder": "https://api.weather.com"
            },
            "endpoint_path": {
                "type": "string",
                "label": "Endpoint Path",
                "description": "Specific API endpoint path",
                "required": True,
                "placeholder": "/v1/current"
            },
            "http_method": {
                "type": "select",
                "label": "HTTP Method",
                "description": "HTTP method for the API call",
                "required": True,
                "options": [
                    {"value": "GET", "label": "GET"},
                    {"value": "POST", "label": "POST"},
                    {"value": "PUT", "label": "PUT"},
                    {"value": "DELETE", "label": "DELETE"},
                    {"value": "PATCH", "label": "PATCH"}
                ],
                "default": "GET"
            },
            "authentication_type": {
                "type": "select",
                "label": "Authentication Type",
                "description": "Authentication method for the API",
                "required": True,
                "options": [
                    {"value": "none", "label": "No Authentication"},
                    {"value": "api_key", "label": "API Key (Header)"},
                    {"value": "bearer_token", "label": "Bearer Token"},
                    {"value": "basic_auth", "label": "Basic Authentication"},
                    {"value": "custom_header", "label": "Custom Header"}
                ],
                "default": "none"
            },
            "auth_header_name": {
                "type": "string",
                "label": "Auth Header Name",
                "description": "Name of the authentication header",
                "required": False,
                "default": "X-API-Key",
                "show_when": {"authentication_type": ["api_key", "custom_header"]}
            },
            "auth_token": {
                "type": "string",
                "label": "API Key/Token",
                "description": "Authentication token or API key",
                "required": False,
                "show_when": {"authentication_type": ["api_key", "bearer_token", "custom_header"]}
            },
            "basic_auth_username": {
                "type": "string",
                "label": "Username",
                "description": "Username for basic authentication",
                "required": False,
                "show_when": {"authentication_type": "basic_auth"}
            },
            "basic_auth_password": {
                "type": "string",
                "label": "Password",
                "description": "Password for basic authentication",
                "required": False,
                "show_when": {"authentication_type": "basic_auth"}
            },
            "request_schema": {
                "type": "json",
                "label": "Request Schema",
                "description": "JSON schema defining the parameters this API accepts",
                "required": True,
                "default": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query parameter"
                        }
                    },
                    "required": ["query"]
                }
            },
            "response_schema": {
                "type": "json",
                "label": "Response Schema",
                "description": "JSON schema defining the expected API response structure",
                "required": False,
                "default": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Response data"
                        }
                    }
                }
            },
            "timeout": {
                "type": "number",
                "label": "Timeout (seconds)",
                "description": "Request timeout in seconds",
                "required": False,
                "default": 30,
                "min": 1,
                "max": 300
            },
            "retry_count": {
                "type": "number",
                "label": "Retry Count",
                "description": "Number of retry attempts on failure",
                "required": False,
                "default": 3,
                "min": 0,
                "max": 10
            },
            "rate_limit": {
                "type": "number",
                "label": "Rate Limit (requests/minute)",
                "description": "Maximum requests per minute (0 = unlimited)",
                "required": False,
                "default": 60,
                "min": 0,
                "max": 1000
            },
            "custom_headers": {
                "type": "json",
                "label": "Custom Headers",
                "description": "Additional HTTP headers to include in requests",
                "required": False,
                "default": {}
            },
            "response_transformation": {
                "type": "textarea",
                "label": "Response Transformation",
                "description": "JavaScript code to transform the API response (optional)",
                "required": False,
                "placeholder": "// Transform the response\n// return response.data.items;"
            },
            "error_handling": {
                "type": "select",
                "label": "Error Handling",
                "description": "How to handle API errors",
                "required": True,
                "options": [
                    {"value": "throw", "label": "Throw Error"},
                    {"value": "return_null", "label": "Return Null"},
                    {"value": "return_error", "label": "Return Error Object"},
                    {"value": "retry", "label": "Retry with Backoff"}
                ],
                "default": "throw"
            },
            "enable_mcp_tool": {
                "type": "boolean",
                "label": "Enable as MCP Tool",
                "description": "Make this API available as a tool for connected Agent nodes",
                "default": True
            },
            "tool_description": {
                "type": "textarea",
                "label": "Tool Description",
                "description": "Description of this API tool for LLM agents",
                "required": False,
                "placeholder": "This tool allows you to search for current weather information by location.",
                "show_when": {"enable_mcp_tool": True}
            }
        }
    }

def get_all_node_schemas() -> List[Dict[str, Any]]:
    """Get all available node schemas for the visual editor"""
    return [
        get_agent_node_schema(),
        get_input_node_schema(),
        get_output_node_schema(),
        get_trigger_node_schema(),
        get_condition_node_schema(),
        get_parallel_node_schema(),
        get_aggregator_node_schema(),
        get_state_node_schema(),
        get_router_node_schema(),
        get_transform_node_schema(),
        get_cache_node_schema(),
        get_api_node_schema()
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
    has_trigger = any(node.get("data", {}).get("type") == "TriggerNode" for node in nodes)
    has_agent = any(node.get("data", {}).get("type") == "AgentNode" for node in nodes)
    has_output = any(node.get("data", {}).get("type") == "OutputNode" for node in nodes)
    
    if not has_input and not has_trigger:
        warnings.append("Workflow has no input or trigger node - will use default input handling")
    
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
    
    # Check cache node configurations and pre-execution cache status
    cache_nodes = [node for node in nodes if node.get("data", {}).get("type") == "CacheNode"]
    cache_info = []
    
    for cache_node in cache_nodes:
        node_id = cache_node.get("id")
        cache_config = cache_node.get("data", {}).get("node", {})
        
        # Validate cache configuration
        cache_key_pattern = cache_config.get("cache_key_pattern", "auto")
        custom_key = cache_config.get("cache_key", "")
        
        if cache_key_pattern == "custom" and not custom_key:
            errors.append(f"Cache node {node_id} has custom key pattern but no custom key specified")
        
        # TTL validation
        ttl = cache_config.get("ttl", 3600)
        if ttl < 0 or ttl > 86400:
            warnings.append(f"Cache node {node_id} has unusual TTL: {ttl} seconds")
        
        # Cache policy validation
        cache_policy = cache_config.get("cache_policy", "always")
        if cache_policy == "conditional":
            cache_condition = cache_config.get("cache_condition", "")
            if not cache_condition:
                warnings.append(f"Cache node {node_id} has conditional policy but no condition specified")
        
        # Check if cache node is properly connected
        cache_has_input = any(edge.get("target") == node_id for edge in edges)
        cache_has_output = any(edge.get("source") == node_id for edge in edges)
        
        if not cache_has_input:
            warnings.append(f"Cache node {node_id} has no input connections")
        if not cache_has_output:
            warnings.append(f"Cache node {node_id} has no output connections")
        
        # Add cache info for pre-execution check
        cache_info.append({
            "node_id": node_id,
            "cache_key_pattern": cache_key_pattern,
            "custom_key": custom_key,
            "ttl": ttl,
            "cache_policy": cache_policy,
            "has_connections": cache_has_input and cache_has_output
        })
    
    validation_result = {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
    
    # Add cache information if cache nodes exist
    if cache_nodes:
        validation_result["cache_info"] = {
            "total_cache_nodes": len(cache_nodes),
            "cache_nodes": cache_info,
            "requires_cache_check": True
        }
    
    return validation_result