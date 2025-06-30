"""
Intelligent Tool Integration

Mode-specific integration functions for easy use across different chat modes.
Respects tool constraints for standard chat, multi-agent, and agentic pipeline modes.
"""

import logging
from typing import Dict, Any, List, Optional
from .intelligent_tool_executor import execute_task_with_intelligent_tools

logger = logging.getLogger(__name__)

async def execute_standard_chat_tools(
    task: str,
    context: Dict[str, Any] = None,
    trace=None,
    stream_callback=None
) -> List[Dict[str, Any]]:
    """
    Execute tools for standard chat mode - all MCP tools available
    
    Args:
        task: The task to accomplish
        context: Additional context
        trace: Langfuse trace for observability
        stream_callback: Optional streaming callback
        
    Returns:
        List of execution events
    """
    logger.info(f"[INTELLIGENT TOOLS] Standard chat execution: {task[:100]}...")
    
    return await execute_task_with_intelligent_tools(
        task=task,
        context=context,
        trace=trace,
        stream_callback=stream_callback,
        mode="standard"
    )

async def execute_multi_agent_tools(
    task: str,
    agent_name: str,
    context: Dict[str, Any] = None,
    trace=None,
    stream_callback=None
) -> List[Dict[str, Any]]:
    """
    Execute tools for multi-agent mode - only agent's assigned tools
    
    Args:
        task: The task to accomplish
        agent_name: Name of the agent (required)
        context: Additional context
        trace: Langfuse trace for observability
        stream_callback: Optional streaming callback
        
    Returns:
        List of execution events
    """
    if not agent_name:
        logger.error("[INTELLIGENT TOOLS] Agent name required for multi-agent mode")
        return [{"type": "error", "error": "Agent name required for multi-agent mode"}]
    
    logger.info(f"[INTELLIGENT TOOLS] Multi-agent execution for {agent_name}: {task[:100]}...")
    
    return await execute_task_with_intelligent_tools(
        task=task,
        context=context,
        trace=trace,
        stream_callback=stream_callback,
        mode="multi_agent",
        agent_name=agent_name
    )


def analyze_agent_tool_constraints(agent_name: str, mode: str = "multi_agent") -> Dict[str, Any]:
    """
    Analyze tool constraints for an agent in a specific mode
    
    Args:
        agent_name: Name of the agent
        mode: "multi_agent" or "pipeline"
        
    Returns:
        Dict with agent tool information and constraints
    """
    try:
        from .intelligent_tool_planner import get_tool_planner
        planner = get_tool_planner()
        
        # Get available tools for this agent/mode
        available_tools = planner.get_enhanced_tool_metadata(mode, agent_name, pipeline_id)
        
        if not available_tools:
            return {
                "agent_name": agent_name,
                "mode": mode,
                "pipeline_id": pipeline_id,
                "available_tools": [],
                "tool_count": 0,
                "error": "No tools available for this agent/mode combination"
            }
        
        # Categorize tools
        tools_by_category = {}
        for tool_name, tool_info in available_tools.items():
            category = tool_info.get("category", "utility")
            if category not in tools_by_category:
                tools_by_category[category] = []
            tools_by_category[category].append({
                "name": tool_name,
                "description": tool_info.get("description", ""),
                "complexity": tool_info.get("complexity", "unknown"),
                "capabilities": tool_info.get("capabilities", [])
            })
        
        return {
            "agent_name": agent_name,
            "mode": mode,
            "pipeline_id": pipeline_id,
            "available_tools": list(available_tools.keys()),
            "tool_count": len(available_tools),
            "tools_by_category": tools_by_category,
            "constraints": {
                "can_use_all_mcp_tools": mode == "standard",
                "restricted_to_assigned_tools": mode in ["multi_agent", "pipeline"],
                "supports_intelligent_planning": True,
                "supports_multi_tool_workflows": True
            }
        }
        
    except Exception as e:
        logger.error(f"[INTELLIGENT TOOLS] Failed to analyze constraints for {agent_name}: {e}")
        return {
            "agent_name": agent_name,
            "mode": mode,
            "pipeline_id": pipeline_id,
            "available_tools": [],
            "tool_count": 0,
            "error": str(e)
        }

def get_mode_specific_tool_summary() -> Dict[str, Any]:
    """
    Get summary of tool availability across different modes
    
    Returns:
        Dict with mode-specific tool information
    """
    try:
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        from app.core.langgraph_agents_cache import get_langgraph_agents
        
        all_mcp_tools = get_enabled_mcp_tools()
        all_agents = get_langgraph_agents()
        
        # Standard chat tools
        standard_tools = list(all_mcp_tools.keys()) if all_mcp_tools else []
        
        # Multi-agent tools summary
        agent_tool_summary = {}
        if all_agents:
            for agent_name, agent_data in all_agents.items():
                agent_tools = agent_data.get("tools", [])
                if isinstance(agent_tools, list):
                    agent_tool_summary[agent_name] = {
                        "tool_count": len(agent_tools),
                        "tools": agent_tools,
                        "is_active": agent_data.get("is_active", False)
                    }
        
        return {
            "standard_chat": {
                "mode": "standard",
                "tool_count": len(standard_tools),
                "all_tools_available": True,
                "constraint": "None - all MCP tools available"
            },
            "multi_agent": {
                "mode": "multi_agent", 
                "agents": agent_tool_summary,
                "total_agents": len(agent_tool_summary),
                "constraint": "Each agent has assigned tools only"
            },
            "agentic_pipeline": {
                "mode": "pipeline",
                "constraint": "Pipeline agents have context-specific tool assignments",
                "note": "Tools can be overridden at pipeline level"
            },
            "intelligent_features": {
                "task_analysis": True,
                "multi_tool_orchestration": True,
                "adaptive_planning": True,
                "error_recovery": True,
                "context_awareness": True,
                "langfuse_tracing": True
            }
        }
        
    except Exception as e:
        logger.error(f"[INTELLIGENT TOOLS] Failed to get mode summary: {e}")
        return {"error": str(e)}

# Legacy compatibility functions for gradual migration
async def intelligent_tool_fallback(task: str, trace=None, **kwargs) -> List[Dict[str, Any]]:
    """
    Fallback function for legacy regex-based tool calls
    Automatically detects mode and calls appropriate function
    """
    # Try to detect mode from context
    context = kwargs.get("context", {})
    agent_name = kwargs.get("agent_name") or context.get("agent_name")
    pipeline_id = kwargs.get("pipeline_id") or context.get("pipeline_id")
    
    if pipeline_id is not None and agent_name:
        # Pipeline mode
        return await execute_pipeline_agent_tools(task, agent_name, pipeline_id, context, trace)
    elif agent_name:
        # Multi-agent mode
        return await execute_multi_agent_tools(task, agent_name, context, trace)
    else:
        # Standard chat mode
        return await execute_standard_chat_tools(task, context, trace)