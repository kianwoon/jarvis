#!/usr/bin/env python3
"""
Enhanced Tool Executor

Drop-in replacement for the current call_mcp_tool function that provides:
1. Support for both HTTP and stdio MCP servers
2. Automatic OAuth token refresh when requests fail due to token expiry
3. Comprehensive error handling and retry logic
4. Backwards compatibility with existing codebase
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from .unified_mcp_service import call_mcp_tool_unified, unified_mcp_service
from .mcp_tools_cache import get_enabled_mcp_tools

logger = logging.getLogger(__name__)

def call_mcp_tool_enhanced(tool_name: str, parameters: Dict[str, Any], 
                         trace=None, _skip_span_creation=False) -> Dict[str, Any]:
    """
    Enhanced MCP tool executor that handles both HTTP and stdio servers
    with automatic OAuth token refresh.
    
    This is a drop-in replacement for the current call_mcp_tool function
    in service.py that provides improved reliability and OAuth handling.
    
    Args:
        tool_name: Name of the tool to call
        parameters: Tool parameters
        trace: Optional trace for Langfuse
        _skip_span_creation: Skip span creation for Langfuse
        
    Returns:
        Tool result or error dictionary
    """
    # Ensure logger is available
    import logging
    logger = logging.getLogger(__name__)
    
    # Create tool span if trace is provided
    tool_span = None
    tracer = None
    if trace and not _skip_span_creation:
        try:
            from app.core.langfuse_integration import get_tracer
            tracer = get_tracer()
            if tracer.is_enabled():
                # Sanitize parameters for Langfuse
                safe_parameters = {}
                if isinstance(parameters, dict):
                    for key, value in parameters.items():
                        safe_key = str(key)[:100]
                        safe_value = str(value)[:500] if value is not None else ""
                        safe_parameters[safe_key] = safe_value
                tool_span = tracer.create_tool_span(trace, str(tool_name), safe_parameters)
        except Exception as e:
            logger.warning(f"Failed to create tool span for {tool_name}: {e}")
            tool_span = None
            tracer = None
    
    # Apply parameter mapping for common mismatches
    try:
        tool_name, parameters = _map_tool_parameters_service(tool_name, parameters)
    except:
        # Fallback if mapping function doesn't exist
        pass
    
    try:
        # Get enabled tools from cache
        enabled_tools = get_enabled_mcp_tools()
        if tool_name not in enabled_tools:
            error_msg = f"Tool {tool_name} is disabled or not available"
            if tool_span and tracer:
                try:
                    tracer.end_span_with_result(tool_span, {"error": error_msg}, False, error_msg)
                except:
                    pass
            return {"error": error_msg}
        
        tool_info = enabled_tools[tool_name]
        
        # Clean up parameters (remove agent parameter that shouldn't be sent to tool)
        clean_parameters = {k: v for k, v in parameters.items() if k != "agent"}
        
        # Handle hostname replacement for Docker/localhost scenarios
        tool_info = _adjust_endpoint_for_environment(tool_info)
        
        logger.info(f"[ENHANCED] Calling {tool_name} via unified service")
        
        # Call the unified MCP service
        if asyncio.iscoroutinefunction(call_mcp_tool_unified):
            # Run async function in sync context
            result = asyncio.run(call_mcp_tool_unified(tool_info, tool_name, clean_parameters))
        else:
            result = call_mcp_tool_unified(tool_info, tool_name, clean_parameters)
        
        # End tool span with result
        if tool_span and tracer:
            try:
                success = "error" not in result
                tracer.end_span_with_result(tool_span, result, success, result.get("error"))
            except Exception as e:
                logger.warning(f"Failed to end tool span for {tool_name}: {e}")
        
        logger.info(f"[ENHANCED] Tool {tool_name} completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Enhanced tool execution failed for {tool_name}: {str(e)}"
        logger.error(error_msg)
        
        if tool_span and tracer:
            try:
                tracer.end_span_with_result(tool_span, None, False, error_msg)
            except:
                pass
        
        return {"error": error_msg}

def _adjust_endpoint_for_environment(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjust endpoint URLs based on whether we're running inside Docker
    """
    try:
        endpoint = tool_info.get("endpoint", "")
        server_hostname = tool_info.get("server_hostname")
        
        # Check if we're running inside Docker
        import os
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        adjusted_tool_info = tool_info.copy()
        
        if in_docker and server_hostname and "localhost" in endpoint:
            adjusted_tool_info["endpoint"] = endpoint.replace("localhost", server_hostname)
        elif not in_docker and server_hostname and server_hostname in endpoint:
            adjusted_tool_info["endpoint"] = endpoint.replace(server_hostname, "localhost")
        
        return adjusted_tool_info
        
    except Exception as e:
        logger.warning(f"Failed to adjust endpoint: {e}")
        return tool_info

def _map_tool_parameters_service(tool_name: str, parameters: Dict[str, Any]) -> tuple:
    """
    Apply parameter mapping for common mismatches between agents and tools
    """
    try:
        # Import the existing mapping function if it exists
        from app.langchain.service import _map_tool_parameters_service as existing_mapper
        return existing_mapper(tool_name, parameters)
    except ImportError:
        # Fallback mapping logic
        if tool_name == "find_email" and "from" in parameters:
            # Map 'from' to 'sender' for Gmail tools
            parameters = parameters.copy()
            parameters["sender"] = parameters.pop("from")
        
        return tool_name, parameters

# Async wrapper for async contexts
async def call_mcp_tool_enhanced_async(tool_name: str, parameters: Dict[str, Any], 
                                     trace=None, _skip_span_creation=False) -> Dict[str, Any]:
    """
    Async version of the enhanced MCP tool executor
    """
    try:
        # Get enabled tools from cache
        enabled_tools = get_enabled_mcp_tools()
        if tool_name not in enabled_tools:
            return {"error": f"Tool {tool_name} is disabled or not available"}
        
        tool_info = enabled_tools[tool_name]
        
        # Clean up parameters
        clean_parameters = {k: v for k, v in parameters.items() if k != "agent"}
        
        # Handle hostname replacement
        tool_info = _adjust_endpoint_for_environment(tool_info)
        
        logger.info(f"[ENHANCED-ASYNC] Calling {tool_name} via unified service")
        
        # Call the unified MCP service
        result = await call_mcp_tool_unified(tool_info, tool_name, clean_parameters)
        
        logger.info(f"[ENHANCED-ASYNC] Tool {tool_name} completed successfully")
        return result
        
    except Exception as e:
        error_msg = f"Enhanced async tool execution failed for {tool_name}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

# Cleanup function for proper shutdown
def cleanup_enhanced_executor():
    """Clean up resources used by the enhanced executor"""
    try:
        asyncio.run(unified_mcp_service.close())
    except Exception as e:
        logger.warning(f"Failed to cleanup enhanced executor: {e}")

# Backwards compatibility aliases
call_mcp_tool = call_mcp_tool_enhanced
call_mcp_tool_async = call_mcp_tool_enhanced_async