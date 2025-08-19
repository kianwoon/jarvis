#!/usr/bin/env python3
"""
MCP Endpoint Resolver
Handles endpoint resolution for both Docker and non-Docker environments
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def resolve_mcp_endpoint(tool_info: Dict[str, Any]) -> str:
    """
    Resolve the correct endpoint URL based on the environment
    
    Args:
        tool_info: Tool information including endpoint and server_hostname
        
    Returns:
        Resolved endpoint URL
    """
    endpoint = tool_info.get("endpoint", "")
    server_hostname = tool_info.get("server_hostname", "")
    
    # Check if we're running inside Docker
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
    
    logger.debug(f"Resolving endpoint: {endpoint}, in_docker: {in_docker}")
    
    # Handle different endpoint formats
    if not endpoint:
        return endpoint
    
    # For non-Docker environments (local development)
    if not in_docker:
        # Replace host.docker.internal with localhost for local execution
        if "host.docker.internal" in endpoint:
            resolved = endpoint.replace("host.docker.internal", "localhost")
            logger.info(f"[LOCAL] Resolved endpoint: {endpoint} -> {resolved}")
            return resolved
        # Keep localhost endpoints as-is
        elif "localhost" in endpoint:
            logger.debug(f"[LOCAL] Keeping localhost endpoint: {endpoint}")
            return endpoint
    
    # For Docker environments
    else:
        # Use server_hostname if available and endpoint has localhost
        if server_hostname and "localhost" in endpoint:
            resolved = endpoint.replace("localhost", server_hostname)
            logger.info(f"[DOCKER] Resolved endpoint: {endpoint} -> {resolved}")
            return resolved
        # Keep host.docker.internal as-is in Docker
        elif "host.docker.internal" in endpoint:
            logger.debug(f"[DOCKER] Keeping host.docker.internal endpoint: {endpoint}")
            return endpoint
    
    # Return endpoint unchanged if no resolution needed
    return endpoint

def get_resolved_tool_info(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get tool info with resolved endpoint
    
    Args:
        tool_info: Original tool information
        
    Returns:
        Tool info with resolved endpoint
    """
    resolved_info = tool_info.copy()
    resolved_info["endpoint"] = resolve_mcp_endpoint(tool_info)
    return resolved_info
