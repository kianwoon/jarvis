#!/usr/bin/env python3
"""
MCP Service Layer

This module provides a service layer for managing MCP servers and tools,
replacing the complex stdio_mcp_handler with proper MCP-compliant communication.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from .mcp_client import MCPClient, MCPServerConfig, MCPServerManager
from .oauth_token_manager import oauth_token_manager

logger = logging.getLogger(__name__)

# Global MCP server manager
mcp_manager = MCPServerManager()

def initialize_mcp_servers():
    """Initialize MCP servers from database configuration"""
    try:
        from .db import SessionLocal, MCPServer
        
        db = SessionLocal()
        try:
            servers = db.query(MCPServer).filter(MCPServer.is_active == True).all()
            
            for server in servers:
                if server.config_type == "command" and server.command == "docker":
                    # Extract container name and command from args
                    args = server.args or []
                    if len(args) >= 4 and args[0] == "exec":
                        container_name = args[2]  # ['exec', '-i', 'container_name', ...]
                        command = args[3:]        # ['node', '/app/index.js']
                        
                        config = MCPServerConfig(
                            name=server.name,
                            container_name=container_name,
                            command=command,
                            environment=server.env or {}
                        )
                        
                        mcp_manager.register_server(config)
                        logger.info(f"[MCP-SERVICE] Initialized server: {server.name}")
                    else:
                        logger.warning(f"[MCP-SERVICE] Invalid args for server {server.name}: {args}")
                else:
                    logger.info(f"[MCP-SERVICE] Skipping non-Docker server: {server.name}")
                    
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Failed to initialize MCP servers: {e}")

def _inject_oauth_credentials(parameters: Dict[str, Any], server_name: str) -> Dict[str, Any]:
    """
    Inject OAuth credentials for Gmail MCP servers
    
    Args:
        parameters: Original tool parameters
        server_name: Name of the MCP server
        
    Returns:
        Parameters with OAuth credentials injected
    """
    try:
        # Only inject for Gmail servers
        if "gmail" not in server_name.lower():
            return parameters
        
        # Get OAuth credentials (using server_id 3 for Gmail as in original code)
        oauth_creds = oauth_token_manager.get_valid_token(3, "gmail")
        
        if oauth_creds:
            enhanced_params = parameters.copy()
            enhanced_params.update({
                "google_client_id": oauth_creds.get("client_id"),
                "google_client_secret": oauth_creds.get("client_secret"),
                "google_access_token": oauth_creds.get("access_token"),
                "google_refresh_token": oauth_creds.get("refresh_token")
            })
            
            logger.debug(f"[MCP-SERVICE] Injected OAuth credentials for {server_name}")
            return enhanced_params
        else:
            logger.warning(f"[MCP-SERVICE] No OAuth credentials found for {server_name}")
            return parameters
            
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Failed to inject OAuth credentials: {e}")
        return parameters

def _fix_gmail_parameters(parameters: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """
    Fix Gmail parameter format issues
    
    Args:
        parameters: Original tool parameters
        tool_name: Name of the tool being called
        
    Returns:
        Parameters with corrected format
    """
    try:
        if tool_name in ["gmail_send", "draft_email", "gmail_update_draft"]:
            fixed_params = parameters.copy()
            
            # Fix 'to' parameter - ensure it's an array
            if "to" in fixed_params:
                if isinstance(fixed_params["to"], str):
                    logger.debug(f"[MCP-SERVICE] Converting 'to' from string to array for {tool_name}")
                    fixed_params["to"] = [fixed_params["to"]]
                elif not isinstance(fixed_params["to"], list):
                    fixed_params["to"] = [str(fixed_params["to"])]
            
            # Fix 'cc' and 'bcc' parameters
            for field in ["cc", "bcc"]:
                if field in fixed_params and fixed_params[field]:
                    if isinstance(fixed_params[field], str):
                        fixed_params[field] = [fixed_params[field]]
                    elif not isinstance(fixed_params[field], list):
                        fixed_params[field] = [str(fixed_params[field])]
            
            return fixed_params
        
        return parameters
        
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Failed to fix Gmail parameters: {e}")
        return parameters

async def call_mcp_tool(server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call an MCP tool on a specific server
    
    Args:
        server_name: Name of the MCP server
        tool_name: Name of the tool to call
        parameters: Tool parameters
        
    Returns:
        Tool result or error
    """
    try:
        logger.info(f"[MCP-SERVICE] Calling {tool_name} on server {server_name}")
        
        # Inject OAuth credentials if needed
        enhanced_params = _inject_oauth_credentials(parameters, server_name)
        
        # Fix parameter format if needed
        enhanced_params = _fix_gmail_parameters(enhanced_params, tool_name)
        
        # Call the tool
        result = await mcp_manager.call_tool_on_server(server_name, tool_name, enhanced_params)
        
        logger.info(f"[MCP-SERVICE] Tool {tool_name} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Failed to call {tool_name} on {server_name}: {e}")
        return {"error": str(e)}

async def list_server_tools(server_name: str) -> Dict[str, Any]:
    """
    List tools available on a specific server
    
    Args:
        server_name: Name of the MCP server
        
    Returns:
        Dictionary containing tools list or error
    """
    try:
        client = mcp_manager.get_server(server_name)
        if not client:
            return {"error": f"Server '{server_name}' not found"}
        
        return await client.list_tools()
        
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Failed to list tools for {server_name}: {e}")
        return {"error": str(e)}

async def health_check_server(server_name: str) -> bool:
    """
    Check if a specific MCP server is healthy
    
    Args:
        server_name: Name of the MCP server
        
    Returns:
        True if server is healthy, False otherwise
    """
    try:
        client = mcp_manager.get_server(server_name)
        if not client:
            return False
        
        return await client.health_check()
        
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Health check failed for {server_name}: {e}")
        return False

def get_server_for_tool(tool_name: str) -> Optional[str]:
    """
    Get the server name that provides a specific tool
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Server name or None if not found
    """
    try:
        from .db import SessionLocal, MCPTool, MCPServer
        
        db = SessionLocal()
        try:
            tool = db.query(MCPTool).filter(
                MCPTool.name == tool_name,
                MCPTool.is_active == True
            ).first()
            
            if tool and tool.server:
                return tool.server.name
            
            return None
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Failed to find server for tool {tool_name}: {e}")
        return None

# Backwards compatibility function for existing code
async def call_stdio_mcp_tool_compat(server_config: Dict[str, Any], tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backwards compatibility wrapper for the old stdio_mcp_handler
    
    This function provides the same interface as the old call_stdio_mcp_tool
    but uses the new MCP-compliant client internally.
    """
    try:
        # Extract server information from config
        args = server_config.get("args", [])
        if len(args) >= 4 and args[0] == "exec":
            container_name = args[2]
            command = args[3:]
            
            # Find existing server or create temporary one
            server_name = None
            for name, client in mcp_manager.servers.items():
                if client.config.container_name == container_name:
                    server_name = name
                    break
            
            if not server_name:
                # Create temporary server config
                temp_config = MCPServerConfig(
                    name=f"temp_{container_name}",
                    container_name=container_name,
                    command=command
                )
                client = mcp_manager.register_server(temp_config)
                server_name = temp_config.name
            
            # Call the tool
            return await call_mcp_tool(server_name, tool_name, parameters)
        else:
            return {"error": f"Invalid server config: {server_config}"}
            
    except Exception as e:
        logger.error(f"[MCP-SERVICE] Compatibility call failed: {e}")
        return {"error": str(e)}