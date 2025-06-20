#!/usr/bin/env python3
"""
Unified MCP Service

Handles both HTTP and stdio MCP servers with automatic OAuth token refresh
and comprehensive error handling as per the current codebase design.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import requests

from .oauth_token_manager import oauth_token_manager
from .mcp_client import MCPClient, MCPServerConfig

logger = logging.getLogger(__name__)

class UnifiedMCPService:
    """
    Unified service for handling both HTTP and stdio MCP servers
    with automatic OAuth token refresh and error handling.
    """
    
    def __init__(self):
        self.http_session = None
        self.stdio_clients: Dict[str, MCPClient] = {}
        
    async def _get_http_session(self):
        """Get or create aiohttp session for HTTP MCP servers"""
        if self.http_session is None or self.http_session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.http_session = aiohttp.ClientSession(timeout=timeout)
        return self.http_session
    
    async def close(self):
        """Close HTTP session"""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
    
    def _get_stdio_client(self, server_config: Dict[str, Any]) -> MCPClient:
        """Get or create stdio MCP client for a server"""
        command = server_config.get("command")
        args = server_config.get("args", [])
        
        if command == "docker" and args:
            # Handle Docker-based stdio servers
            if len(args) >= 4 and args[0] == "exec":
                container_name = args[2]
                docker_command = args[3:]
                
                client_key = f"{container_name}:{':'.join(docker_command)}"
                
                if client_key not in self.stdio_clients:
                    config = MCPServerConfig(
                        name=f"stdio_{container_name}",
                        container_name=container_name,
                        command=docker_command,
                        environment=server_config.get("env", {})
                    )
                    self.stdio_clients[client_key] = MCPClient(config)
                
                return self.stdio_clients[client_key]
        elif command in ['npx', 'node', 'python', 'python3']:
            # Handle direct stdio commands (npx, node, etc.)
            client_key = f"{command}:{':'.join(args)}"
            
            if client_key not in self.stdio_clients:
                # For direct commands, we need to use the stdio bridge directly
                # rather than the MCPClient which is designed for Docker containers
                from .mcp_stdio_bridge import call_mcp_tool_via_stdio
                # Store server config for later use
                self.stdio_clients[client_key] = server_config
            
            return self.stdio_clients[client_key]
        
        raise ValueError(f"Invalid stdio server config: {server_config}")
    
    def _inject_oauth_credentials(self, parameters: Dict[str, Any], server_id: int, 
                                service_name: str = "gmail") -> Dict[str, Any]:
        """
        Inject OAuth credentials with automatic refresh
        
        Args:
            parameters: Original tool parameters
            server_id: MCP server ID
            service_name: Service name (gmail, outlook, etc.)
            
        Returns:
            Parameters with OAuth credentials injected
        """
        try:
            # Ensure parameters is not None
            if parameters is None:
                parameters = {}
            
            # Get valid OAuth token (automatically refreshes if needed)
            oauth_creds = oauth_token_manager.get_valid_token(server_id, service_name)
            
            if oauth_creds:
                enhanced_params = parameters.copy()
                
                if service_name == "gmail":
                    enhanced_params.update({
                        "google_client_id": oauth_creds.get("client_id"),
                        "google_client_secret": oauth_creds.get("client_secret"),
                        "google_access_token": oauth_creds.get("access_token"),
                        "google_refresh_token": oauth_creds.get("refresh_token")
                    })
                elif service_name == "outlook":
                    enhanced_params.update({
                        "microsoft_client_id": oauth_creds.get("client_id"),
                        "microsoft_client_secret": oauth_creds.get("client_secret"),
                        "microsoft_access_token": oauth_creds.get("access_token"),
                        "microsoft_refresh_token": oauth_creds.get("refresh_token")
                    })
                
                logger.debug(f"Injected OAuth credentials for {service_name}")
                return enhanced_params
            else:
                logger.warning(f"No OAuth credentials found for server {server_id}, service {service_name}")
                return parameters
                
        except Exception as e:
            logger.error(f"Failed to inject OAuth credentials: {e}")
            return parameters
    
    def _fix_parameter_format(self, parameters: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """
        Fix parameter format issues for specific tools
        
        Args:
            parameters: Original tool parameters
            tool_name: Name of the tool being called
            
        Returns:
            Parameters with corrected format
        """
        try:
            # Ensure parameters is not None
            if parameters is None:
                parameters = {}
                
            # Gmail tools parameter fixes
            if tool_name in ["gmail_send", "draft_email", "gmail_update_draft"]:
                fixed_params = parameters.copy()
                
                # Fix 'to' parameter - ensure it's an array
                if "to" in fixed_params:
                    if isinstance(fixed_params["to"], str):
                        logger.debug(f"Converting 'to' from string to array for {tool_name}")
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
                
                # Fix 'message' parameter - should be 'body' for Gmail tools
                if "message" in fixed_params and "body" not in fixed_params:
                    logger.debug(f"Converting 'message' to 'body' parameter for {tool_name}")
                    fixed_params["body"] = fixed_params["message"]
                    del fixed_params["message"]
                
                return fixed_params
            
            return parameters
            
        except Exception as e:
            logger.error(f"Failed to fix parameters for {tool_name}: {e}")
            return parameters
    
    async def _handle_token_expiry_error(self, error_response: Dict[str, Any], 
                                       server_id: int, service_name: str) -> bool:
        """
        Handle token expiry errors by refreshing tokens
        
        Args:
            error_response: Error response from tool execution
            server_id: MCP server ID
            service_name: Service name
            
        Returns:
            True if token was refreshed successfully, False otherwise
        """
        try:
            error_msg = str(error_response.get("error", "")).lower()
            
            # Check for common token expiry indicators
            token_expiry_indicators = [
                "invalid_token", "token_expired", "unauthorized", 
                "invalid_client", "invalid_grant", "401", "403"
            ]
            
            if any(indicator in error_msg for indicator in token_expiry_indicators):
                logger.info(f"Detected token expiry for server {server_id}, service {service_name}")
                
                # Invalidate cached token
                oauth_token_manager.invalidate_token(server_id, service_name)
                
                # Force refresh by getting a new token
                new_creds = oauth_token_manager.get_valid_token(server_id, service_name)
                
                if new_creds:
                    logger.info(f"Successfully refreshed token for {service_name}")
                    return True
                else:
                    logger.error(f"Failed to refresh token for {service_name}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling token expiry: {e}")
            return False
    
    async def call_stdio_tool(self, server_config: Dict[str, Any], tool_name: str, 
                            parameters: Dict[str, Any], server_id: int = None,
                            service_name: str = "gmail") -> Dict[str, Any]:
        """
        Call a tool on a stdio MCP server with OAuth handling
        
        Args:
            server_config: Server configuration (command, args, env)
            tool_name: Name of the tool to call
            parameters: Tool parameters
            server_id: Server ID for OAuth credential lookup
            service_name: Service name for OAuth (gmail, outlook, etc.)
            
        Returns:
            Tool result or error
        """
        try:
            logger.info(f"[STDIO] Calling {tool_name} on stdio server")
            
            # Get stdio client
            client = self._get_stdio_client(server_config)
            
            # Fix parameter format
            fixed_params = self._fix_parameter_format(parameters, tool_name)
            
            # Inject OAuth credentials if server_id provided and service needs OAuth
            if server_id and service_name != "general":
                enhanced_params = self._inject_oauth_credentials(fixed_params, server_id, service_name)
            else:
                enhanced_params = fixed_params
            
            # Call the tool - handle both MCPClient and direct stdio
            if isinstance(client, dict):
                # Direct stdio command (npx, node, etc.) - use stdio bridge
                from .mcp_stdio_bridge import call_mcp_tool_via_stdio
                result = await call_mcp_tool_via_stdio(client, tool_name, enhanced_params)
            else:
                # MCPClient (Docker-based)
                result = await client.call_tool(tool_name, enhanced_params)
            
            # Check for token expiry and retry if needed
            if "error" in result and server_id:
                token_refreshed = await self._handle_token_expiry_error(result, server_id, service_name)
                
                if token_refreshed:
                    logger.info(f"Retrying {tool_name} with refreshed token")
                    # Retry with refreshed credentials
                    enhanced_params = self._inject_oauth_credentials(fixed_params, server_id, service_name)
                    
                    # Retry the call with the same logic
                    if isinstance(client, dict):
                        result = await call_mcp_tool_via_stdio(client, tool_name, enhanced_params)
                    else:
                        result = await client.call_tool(tool_name, enhanced_params)
            
            logger.info(f"[STDIO] Tool {tool_name} completed")
            return result
            
        except Exception as e:
            logger.error(f"[STDIO] Failed to call {tool_name}: {e}")
            return {"error": str(e)}
    
    async def call_http_tool(self, endpoint: str, tool_name: str, parameters: Dict[str, Any],
                           method: str = "POST", headers: Dict[str, str] = None,
                           server_id: int = None, service_name: str = "gmail") -> Dict[str, Any]:
        """
        Call a tool on an HTTP MCP server with OAuth handling
        
        Args:
            endpoint: HTTP endpoint URL
            tool_name: Name of the tool to call
            parameters: Tool parameters
            method: HTTP method (GET, POST)
            headers: Additional headers
            server_id: Server ID for OAuth credential lookup
            service_name: Service name for OAuth
            
        Returns:
            Tool result or error
        """
        try:
            logger.info(f"[HTTP] Calling {tool_name} at {endpoint}")
            
            # Fix parameter format
            fixed_params = self._fix_parameter_format(parameters, tool_name)
            
            # Inject OAuth credentials if server_id provided and service needs OAuth
            if server_id and service_name != "general":
                enhanced_params = self._inject_oauth_credentials(fixed_params, server_id, service_name)
            else:
                enhanced_params = fixed_params
            
            # Prepare request
            request_headers = {"Content-Type": "application/json"}
            if headers:
                request_headers.update(headers)
            
            # Determine payload format based on endpoint
            if "/invoke" in endpoint:
                # Standard MCP format
                payload = {
                    "name": tool_name,
                    "arguments": enhanced_params
                }
            else:
                # Direct parameters
                payload = enhanced_params
            
            # Execute HTTP request
            session = await self._get_http_session()
            
            if method.upper() == "GET":
                async with session.get(endpoint, params=payload, headers=request_headers) as response:
                    result = await self._process_http_response(response, tool_name)
            else:
                async with session.post(endpoint, json=payload, headers=request_headers) as response:
                    result = await self._process_http_response(response, tool_name)
            
            # Check for token expiry and retry if needed
            if "error" in result and server_id:
                token_refreshed = await self._handle_token_expiry_error(result, server_id, service_name)
                
                if token_refreshed:
                    logger.info(f"Retrying {tool_name} with refreshed token")
                    # Retry with refreshed credentials
                    enhanced_params = self._inject_oauth_credentials(fixed_params, server_id, service_name)
                    
                    if "/invoke" in endpoint:
                        payload = {"name": tool_name, "arguments": enhanced_params}
                    else:
                        payload = enhanced_params
                    
                    if method.upper() == "GET":
                        async with session.get(endpoint, params=payload, headers=request_headers) as response:
                            result = await self._process_http_response(response, tool_name)
                    else:
                        async with session.post(endpoint, json=payload, headers=request_headers) as response:
                            result = await self._process_http_response(response, tool_name)
            
            logger.info(f"[HTTP] Tool {tool_name} completed")
            return result
            
        except Exception as e:
            logger.error(f"[HTTP] Failed to call {tool_name}: {e}")
            return {"error": str(e)}
    
    async def _process_http_response(self, response, tool_name: str) -> Dict[str, Any]:
        """Process HTTP response from MCP server"""
        try:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                logger.error(f"HTTP {response.status} for {tool_name}: {error_text}")
                return {"error": f"HTTP {response.status}: {error_text}"}
                
        except Exception as e:
            logger.error(f"Failed to process response for {tool_name}: {e}")
            return {"error": f"Response processing error: {e}"}
    
    async def call_remote_tool(self, server_config: Dict[str, Any], tool_name: str, 
                             parameters: Dict[str, Any], server_id: int = None, 
                             service_name: str = "general") -> Dict[str, Any]:
        """
        Call a tool on a remote MCP server using MCP protocol over HTTP/SSE
        
        Args:
            server_config: Remote server configuration
            tool_name: Name of the tool to call
            parameters: Tool parameters
            server_id: Server ID for OAuth credential lookup
            service_name: Service name for OAuth
            
        Returns:
            Tool result or error
        """
        try:
            logger.info(f"[REMOTE] Calling {tool_name} on remote server {server_config.get('name')}")
            
            # Fix parameter format
            fixed_params = self._fix_parameter_format(parameters, tool_name)
            
            # Inject OAuth credentials if server_id provided and service needs OAuth
            if server_id and service_name != "general":
                enhanced_params = self._inject_oauth_credentials(fixed_params, server_id, service_name)
            else:
                enhanced_params = fixed_params
            
            # Use remote MCP client to execute tool
            from .remote_mcp_client import remote_mcp_manager
            
            result = await remote_mcp_manager.call_tool(server_config, tool_name, enhanced_params)
            
            # Handle token expiry errors for OAuth-enabled tools
            if server_id and service_name != "general" and "error" in result:
                token_refreshed = await self._handle_token_expiry_error(result, server_id, service_name)
                if token_refreshed:
                    # Retry with fresh credentials
                    logger.info(f"[REMOTE] Retrying {tool_name} with refreshed token")
                    fresh_params = self._inject_oauth_credentials(fixed_params, server_id, service_name)
                    result = await remote_mcp_manager.call_tool(server_config, tool_name, fresh_params)
            
            logger.info(f"[REMOTE] Successfully called {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"[REMOTE] Failed to call {tool_name}: {e}")
            return {"error": str(e)}

# Global instance
unified_mcp_service = UnifiedMCPService()

async def call_mcp_tool_unified(tool_info: Dict[str, Any], tool_name: str, 
                              parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified entry point for calling MCP tools (both HTTP and stdio)
    
    Args:
        tool_info: Tool information from cache (includes endpoint, server_id, etc.)
        tool_name: Name of the tool to call
        parameters: Tool parameters
        
    Returns:
        Tool result or error
    """
    try:
        endpoint = tool_info.get("endpoint", "")
        server_id = tool_info.get("server_id")
        
        # Determine service type based on tool name or server info
        service_name = "general"  # Default for non-OAuth tools
        if any(gmail_term in tool_name.lower() for gmail_term in ["gmail", "email", "mail"]):
            service_name = "gmail"
        elif "outlook" in tool_name.lower() or "outlook" in endpoint.lower():
            service_name = "outlook"
        elif "jira" in tool_name.lower() or "jira" in endpoint.lower():
            service_name = "jira"
        
        if endpoint.startswith("stdio://"):
            # Stdio MCP server
            from .db import SessionLocal, MCPServer
            
            if not server_id:
                return {"error": "No server_id for stdio tool"}
            
            db = SessionLocal()
            try:
                server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
                if not server or not server.command:
                    return {"error": f"Server {server_id} not found or has no command"}
                
                server_config = {
                    "command": server.command,
                    "args": server.args if server.args else [],
                    "env": server.env if server.env else {}
                }
                
                # Get unified service instance
                service = UnifiedMCPService()
                return await service.call_stdio_tool(
                    server_config, tool_name, parameters, server_id, service_name
                )
            finally:
                db.close()
        
        elif endpoint.startswith("http://") or endpoint.startswith("https://"):
            # HTTP MCP server
            method = tool_info.get("method", "POST")
            headers = tool_info.get("headers") or {}
            
            # Add API key if available
            api_key = tool_info.get("api_key")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            # Get unified service instance
            service = UnifiedMCPService()
            return await service.call_http_tool(
                endpoint, tool_name, parameters, method, headers, server_id, service_name
            )
        
        elif endpoint.startswith("remote://"):
            # Remote MCP server (HTTP/SSE MCP protocol)
            from .db import SessionLocal, MCPServer
            
            if not server_id:
                return {"error": "No server_id for remote tool"}
            
            db = SessionLocal()
            try:
                server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
                if not server or not server.remote_config:
                    return {"error": f"Remote server {server_id} not found or has no remote_config"}
                
                server_config = {
                    "id": server.id,
                    "name": server.name,
                    "remote_config": server.remote_config
                }
                
                # Get unified service instance
                service = UnifiedMCPService()
                return await service.call_remote_tool(
                    server_config, tool_name, parameters, server_id, service_name
                )
            finally:
                db.close()
        
        else:
            return {"error": f"Unsupported endpoint format: {endpoint}"}
            
    except Exception as e:
        import traceback
        logger.error(f"Unified MCP tool call failed for {tool_name}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}