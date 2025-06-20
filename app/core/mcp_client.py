#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client Implementation

This module provides a proper MCP-compliant client for communicating with
MCP servers running in Docker containers using stdio transport.
"""

import json
import logging
import asyncio
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    container_name: str
    command: List[str]
    environment: Dict[str, str] = None
    working_directory: str = None
    
    def __post_init__(self):
        if self.environment is None:
            self.environment = {}

class MCPClient:
    """
    MCP-compliant client for communicating with stdio-based MCP servers
    running in Docker containers.
    """
    
    def __init__(self, server_config: MCPServerConfig):
        self.config = server_config
        self.request_id = 0
        
    def _next_request_id(self) -> int:
        """Generate next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def _execute_docker_command(self, json_request: str) -> Tuple[str, str, int]:
        """
        Execute MCP command in Docker container using Docker SDK
        
        Args:
            json_request: JSON-RPC request string
            
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        logger.debug(f"[MCP-CLIENT] Sending to container: {self.config.container_name}")
        logger.debug(f"[MCP-CLIENT] Command: {' '.join(self.config.command)}")
        logger.debug(f"[MCP-CLIENT] JSON request: {json_request}")
        
        try:
            import docker
            client = docker.from_env()
            
            # Get the container
            try:
                container = client.containers.get(self.config.container_name)
            except docker.errors.NotFound:
                raise Exception(f"Container {self.config.container_name} not found")
            
            # Send JSON directly to the existing MCP server process (PID 1)
            # This uses the same approach as our fixed stdio handler
            escaped_json = json_request.replace("'", "'\"'\"'")
            
            # Execute command to send JSON to the running MCP process
            exec_result = container.exec_run(
                cmd=["sh", "-c", f"echo '{escaped_json}' > /proc/1/fd/0"],
                stdout=True,
                stderr=True,
                demux=True,
                environment=self.config.environment
            )
            
            # Wait for MCP server to process the request
            import time
            time.sleep(1.0)  # Increased wait time for reliable response
            
            # Get fresh output from container logs after our request
            # Use larger tail without since parameter to ensure we catch the response
            stdout = container.logs(stdout=True, stderr=False, tail=100).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True, tail=100).decode('utf-8')
            
            # Extract JSON response from the logs
            # Look for the most recent JSON-RPC response that matches our request ID
            request_id = json.loads(json_request).get("id")
            json_response = ""
            
            # Debug: Log the stdout content for troubleshooting
            logger.debug(f"[MCP-CLIENT] Looking for request ID {request_id} in stdout:")
            logger.debug(f"[MCP-CLIENT] Stdout content: {stdout[:500]}...")
            
            # Search for JSON-RPC response with matching ID
            for line in reversed(stdout.split('\n')):
                line = line.strip()
                if line.startswith('{"') and '"jsonrpc"' in line:
                    try:
                        parsed = json.loads(line)
                        if parsed.get("id") == request_id:
                            json_response = line
                            logger.debug(f"[MCP-CLIENT] Found matching response for ID {request_id}")
                            break
                    except Exception as e:
                        logger.debug(f"[MCP-CLIENT] Failed to parse JSON line: {e}")
                        continue
            
            # If no ID match, use the most recent JSON response
            if not json_response:
                logger.debug(f"[MCP-CLIENT] No matching ID found, looking for any JSON response")
                for line in reversed(stdout.split('\n')):
                    line = line.strip()
                    if line.startswith('{"') and '"jsonrpc"' in line:
                        json_response = line
                        logger.debug(f"[MCP-CLIENT] Using most recent JSON response")
                        break
            
            if json_response:
                stdout = json_response
            else:
                # No JSON found, check if there's any useful output
                stdout = stdout.strip()
            
            return (stdout, stderr, 0)
            
        except Exception as e:
            logger.error(f"[MCP-CLIENT] Docker SDK execution failed: {e}")
            raise
    
    async def list_tools(self) -> Dict[str, Any]:
        """
        List available tools from the MCP server
        
        Returns:
            Dictionary containing tools list or error
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_request_id()
        }
        
        try:
            stdout, stderr, exit_code = await self._execute_docker_command(
                json.dumps(request)
            )
            
            if exit_code != 0:
                return {"error": f"Docker command failed with exit code {exit_code}: {stderr}"}
            
            if not stdout.strip():
                return {"error": f"No response from MCP server. Stderr: {stderr}"}
            
            # Parse JSON-RPC response
            response = json.loads(stdout.strip())
            
            if "error" in response:
                return {"error": response["error"]}
            
            return response.get("result", {})
            
        except json.JSONDecodeError as e:
            logger.error(f"[MCP-CLIENT] Invalid JSON response: {stdout}")
            return {"error": f"Invalid JSON response from MCP server: {e}"}
        except Exception as e:
            logger.error(f"[MCP-CLIENT] list_tools failed: {e}")
            return {"error": str(e)}
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a specific tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Dictionary containing tool result or error
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._next_request_id()
        }
        
        try:
            stdout, stderr, exit_code = await self._execute_docker_command(
                json.dumps(request)
            )
            
            if exit_code != 0:
                return {"error": f"Docker command failed with exit code {exit_code}: {stderr}"}
            
            if not stdout.strip():
                logger.warning(f"[MCP-CLIENT] No stdout from {tool_name}. Stderr: {stderr}")
                return {"error": f"No response from MCP server for {tool_name}"}
            
            # Parse JSON-RPC response
            response = json.loads(stdout.strip())
            
            if "error" in response:
                return {"error": response["error"]}
            
            # Return the result content
            result = response.get("result", {})
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"[MCP-CLIENT] Invalid JSON response for {tool_name}: {stdout}")
            return {"error": f"Invalid JSON response from MCP server: {e}"}
        except Exception as e:
            logger.error(f"[MCP-CLIENT] call_tool({tool_name}) failed: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """
        Check if the MCP server is healthy
        
        Returns:
            True if server is responding, False otherwise
        """
        try:
            result = await self.list_tools()
            return "error" not in result
        except Exception:
            return False


class MCPServerManager:
    """
    Manager for multiple MCP servers
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPClient] = {}
    
    def register_server(self, server_config: MCPServerConfig) -> MCPClient:
        """
        Register a new MCP server
        
        Args:
            server_config: Server configuration
            
        Returns:
            MCPClient instance
        """
        client = MCPClient(server_config)
        self.servers[server_config.name] = client
        logger.info(f"[MCP-MANAGER] Registered server: {server_config.name}")
        return client
    
    def get_server(self, name: str) -> Optional[MCPClient]:
        """Get MCP server by name"""
        return self.servers.get(name)
    
    async def list_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        List tools from all registered servers
        
        Returns:
            Dictionary mapping server names to their tools
        """
        all_tools = {}
        
        for server_name, client in self.servers.items():
            try:
                tools = await client.list_tools()
                all_tools[server_name] = tools
            except Exception as e:
                logger.error(f"[MCP-MANAGER] Failed to list tools for {server_name}: {e}")
                all_tools[server_name] = {"error": str(e)}
        
        return all_tools
    
    async def call_tool_on_server(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on a specific server
        
        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Tool result or error
        """
        client = self.get_server(server_name)
        if not client:
            return {"error": f"Server '{server_name}' not found"}
        
        return await client.call_tool(tool_name, arguments)