"""
MCP stdio bridge for Docker-based MCP servers
Implements the Model Context Protocol over stdio transport
"""
import asyncio
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class MCPStdioBridge:
    """Bridge between HTTP requests and stdio-based MCP servers"""
    
    def __init__(self, command: str, args: List[str]):
        self.command = command
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        
    async def start(self):
        """Start the MCP server process"""
        try:
            full_command = [self.command] + self.args
            logger.info(f"Starting MCP server: {' '.join(full_command)}")
            
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Initialize the MCP connection
            await self._initialize_mcp()
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def _initialize_mcp(self):
        """Initialize MCP connection with the server"""
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "jarvis-mcp-bridge",
                    "version": "1.0.0"
                }
            },
            "id": self._next_id()
        }
        
        response = await self._send_request(init_request)
        logger.info(f"MCP server initialized: {response}")
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self._send_notification(initialized_notification)
        
    def _next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and wait for response"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server process not running")
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()
        
        logger.debug(f"Sent MCP request: {request}")
        
        # Read response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())
        
        logger.debug(f"Received MCP response: {response}")
        
        if "error" in response:
            raise Exception(f"MCP error: {response['error']}")
        
        return response
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send a notification (no response expected)"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server process not running")
        
        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        await self.process.stdin.drain()
        
        logger.debug(f"Sent MCP notification: {notification}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_id()
        }
        
        response = await self._send_request(request)
        return response.get("result", {}).get("tools", [])
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        # Debug: Log OAuth credentials being passed
        if "gmail" in tool_name.lower():
            logger.info(f"[DEBUG] Gmail tool call arguments:")
            for key in ["google_access_token", "google_refresh_token", "google_client_id", "google_client_secret", 
                       "access_token", "refresh_token", "client_id", "client_secret", "token_uri"]:
                if key in arguments:
                    val = arguments[key]
                    if key in ["access_token", "google_access_token", "refresh_token", "google_refresh_token", 
                              "client_secret", "google_client_secret"]:
                        logger.info(f"  {key}: {val[:10]}... (length: {len(val)})")
                    else:
                        logger.info(f"  {key}: {val}")
                else:
                    logger.info(f"  {key}: NOT PROVIDED")
        
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._next_id()
        }
        
        response = await self._send_request(request)
        return response.get("result", {})
    
    async def stop(self):
        """Stop the MCP server process"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None


class MCPDockerBridge(MCPStdioBridge):
    """Specialized bridge for Docker-based MCP servers"""
    
    def __init__(self, container_name: str, command: List[str]):
        # For Docker exec, we use: docker exec -i <container> <command>
        super().__init__("docker", ["exec", "-i", container_name] + command)
        self.container_name = container_name
        
    async def is_container_running(self) -> bool:
        """Check if the Docker container is running"""
        try:
            result = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}", self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await result.communicate()
            return stdout.decode().strip() == "true"
        except Exception:
            return False


async def call_mcp_tool_via_stdio(
    server_config: Dict[str, Any],
    tool_name: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call an MCP tool via stdio transport
    
    This is a simplified version that creates a new connection for each call.
    In production, you'd want to maintain persistent connections.
    """
    try:
        import os
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        
        if command == "docker" and "exec" in args:
            # Extract container name
            container_idx = args.index("-i") + 1
            if container_idx < len(args):
                container_name = args[container_idx]
                # Get the actual command after container name
                mcp_command = args[container_idx + 1:]
                
                if in_docker:
                    # When running in Docker, use simple docker exec
                    logger.info(f"Using docker exec for {container_name}")
                    try:
                        from app.core.mcp_docker_exec import call_mcp_tool_in_docker
                        result = await call_mcp_tool_in_docker(
                            container_name,
                            mcp_command,
                            tool_name,
                            parameters
                        )
                        return result
                    except Exception as e:
                        logger.error(f"Docker exec error: {e}")
                        return {"error": f"Docker exec error: {str(e)}"}
                
                bridge = MCPDockerBridge(container_name, mcp_command)
                
                # Check if container is running
                if not await bridge.is_container_running():
                    return {"error": f"Docker container {container_name} is not running"}
                
                # Start the bridge
                await bridge.start()
                
                try:
                    # Call the tool
                    result = await bridge.call_tool(tool_name, parameters)
                    return result
                finally:
                    # Clean up
                    await bridge.stop()
            else:
                return {"error": "Invalid Docker exec command format"}
        else:
            # Generic stdio bridge
            bridge = MCPStdioBridge(command, args)
            await bridge.start()
            
            try:
                result = await bridge.call_tool(tool_name, parameters)
                return result
            finally:
                await bridge.stop()
                
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name} via stdio: {e}")
        return {"error": str(e)}