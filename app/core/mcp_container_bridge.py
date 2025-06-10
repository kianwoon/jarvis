"""
Bridge for connecting to MCP servers running in Docker containers
using Docker SDK to establish stdio communication
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import docker
from docker.errors import NotFound, APIError

logger = logging.getLogger(__name__)


class MCPContainerBridge:
    """
    Bridge for MCP servers running in Docker containers.
    Uses Docker SDK to exec into containers and communicate via stdio.
    """
    
    def __init__(self, container_name: str, command: List[str]):
        self.container_name = container_name
        self.command = command
        self.client = docker.from_env()
        self.exec_instance = None
        self.socket = None
        self.reader = None
        self.writer = None
        self.request_id = 0
        
    async def start(self):
        """Start the MCP server connection via Docker exec"""
        try:
            # Get the container
            container = self.client.containers.get(self.container_name)
            if container.status != 'running':
                raise RuntimeError(f"Container {self.container_name} is not running")
            
            # Create exec instance with stdin and stdout
            exec_command = self.command
            self.exec_instance = container.exec_run(
                exec_command,
                stdin=True,
                stdout=True,
                stderr=True,
                tty=False,
                stream=True,
                socket=True,
                demux=True
            )
            
            # Get the socket for bidirectional communication
            self.socket = self.exec_instance.output
            
            # Initialize the MCP connection
            await self._initialize()
            
        except NotFound:
            raise RuntimeError(f"Container {self.container_name} not found")
        except Exception as e:
            logger.error(f"Failed to start MCP container bridge: {e}")
            raise
    
    async def _initialize(self):
        """Send MCP initialization message"""
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "jarvis-mcp-client",
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
            "method": "notifications/initialized",
            "params": {}
        }
        await self._send_notification(initialized_notification)
    
    def _next_id(self) -> int:
        """Generate next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and wait for response"""
        if not self.socket:
            raise RuntimeError("MCP container bridge not connected")
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.socket._sock.send(request_str.encode())
        
        # Read response
        response_data = b""
        while True:
            chunk = self.socket._sock.recv(4096)
            if not chunk:
                break
            response_data += chunk
            
            # Check if we have a complete JSON response
            try:
                lines = response_data.decode().strip().split('\n')
                for line in lines:
                    if line.strip():
                        response = json.loads(line)
                        if response.get("id") == request["id"]:
                            return response
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Continue reading
                continue
            
            # Check for newline to see if message is complete
            if b'\n' in response_data:
                break
        
        # Try to parse the response
        try:
            lines = response_data.decode().strip().split('\n')
            for line in lines:
                if line.strip():
                    response = json.loads(line)
                    if response.get("id") == request["id"]:
                        return response
        except Exception as e:
            logger.error(f"Failed to parse response: {e}, data: {response_data}")
            raise
        
        raise RuntimeError(f"No response received for request {request['id']}")
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send a notification (no response expected)"""
        if not self.socket:
            raise RuntimeError("MCP container bridge not connected")
        
        notification_str = json.dumps(notification) + "\n"
        self.socket._sock.send(notification_str.encode())
    
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
        """Stop the MCP server connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        if self.exec_instance:
            self.exec_instance = None


# Global connection pool for persistent connections
container_connections = {}


async def get_or_create_container_bridge(container_name: str, command: List[str]) -> MCPContainerBridge:
    """Get existing bridge or create new one"""
    global container_connections
    
    key = f"{container_name}:{' '.join(command)}"
    
    if key not in container_connections:
        bridge = MCPContainerBridge(container_name, command)
        await bridge.start()
        container_connections[key] = bridge
    
    return container_connections[key]


async def call_mcp_tool_via_container(
    container_name: str,
    command: List[str],
    tool_name: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call an MCP tool in a Docker container
    """
    try:
        # Get or create bridge
        bridge = await get_or_create_container_bridge(container_name, command)
        
        # Call the tool
        result = await bridge.call_tool(tool_name, parameters)
        return result
        
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name} in container {container_name}: {e}")
        return {"error": str(e)}