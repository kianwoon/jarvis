"""
Remote MCP Client for HTTP/SSE transport
Implements MCP protocol over HTTP and Server-Sent Events
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MCPMessage:
    """MCP protocol message structure"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

class RemoteMCPClient:
    """
    MCP client for remote HTTP/SSE servers
    Implements the Model Context Protocol over HTTP transport
    """
    
    def __init__(self, server_url: str, transport_type: str = "http", 
                 auth_headers: Optional[Dict[str, str]] = None,
                 client_info: Optional[Dict[str, str]] = None,
                 connection_timeout: int = 30):
        self.server_url = server_url.rstrip('/')
        self.transport_type = transport_type
        self.auth_headers = auth_headers or {}
        self.client_info = client_info or {"name": "jarvis-mcp-client", "version": "1.0.0"}
        self.connection_timeout = connection_timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.initialized = False
        self.server_capabilities = {}
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
        
    async def connect(self):
        """Establish connection to remote MCP server"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            
            # Prepare headers for MCP over HTTP/SSE
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": f"{self.client_info.get('name', 'mcp-client')}/{self.client_info.get('version', '1.0.0')}"
            }
            headers.update(self.auth_headers)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
            logger.debug(f"Created new HTTP session for remote MCP server: {self.server_url}")
        
        try:
            # Initialize MCP connection
            await self._initialize()
        except Exception as e:
            # CRITICAL FIX: Clean up session if initialization fails to prevent resource leaks
            logger.error(f"MCP initialization failed, cleaning up session: {e}")
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
            raise
        
    async def disconnect(self):
        """Close connection to remote MCP server"""
        if self.session and not self.session.closed:
            logger.debug(f"Closing HTTP session for remote MCP server: {self.server_url}")
            await self.session.close()
            # CRITICAL FIX: Ensure session is set to None to prevent reuse of closed session
            self.session = None
        self.initialized = False
        
    async def _initialize(self):
        """Initialize MCP protocol connection"""
        try:
            init_message = MCPMessage(
                id=str(uuid.uuid4()),
                method="initialize",
                params={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": self.client_info
                }
            )
            
            logger.info(f"Initializing MCP connection to {self.server_url}")
            response = await self._send_message(init_message)
            
            if response.error:
                raise Exception(f"MCP initialization failed: {response.error}")
                
            # Store server capabilities
            if response.result and "capabilities" in response.result:
                self.server_capabilities = response.result["capabilities"]
                
            self.initialized = True
            logger.info(f"MCP connection initialized successfully: {response.result}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP connection: {str(e)}")
            raise
            
    async def _send_message(self, message: MCPMessage) -> MCPMessage:
        """Send MCP message and receive response"""
        if not self.session:
            raise Exception("MCP client not connected")
            
        # Convert message to JSON
        message_data = {
            "jsonrpc": message.jsonrpc,
            "id": message.id,
            "method": message.method,
            "params": message.params
        }
        
        # Remove None values
        message_data = {k: v for k, v in message_data.items() if v is not None}
        
        try:
            # Determine the correct endpoint based on transport type and URL
            endpoint = self._get_mcp_endpoint()
                
            logger.debug(f"Sending MCP message to {endpoint}: {message_data}")
            
            if self.transport_type == "sse":
                # For SSE transport, we need to handle streaming connection
                return await self._send_message_sse(endpoint, message_data, message.id)
            else:
                # For HTTP transport, use regular POST
                return await self._send_message_http(endpoint, message_data, message.id)
                
        except Exception as e:
            logger.error(f"Failed to send MCP message: {str(e)}")
            return MCPMessage(
                jsonrpc="2.0",
                id=message.id,
                error={"code": -1, "message": str(e)}
            )
    
    def _get_mcp_endpoint(self) -> str:
        """Get the correct MCP endpoint based on URL and transport type"""
        url = self.server_url.rstrip('/')
        
        # Smart endpoint detection for Zapier and other MCP providers
        if 'mcp.zapier.com' in url:
            # For Zapier, always use /mcp endpoint regardless of transport type
            # Based on official Zapier example using StreamableHTTPClientTransport
            if url.endswith('/sse'):
                # Convert /sse to /mcp for Zapier compatibility
                url = url[:-4] + '/mcp'
                logger.info(f"Auto-corrected Zapier endpoint from /sse to /mcp: {url}")
            elif not url.endswith('/mcp'):
                url = f"{url}/mcp"
            return url
        
        # For other providers, respect the configured transport type
        if url.endswith(('/mcp', '/sse')):
            return url
            
        # Otherwise, append the appropriate endpoint based on transport type
        if self.transport_type == "sse":
            return f"{url}/sse"
        else:
            return f"{url}/mcp"
    
    async def _send_message_http(self, endpoint: str, message_data: dict, message_id: str) -> MCPMessage:
        """Send message via HTTP transport (like Zapier's StreamableHTTPClientTransport)"""
        try:
            async with self.session.post(endpoint, json=message_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                # Handle JSON response (standard for HTTP MCP)
                response_data = await response.json()
                logger.debug(f"Received HTTP MCP response: {response_data}")
                
                return MCPMessage(
                    jsonrpc=response_data.get("jsonrpc", "2.0"),
                    id=response_data.get("id"),
                    result=response_data.get("result"),
                    error=response_data.get("error")
                )
        except Exception as e:
            # CRITICAL FIX: Log HTTP errors with context for debugging
            logger.error(f"HTTP MCP request failed for endpoint {endpoint}: {e}")
            raise
    
    async def _send_message_sse(self, endpoint: str, message_data: dict, message_id: str) -> MCPMessage:
        """Send message via SSE transport"""
        # For SSE, we typically need to establish a persistent connection
        # and send messages via POST while receiving responses via SSE stream
        
        try:
            # First, try to send the message via POST
            async with self.session.post(endpoint, json=message_data) as response:
                if response.status == 405:  # Method Not Allowed
                    # Some SSE endpoints might not support POST directly
                    # Try to establish SSE connection instead
                    return await self._handle_sse_connection(endpoint, message_data, message_id)
                elif response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                
                content_type = response.headers.get('content-type', '').lower()
                
                if 'text/event-stream' in content_type:
                    # Handle Server-Sent Events response
                    response_text = await response.text()
                    logger.debug(f"Received SSE response: {response_text}")
                    
                    # Parse SSE format: data: {json}\n\n
                    response_data = None
                    for line in response_text.split('\n'):
                        if line.startswith('data: '):
                            json_str = line[6:]  # Remove 'data: ' prefix
                            try:
                                response_data = json.loads(json_str)
                                break
                            except json.JSONDecodeError:
                                continue
                    
                    if not response_data:
                        raise Exception(f"Failed to parse SSE response: {response_text}")
                        
                else:
                    # Fallback to JSON
                    response_data = await response.json()
                
                logger.debug(f"Received SSE MCP response: {response_data}")
                
                return MCPMessage(
                    jsonrpc=response_data.get("jsonrpc", "2.0"),
                    id=response_data.get("id"),
                    result=response_data.get("result"),
                    error=response_data.get("error")
                )
        except Exception as e:
            # CRITICAL FIX: Log SSE errors with context for debugging
            logger.error(f"SSE MCP request failed for endpoint {endpoint}: {e}")
            raise
    
    async def _handle_sse_connection(self, endpoint: str, message_data: dict, message_id: str) -> MCPMessage:
        """Handle SSE connection for endpoints that don't support POST"""
        # This is a fallback for SSE endpoints that require different handling
        
        if 'mcp.zapier.com' in endpoint:
            # Zapier-specific guidance
            raise Exception(
                "Zapier MCP servers use HTTP transport with /mcp endpoint. "
                "The SSE endpoint is not supported for MCP protocol messages. "
                "Please use HTTP transport type instead of SSE."
            )
        else:
            # General guidance for other providers
            raise Exception(
                "SSE endpoint does not support POST requests. "
                "This MCP server may require HTTP transport instead of SSE. "
                "Try changing the transport type to 'http' and ensure the URL ends with '/mcp'"
            )
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from remote MCP server"""
        if not self.initialized:
            raise Exception("MCP client not initialized")
            
        message = MCPMessage(
            id=str(uuid.uuid4()),
            method="tools/list",
            params={}
        )
        
        response = await self._send_message(message)
        
        if response.error:
            raise Exception(f"Failed to list tools: {response.error}")
            
        tools = response.result.get("tools", []) if response.result else []
        logger.info(f"Discovered {len(tools)} tools from remote MCP server")
        
        return tools
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the remote MCP server"""
        if not self.initialized:
            raise Exception("MCP client not initialized")
            
        message = MCPMessage(
            id=str(uuid.uuid4()),
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        response = await self._send_message(message)
        
        if response.error:
            raise Exception(f"Tool execution failed: {response.error}")
            
        return response.result or {}
        
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and capabilities"""
        return {
            "server_url": self.server_url,
            "transport_type": self.transport_type,
            "initialized": self.initialized,
            "capabilities": self.server_capabilities,
            "client_info": self.client_info
        }

class RemoteMCPManager:
    """
    Manager for remote MCP server connections
    Handles connection pooling and tool discovery
    """
    
    def __init__(self):
        self.clients: Dict[int, RemoteMCPClient] = {}
        
    async def get_client(self, server_config: Dict[str, Any]) -> RemoteMCPClient:
        """Get or create MCP client for server"""
        server_id = server_config.get("id")
        
        if server_id in self.clients:
            return self.clients[server_id]
            
        # Create new client
        remote_config = server_config.get("remote_config", {})
        
        client = RemoteMCPClient(
            server_url=remote_config.get("server_url"),
            transport_type=remote_config.get("transport_type", "http"),
            auth_headers=remote_config.get("auth_headers", {}),
            client_info=remote_config.get("client_info", {}),
            connection_timeout=remote_config.get("connection_timeout", 30)
        )
        
        # Store client
        self.clients[server_id] = client
        
        return client
        
    async def discover_tools(self, server_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover tools from remote MCP server"""
        try:
            client = await self.get_client(server_config)
            
            async with client:
                tools = await client.list_tools()
                return tools
                
        except Exception as e:
            logger.error(f"Failed to discover tools from remote server {server_config.get('id')}: {str(e)}")
            raise
            
    async def call_tool(self, server_config: Dict[str, Any], 
                       tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool on remote MCP server"""
        try:
            client = await self.get_client(server_config)
            
            async with client:
                result = await client.call_tool(tool_name, arguments)
                return result
                
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on remote server {server_config.get('id')}: {str(e)}")
            raise
            
    async def close_client(self, server_id: int):
        """Close and remove client for server"""
        if server_id in self.clients:
            client = self.clients[server_id]
            await client.disconnect()
            del self.clients[server_id]
            
    async def close_all_clients(self):
        """Close all client connections with proper error handling"""
        # CRITICAL FIX: Handle errors during cleanup to prevent resource leaks
        cleanup_errors = []
        for server_id, client in list(self.clients.items()):
            try:
                await client.disconnect()
                logger.debug(f"Successfully closed remote MCP client for server {server_id}")
            except Exception as e:
                logger.error(f"Error closing remote MCP client for server {server_id}: {e}")
                cleanup_errors.append((server_id, e))
            
        self.clients.clear()
        
        # Log summary of cleanup results
        if cleanup_errors:
            logger.warning(f"Remote MCP client cleanup completed with {len(cleanup_errors)} errors")
        else:
            logger.debug("All remote MCP clients closed successfully")

# Global instance
remote_mcp_manager = RemoteMCPManager()