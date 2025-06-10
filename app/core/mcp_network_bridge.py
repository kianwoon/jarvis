"""
Network-based MCP bridge for connecting to MCP servers in other containers
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
import aiohttp
import socket

logger = logging.getLogger(__name__)


class MCPNetworkBridge:
    """
    Bridge for connecting to MCP servers running in other Docker containers
    via TCP/HTTP when direct stdio is not available (e.g., when running in Docker)
    """
    
    def __init__(self, container_name: str, command: List[str]):
        self.container_name = container_name
        self.command = command
        # For Gmail MCP, we know it's accessible via the container name on the Docker network
        self.base_url = f"http://{container_name}:8080"  # Adjust port as needed
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def start(self):
        """Initialize the network connection"""
        self.session = aiohttp.ClientSession()
        
    async def stop(self):
        """Close the network connection"""
        if self.session:
            await self.session.close()
            
    async def is_reachable(self) -> bool:
        """Check if the MCP server is reachable via network"""
        if not self.session:
            return False
            
        try:
            # Try to connect to the container by name (Docker network resolution)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((self.container_name, 8080))
            sock.close()
            return result == 0
        except Exception as e:
            logger.error(f"Cannot reach {self.container_name}: {e}")
            return False
            
    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        Fallback implementation that returns known Gmail tools
        when we can't use stdio from within Docker
        """
        # Since we can't actually connect via stdio from within Docker,
        # return a predefined list of Gmail tools
        return [
            {
                "name": "gmail_send",
                "description": "Send an email",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email address"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body (plain text or HTML)"},
                        "cc": {"type": "string", "description": "CC recipients (optional)"},
                        "bcc": {"type": "string", "description": "BCC recipients (optional)"}
                    },
                    "required": ["to", "subject", "body"]
                }
            },
            {
                "name": "gmail_search",
                "description": "Search for emails",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Gmail search query"},
                        "max_results": {"type": "number", "description": "Maximum number of results (default: 10)"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "gmail_get_thread",
                "description": "Get all messages in a thread",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "thread_id": {"type": "string", "description": "The thread ID"}
                    },
                    "required": ["thread_id"]
                }
            },
            {
                "name": "gmail_get_message",
                "description": "Get a specific email message",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string", "description": "The message ID"}
                    },
                    "required": ["message_id"]
                }
            },
            {
                "name": "gmail_trash_message",
                "description": "Move a message to trash",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string", "description": "The message ID to trash"}
                    },
                    "required": ["message_id"]
                }
            },
            {
                "name": "gmail_create_draft",
                "description": "Create a new email draft",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email address"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"}
                    },
                    "required": ["to", "subject", "body"]
                }
            },
            {
                "name": "gmail_update_draft",
                "description": "Update an existing draft",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "draft_id": {"type": "string", "description": "The draft ID"},
                        "to": {"type": "string", "description": "Recipient email address"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"}
                    },
                    "required": ["draft_id"]
                }
            },
            {
                "name": "gmail_send_draft",
                "description": "Send an existing draft",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "draft_id": {"type": "string", "description": "The draft ID to send"}
                    },
                    "required": ["draft_id"]
                }
            },
            {
                "name": "get_latest_emails",
                "description": "Retrieves the latest emails from Gmail inbox",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "number", "description": "Number of latest emails to retrieve (default: 10)"},
                        "includeBody": {"type": "boolean", "description": "Whether to include email body content (default: false)"},
                        "labelIds": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by specific label IDs (e.g., ['INBOX'])"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "search_emails",
                "description": "Searches for emails using Gmail search syntax",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Gmail search query"},
                        "maxResults": {"type": "number", "description": "Max results to return"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool (not implemented for network bridge)"""
        raise NotImplementedError("Network bridge does not support direct tool calls yet")