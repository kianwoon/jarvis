"""
HTTP-based Tool Executor - Reliable alternative to stdio Docker execution
"""
import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class HTTPToolExecutor:
    """Execute tools via HTTP instead of unreliable Docker stdio"""
    
    def __init__(self):
        self.session = None
        self.tool_servers = {
            "gmail_send": {"url": "http://localhost:8001/tools/gmail_send", "type": "gmail"},
            "find_email": {"url": "http://localhost:8001/tools/find_email", "type": "gmail"},
            "search_emails": {"url": "http://localhost:8001/tools/search_emails", "type": "gmail"},
            "read_email": {"url": "http://localhost:8001/tools/read_email", "type": "gmail"},
            "draft_email": {"url": "http://localhost:8001/tools/draft_email", "type": "gmail"},
            "gmail_send_draft": {"url": "http://localhost:8001/tools/gmail_send_draft", "type": "gmail"},
            "delete_email": {"url": "http://localhost:8001/tools/delete_email", "type": "gmail"},
            "gmail_trash_message": {"url": "http://localhost:8001/tools/gmail_trash_message", "type": "gmail"},
            "gmail_get_thread": {"url": "http://localhost:8001/tools/gmail_get_thread", "type": "gmail"},
            "gmail_update_draft": {"url": "http://localhost:8001/tools/gmail_update_draft", "type": "gmail"},
            "list_email_labels": {"url": "http://localhost:8001/tools/list_email_labels", "type": "gmail"},
            "modify_email": {"url": "http://localhost:8001/tools/modify_email", "type": "gmail"}
        }
    
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def _inject_oauth_credentials(self, parameters: Dict[str, Any], tool_type: str) -> Dict[str, Any]:
        """Inject OAuth credentials for Gmail tools"""
        if tool_type != "gmail":
            return parameters
            
        try:
            from app.core.oauth_token_manager import oauth_token_manager
            
            # Get valid OAuth token for Gmail (server_id 3)
            oauth_creds = oauth_token_manager.get_valid_token(3, "gmail")
            
            if oauth_creds:
                enhanced_params = parameters.copy()
                enhanced_params.update({
                    "google_client_id": oauth_creds.get("client_id"),
                    "google_client_secret": oauth_creds.get("client_secret"), 
                    "google_access_token": oauth_creds.get("access_token"),
                    "google_refresh_token": oauth_creds.get("refresh_token")
                })
                logger.debug(f"Injected OAuth credentials for {tool_type} tool")
                return enhanced_params
            else:
                logger.warning(f"No OAuth credentials found for {tool_type}")
                return parameters
                
        except Exception as e:
            logger.error(f"Failed to inject OAuth credentials: {e}")
            return parameters
    
    def _fix_gmail_parameters(self, parameters: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Fix Gmail parameter format issues"""
        if tool_name not in ["gmail_send", "draft_email", "gmail_update_draft"]:
            return parameters
            
        fixed_params = parameters.copy()
        
        # Fix 'to' parameter - ensure it's an array
        if "to" in fixed_params:
            if isinstance(fixed_params["to"], str):
                logger.info(f"[HTTP-EXECUTOR] Converting 'to' from string to array for {tool_name}")
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
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via HTTP"""
        start_time = datetime.now()
        
        try:
            # Check if tool is supported
            if tool_name not in self.tool_servers:
                return {"error": f"Tool {tool_name} not supported by HTTP executor"}
            
            server_config = self.tool_servers[tool_name]
            tool_type = server_config["type"]
            url = server_config["url"]
            
            logger.info(f"[HTTP-EXECUTOR] Executing {tool_name} via HTTP: {url}")
            
            # Fix parameters and inject credentials
            fixed_params = self._fix_gmail_parameters(parameters, tool_name)
            enhanced_params = self._inject_oauth_credentials(fixed_params, tool_type)
            
            # Prepare request payload
            payload = {
                "tool": tool_name,
                "parameters": enhanced_params,
                "timestamp": start_time.isoformat()
            }
            
            # Execute HTTP request
            session = await self._get_session()
            
            logger.debug(f"[HTTP-EXECUTOR] Sending request to {url}")
            logger.debug(f"[HTTP-EXECUTOR] Payload size: {len(json.dumps(payload))} chars")
            
            async with session.post(url, json=payload) as response:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"[HTTP-EXECUTOR] {tool_name} completed successfully in {execution_time:.2f}s")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"[HTTP-EXECUTOR] {tool_name} failed with status {response.status}: {error_text}")
                    return {"error": f"HTTP {response.status}: {error_text}"}
                    
        except aiohttp.ClientConnectorError as e:
            logger.error(f"[HTTP-EXECUTOR] Connection failed for {tool_name}: {e}")
            return {"error": f"Connection failed: {e}"}
        except asyncio.TimeoutError:
            logger.error(f"[HTTP-EXECUTOR] Timeout for {tool_name}")
            return {"error": "Request timeout"}
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"[HTTP-EXECUTOR] {tool_name} failed after {execution_time:.2f}s: {e}")
            return {"error": str(e)}

# Global instance
http_tool_executor = HTTPToolExecutor()

async def execute_tool_via_http(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute tool via HTTP - main entry point"""
    return await http_tool_executor.execute_tool(tool_name, parameters)