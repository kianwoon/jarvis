"""
Enhanced Tool Error Handler

Provides comprehensive error handling, token renewal, and retry logic for tool execution.
Supports various authentication mechanisms and error recovery strategies.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Classification of tool execution errors"""
    AUTHENTICATION = "authentication"
    TOKEN_EXPIRED = "token_expired"
    RATE_LIMITED = "rate_limited" 
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    INVALID_PARAMS = "invalid_params"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_backoff: bool = True
    jitter: bool = True
    
    # Error-specific retry counts
    auth_retries: int = 2
    rate_limit_retries: int = 5
    connection_retries: int = 3

@dataclass
class ErrorInfo:
    """Information about a tool execution error"""
    error_type: ErrorType
    message: str
    status_code: Optional[int] = None
    retry_after: Optional[int] = None  # For rate limiting
    server_id: Optional[str] = None
    tool_name: Optional[str] = None
    original_error: Optional[Exception] = None

class ToolErrorHandler:
    """Enhanced error handler with authentication renewal and retry logic"""
    
    def __init__(self, retry_config: RetryConfig = None):
        self.retry_config = retry_config or RetryConfig()
        self.token_refresh_cache = {}  # Cache recent refresh attempts
        
    def classify_error(self, error: Exception, response_text: str = "", status_code: int = None, tool_name: str = "") -> ErrorInfo:
        """Classify an error to determine appropriate recovery strategy"""
        
        error_msg = str(error).lower()
        response_lower = response_text.lower()
        
        # Authentication and token errors
        if (status_code == 401 or 
            any(phrase in error_msg for phrase in ["unauthorized", "authentication", "invalid token", "expired token"]) or
            any(phrase in response_lower for phrase in ["credentials do not contain", "token expired", "invalid_grant"])):
            
            if any(phrase in response_lower for phrase in ["token expired", "expired token", "refresh"]):
                return ErrorInfo(ErrorType.TOKEN_EXPIRED, error_msg, status_code, tool_name=tool_name)
            else:
                return ErrorInfo(ErrorType.AUTHENTICATION, error_msg, status_code, tool_name=tool_name)
        
        # Rate limiting
        if (status_code == 429 or 
            any(phrase in error_msg for phrase in ["rate limit", "too many requests", "quota exceeded"])):
            
            # Try to extract retry-after header value
            retry_after = None
            if "retry-after" in response_lower:
                import re
                match = re.search(r'retry-after[:\s]+(\d+)', response_lower)
                if match:
                    retry_after = int(match.group(1))
            
            return ErrorInfo(ErrorType.RATE_LIMITED, error_msg, status_code, retry_after=retry_after, tool_name=tool_name)
        
        # Connection errors
        if (any(phrase in error_msg for phrase in ["connection", "network", "dns", "host"]) or
            status_code in [502, 503, 504]):
            return ErrorInfo(ErrorType.CONNECTION, error_msg, status_code, tool_name=tool_name)
        
        # Timeout errors
        if any(phrase in error_msg for phrase in ["timeout", "timed out"]):
            return ErrorInfo(ErrorType.TIMEOUT, error_msg, status_code, tool_name=tool_name)
        
        # Permission errors
        if (status_code == 403 or 
            any(phrase in error_msg for phrase in ["forbidden", "permission", "access denied"])):
            return ErrorInfo(ErrorType.PERMISSION, error_msg, status_code, tool_name=tool_name)
        
        # Not found errors
        if status_code == 404:
            return ErrorInfo(ErrorType.NOT_FOUND, error_msg, status_code, tool_name=tool_name)
        
        # Invalid parameters
        if (status_code == 400 or 
            any(phrase in error_msg for phrase in ["bad request", "invalid", "missing parameter"])):
            return ErrorInfo(ErrorType.INVALID_PARAMS, error_msg, status_code, tool_name=tool_name)
        
        # Server errors
        if status_code and 500 <= status_code < 600:
            return ErrorInfo(ErrorType.SERVER_ERROR, error_msg, status_code, tool_name=tool_name)
        
        # Unknown error
        return ErrorInfo(ErrorType.UNKNOWN, error_msg, status_code, tool_name=tool_name, original_error=error)
    
    def should_retry(self, error_info: ErrorInfo, attempt: int) -> bool:
        """Determine if an error should trigger a retry"""
        
        max_retries = self.retry_config.max_retries
        
        # Error-specific retry limits
        if error_info.error_type == ErrorType.AUTHENTICATION:
            max_retries = self.retry_config.auth_retries
        elif error_info.error_type == ErrorType.RATE_LIMITED:
            max_retries = self.retry_config.rate_limit_retries
        elif error_info.error_type == ErrorType.CONNECTION:
            max_retries = self.retry_config.connection_retries
        elif error_info.error_type in [ErrorType.PERMISSION, ErrorType.NOT_FOUND, ErrorType.INVALID_PARAMS]:
            # Don't retry these errors - they won't improve with retries
            return False
        
        return attempt < max_retries
    
    def calculate_delay(self, attempt: int, error_info: ErrorInfo) -> float:
        """Calculate delay before retry"""
        
        # Use retry-after header for rate limiting
        if error_info.error_type == ErrorType.RATE_LIMITED and error_info.retry_after:
            return min(error_info.retry_after, self.retry_config.max_delay)
        
        # Calculate exponential backoff
        if self.retry_config.exponential_backoff:
            delay = self.retry_config.base_delay * (2 ** attempt)
        else:
            delay = self.retry_config.base_delay
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return min(delay, self.retry_config.max_delay)
    
    async def refresh_authentication(self, tool_name: str, tool_info: Dict[str, Any], error_info: ErrorInfo) -> Dict[str, Any]:
        """Attempt to refresh authentication tokens using server configuration"""
        
        server_id = tool_info.get("server_id")
        if not server_id:
            return {"error": "No server_id available for token refresh"}
        
        # Check if we recently tried to refresh this token
        cache_key = f"{server_id}_{tool_name}"
        now = time.time()
        
        if cache_key in self.token_refresh_cache:
            last_refresh_time, last_result = self.token_refresh_cache[cache_key]
            # Don't retry refresh within 30 seconds
            if now - last_refresh_time < 30:
                logger.info(f"[TOKEN REFRESH] Skipping refresh for {tool_name} - recently attempted")
                return last_result
        
        logger.info(f"[TOKEN REFRESH] Attempting to refresh token for {tool_name} (server_id: {server_id})")
        
        try:
            # Get server configuration from database to determine refresh strategy
            server_config = await self._get_server_config(server_id)
            if not server_config:
                return {"error": f"Server configuration not found for server_id: {server_id}"}
            
            # Determine refresh strategy based on server configuration
            result = await self._refresh_token_by_server_type(server_config, tool_info)
            
            # Cache the result
            self.token_refresh_cache[cache_key] = (now, result)
            
            if result and not result.get("error"):
                logger.info(f"[TOKEN REFRESH] Successfully refreshed token for {tool_name}")
                # Invalidate tool cache to reload with fresh credentials
                from app.core.mcp_tools_cache import reload_enabled_mcp_tools
                reload_enabled_mcp_tools()
            else:
                logger.error(f"[TOKEN REFRESH] Failed to refresh token for {tool_name}: {result}")
            
            return result
            
        except Exception as e:
            error_result = {"error": f"Token refresh failed: {str(e)}"}
            self.token_refresh_cache[cache_key] = (now, error_result)
            logger.error(f"[TOKEN REFRESH] Exception during token refresh for {tool_name}: {e}")
            return error_result
    
    async def _get_server_config(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get server configuration from database"""
        try:
            from app.core.db import SessionLocal, MCPServer
            db = SessionLocal()
            try:
                server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
                if server:
                    # Extract server type and auth type from enhanced configurations
                    auth_refresh_config = server.auth_refresh_config or {}
                    server_type = auth_refresh_config.get('server_type', 'unknown')
                    auth_type = auth_refresh_config.get('auth_type', 'unknown')
                    
                    return {
                        "id": server.id,
                        "name": server.name,
                        "server_type": server_type,
                        "auth_type": auth_type,
                        "config": {
                            "refresh_endpoint": auth_refresh_config.get('refresh_endpoint', ''),
                            "refresh_method": auth_refresh_config.get('refresh_method', 'POST'),
                            "refresh_headers": auth_refresh_config.get('refresh_headers', {}),
                            "refresh_data_template": auth_refresh_config.get('refresh_data_template', {}),
                            "token_expiry_buffer_minutes": auth_refresh_config.get('token_expiry_buffer_minutes', 5),
                        },
                        "credentials": server.oauth_credentials or {},
                        "command": server.command,
                        "args": server.args,
                        "enhanced_error_handling_config": server.enhanced_error_handling_config,
                        "auth_refresh_config": server.auth_refresh_config
                    }
                return None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to get server config for {server_id}: {e}")
            return None
    
    async def _refresh_token_by_server_type(self, server_config: Dict[str, Any], tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh token based on server type and authentication method"""
        
        server_id = server_config.get("id")
        server_type = server_config.get("server_type", "").lower()
        auth_type = server_config.get("auth_type", "").lower()
        
        logger.info(f"[TOKEN REFRESH] Server {server_id} type: {server_type}, auth: {auth_type}")
        
        # Check if server has refresh capabilities
        refresh_endpoint = server_config.get("config", {}).get("refresh_endpoint")
        refresh_token = server_config.get("credentials", {}).get("refresh_token")
        
        # OAuth-based refresh (covers Gmail, Jira, Confluence, etc.)
        if auth_type in ["oauth", "oauth2"] or refresh_token:
            return await self._refresh_oauth_token(server_config)
        
        # API key rotation (if supported by the service)
        elif auth_type in ["api_key", "token"] and refresh_endpoint:
            return await self._refresh_api_key(server_config)
        
        # Service-specific implementations
        elif "gmail" in server_type or "google" in server_type:
            return await self._refresh_gmail_token(server_id)
        
        # Generic refresh attempt
        elif refresh_endpoint:
            return await self._generic_token_refresh(server_config)
        
        else:
            logger.warning(f"[TOKEN REFRESH] No refresh strategy available for server {server_id} (type: {server_type}, auth: {auth_type})")
            return {"error": f"Token refresh not supported for server type: {server_type}"}
    
    async def _refresh_oauth_token(self, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic OAuth token refresh"""
        server_id = server_config.get("id")
        credentials = server_config.get("credentials", {})
        config = server_config.get("config", {})
        
        refresh_token = credentials.get("refresh_token")
        client_id = credentials.get("client_id")
        client_secret = credentials.get("client_secret")
        token_endpoint = config.get("token_endpoint")
        
        if not all([refresh_token, client_id, client_secret, token_endpoint]):
            missing = [k for k, v in {
                "refresh_token": refresh_token,
                "client_id": client_id, 
                "client_secret": client_secret,
                "token_endpoint": token_endpoint
            }.items() if not v]
            return {"error": f"Missing OAuth credentials: {missing}"}
        
        try:
            import requests
            
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret
            }
            
            response = requests.post(token_endpoint, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            
            # Update credentials in database
            await self._update_server_credentials(server_id, token_data)
            
            return {"success": True, "token_data": token_data}
            
        except Exception as e:
            logger.error(f"OAuth token refresh failed for server {server_id}: {e}")
            return {"error": f"OAuth refresh failed: {str(e)}"}
    
    async def _refresh_api_key(self, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Refresh API key using service-specific endpoint"""
        server_id = server_config.get("id")
        config = server_config.get("config", {})
        credentials = server_config.get("credentials", {})
        
        refresh_endpoint = config.get("refresh_endpoint")
        api_key = credentials.get("api_key")
        
        if not refresh_endpoint:
            return {"error": "No refresh endpoint configured"}
        
        try:
            import requests
            
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            response = requests.post(refresh_endpoint, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # Update credentials if new key provided
            if "api_key" in result or "access_token" in result:
                await self._update_server_credentials(server_id, result)
                return {"success": True, "new_credentials": result}
            else:
                return {"success": True, "message": "API key refreshed"}
                
        except Exception as e:
            logger.error(f"API key refresh failed for server {server_id}: {e}")
            return {"error": f"API key refresh failed: {str(e)}"}
    
    async def _refresh_gmail_token(self, server_id: str) -> Dict[str, Any]:
        """Refresh Gmail OAuth token using existing service function"""
        try:
            from app.langchain.service import refresh_gmail_token
            return refresh_gmail_token(server_id)
        except Exception as e:
            return {"error": f"Gmail token refresh failed: {str(e)}"}
    
    async def _generic_token_refresh(self, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic token refresh for custom implementations"""
        server_id = server_config.get("id")
        config = server_config.get("config", {})
        
        refresh_endpoint = config.get("refresh_endpoint")
        refresh_method = config.get("refresh_method", "POST").upper()
        refresh_headers = config.get("refresh_headers", {})
        refresh_data = config.get("refresh_data", {})
        
        if not refresh_endpoint:
            return {"error": "No refresh endpoint configured"}
        
        try:
            import requests
            
            if refresh_method == "GET":
                response = requests.get(refresh_endpoint, headers=refresh_headers, timeout=30)
            else:
                response = requests.request(
                    refresh_method, 
                    refresh_endpoint, 
                    headers=refresh_headers,
                    json=refresh_data,
                    timeout=30
                )
            
            response.raise_for_status()
            result = response.json()
            
            # Update credentials if provided
            if any(key in result for key in ["access_token", "api_key", "token"]):
                await self._update_server_credentials(server_id, result)
            
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Generic token refresh failed for server {server_id}: {e}")
            return {"error": f"Generic refresh failed: {str(e)}"}
    
    async def _update_server_credentials(self, server_id: str, new_credentials: Dict[str, Any]) -> bool:
        """Update server credentials in database"""
        try:
            from app.core.db import SessionLocal, MCPServer
            db = SessionLocal()
            try:
                server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
                if server:
                    # Update credentials
                    current_credentials = getattr(server, 'credentials', {}) or {}
                    
                    # Map common token field names
                    token_mapping = {
                        "access_token": "access_token",
                        "api_key": "api_key", 
                        "token": "access_token",
                        "refresh_token": "refresh_token"
                    }
                    
                    for old_key, new_key in token_mapping.items():
                        if old_key in new_credentials:
                            current_credentials[new_key] = new_credentials[old_key]
                    
                    # Update other credential fields
                    for key, value in new_credentials.items():
                        if key not in token_mapping:
                            current_credentials[key] = value
                    
                    server.credentials = current_credentials
                    db.commit()
                    
                    logger.info(f"[TOKEN REFRESH] Updated credentials for server {server_id}")
                    return True
                else:
                    logger.error(f"[TOKEN REFRESH] Server {server_id} not found for credential update")
                    return False
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to update credentials for server {server_id}: {e}")
            return False
    
    async def execute_with_retry(
        self, 
        tool_call_func: Callable, 
        tool_name: str, 
        parameters: Dict[str, Any], 
        tool_info: Dict[str, Any] = None,
        trace=None
    ) -> Dict[str, Any]:
        """
        Execute a tool call with comprehensive error handling and retry logic
        
        Args:
            tool_call_func: The function to call the tool
            tool_name: Name of the tool being called
            parameters: Parameters for the tool
            tool_info: Information about the tool from cache
            trace: Langfuse trace for logging
            
        Returns:
            Tool execution result or error information
        """
        
        last_error_info = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.info(f"[TOOL RETRY] Attempt {attempt + 1}/{self.retry_config.max_retries + 1} for {tool_name}")
                
                # Call the tool
                result = tool_call_func(tool_name, parameters, trace=trace, _skip_span_creation=True)
                
                # Check if result indicates an error
                if isinstance(result, dict) and "error" in result:
                    # Create error info from result
                    error_info = self.classify_error(
                        Exception(result["error"]), 
                        result.get("error", ""), 
                        tool_name=tool_name
                    )
                    last_error_info = error_info
                    
                    # Handle authentication errors
                    if error_info.error_type in [ErrorType.AUTHENTICATION, ErrorType.TOKEN_EXPIRED]:
                        if tool_info:
                            logger.info(f"[TOOL RETRY] Authentication error for {tool_name}, attempting token refresh")
                            refresh_result = await self.refresh_authentication(tool_name, tool_info, error_info)
                            
                            if refresh_result and not refresh_result.get("error"):
                                logger.info(f"[TOOL RETRY] Token refreshed for {tool_name}, retrying immediately")
                                continue  # Retry immediately after successful token refresh
                    
                    # Check if we should retry this error
                    if not self.should_retry(error_info, attempt):
                        logger.info(f"[TOOL RETRY] Not retrying {tool_name} - error type {error_info.error_type} or max attempts reached")
                        return result
                    
                    # Calculate delay before retry
                    if attempt < self.retry_config.max_retries:
                        delay = self.calculate_delay(attempt, error_info)
                        logger.info(f"[TOOL RETRY] Retrying {tool_name} in {delay:.2f} seconds (error: {error_info.error_type})")
                        await asyncio.sleep(delay)
                        continue
                    
                    return result
                else:
                    # Success
                    if attempt > 0:
                        logger.info(f"[TOOL RETRY] {tool_name} succeeded on attempt {attempt + 1}")
                    return result
                    
            except Exception as e:
                # Classify the exception
                error_info = self.classify_error(e, tool_name=tool_name)
                last_error_info = error_info
                
                logger.error(f"[TOOL RETRY] Exception on attempt {attempt + 1} for {tool_name}: {e}")
                
                # Handle authentication errors
                if error_info.error_type in [ErrorType.AUTHENTICATION, ErrorType.TOKEN_EXPIRED] and tool_info:
                    logger.info(f"[TOOL RETRY] Authentication exception for {tool_name}, attempting token refresh")
                    refresh_result = await self.refresh_authentication(tool_name, tool_info, error_info)
                    
                    if refresh_result and not refresh_result.get("error"):
                        logger.info(f"[TOOL RETRY] Token refreshed for {tool_name}, retrying immediately")
                        continue  # Retry immediately after successful token refresh
                
                # Check if we should retry this error
                if not self.should_retry(error_info, attempt):
                    logger.info(f"[TOOL RETRY] Not retrying {tool_name} - error type {error_info.error_type} or max attempts reached")
                    return {"error": f"Tool execution failed after {attempt + 1} attempts: {str(e)}", "error_type": error_info.error_type.value}
                
                # Calculate delay before retry
                if attempt < self.retry_config.max_retries:
                    delay = self.calculate_delay(attempt, error_info)
                    logger.info(f"[TOOL RETRY] Retrying {tool_name} in {delay:.2f} seconds (error: {error_info.error_type})")
                    await asyncio.sleep(delay)
                    continue
        
        # All retries exhausted
        error_msg = f"Tool {tool_name} failed after {self.retry_config.max_retries + 1} attempts"
        if last_error_info:
            error_msg += f" (last error: {last_error_info.error_type.value})"
        
        return {
            "error": error_msg,
            "error_type": last_error_info.error_type.value if last_error_info else "unknown",
            "attempts": self.retry_config.max_retries + 1
        }

# Global error handler instance
_global_error_handler = None

def get_tool_error_handler(retry_config: RetryConfig = None) -> ToolErrorHandler:
    """Get the global tool error handler instance"""
    global _global_error_handler
    if _global_error_handler is None or retry_config is not None:
        _global_error_handler = ToolErrorHandler(retry_config)
    return _global_error_handler

async def get_server_retry_config(tool_name: str) -> RetryConfig:
    """Get retry configuration from MCP server settings"""
    try:
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        from app.core.db import SessionLocal, MCPServer
        
        # Get tool info
        tools = get_enabled_mcp_tools()
        tool_info = tools.get(tool_name, {})
        server_id = tool_info.get("server_id")
        
        if not server_id:
            logger.debug(f"No server_id found for tool {tool_name}, using default config")
            return RetryConfig()
        
        # Get server configuration
        db = SessionLocal()
        try:
            server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
            if server and server.enhanced_error_handling_config:
                config = server.enhanced_error_handling_config
                
                if not config.get("enabled", True):
                    # Return minimal retry config if disabled
                    return RetryConfig(max_retries=0)
                
                logger.info(f"[SERVER CONFIG] Using enhanced error handling for {tool_name} from server {server_id}")
                return RetryConfig(
                    max_retries=config.get("max_tool_retries", 3),
                    base_delay=config.get("retry_base_delay", 1.0),
                    max_delay=config.get("retry_max_delay", 60.0),
                    exponential_backoff=True,
                    jitter=True
                )
        except Exception as e:
            logger.error(f"Error reading server config for {tool_name}: {e}")
        finally:
            db.close()
        
        # Return default config if anything fails
        return RetryConfig()
        
    except Exception as e:
        logger.error(f"Error getting server retry config for {tool_name}: {e}")
        return RetryConfig()

# Convenience function for enhanced tool calling
async def call_mcp_tool_with_retry(
    tool_name: str, 
    parameters: Dict[str, Any], 
    trace=None,
    retry_config: RetryConfig = None
) -> Dict[str, Any]:
    """
    Call an MCP tool with enhanced error handling and retry logic
    
    Args:
        tool_name: Name of the tool to call
        parameters: Parameters for the tool
        trace: Langfuse trace for logging
        retry_config: Optional retry configuration (uses server config if None)
        
    Returns:
        Tool execution result
    """
    
    # Use provided config or get from server settings
    if retry_config is None:
        retry_config = await get_server_retry_config(tool_name)
    
    # Get tool info from cache
    try:
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        enabled_tools = get_enabled_mcp_tools()
        tool_info = enabled_tools.get(tool_name)
    except:
        tool_info = None
    
    # Get error handler
    error_handler = get_tool_error_handler(retry_config)
    
    # Import the original tool call function
    from app.langchain.service import call_mcp_tool
    
    # Execute with retry logic
    return await error_handler.execute_with_retry(
        call_mcp_tool, 
        tool_name, 
        parameters, 
        tool_info, 
        trace
    )