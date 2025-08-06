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
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

from .oauth_token_manager import oauth_token_manager
from .mcp_client import MCPClient, MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPSubprocessInfo:
    """Information about an MCP subprocess"""
    process_id: str
    server_config: Dict[str, Any]
    client: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    is_active: bool = True


class MCPSubprocessPool:
    """Pool manager for MCP subprocesses to prevent resource exhaustion"""
    
    def __init__(self, max_processes: int = 50, max_idle_time: int = 60):
        self.max_processes = max_processes
        self.max_idle_time = max_idle_time  # 5 minutes
        self.processes: Dict[str, MCPSubprocessInfo] = {}
        self.process_usage: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of idle processes"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_processes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MCP subprocess pool cleanup error: {e}")
    
    def _generate_process_key(self, server_config: Dict[str, Any]) -> str:
        """Generate unique key for server configuration"""
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        # Add timestamp to prevent collisions in high-concurrency scenarios
        import time
        return f"{command}:{':'.join(args)}:{int(time.time() * 1000) % 10000}"
    
    async def _validate_subprocess_health(self, process_info: MCPSubprocessInfo) -> bool:
        """Validate that a subprocess is healthy and responsive"""
        try:
            client = process_info.client
            
            # For dict-type clients (direct stdio), check if process is still alive
            if isinstance(client, dict):
                # This is a direct stdio config, always create new
                return False
            
            # For MCPClient instances, check if they have a health check method
            if hasattr(client, 'is_healthy'):
                return await client.is_healthy()
            
            # For subprocess.Popen-like objects, check if process is alive
            if hasattr(client, 'poll'):
                return client.poll() is None
            
            # Default: assume healthy for unknown client types
            logger.warning(f"Unknown client type for health check: {type(client)}")
            return True
            
        except Exception as e:
            logger.warning(f"Subprocess health check failed: {e}")
            return False
    
    async def _check_pool_health(self):
        """Check and clean up unhealthy processes in the pool"""
        try:
            async with self._lock:
                unhealthy_keys = []
                
                for process_key, process_info in self.processes.items():
                    if process_info.is_active:
                        if not await self._validate_subprocess_health(process_info):
                            unhealthy_keys.append(process_key)
                
                # Remove unhealthy processes
                for key in unhealthy_keys:
                    logger.warning(f"Removing unhealthy process from pool: {key}")
                    await self._remove_process(key)
                
                # Check if pool is getting full
                active_count = len([p for p in self.processes.values() if p.is_active])
                if active_count > self.max_processes * 0.8:
                    logger.warning(f"MCP subprocess pool nearing capacity: {active_count}/{self.max_processes}")
                    
        except Exception as e:
            logger.error(f"Pool health check failed: {e}")
    
    async def get_or_create_process(self, server_config: Dict[str, Any]) -> Optional[Any]:
        """Get existing process or create new one from pool"""
        async with self._lock:
            process_key = self._generate_process_key(server_config)
            
            # EMERGENCY FIX: Force fresh subprocess creation to prevent corruption
            # This disables reuse but prevents 3rd agent failures
            logger.debug(f"MCP subprocess pool: EMERGENCY MODE - Creating fresh subprocess for {process_key}")
            
            # Remove any existing process with this key to force fresh creation
            if process_key in self.processes:
                logger.warning(f"MCP subprocess pool: Removing existing process {process_key} to force fresh creation")
                await self._remove_process(process_key)
            
            # Check if we're at capacity - be more aggressive about cleanup
            active_processes = [p for p in self.processes.values() if p.is_active]
            if len(active_processes) >= self.max_processes:
                logger.warning(f"MCP subprocess pool at capacity: {len(active_processes)}/{self.max_processes}")
                # Remove multiple old processes
                for _ in range(min(5, len(active_processes))):
                    await self._remove_lru_process()
            
            # Create new process
            try:
                if server_config.get("command") == "docker":
                    # Docker-based client
                    from .mcp_client import MCPClient
                    client = MCPClient(MCPServerConfig(**server_config))
                    await client.start()
                else:
                    # Store config for stdio bridge usage
                    client = server_config
                
                process_info = MCPSubprocessInfo(
                    process_id=process_key,
                    server_config=server_config,
                    client=client
                )
                
                self.processes[process_key] = process_info
                logger.debug(f"MCP subprocess pool: Created new process {process_key} ({len(active_processes) + 1}/{self.max_processes})")
                return client
                
            except Exception as e:
                logger.error(f"MCP subprocess pool: Failed to create process for {process_key}: {e}")
                return None
    
    async def _remove_lru_process(self):
        """Remove least recently used process"""
        if not self.processes:
            return
        
        # Find LRU process
        lru_key = min(
            self.processes.keys(),
            key=lambda k: self.processes[k].last_used
        )
        
        await self._remove_process(lru_key)
    
    async def _remove_process(self, process_key: str):
        """Remove specific process from pool"""
        if process_key not in self.processes:
            return
        
        process_info = self.processes[process_key]
        try:
            # Cleanup process
            if hasattr(process_info.client, 'stop'):
                await process_info.client.stop()
            elif hasattr(process_info.client, 'close'):
                await process_info.client.close()
        except Exception as e:
            logger.warning(f"MCP subprocess pool: Error stopping process {process_key}: {e}")
        
        # Remove from pool
        del self.processes[process_key]
        logger.debug(f"MCP subprocess pool: Removed process {process_key}")
    
    async def _cleanup_idle_processes(self):
        """Clean up processes that have been idle too long"""
        async with self._lock:
            current_time = datetime.now()
            to_remove = []
            
            for process_key, process_info in self.processes.items():
                idle_time = (current_time - process_info.last_used).total_seconds()
                if idle_time > self.max_idle_time:
                    to_remove.append(process_key)
            
            for process_key in to_remove:
                await self._remove_process(process_key)
                logger.debug(f"MCP subprocess pool: Cleaned up idle process {process_key}")
    
    async def cleanup_all(self):
        """Clean up all processes in the pool"""
        async with self._lock:
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            process_keys = list(self.processes.keys())
            for process_key in process_keys:
                await self._remove_process(process_key)
            
            logger.debug("MCP subprocess pool: All processes cleaned up")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics for monitoring"""
        active_processes = [p for p in self.processes.values() if p.is_active]
        return {
            "total_processes": len(self.processes),
            "active_processes": len(active_processes),
            "max_processes": self.max_processes,
            "process_usage": dict(self.process_usage),
            "pool_utilization": len(active_processes) / self.max_processes if self.max_processes > 0 else 0
        }


# Global subprocess pool instance
mcp_subprocess_pool = MCPSubprocessPool()


class CircuitBreaker:
    """Simple circuit breaker for MCP servers to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: Dict[str, int] = {}
        self.last_failure_time: Dict[str, datetime] = {}
        self.circuit_open: Dict[str, bool] = {}
    
    def _get_server_key(self, server_config: Dict[str, Any], tool_name: str) -> str:
        """Generate a unique key for the server/tool combination"""
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        return f"{command}:{':'.join(args)}:{tool_name}"
    
    def is_circuit_open(self, server_config: Dict[str, Any], tool_name: str) -> bool:
        """Check if circuit is open (server temporarily disabled)"""
        key = self._get_server_key(server_config, tool_name)
        
        if key not in self.circuit_open:
            return False
        
        if not self.circuit_open[key]:
            return False
        
        # Check if recovery timeout has passed
        if key in self.last_failure_time:
            time_since_failure = datetime.now() - self.last_failure_time[key]
            if time_since_failure.total_seconds() > self.recovery_timeout:
                # Reset circuit
                self.circuit_open[key] = False
                self.failures[key] = 0
                logger.info(f"Circuit breaker reset for {key}")
                return False
        
        return True
    
    def record_failure(self, server_config: Dict[str, Any], tool_name: str):
        """Record a failure and potentially open the circuit"""
        key = self._get_server_key(server_config, tool_name)
        
        self.failures[key] = self.failures.get(key, 0) + 1
        self.last_failure_time[key] = datetime.now()
        
        if self.failures[key] >= self.failure_threshold:
            self.circuit_open[key] = True
            logger.warning(f"Circuit breaker opened for {key} after {self.failures[key]} failures")
    
    def record_success(self, server_config: Dict[str, Any], tool_name: str):
        """Record a success and reset failure count"""
        key = self._get_server_key(server_config, tool_name)
        
        if key in self.failures:
            self.failures[key] = 0
        if key in self.circuit_open:
            self.circuit_open[key] = False


# Global circuit breaker instance
mcp_circuit_breaker = CircuitBreaker()

class UnifiedMCPService:
    """
    Unified service for handling both HTTP and stdio MCP servers
    with automatic OAuth token refresh and error handling.
    """
    
    def __init__(self):
        self.http_session = None
        self.stdio_clients: Dict[str, MCPClient] = {}
    
    async def _direct_google_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        EMERGENCY BYPASS: Direct Google Search API call to avoid MCP subprocess issues
        """
        try:
            import os
            import aiohttp
            
            # Get Google Search API credentials from environment (with fallbacks)
            api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "AIzaSyA2U7MBpH7cNDykiZ_OlGsdJJlXumsMps4")
            search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "d77ac8c3d3e124c3c")
            
            logger.debug(f"Using Google API key: {api_key[:10]}... and search engine: {search_engine_id}")
            
            if not api_key or not search_engine_id:
                logger.error("Google Search API credentials not found in environment")
                return {
                    "error": "Google Search API credentials not configured"
                }
            
            query = parameters.get("query", "")
            num_results = parameters.get("num_results", 5)
            
            if not query:
                return {"error": "No search query provided"}
            
            logger.debug(f"Searching for: {query}")
            
            # Direct Google Custom Search API call
            search_url = "https://www.googleapis.com/customsearch/v1"
            search_params = {
                "key": api_key,
                "cx": search_engine_id,
                "q": query,
                "num": min(num_results, 10)  # Google API max is 10
            }
            
            session = await self._get_http_session()
            async with session.get(search_url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Format results in expected MCP format
                    results = []
                    for item in data.get("items", []):
                        results.append({
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "snippet": item.get("snippet", "")
                        })
                    
                    logger.debug(f"Found {len(results)} results")
                    
                    return {
                        "content": [{
                            "type": "text", 
                            "text": f"Found {len(results)} search results for '{query}':\n\n" + 
                                   "\n\n".join([f"**{r['title']}**\n{r['snippet']}\n{r['link']}" for r in results])
                        }]
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Google Search API error {response.status}: {error_text}")
                    return {"error": f"Google Search API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Direct Google Search failed: {e}")
            return {"error": f"Direct Google Search failed: {str(e)}"}
        
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
    
    async def _get_stdio_client(self, server_config: Dict[str, Any]) -> Any:
        """Get or create stdio MCP client for a server using subprocess pool"""
        command = server_config.get("command")
        args = server_config.get("args", [])
        
        if command == "docker" and args:
            # Handle Docker-based stdio servers
            if len(args) >= 4 and args[0] == "exec":
                return await mcp_subprocess_pool.get_or_create_process(server_config)
        elif command in ['npx', 'node', 'python', 'python3']:
            # Handle direct stdio commands (npx, node, etc.)
            return await mcp_subprocess_pool.get_or_create_process(server_config)
        
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
            # CRITICAL: Check pool health before attempting tool call
            await mcp_subprocess_pool._check_pool_health()
            # Check circuit breaker first
            if mcp_circuit_breaker.is_circuit_open(server_config, tool_name):
                error_msg = f"MCP server circuit breaker is open for {tool_name} - server temporarily disabled"
                logger.warning(error_msg)
                return {"error": error_msg}
            
            logger.debug(f"Calling {tool_name} on stdio server")
            
            # Get stdio client from pool
            client = await self._get_stdio_client(server_config)
            
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
                    logger.debug(f"Retrying {tool_name} with refreshed token")
                    # Retry with refreshed credentials
                    enhanced_params = self._inject_oauth_credentials(fixed_params, server_id, service_name)
                    
                    # Retry the call with the same logic
                    if isinstance(client, dict):
                        result = await call_mcp_tool_via_stdio(client, tool_name, enhanced_params)
                    else:
                        result = await client.call_tool(tool_name, enhanced_params)
            
            logger.debug(f"Tool {tool_name} completed")
            
            # Record success if no error in result
            if "error" not in result:
                mcp_circuit_breaker.record_success(server_config, tool_name)
            else:
                mcp_circuit_breaker.record_failure(server_config, tool_name)
            
            # EMERGENCY FIX: Immediate cleanup after tool execution
            try:
                process_key = mcp_subprocess_pool._generate_process_key(server_config)
                logger.info(f"[STDIO] Immediately cleaning up subprocess {process_key} after tool completion")
                await mcp_subprocess_pool._remove_process(process_key)
            except Exception as cleanup_error:
                logger.warning(f"[STDIO] Failed to cleanup subprocess after tool: {cleanup_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"[STDIO] Failed to call {tool_name}: {e}")
            # Record failure in circuit breaker
            mcp_circuit_breaker.record_failure(server_config, tool_name)
            
            # Special fallback for get_datetime tool
            if tool_name == "get_datetime":
                logger.info(f"[STDIO] Using datetime fallback for {tool_name}")
                try:
                    from .datetime_fallback import get_current_datetime
                    fallback_result = get_current_datetime()
                    logger.info(f"[STDIO] Datetime fallback successful")
                    return {"content": [{"type": "text", "text": str(fallback_result)}]}
                except Exception as fallback_error:
                    logger.error(f"[STDIO] Datetime fallback also failed: {fallback_error}")
                    return {"error": f"MCP server failed and fallback failed: {str(e)}"}
            
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
                    logger.debug(f"Retrying {tool_name} with refreshed token")
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

async def cleanup_mcp_subprocesses():
    """Clean up MCP subprocesses using the subprocess pool and unified service"""
    try:
        logger.info("Starting enhanced MCP subprocess cleanup")
        
        # 1. Clean up subprocess pool (primary cleanup mechanism)
        await mcp_subprocess_pool.cleanup_all()
        logger.info("Subprocess pool cleanup completed")
        
        # 2. Close HTTP session if it exists
        if unified_mcp_service.http_session and not unified_mcp_service.http_session.closed:
            await unified_mcp_service.http_session.close()
            logger.info("Closed HTTP session")
        
        # 3. Clean up any remaining stdio clients (legacy cleanup)
        if unified_mcp_service.stdio_clients:
            logger.info(f"Cleaning up {len(unified_mcp_service.stdio_clients)} legacy stdio clients")
            clients_to_remove = list(unified_mcp_service.stdio_clients.keys())
            
            for client_key in clients_to_remove:
                try:
                    client = unified_mcp_service.stdio_clients[client_key]
                    # If client has a stop method, call it
                    if hasattr(client, 'stop'):
                        await client.stop()
                        logger.info(f"Stopped legacy stdio client: {client_key}")
                    elif hasattr(client, 'close'):
                        await client.close()
                        logger.info(f"Closed legacy stdio client: {client_key}")
                    
                    # Remove from cache
                    del unified_mcp_service.stdio_clients[client_key]
                    
                except Exception as e:
                    logger.warning(f"Failed to cleanup legacy stdio client {client_key}: {e}")
                    # Still remove from cache even if cleanup failed
                    try:
                        del unified_mcp_service.stdio_clients[client_key]
                    except:
                        pass
            
            logger.info("Completed legacy stdio client cleanup")
        
        # 4. Force garbage collection to help clean up closed resources
        import gc
        gc.collect()
        
        logger.info("Enhanced MCP subprocess cleanup completed")
        
    except Exception as e:
        logger.error(f"Enhanced MCP subprocess cleanup failed: {e}")


def get_mcp_pool_stats() -> Dict[str, Any]:
    """Get MCP subprocess pool statistics for monitoring"""
    try:
        pool_stats = mcp_subprocess_pool.get_pool_stats()
        
        # Add unified service stats
        unified_stats = {
            "legacy_stdio_clients": len(unified_mcp_service.stdio_clients),
            "http_session_active": unified_mcp_service.http_session is not None and not unified_mcp_service.http_session.closed
        }
        
        return {
            "subprocess_pool": pool_stats,
            "unified_service": unified_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get MCP pool stats: {e}")
        return {"error": str(e)}

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
        logger.debug(f"Starting unified tool call for {tool_name}")
        endpoint = tool_info.get("endpoint", "")
        server_id = tool_info.get("server_id")
        logger.debug(f"Tool {tool_name} - endpoint: {endpoint}, server_id: {server_id}")
        
        # CRITICAL FIX: Direct routing bypass for google_search to avoid subprocess issues
        if tool_name == "google_search":
            logger.info(f"[BYPASS] Routing google_search directly to avoid subprocess issues")
            service = UnifiedMCPService()
            result = await service._direct_google_search(parameters)
            logger.info(f"[BYPASS] Direct google_search completed successfully")
            return result
        
        # CRITICAL FIX: Direct routing bypass for get_datetime to avoid MCP server issues
        if tool_name == "get_datetime":
            logger.info(f"[BYPASS] Routing get_datetime directly to fallback to avoid server issues")
            try:
                from .datetime_fallback import get_current_datetime
                fallback_result = get_current_datetime()
                logger.info(f"[BYPASS] get_datetime fallback successful")
                return {"content": [{"type": "text", "text": str(fallback_result)}]}
            except Exception as fallback_error:
                logger.error(f"[BYPASS] get_datetime fallback failed: {fallback_error}")
                return {"error": f"Datetime fallback failed: {str(fallback_error)}"}
        
        # CRITICAL FIX: Handle internal:// RAG service calls
        if endpoint.startswith("internal://"):
            logger.info(f"[INTERNAL] Routing {tool_name} to internal service: {endpoint}")
            
            if endpoint == "internal://rag_mcp_service" and tool_name == "rag_knowledge_search":
                logger.info(f"[INTERNAL RAG] Processing RAG knowledge search request")
                try:
                    from ..mcp_services.rag_mcp_service import execute_rag_search_sync
                    
                    # Extract parameters
                    query = parameters.get('query', '')
                    collections = parameters.get('collections')
                    
                    # Use same max_documents as standard chat for consistency
                    from ..core.rag_settings_cache import get_document_retrieval_settings
                    doc_settings = get_document_retrieval_settings()
                    max_documents = parameters.get('max_documents') or doc_settings.get('max_documents_mcp', 8)
                    include_content = parameters.get('include_content', True)
                    
                    logger.info(f"[INTERNAL RAG] Calling execute_rag_search_sync with query: '{query[:100]}...'")
                    result = execute_rag_search_sync(
                        query=query,
                        collections=collections,
                        max_documents=max_documents,
                        include_content=include_content
                    )
                    logger.info(f"[INTERNAL RAG] RAG search completed successfully")
                    return result
                except Exception as rag_error:
                    logger.error(f"[INTERNAL RAG] RAG search failed: {rag_error}")
                    return {"error": f"RAG search failed: {str(rag_error)}"}
            
            return {"error": f"Unknown internal service: {endpoint}"}
        
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
            logger.debug(f"Using stdio protocol for {tool_name}")
            from .db import SessionLocal, MCPServer
            
            if not server_id:
                logger.error(f"No server_id for stdio tool {tool_name}")
                return {"error": "No server_id for stdio tool"}
            
            logger.debug(f"Looking up server {server_id} in database")
            db = SessionLocal()
            try:
                server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
                if not server or not server.command:
                    logger.error(f"Server {server_id} not found or has no command")
                    return {"error": f"Server {server_id} not found or has no command"}
                
                server_config = {
                    "command": server.command,
                    "args": server.args if server.args else [],
                    "env": server.env if server.env else {}
                }
                logger.debug(f"Server config for {tool_name}: {server_config}")
                
                # Get unified service instance
                service = UnifiedMCPService()
                logger.debug(f"Calling stdio tool {tool_name}")
                result = await service.call_stdio_tool(
                    server_config, tool_name, parameters, server_id, service_name
                )
                logger.info(f"[UNIFIED MCP] Stdio tool {tool_name} completed successfully")
                logger.debug(f"[UNIFIED MCP] Result preview: {str(result)[:300]}..." if result and len(str(result)) > 300 else f"[UNIFIED MCP] Result: {result}")
                logger.info(f"[UNIFIED MCP] ========== Unified MCP Tool Call Complete ==========")
                return result
            except Exception as e:
                logger.error(f"[UNIFIED MCP] Failed to call stdio tool {tool_name}: {e}")
                logger.info(f"[UNIFIED MCP] ========== Unified MCP Tool Call Failed ==========")
                return {"error": str(e)}
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