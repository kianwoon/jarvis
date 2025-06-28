"""
MCP stdio bridge for Docker-based MCP servers
Implements the Model Context Protocol over stdio transport
"""
import asyncio
import json
import logging
import subprocess
import threading
import queue
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class MCPStdioBridge:
    """Bridge between HTTP requests and stdio-based MCP servers"""
    
    def __init__(self, command: str, args: List[str], env_vars: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args
        self.env_vars = env_vars or {}
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.initialized = False
        self.server_capabilities = {}
        self.client_info = {
            "name": "jarvis-mcp-bridge",
            "version": "1.0.0"
        }
        self.response_queue = queue.Queue()
        self.response_thread = None
        
    async def start(self):
        """Start the MCP server process"""
        try:
            full_command = [self.command] + self.args
            logger.info(f"[STDIO DEBUG] Starting MCP server: {' '.join(full_command)}")
            logger.info(f"[STDIO DEBUG] Command: {self.command}, Args: {self.args}")
            
            # Set up environment with common Node.js paths and MCP server env vars
            import os
            env = os.environ.copy()
            
            # Add MCP server environment variables first
            env.update(self.env_vars)
            
            # Ensure minimal required environment variables for MCP servers
            required_defaults = {
                'JIRA_URL': 'https://dummy.atlassian.net',
                'JIRA_USER': 'dummy@example.com', 
                'JIRA_TOKEN': 'dummy-token',
                'MS_GRAPH_TOKEN': 'dummy-token'
            }
            
            for key, default_value in required_defaults.items():
                if key not in env:
                    env[key] = default_value
                    logger.debug(f"[STDIO DEBUG] Set default env var {key}")
            
            # Ensure PATH includes common Node.js installation paths
            current_path = env.get('PATH', '')
            node_paths = [
                '/opt/homebrew/bin',  # Homebrew on Apple Silicon
                '/usr/local/bin',     # Homebrew on Intel
                '/usr/bin',           # System Node.js
                '/bin'                # Basic system paths
            ]
            additional_paths = [path for path in node_paths if path not in current_path]
            if additional_paths:
                env['PATH'] = ':'.join(additional_paths) + ':' + current_path
            
            # For npx/npm commands, verify npx is accessible and log detailed info
            if self.command == 'npx':
                npx_found = False
                logger.info(f"Checking for npx in PATH: {env['PATH']}")
                
                for path in env['PATH'].split(':'):
                    if not path:  # Skip empty path components
                        continue
                    npx_path = os.path.join(path, 'npx')
                    logger.info(f"Checking {npx_path} - exists: {os.path.exists(npx_path)}")
                    if os.path.exists(npx_path):
                        logger.info(f"Found npx at {npx_path}")
                        npx_found = True
                        break
                
                if not npx_found:
                    logger.warning(f"npx not found in PATH: {env['PATH']}")
                    # Try to find npx in common locations
                    npx_locations = [
                        '/opt/homebrew/bin/npx',
                        '/usr/local/bin/npx', 
                        '/usr/bin/npx'
                    ]
                    for npx_path in npx_locations:
                        logger.info(f"Checking fallback location {npx_path} - exists: {os.path.exists(npx_path)}")
                        if os.path.exists(npx_path):
                            logger.info(f"Found npx at {npx_path}, adding to PATH")
                            env['PATH'] = os.path.dirname(npx_path) + ':' + env['PATH']
                            npx_found = True
                            break
                    
                    if not npx_found:
                        logger.error("npx not found anywhere, MCP server will likely fail to start")
                else:
                    logger.info("npx found and accessible")
                
                # If npx command and not found, try using full path
                if not npx_found and self.command == 'npx':
                    for npx_path in ['/opt/homebrew/bin/npx', '/usr/local/bin/npx', '/usr/bin/npx']:
                        if os.path.exists(npx_path):
                            logger.info(f"Using full path to npx: {npx_path}")
                            self.command = npx_path
                            break
                
                # For npx commands, test if the package exists
                if self.command == 'npx' or self.command.endswith('/npx'):
                    if len(self.args) > 0:
                        package_name = self.args[0]
                        logger.info(f"Testing if npm package '{package_name}' is available")
                        try:
                            # Test if package can be found
                            test_proc = await asyncio.create_subprocess_exec(
                                'npm', 'list', '-g', package_name,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                env=env
                            )
                            stdout, stderr = await test_proc.communicate()
                            if test_proc.returncode == 0:
                                logger.info(f"Package '{package_name}' is globally installed")
                            else:
                                logger.warning(f"Package '{package_name}' not found globally: {stderr.decode()}")
                        except Exception as e:
                            logger.warning(f"Could not check package availability: {e}")
            
            # Create subprocess with timeout
            logger.info(f"[STDIO DEBUG] About to create subprocess")
            logger.info(f"[STDIO DEBUG] Full command: {self.command} {' '.join(self.args)}")
            # Log working directory info
            cwd_info = '/mcp-servers'
            if not os.path.exists('/mcp-servers'):
                if self.args and os.path.dirname(self.args[0]):
                    cwd_info = os.path.dirname(self.args[0])
                else:
                    cwd_info = 'current working directory'
            logger.info(f"[STDIO DEBUG] Working directory: {cwd_info}")
            logger.info(f"[STDIO DEBUG] Environment variables being passed: {list(env.keys())}")
            logger.info(f"[STDIO DEBUG] MCP env vars: {list(self.env_vars.keys())}")
            logger.info(f"[STDIO DEBUG] Required MCP env vars present: JIRA_URL={env.get('JIRA_URL', 'NOT SET')[:20]}...")
            
            # Test if the command exists and is executable
            try:
                import os
                node_path = env.get('PATH', '').split(':')
                logger.info(f"[STDIO DEBUG] PATH: {node_path}")
                node_exists = any(os.path.exists(os.path.join(p, 'node')) for p in node_path if p)
                logger.info(f"[STDIO DEBUG] Node.js exists in PATH: {node_exists}")
            except Exception as e:
                logger.warning(f"[STDIO DEBUG] Could not check node existence: {e}")
            
            try:
                logger.info(f"[STDIO DEBUG] Using subprocess.Popen instead of asyncio")
                import subprocess
                
                # Determine working directory - use actual MCP server directory if outside Docker
                import os
                cwd = '/mcp-servers'
                if not os.path.exists(cwd):
                    # Not in Docker environment, use MCP server directory from args
                    if self.args and os.path.dirname(self.args[0]):
                        cwd = os.path.dirname(self.args[0])
                        logger.info(f"[STDIO DEBUG] Using MCP server directory: {cwd}")
                    else:
                        cwd = None  # Use current working directory
                        logger.info(f"[STDIO DEBUG] Using current working directory")
                
                self.process = subprocess.Popen(
                    [self.command] + self.args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=cwd,
                    text=False  # Use bytes
                )
                logger.info(f"[STDIO DEBUG] Subprocess created successfully, PID: {self.process.pid}")
                
                # Start response reader thread
                self._start_response_thread()
                
            except Exception as e:
                logger.error(f"[STDIO DEBUG] Process creation failed: {e}")
                logger.error(f"[STDIO DEBUG] Command that failed: {self.command} {' '.join(self.args)}")
                logger.error(f"[STDIO DEBUG] Working directory: /mcp-servers")
                raise
            
            # Initialize the MCP connection
            logger.info(f"[STDIO DEBUG] Starting MCP initialization")
            try:
                await self._initialize_mcp()
                logger.info(f"[STDIO DEBUG] MCP initialization completed successfully")
            except Exception as e:
                logger.error(f"[STDIO DEBUG] MCP initialization failed: {e}")
                if self.process:
                    self.process.terminate()
                raise
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            logger.error(f"Command: {self.command} {' '.join(self.args)}")
            logger.error(f"Environment variables: {list(self.env_vars.keys())}")
            if self.process and self.process.stderr:
                try:
                    # Read stderr synchronously for subprocess.Popen
                    import select
                    import sys
                    if sys.platform != 'win32':
                        # Use select for non-blocking read on Unix systems
                        ready, _, _ = select.select([self.process.stderr], [], [], 1.0)
                        if ready:
                            stderr_text = self.process.stderr.read(1024).decode()
                            if stderr_text:
                                logger.error(f"MCP server stderr: {stderr_text}")
                    else:
                        # Windows fallback - just try to read
                        try:
                            stderr_text = self.process.stderr.read(1024).decode()
                            if stderr_text:
                                logger.error(f"MCP server stderr: {stderr_text}")
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"Could not read stderr: {e}")
            raise
    
    def _start_response_thread(self):
        """Start thread to read responses from stdout"""
        def response_reader():
            while True:
                try:
                    if self.process and self.process.stdout:
                        line = self.process.stdout.readline()
                        if line:
                            self.response_queue.put(line.decode().strip())
                        else:
                            break  # Process closed
                    else:
                        break
                except Exception as e:
                    logger.error(f"[STDIO DEBUG] Response reader error: {e}")
                    break
        
        self.response_thread = threading.Thread(target=response_reader, daemon=True)
        self.response_thread.start()
        logger.info(f"[STDIO DEBUG] Response reader thread started")
    
    async def _initialize_mcp(self):
        """Initialize MCP connection with the server following MCP spec"""
        logger.info(f"[STDIO DEBUG] Building initialize request")
        # Send initialize request with proper client capabilities
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
                "clientInfo": self.client_info
            },
            "id": self._next_id()
        }
        
        logger.info(f"[STDIO DEBUG] Sending initialize request")
        response = await self._send_request(init_request)
        logger.info(f"[STDIO DEBUG] Initialize request completed")
        
        # Store server capabilities for future reference
        if "result" in response:
            result = response["result"]
            self.server_capabilities = result.get("capabilities", {})
            server_info = result.get("serverInfo", {})
            logger.info(f"MCP server initialized: {server_info.get('name', 'unknown')} v{server_info.get('version', 'unknown')}")
            logger.debug(f"Server capabilities: {self.server_capabilities}")
        
        # Send initialized notification to complete the handshake
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self._send_notification(initialized_notification)
        
        self.initialized = True
        
    def _next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and wait for response"""
        logger.info(f"[STDIO DEBUG] _send_request called for method: {request.get('method', 'unknown')}")
        if not self.process or not self.process.stdin:
            logger.error(f"[STDIO DEBUG] MCP server process not running or stdin not available")
            raise RuntimeError("MCP server process not running")
        
        # Send request
        request_str = json.dumps(request) + "\n"
        logger.info(f"[STDIO DEBUG] Writing request to stdin: {len(request_str)} bytes")
        self.process.stdin.write(request_str.encode())
        self.process.stdin.flush()  # Use sync flush instead of async drain
        logger.info(f"[STDIO DEBUG] Request written and flushed")
        
        logger.debug(f"Sent MCP request: {request}")
        
        # Read response with timeout using queue
        logger.info(f"[STDIO DEBUG] Waiting for response from queue")
        try:
            # Use asyncio to wait for response from queue
            import time
            start_time = time.time()
            timeout = 3.0
            
            while time.time() - start_time < timeout:
                try:
                    response_text = self.response_queue.get_nowait()
                    logger.info(f"[STDIO DEBUG] Response received from queue: {len(response_text)} chars")
                    break
                except queue.Empty:
                    await asyncio.sleep(0.1)  # Small delay before checking again
            else:
                logger.error(f"[STDIO DEBUG] MCP server response timed out after {timeout} seconds")
                raise Exception("MCP server response timeout - server may be unresponsive")
                
        except Exception as e:
            logger.error(f"[STDIO DEBUG] Error reading from queue: {e}")
            raise
        
        logger.debug(f"Raw MCP response: '{response_text}'")
        
        if not response_text:
            raise Exception("Empty response from MCP server")
        
        try:
            response = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: '{response_text}' - Error: {e}")
            # Check if there are error messages on stderr
            if self.process.stderr:
                try:
                    import select
                    import sys
                    if sys.platform != 'win32':
                        # Use select for non-blocking read
                        ready, _, _ = select.select([self.process.stderr], [], [], 0.5)
                        if ready:
                            stderr_text = self.process.stderr.readline().decode().strip()
                            if stderr_text:
                                logger.error(f"MCP server stderr: {stderr_text}")
                    else:
                        # Windows fallback
                        try:
                            stderr_text = self.process.stderr.readline().decode().strip()
                            if stderr_text:
                                logger.error(f"MCP server stderr: {stderr_text}")
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"Could not read stderr: {e}")
            raise Exception(f"Invalid JSON response from MCP server: {response_text}")
        
        logger.debug(f"Parsed MCP response: {response}")
        
        if "error" in response:
            error = response["error"]
            error_code = error.get("code", -32603)
            error_message = error.get("message", "Unknown error")
            error_data = error.get("data")
            
            # Log detailed error information
            logger.error(f"MCP server error [{error_code}]: {error_message}")
            if error_data:
                logger.error(f"Error data: {error_data}")
            
            # Raise with proper error context
            raise Exception(f"MCP server error [{error_code}]: {error_message}")
        
        return response
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send a notification (no response expected)"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server process not running")
        
        notification_str = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_str.encode())
        self.process.stdin.flush()  # Use sync flush for subprocess.Popen
        
        logger.debug(f"Sent MCP notification: {notification}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        if not self.initialized:
            raise RuntimeError("MCP server not initialized. Call start() first.")
        
        # Check if server supports tools
        if "tools" not in self.server_capabilities:
            logger.warning("MCP server does not advertise tool support")
            return []
        
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
        if not self.initialized:
            raise RuntimeError("MCP server not initialized. Call start() first.")
        
        # Validate tool capability
        if "tools" not in self.server_capabilities:
            raise RuntimeError("MCP server does not support tools")
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
        
        # Call tool with timeout
        try:
            response = await asyncio.wait_for(
                self._send_request(request),
                timeout=5.0  # 5 second timeout for tool execution
            )
            return response.get("result", {})
        except asyncio.TimeoutError:
            logger.error(f"MCP tool call {tool_name} timed out after 5 seconds")
            raise Exception(f"MCP tool '{tool_name}' execution timeout")
    
    async def stop(self):
        """Stop the MCP server process with proper cleanup"""
        if self.process and self.initialized:
            try:
                # Send shutdown notification if server supports it
                if "notifications" in self.server_capabilities:
                    shutdown_notification = {
                        "jsonrpc": "2.0",
                        "method": "notifications/shutdown"
                    }
                    await self._send_notification(shutdown_notification)
                    await asyncio.sleep(0.1)  # Give server time to cleanup
            except Exception as e:
                logger.warning(f"Failed to send shutdown notification: {e}")
            
            # Graceful termination
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    # Use threading for process wait with timeout
                    import threading
                    import time
                    
                    def wait_for_process():
                        return self.process.wait()
                    
                    wait_thread = threading.Thread(target=wait_for_process)
                    wait_thread.start()
                    wait_thread.join(timeout=5.0)
                    
                    if wait_thread.is_alive():
                        logger.warning("MCP server did not terminate gracefully, killing process")
                        self.process.kill()
                        wait_thread.join(timeout=2.0)
                    
                except Exception as e:
                    logger.error(f"Error during process termination: {e}")
                    try:
                        self.process.kill()
                    except:
                        pass
            
            self.process = None
            self.initialized = False


class MCPDockerBridge(MCPStdioBridge):
    """Specialized bridge for Docker-based MCP servers"""
    
    def __init__(self, container_name: str, command: List[str], env_vars: Optional[Dict[str, str]] = None):
        # For Docker exec, we use: docker exec -i <container> <command>
        super().__init__("docker", ["exec", "-i", container_name] + command, env_vars)
        self.container_name = container_name
        
    async def is_container_running(self) -> bool:
        """Check if the Docker container is running"""
        try:
            # Check container status with timeout
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "docker", "inspect", "-f", "{{.State.Running}}", self.container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=5.0  # 5 second timeout for docker inspect
            )
            stdout, _ = await asyncio.wait_for(result.communicate(), timeout=5.0)
            return stdout.decode().strip() == "true"
        except (Exception, asyncio.TimeoutError):
            logger.warning(f"Failed to check container {self.container_name} status (timeout or error)")
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
                
                bridge = MCPDockerBridge(container_name, mcp_command, server_config.get("env", {}))
                
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
            bridge = MCPStdioBridge(command, args, server_config.get("env", {}))
            await bridge.start()
            
            try:
                result = await bridge.call_tool(tool_name, parameters)
                return result
            finally:
                await bridge.stop()
                
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name} via stdio: {e}")
        return {"error": str(e)}