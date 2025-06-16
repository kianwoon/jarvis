"""
Handler for stdio-based MCP servers (like Gmail MCP running in Docker)
"""
import json
import subprocess
import logging
import requests
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

def _inject_oauth_credentials(parameters: Dict[str, Any], server_id: int) -> Dict[str, Any]:
    """
    Inject OAuth credentials into parameters for Gmail MCP tools
    
    Args:
        parameters: Original tool parameters
        server_id: MCP server ID to get credentials for
        
    Returns:
        Parameters with OAuth credentials injected
    """
    try:
        from app.core.oauth_token_manager import oauth_token_manager
        
        # Get valid OAuth token
        oauth_creds = oauth_token_manager.get_valid_token(server_id, "gmail")
        
        if oauth_creds:
            # Create a copy of parameters to avoid modifying the original
            enhanced_params = parameters.copy()
            
            # Add OAuth credentials as expected by Gmail MCP server
            enhanced_params.update({
                "google_client_id": oauth_creds.get("client_id"),
                "google_client_secret": oauth_creds.get("client_secret"),
                "google_access_token": oauth_creds.get("access_token"),
                "google_refresh_token": oauth_creds.get("refresh_token")
            })
            
            logger.debug(f"Injected OAuth credentials for Gmail MCP tool")
            return enhanced_params
        else:
            logger.warning(f"No OAuth credentials found for server {server_id}")
            return parameters
            
    except Exception as e:
        logger.error(f"Failed to inject OAuth credentials: {e}")
        return parameters

def call_stdio_mcp_tool(server_config: Dict[str, Any], tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call an MCP tool that uses stdio transport (typically Docker-based servers)
    
    Args:
        server_config: Server configuration with command, args, etc.
        tool_name: Name of the tool to call
        parameters: Parameters to pass to the tool
        
    Returns:
        Tool response or error
    """
    try:
        # Build the command
        command = server_config.get("command", "")
        args = server_config.get("args", [])
        
        # For Gmail MCP tools, inject OAuth credentials
        if "mcp-gmail" in str(args):
            parameters = _inject_oauth_credentials(parameters, server_id=3)  # Gmail server ID is 3
        
        if command == "docker" and "exec" in args:
            # For Docker exec commands, we need to send the MCP request via stdin
            # The Gmail MCP server expects JSON-RPC format
            
            # Build JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                },
                "id": 1
            }
            
            # Try Docker CLI first, fallback to HTTP API if not available
            try:
                # Build full command
                full_command = [command] + args
                
                logger.info(f"Calling stdio MCP tool {tool_name} via command: {' '.join(full_command)}")
                logger.debug(f"Request: {json.dumps(request)}")
                
                # Execute the command with JSON input
                process = subprocess.Popen(
                    full_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Send request and get response
                stdout, stderr = process.communicate(input=json.dumps(request) + "\n", timeout=30)
                
                if stderr:
                    logger.warning(f"Stderr from MCP tool: {stderr}")
                
                if stdout:
                    # Parse JSON-RPC response
                    try:
                        response = json.loads(stdout)
                        if "error" in response:
                            return {"error": response["error"].get("message", "Unknown error")}
                        elif "result" in response:
                            return response["result"]
                        else:
                            return {"result": stdout}
                    except json.JSONDecodeError:
                        # If not JSON, return as plain text
                        return {"result": stdout}
                else:
                    return {"error": "No response from MCP server"}
                    
            except FileNotFoundError:
                # Docker CLI not available, try Docker HTTP API
                logger.info(f"Docker CLI not available, trying HTTP API for {tool_name}")
                return _call_docker_exec_via_api(args, request, tool_name)
                
        else:
            return {"error": f"Unsupported stdio command type: {command}"}
            
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout calling {tool_name} (30s exceeded)"}
    except Exception as e:
        logger.error(f"Error calling stdio MCP tool {tool_name}: {str(e)}")
        return {"error": str(e)}


def _call_docker_exec_via_api(args: list, request: dict, tool_name: str) -> Dict[str, Any]:
    """
    Call docker exec using Docker Python SDK instead of CLI
    
    Args:
        args: Docker command args (e.g., ['exec', '-i', 'mcp-gmail', 'node', '/app/index.js'])
        request: JSON-RPC request to send
        tool_name: Name of the tool being called
        
    Returns:
        Tool response or error
    """
    try:
        import docker
        
        # Extract container name and command from args
        # args format: ['exec', '-i', 'container_name', 'command', 'arg1', 'arg2', ...]
        if len(args) < 4 or args[0] != "exec":
            return {"error": "Invalid docker exec arguments"}
        
        container_name = args[2]  # Skip 'exec' and '-i'
        exec_cmd = args[3:]       # The actual command to run
        
        logger.info(f"Using Docker SDK to exec in container {container_name} with command: {' '.join(exec_cmd)}")
        logger.debug(f"JSON-RPC request: {json.dumps(request)}")
        
        # Initialize Docker client
        client = docker.from_env()
        
        # Get the container
        try:
            container = client.containers.get(container_name)
        except docker.errors.NotFound:
            return {"error": f"Container {container_name} not found"}
        
        # For simplicity, let's use a direct approach with container.exec_run
        # We'll write the JSON to a temporary file and then cat it to the command
        json_input = json.dumps(request)
        
        # Create a command that feeds JSON to the MCP server
        # Escape the JSON properly for shell execution
        import shlex
        escaped_json = shlex.quote(json_input)
        full_cmd = f"echo {escaped_json} | {' '.join(exec_cmd)}"
        
        logger.info(f"Executing command: {full_cmd}")
        
        # Execute the command
        exec_result = container.exec_run(
            cmd=["sh", "-c", full_cmd],
            stdout=True,
            stderr=True,
            demux=True  # Separate stdout and stderr
        )
        
        # Get the result
        exit_code = exec_result.exit_code
        stdout_bytes, stderr_bytes = exec_result.output
        stdout = stdout_bytes.decode('utf-8') if stdout_bytes else ""
        stderr = stderr_bytes.decode('utf-8') if stderr_bytes else ""
        
        if stderr:
            logger.warning(f"Stderr from command: {stderr}")
        
        logger.info(f"Command exited with code {exit_code}")
        logger.debug(f"Output: {stdout[:500]}...")
        
        if exit_code != 0:
            return {"error": f"Command failed with exit code {exit_code}: {stdout}"}
        
        if stdout:
            # Parse JSON-RPC response
            try:
                # Clean up the output - remove any extra whitespace/newlines
                stdout = stdout.strip()
                
                # Handle cases where there's extra text before the JSON
                # Look for JSON starting with { and ending with }
                import re
                json_match = re.search(r'\{.*\}', stdout, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    logger.info(f"Extracted JSON from output: {json_text[:200]}...")
                    response = json.loads(json_text)
                else:
                    # If no JSON found, try the original stdout
                    logger.warning(f"No JSON found in output, trying original: {stdout[:200]}...")
                    response = json.loads(stdout)
                    
                if "error" in response:
                    return {"error": response["error"].get("message", "Unknown error")}
                elif "result" in response:
                    return response["result"]
                else:
                    return {"result": response}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}. Raw response: {stdout}")
                # If not JSON, return as plain text
                return {"result": stdout}
        else:
            return {"error": "No response from MCP server"}
            
    except ImportError:
        logger.error("Docker SDK not available - cannot use Docker API fallback")
        return {"error": "Docker SDK not available"}
    except Exception as e:
        logger.error(f"Error calling Docker API for {tool_name}: {str(e)}")
        return {"error": str(e)}