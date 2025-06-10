"""
Simple Docker exec wrapper for MCP stdio communication
"""
import asyncio
import json
import logging
import subprocess
from typing import Dict, List, Any
from app.core.docker_api_bridge import get_docker_api

logger = logging.getLogger(__name__)


async def _call_via_docker_api(
    docker_api,
    container_name: str,
    command: List[str],
    tool_name: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Call MCP tool using Docker API"""
    # Create the MCP protocol messages
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
        "id": 1
    }
    
    initialized_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }
    
    tool_request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": parameters
        },
        "id": 2
    }
    
    # Combine all messages
    messages = "\n".join([
        json.dumps(init_request),
        json.dumps(initialized_notification),
        json.dumps(tool_request),
        ""  # Empty line at end
    ])
    
    # Debug logging
    logger.info(f"[DEBUG] MCP Docker API call:")
    logger.info(f"  Container: {container_name}")
    logger.info(f"  Command: {command}")
    logger.info(f"  Tool: {tool_name}")
    logger.info(f"  Parameters keys: {list(parameters.keys())}")
    
    # Log OAuth params if present (masked)
    oauth_keys = ["google_access_token", "google_refresh_token", "google_client_id", "google_client_secret"]
    for key in oauth_keys:
        if key in parameters:
            val = parameters[key]
            if "token" in key or "secret" in key:
                logger.info(f"  {key}: {val[:10]}... (length: {len(val)})")
            else:
                logger.info(f"  {key}: {val}")
    
    # Execute via Docker API (run in executor for async)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        docker_api.exec_in_container,
        container_name,
        command,
        messages  # Pass as stdin_data
    )
    
    if result.get("error"):
        return {"error": result["error"]}
    
    # Parse output
    output = result.get("stdout", "")
    tool_result = None
    
    for line in output.split('\n'):
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
            if msg.get("id") == 2 and "result" in msg:
                tool_result = msg["result"]
                break
        except json.JSONDecodeError:
            continue
    
    if tool_result is not None:
        return tool_result
    else:
        return {
            "error": "No valid tool response received",
            "raw_output": output[:500]
        }


async def call_mcp_tool_in_docker(
    container_name: str,
    command: List[str],
    tool_name: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call an MCP tool in a Docker container using subprocess
    """
    try:
        # Try Docker API first, fall back to CLI if needed
        docker_api = get_docker_api()
        
        if docker_api.client and docker_api.is_container_running(container_name):
            # Use Docker API
            return await _call_via_docker_api(docker_api, container_name, command, tool_name, parameters)
        else:
            # Fall back to Docker CLI
            logger.info("Docker API not available, using Docker CLI")
            docker_cmd = ["docker", "exec", "-i", container_name] + command
        
        # Create the MCP protocol messages
        # 1. Initialize
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
            "id": 1
        }
        
        # 2. Initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        }
        
        # 3. Tool call
        tool_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters
            },
            "id": 2
        }
        
        # Combine all messages
        messages = [
            json.dumps(init_request) + "\n",
            json.dumps(initialized_notification) + "\n",
            json.dumps(tool_request) + "\n"
        ]
        
        # Execute docker command
        process = await asyncio.create_subprocess_exec(
            *docker_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send all messages
        stdin_data = "".join(messages).encode()
        stdout, stderr = await process.communicate(input=stdin_data)
        
        if stderr:
            logger.warning(f"Docker exec stderr: {stderr.decode()}")
        
        # Parse the output to find the tool response
        output = stdout.decode()
        tool_result = None
        
        for line in output.split('\n'):
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
                # Look for our tool call response
                if msg.get("id") == 2 and "result" in msg:
                    tool_result = msg["result"]
                    break
            except json.JSONDecodeError:
                continue
        
        if tool_result is not None:
            return tool_result
        else:
            # If no proper response, return the raw output for debugging
            return {
                "error": "No valid tool response received",
                "raw_output": output[:500]  # First 500 chars for debugging
            }
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker exec failed: {e}")
        return {"error": f"Docker exec failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Error calling MCP tool in Docker: {e}")
        return {"error": str(e)}