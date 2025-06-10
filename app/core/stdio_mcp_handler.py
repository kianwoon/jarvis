"""
Handler for stdio-based MCP servers (like Gmail MCP running in Docker)
"""
import json
import subprocess
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

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
                
        else:
            return {"error": f"Unsupported stdio command type: {command}"}
            
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout calling {tool_name} (30s exceeded)"}
    except Exception as e:
        logger.error(f"Error calling stdio MCP tool {tool_name}: {str(e)}")
        return {"error": str(e)}