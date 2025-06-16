"""
Tool Execution Module for Multi-Agent System
Handles tool call parsing and execution without hardcoding
"""
import json
import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def _is_tool_result_successful(result) -> bool:
    """
    Determine if a tool execution result indicates success
    
    Args:
        result: The tool execution result
        
    Returns:
        bool: True if successful, False if failed
    """
    if not isinstance(result, dict):
        return True  # Non-dict results are considered successful
    
    # Check for explicit error indicators
    if "error" in result and isinstance(result["error"], str) and result["error"].strip():
        return False
    
    # Check for successful content patterns
    if "content" in result:
        content = result["content"]
        if isinstance(content, list) and len(content) > 0:
            # Check if first content item indicates success
            first_item = content[0]
            if isinstance(first_item, dict) and "text" in first_item:
                text = first_item["text"].lower()
                # Look for explicit success indicators
                if any(indicator in text for indicator in ["✅", "success", "sent successfully", "completed successfully"]):
                    return True
                # Look for explicit error indicators
                if any(indicator in text for indicator in ["❌", "failed", "error:", "exception:"]):
                    return False
        return True  # Content exists, assume success
    
    # Check for result field
    if "result" in result:
        return True  # Has result field, assume success
    
    # Default to success for unrecognized formats
    return True

class ToolExecutor:
    """Executes tools based on agent responses"""
    
    def __init__(self):
        # Import at runtime to avoid circular imports
        try:
            from app.langchain.service import call_mcp_tool
            self.call_mcp_tool = call_mcp_tool
        except ImportError as e:
            logger.error(f"Failed to import call_mcp_tool: {e}")
            self.call_mcp_tool = None
    
    def extract_tool_calls(self, agent_response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from agent response text
        Supports multiple formats without hardcoding specific tools
        """
        tool_calls = []
        
        # Format 1: JSON tool call format - improved pattern for complex nested JSON
        logger.info(f"Analyzing response for tool calls. Response length: {len(agent_response)}")
        logger.info(f"Response preview: {agent_response[:500]}")
        
        # First try to find complete JSON objects with tool and parameters
        json_blocks = []
        
        # Look for JSON blocks that start with {"tool"
        json_pattern = r'\{\s*"tool"[^}]*\}'
        potential_jsons = re.findall(json_pattern, agent_response, re.DOTALL)
        
        for potential_json in potential_jsons:
            # Try to find the complete JSON by counting braces
            start_idx = agent_response.find(potential_json)
            if start_idx != -1:
                # Find the complete JSON by balancing braces
                brace_count = 0
                for i, char in enumerate(agent_response[start_idx:]):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            complete_json = agent_response[start_idx:start_idx + i + 1]
                            json_blocks.append(complete_json)
                            break
        
        logger.info(f"Found {len(json_blocks)} potential JSON blocks: {json_blocks}")
        
        for json_block in json_blocks:
            try:
                # First try direct parsing
                parsed = json.loads(json_block)
                if isinstance(parsed, dict) and "tool" in parsed and "parameters" in parsed:
                    tool_calls.append({
                        "tool": parsed["tool"].strip(),
                        "parameters": parsed["parameters"]
                    })
                    logger.info(f"Extracted tool call: {parsed['tool']}")
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues with multi-line strings
                try:
                    # More sophisticated approach: properly escape newlines in JSON strings
                    # First, let's try a simpler fix - replace actual newlines within quoted strings
                    lines = json_block.split('\n')
                    fixed_lines = []
                    in_string = False
                    current_string = []
                    
                    for line in lines:
                        # Simple heuristic: count quotes to track if we're in a string
                        quote_count = line.count('"') - line.count('\\"')
                        
                        if in_string:
                            # We're continuing a multi-line string
                            if quote_count % 2 == 1:  # Odd number of quotes means string ends
                                in_string = False
                                current_string.append(line)
                                # Join the string parts with \n
                                joined = '\\n'.join(current_string)
                                fixed_lines.append(joined)
                                current_string = []
                            else:
                                current_string.append(line.rstrip())
                        else:
                            # Check if we're starting a multi-line string
                            if quote_count % 2 == 1 and not line.rstrip().endswith('",') and not line.rstrip().endswith('"}'):
                                in_string = True
                                current_string = [line.rstrip()]
                            else:
                                fixed_lines.append(line)
                    
                    # Rejoin and try parsing
                    fixed_json = '\n'.join(fixed_lines)
                    parsed = json.loads(fixed_json)
                    
                    if isinstance(parsed, dict) and "tool" in parsed and "parameters" in parsed:
                        tool_calls.append({
                            "tool": parsed["tool"].strip(),
                            "parameters": parsed["parameters"]
                        })
                        logger.info(f"Extracted tool call (after fixing multi-line strings): {parsed['tool']}")
                except Exception as fix_error:
                    logger.warning(f"Failed to parse JSON block even after fixes: {json_block[:100]}... Original error: {e}")
        
        # Fallback: Simple regex pattern
        if not tool_calls:
            tool_json_pattern = r'\{\s*"tool":\s*"([^"]+)"\s*,\s*"parameters":\s*\{([^}]*)\}\s*\}'
            matches = re.findall(tool_json_pattern, agent_response, re.DOTALL)
            
            for tool_name, params_content in matches:
                try:
                    # Build the parameters JSON
                    params_json = "{" + params_content + "}"
                    parameters = json.loads(params_json)
                    tool_calls.append({
                        "tool": tool_name.strip(),
                        "parameters": parameters
                    })
                    logger.info(f"Extracted tool call (fallback): {tool_name}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse parameters for {tool_name}: {e}")
        
        # If simple pattern didn't work, try finding any valid JSON with tool/parameters
        if not tool_calls:
            # Look for JSON blocks that might contain tool calls
            json_blocks = re.findall(r'\{[^}]+\}', agent_response.replace('\n', ' '))
            for block in json_blocks:
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, dict) and "tool" in parsed and "parameters" in parsed:
                        tool_calls.append({
                            "tool": parsed["tool"].strip(),
                            "parameters": parsed["parameters"]
                        })
                        logger.info(f"Extracted tool call (fallback): {parsed['tool']}")
                except json.JSONDecodeError:
                    continue
        
        # Format 2: XML-style: <tool>tool_name(parameters)</tool>
        if not tool_calls:
            xml_pattern = r'<tool>([^(]+)\(([^)]*)\)</tool>'
            xml_matches = re.findall(xml_pattern, agent_response, re.DOTALL)
            
            for tool_name, params_str in xml_matches:
                try:
                    tool_name = tool_name.strip()
                    params_str = params_str.strip()
                    
                    if params_str in ["{}", ""]:
                        parameters = {}
                    else:
                        parameters = json.loads(params_str)
                    
                    tool_calls.append({
                        "tool": tool_name,
                        "parameters": parameters
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse XML parameters for {tool_name}: {params_str}")
        
        # Format 3: Function call style: tool_name(param1="value1", param2="value2")
        if not tool_calls:
            func_pattern = r'(\w+)\(([^)]*)\)'
            func_matches = re.findall(func_pattern, agent_response)
            
            # Only consider matches that look like tool calls (contain known patterns)
            for tool_name, params_str in func_matches:
                if self._looks_like_tool_call(tool_name, params_str):
                    parameters = self._parse_function_params(params_str)
                    tool_calls.append({
                        "tool": tool_name,
                        "parameters": parameters
                    })
        
        return tool_calls
    
    def _looks_like_tool_call(self, tool_name: str, params_str: str) -> bool:
        """Check if a function call looks like a tool call"""
        # Check if tool name follows common tool naming patterns
        tool_indicators = [
            "search", "send", "get", "list", "create", "read", "write", 
            "delete", "update", "fetch", "call", "execute"
        ]
        
        tool_name_lower = tool_name.lower()
        has_tool_indicator = any(indicator in tool_name_lower for indicator in tool_indicators)
        
        # Check if parameters contain key-value pairs or JSON-like structure
        has_params = "=" in params_str or ":" in params_str or params_str.strip() == ""
        
        return has_tool_indicator and has_params
    
    def _parse_function_params(self, params_str: str) -> Dict[str, Any]:
        """Parse function-style parameters: param1="value1", param2="value2" """
        parameters = {}
        
        if not params_str.strip():
            return parameters
        
        # Simple parsing for key="value" or key=value patterns
        param_pattern = r'(\w+)\s*=\s*["\']?([^,"\']+)["\']?'
        param_matches = re.findall(param_pattern, params_str)
        
        for key, value in param_matches:
            # Try to parse as JSON value, fallback to string
            try:
                if value.lower() in ['true', 'false']:
                    parameters[key] = value.lower() == 'true'
                elif value.isdigit():
                    parameters[key] = int(value)
                else:
                    parameters[key] = value.strip('"\'')
            except:
                parameters[key] = value.strip('"\'')
        
        return parameters
    
    async def execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a list of tool calls and return results"""
        if not self.call_mcp_tool:
            logger.error("Tool execution not available - call_mcp_tool not imported")
            return []
        
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            
            if not tool_name:
                continue
            
            try:
                logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
                result = self.call_mcp_tool(tool_name, parameters)
                
                results.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "success": _is_tool_result_successful(result)
                })
                
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                results.append({
                    "tool": tool_name,
                    "parameters": parameters,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    async def process_agent_response(self, agent_response: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Process agent response, execute any tool calls, and return enhanced response
        Returns: (enhanced_response, tool_results)
        """
        # Extract tool calls from response
        tool_calls = self.extract_tool_calls(agent_response)
        
        if not tool_calls:
            return agent_response, []
        
        # Execute tools
        tool_results = await self.execute_tools(tool_calls)
        
        # Create enhanced response with tool results
        enhanced_response = agent_response
        
        if tool_results:
            enhanced_response += "\n\n**Tool Execution Results:**\n"
            for result in tool_results:
                if result["success"]:
                    enhanced_response += f"✅ {result['tool']}: {json.dumps(result['result'], indent=2)}\n"
                else:
                    enhanced_response += f"❌ {result['tool']}: {result.get('error', 'Unknown error')}\n"
        
        return enhanced_response, tool_results

# Global instance for easy access
tool_executor = ToolExecutor()

def extract_and_execute_tools(agent_response: str) -> tuple[str, List[Dict[str, Any]]]:
    """Convenience function for processing agent responses"""
    import asyncio
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(tool_executor.process_agent_response(agent_response))