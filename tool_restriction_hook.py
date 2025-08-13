#!/usr/bin/env python3
"""
Tool Restriction Hook
=====================
This hook enforces tool restrictions to ensure Claude uses agents instead of direct execution.
It intercepts tool calls and blocks restricted operations.
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Logging configuration
LOG_FILE = "/Users/kianwoonwong/Downloads/jarvis/tool_restriction.log"

def log_message(level: str, message: str):
    """Log message to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    except:
        pass  # Silent fail for logging

class ToolRestrictionEnforcer:
    """Enforces tool usage restrictions"""
    
    # Tools that are completely denied
    DENIED_TOOLS = {
        "Write",
        "Edit", 
        "MultiEdit",
        "NotebookEdit",
        "TodoWrite"
    }
    
    # Tools that are allowed for read/analysis only
    ALLOWED_TOOLS = {
        "Read",
        "Grep",
        "Glob",
        "LS",
        "WebSearch",
        "WebFetch",
        "BashOutput",
        "KillBash"
    }
    
    # Bash command patterns that are restricted
    RESTRICTED_BASH_PATTERNS = [
        "python",
        "node",
        "npm",
        "docker",
        "git add",
        "git commit",
        "git push",
        "psql",
        "redis-cli",
        "rm",
        "mv",
        "cp",
        "chmod",
        "chown",
        "sudo",
        "pip install",
        "apt",
        "brew",
        "curl -X POST",
        "curl -X PUT",
        "curl -X DELETE",
        "wget",
        ">",  # File redirection
        ">>",  # File append
    ]
    
    def __init__(self):
        self.session_id = os.getenv("CLAUDE_SESSION_ID", "unknown")
        
    def check_tool_permission(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a tool call is permitted"""
        
        # Check if tool is explicitly denied
        if tool_name in self.DENIED_TOOLS:
            log_message("BLOCKED", f"Denied tool '{tool_name}' - must use agents")
            return {
                "allowed": False,
                "reason": f"Tool '{tool_name}' is restricted. Please use request_agent_work.py to delegate this task to appropriate agents.",
                "suggestion": self._get_agent_suggestion(tool_name),
                "enforcement": "strict"
            }
        
        # Special handling for Bash commands
        if tool_name == "Bash":
            return self._check_bash_permission(parameters)
        
        # Check if tool is in allowed list
        if tool_name in self.ALLOWED_TOOLS:
            log_message("ALLOWED", f"Permitted tool '{tool_name}'")
            return {
                "allowed": True,
                "reason": f"Tool '{tool_name}' is allowed for analysis"
            }
        
        # Check MCP tools (usually safe for read operations)
        if tool_name.startswith("mcp__"):
            # Block execute operations
            if "execute" in tool_name.lower() and "Code" in tool_name:
                log_message("BLOCKED", f"Denied MCP execute tool '{tool_name}'")
                return {
                    "allowed": False,
                    "reason": "Direct code execution is restricted. Use agents for execution.",
                    "suggestion": "Use 'Code Agent' via request_agent_work.py"
                }
            
            log_message("ALLOWED", f"Permitted MCP tool '{tool_name}'")
            return {
                "allowed": True,
                "reason": f"MCP tool '{tool_name}' allowed"
            }
        
        # Default deny for unknown tools
        log_message("BLOCKED", f"Unknown tool '{tool_name}' denied by default")
        return {
            "allowed": False,
            "reason": f"Tool '{tool_name}' not in allowed list",
            "suggestion": "Use request_agent_work.py for this operation"
        }
    
    def _check_bash_permission(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a Bash command is permitted"""
        command = parameters.get("command", "")
        
        # Check for restricted patterns
        for pattern in self.RESTRICTED_BASH_PATTERNS:
            if pattern in command.lower():
                log_message("BLOCKED", f"Denied Bash command with pattern '{pattern}': {command[:100]}")
                return {
                    "allowed": False,
                    "reason": f"Bash command contains restricted pattern '{pattern}'",
                    "suggestion": self._get_bash_agent_suggestion(command),
                    "enforcement": "strict"
                }
        
        # Allow safe read-only commands
        safe_commands = ["ls", "pwd", "echo", "cat", "head", "tail", "grep", "find", "which", "env"]
        first_word = command.split()[0] if command else ""
        
        if first_word in safe_commands:
            log_message("ALLOWED", f"Permitted safe Bash command: {command[:100]}")
            return {
                "allowed": True,
                "reason": "Read-only Bash command allowed"
            }
        
        # Default deny for other Bash commands
        log_message("BLOCKED", f"Denied Bash command by default: {command[:100]}")
        return {
            "allowed": False,
            "reason": "Bash command not in safe list. Use agents for execution.",
            "suggestion": self._get_bash_agent_suggestion(command)
        }
    
    def _get_agent_suggestion(self, tool_name: str) -> str:
        """Get suggestion for which agent to use"""
        suggestions = {
            "Write": "Use 'Code Agent' to create files",
            "Edit": "Use 'Code Agent' to modify files", 
            "MultiEdit": "Use 'Code Agent' for multiple file edits",
            "NotebookEdit": "Use 'Data Agent' for notebook operations",
            "TodoWrite": "Use 'Planning Agent' for task management"
        }
        return suggestions.get(tool_name, "Use appropriate agent via request_agent_work.py")
    
    def _get_bash_agent_suggestion(self, command: str) -> str:
        """Get agent suggestion based on Bash command"""
        if "python" in command or "node" in command:
            return "Use 'Code Agent' to execute code"
        elif "docker" in command:
            return "Use 'Integration Agent' for container operations"
        elif "git" in command:
            return "Use 'Code Agent' for version control"
        elif "psql" in command or "redis" in command:
            return "Use 'Data Agent' for database operations"
        else:
            return "Use appropriate agent via request_agent_work.py"
    
    def process_hook(self, hook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the hook and return response"""
        tool_name = hook_data.get("tool", "")
        parameters = hook_data.get("parameters", {})
        
        # Check permission
        permission_result = self.check_tool_permission(tool_name, parameters)
        
        # Format response
        if not permission_result["allowed"]:
            return {
                "action": "block",
                "message": permission_result["reason"],
                "suggestion": permission_result.get("suggestion", ""),
                "alternative": "python /Users/kianwoonwong/Downloads/jarvis/request_agent_work.py --help"
            }
        
        return {
            "action": "allow",
            "message": permission_result["reason"]
        }

def main():
    """Main hook entry point"""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        
        if not input_data:
            # No input, allow by default
            print(json.dumps({"action": "allow"}))
            return
        
        # Parse input
        try:
            hook_data = json.loads(input_data)
        except json.JSONDecodeError:
            # Invalid JSON, allow by default (don't break Claude)
            print(json.dumps({"action": "allow"}))
            return
        
        # Create enforcer and process
        enforcer = ToolRestrictionEnforcer()
        result = enforcer.process_hook(hook_data)
        
        # Output result
        print(json.dumps(result))
        
    except Exception as e:
        # On any error, log but allow (don't break Claude)
        log_message("ERROR", f"Hook error: {e}")
        print(json.dumps({"action": "allow"}))

if __name__ == "__main__":
    main()