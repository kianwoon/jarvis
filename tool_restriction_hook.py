#!/usr/bin/env python3
"""
Tool Restriction Hook
=====================
This hook enforces tool restrictions to ensure Claude uses Claude Code agents via Task tool 
instead of direct execution. It intercepts tool calls and blocks restricted operations.

CRITICAL SYSTEM SEPARATION:
- Claude Code agents: .claude/agents/*.md files - FOR CLAUDE'S INTERNAL USE
- Jarvis agents: PostgreSQL database - FOR END USER @agent FEATURE

This hook enforces Claude Code agent usage ONLY, not Jarvis agent usage.
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
    
    # Tools that are completely denied - MUST USE CLAUDE CODE AGENTS VIA TASK TOOL
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
        # Removed psql and redis-cli to allow read operations
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
            log_message("BLOCKED", f"Denied tool '{tool_name}' - MUST use Claude Code agents from .claude/agents/")
            
            # Log violation for enforcement monitoring
            violation_log = "/Users/kianwoonwong/Downloads/jarvis/.claude/logs/violations.log"
            os.makedirs(os.path.dirname(violation_log), exist_ok=True)
            with open(violation_log, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] VIOLATION: Attempted to use {tool_name}\n")
                f.write(f"  MUST USE: Claude Code agent from .claude/agents/\n\n")
            
            return {
                "allowed": False,
                "reason": f"🚨 CRITICAL: Tool '{tool_name}' is STRICTLY FORBIDDEN! You MUST use Claude Code agents from /Users/kianwoonwong/Downloads/jarvis/.claude/agents/",
                "suggestion": self._get_claude_agent_suggestion(tool_name),
                "enforcement": "MANDATORY",
                "violation_logged": True,
                "agent_path": "/Users/kianwoonwong/Downloads/jarvis/.claude/agents/"
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
                    "suggestion": "Use Claude Code agent via Task tool (e.g. Task(task='...', subagent_type='coder'))"
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
            "suggestion": "Use Claude Code agent via Task tool with appropriate subagent_type"
        }
    
    def _is_database_read_command(self, command: str) -> bool:
        """Check if command is a read-only database operation"""
        command_lower = command.lower().strip()
        
        # PostgreSQL read operations
        if "psql" in command_lower:
            # Allow PGPASSWORD environment variable pattern
            if command_lower.startswith("pgpassword="):
                # Check for SELECT or describe commands
                if "select " in command_lower or any(pattern in command_lower for pattern in ["\\d", "\\dt", "\\l"]):
                    return True
            
            # Allow psql with SELECT statements
            if "select " in command_lower:
                return True
            
            # Allow psql describe/list commands
            psql_read_patterns = [
                "\\d",      # Describe tables/relations
                "\\dt",     # List tables
                "\\l",      # List databases
                "\\du",     # List users
                "\\dn",     # List schemas
                "\\df",     # List functions
                "\\dv",     # List views
                "\\di",     # List indexes
                "\\ds",     # List sequences
            ]
            
            for pattern in psql_read_patterns:
                if pattern in command_lower:
                    return True
        
        # Redis read operations
        if "redis-cli" in command_lower:
            redis_read_commands = [
                "get ",
                "hget ",
                "hgetall ",
                "keys ",
                "scan ",
                "ttl ",
                "exists ",
                "type ",
                "info",
                "ping",
                "dbsize",
                "llen ",
                "scard ",
                "zcard ",
                "strlen ",
                "mget ",
                "lrange ",
                "smembers ",
                "zrange ",
            ]
            
            for cmd in redis_read_commands:
                if cmd in command_lower:
                    return True
        
        return False
    
    def _check_bash_permission(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a Bash command is permitted"""
        command = parameters.get("command", "")
        
        # Check for database read commands first (before restricted patterns)
        if self._is_database_read_command(command):
            log_message("ALLOWED", f"Permitted database read command: {command[:100]}")
            return {
                "allowed": True,
                "reason": "Database read-only command allowed"
            }
        
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
        """Get suggestion for which agent to use (deprecated - use _get_claude_agent_suggestion)"""
        return self._get_claude_agent_suggestion(tool_name)
    
    def _get_claude_agent_suggestion(self, tool_name: str) -> str:
        """Get CLAUDE CODE agent suggestion for blocked tools"""
        claude_agents = {
            "Write": "🔴 MUST USE: ./delegate_to_agent.sh coder 'Create the file' OR ./delegate_to_agent.sh senior-coder 'Implement the feature'",
            "Edit": "🔴 MUST USE: ./delegate_to_agent.sh coder 'Modify the file' OR ./delegate_to_agent.sh senior-coder 'Fix the issue'", 
            "MultiEdit": "🔴 MUST USE: ./delegate_to_agent.sh senior-coder 'Make multiple edits to the file'",
            "NotebookEdit": "🔴 MUST USE: ./delegate_to_agent.sh database-administrator 'Modify the notebook'",
            "TodoWrite": "🔴 MUST USE: ./delegate_to_agent.sh general-purpose 'Manage the task list'"
        }
        default = "🔴 MUST USE: ./delegate_to_agent.sh --list (to see available Claude Code agents)"
        return claude_agents.get(tool_name, default)
    
    def _get_bash_agent_suggestion(self, command: str) -> str:
        """Get CLAUDE CODE agent suggestion based on Bash command"""
        if "python" in command or "node" in command:
            return "🔴 MUST USE: ./delegate_to_agent.sh coder 'Execute the code' OR senior-coder for complex tasks"
        elif "docker" in command:
            return "🔴 MUST USE: ./delegate_to_agent.sh senior-coder 'Handle container operations'"
        elif "git" in command:
            return "🔴 MUST USE: ./delegate_to_agent.sh coder 'Perform git operations'"
        else:
            return "🔴 MUST USE: ./delegate_to_agent.sh --list (see Claude Code agents in .claude/agents/)"
    
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
                "alternative": "./delegate_to_agent.sh --list",
                "critical_reminder": "⚠️ YOU MUST USE CLAUDE CODE AGENTS FROM .claude/agents/ - NO EXCEPTIONS!",
                "agent_location": "/Users/kianwoonwong/Downloads/jarvis/.claude/agents/"
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