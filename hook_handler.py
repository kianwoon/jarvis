#!/usr/bin/env python3
"""
Hook handler script for Claude Code
Properly processes JSON input and executes appropriate actions
"""

import json
import sys
import subprocess
import os

def safe_run_command(command):
    """Safely run a command and return success"""
    try:
        subprocess.run(command, shell=True, check=False, capture_output=True, timeout=2)
        return True
    except Exception as e:
        print(f"Hook command error (non-fatal): {e}", file=sys.stderr)
        return True  # Return True to prevent hook failures from blocking Claude

def handle_bash_hook():
    """Handle Bash tool hooks"""
    try:
        data = json.load(sys.stdin)
        cmd = data.get('tool_input', {}).get('command', '')
        
        # Truncate long commands for display
        display_cmd = f"{cmd[:50]}..." if len(cmd) > 50 else cmd
        print(f"🔧 Running: {display_cmd}")
        
        # Speak for important commands only
        important_commands = ['docker', 'npm', 'git commit', 'psql', 'python', 'node']
        if any(keyword in cmd.lower() for keyword in important_commands):
            safe_run_command("say 'Running command'")
            
    except Exception as e:
        print(f"🔧 Running Bash command")
    
    return 0

def handle_task_hook():
    """Handle Task tool hooks"""
    try:
        data = json.load(sys.stdin)
        task_desc = data.get('tool_input', {}).get('description', 'Unknown task')
        subagent = data.get('tool_input', {}).get('subagent_type', 'agent')
        
        # Truncate long descriptions
        display_desc = f"{task_desc[:60]}..." if len(task_desc) > 60 else task_desc
        print(f"🤖 Subagent [{subagent}]: {display_desc}")
        
        safe_run_command("say 'Subagent working'")
        
    except Exception as e:
        print(f"🤖 Starting subagent task")
    
    return 0

def handle_notification():
    """Handle notification hooks"""
    safe_run_command("say 'Claude needs your input'")
    safe_run_command("osascript -e 'display notification \"Claude needs your attention\" with title \"Claude Code\" sound name \"Ping\"'")
    return 0

def handle_stop():
    """Handle stop hooks"""
    safe_run_command("say 'Task complete'")
    safe_run_command("osascript -e 'display notification \"Task completed successfully\" with title \"Claude Code\" sound name \"Pop\"'")
    return 0

def handle_subagent_stop():
    """Handle subagent stop hooks"""
    safe_run_command("say 'Sub task done'")
    safe_run_command("osascript -e 'display notification \"Subagent task completed\" with title \"Claude Code\" sound name \"Blow\"'")
    return 0

def main():
    """Main entry point"""
    # Get the hook type from command line argument
    if len(sys.argv) < 2:
        print("Usage: hook_handler.py <hook_type>")
        return 0  # Return success to not block Claude
    
    hook_type = sys.argv[1]
    
    # Route to appropriate handler
    handlers = {
        'bash': handle_bash_hook,
        'task': handle_task_hook,
        'notification': handle_notification,
        'stop': handle_stop,
        'subagent_stop': handle_subagent_stop
    }
    
    handler = handlers.get(hook_type)
    if handler:
        return handler()
    else:
        print(f"Unknown hook type: {hook_type}")
        return 0  # Return success to not block Claude

if __name__ == "__main__":
    sys.exit(main())