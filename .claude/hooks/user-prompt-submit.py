#!/usr/bin/env python3
"""
Claude Code User Prompt Submit Hook
====================================
This hook is triggered for every user message in Claude Code.
It ensures that Claude delegates all execution tasks to appropriate agents.
"""

import sys
import os
import json
from datetime import datetime

# Add the hooks directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the analyzer
from user_prompt_analyzer import create_hook_response, UserPromptAnalyzer

def main():
    """Main hook entry point"""
    
    # Read user prompt from stdin (Claude Code passes it this way)
    try:
        user_prompt = sys.stdin.read().strip()
    except:
        user_prompt = ""
    
    # If no prompt provided as stdin, check command line args
    if not user_prompt and len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    
    if not user_prompt:
        # Return empty response if no prompt
        print(json.dumps({
            "status": "no_prompt",
            "message": "No user prompt provided"
        }))
        return
    
    # Analyze the prompt and get recommendations
    response = create_hook_response(user_prompt)
    
    # Output the hook response
    # Claude Code expects JSON output from hooks
    if response.get("status") == "success":
        analysis = response.get("analysis", {})
        
        # Print the reminder to stderr so it appears in Claude's context
        if response.get("display_reminder"):
            print(analysis.get("reminder", ""), file=sys.stderr)
        
        # Print the delegation command for easy copying
        print("\n" + "="*60, file=sys.stderr)
        print("DELEGATION COMMAND:", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(analysis.get("delegation_command", ""), file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        
        # Return structured response for Claude Code
        output = {
            "hook": "user-prompt-submit",
            "status": "success",
            "agents_selected": [a["name"] for a in analysis.get("selected_agents", [])],
            "task_type": analysis.get("analysis", {}).get("task_type", "general"),
            "priority": analysis.get("analysis", {}).get("priority", "normal"),
            "command": analysis.get("delegation_command", ""),
            "reminder_displayed": True
        }
        
        print(json.dumps(output, indent=2))
    else:
        # Error case
        print(json.dumps(response), file=sys.stderr)

if __name__ == "__main__":
    main()