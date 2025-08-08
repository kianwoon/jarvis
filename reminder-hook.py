#!/usr/bin/env python3
"""
UserPromptSubmit hook script to automatically append "get agent to work on" reminder
to every message the user sends in Claude Code.

This script reads JSON input from stdin and outputs the reminder as additionalContext.
"""

import json
import sys
import os
from datetime import datetime


def log_event(message, level="INFO"):
    """Log events to the hook log file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "reminder-hook.log")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {level}: {message}\n"
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Failed to write to log: {e}", file=sys.stderr)


def main():
    """Main function to process UserPromptSubmit hook input and append reminder."""
    try:
        # Read JSON input from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            log_event("No input received from stdin", "WARNING")
            sys.exit(0)
        
        # Parse the JSON input from Claude Code
        try:
            hook_data = json.loads(input_data)
        except json.JSONDecodeError as e:
            log_event(f"Failed to parse JSON input: {e}", "ERROR")
            sys.exit(1)
        
        # Validate that this is a UserPromptSubmit hook
        if hook_data.get("hook_event_name") != "UserPromptSubmit":
            log_event(f"Unexpected hook event: {hook_data.get('hook_event_name', 'unknown')}", "WARNING")
            sys.exit(0)
        
        # Get the user's prompt
        user_prompt = hook_data.get("prompt", "")
        session_id = hook_data.get("session_id", "unknown")
        
        log_event(f"Processing UserPromptSubmit for session {session_id}")
        
        # Check if the reminder is already present to avoid duplication
        reminder_text = "get agent from agents list to work on"
        
        if reminder_text.lower() in user_prompt.lower():
            log_event(f"Reminder already present in prompt, skipping", "INFO")
            sys.exit(0)
        
        # Create the reminder message to append as additional context
        # This will be added to the context that Claude sees
        reminder_context = f"""

IMPORTANT REMINDER: Always get agent to work on this task. Follow the multi-agent coordination patterns in the codebase and ensure work is executed by agents from the agents list - NO EXCEPTIONS!"""
        
        # Output JSON with hookSpecificOutput.additionalContext for Claude Code
        hook_output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": reminder_context
            }
        }
        
        # Print JSON output that Claude Code expects
        print(json.dumps(hook_output))
        
        log_event(f"Successfully appended reminder to user prompt via JSON output", "SUCCESS")
        
    except Exception as e:
        log_event(f"Unexpected error: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()