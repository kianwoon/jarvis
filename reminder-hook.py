#!/usr/bin/env python3
"""
UserPromptSubmit hook script to automatically append Claude Code agent enforcement reminder
to every message the user sends in Claude Code.

CRITICAL SYSTEM SEPARATION:
- Claude Code agents: .claude/agents/*.md files - FOR CLAUDE'S INTERNAL USE
- Jarvis agents: PostgreSQL database - FOR END USER @agent FEATURE

This script enforces Claude Code agent usage ONLY, not Jarvis agent usage.

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
            # Still output empty hook response
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}}))
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
            # Still output hook response for unexpected events
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}}))
            sys.exit(0)
        
        # Get the user's prompt
        user_prompt = hook_data.get("prompt", "")
        session_id = hook_data.get("session_id", "unknown")
        
        log_event(f"Processing UserPromptSubmit for session {session_id}")
        
        # Check if the reminder is already present to avoid duplication
        reminder_text = "Task tool"
        claude_agents_text = ".claude/agents"
        
        # Skip if Claude Code agent enforcement is already mentioned
        if claude_agents_text in user_prompt.lower() or reminder_text in user_prompt.lower():
            log_event(f"Claude Code agent enforcement already mentioned, skipping additional reminder", "INFO")
            # Still output hook response when reminder already present
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}}))
            sys.exit(0)
        
        # Create the STRONGEST possible reminder message
        reminder_context = f"""

🚨 CRITICAL CLAUDE CODE AGENT ENFORCEMENT 🚨

YOU MUST USE TASK TOOL WITH CLAUDE CODE AGENTS!

⚠️ SYSTEM SEPARATION:
• Claude Code agents (.claude/agents/*.md) - FOR CLAUDE'S INTERNAL USE
• Jarvis agents (database) - FOR END USER @agent FEATURE
• DO NOT CONFUSE THESE TWO SYSTEMS!

CLAUDE CODE AGENTS (use Task tool):
• senior-coder ⭐ (RECOMMENDED for complex tasks)  
• coder (general coding)
• database-administrator (database tasks)
• codebase-error-analyzer (debugging)
• general-purpose (planning)
• llm-ai-architect (AI/ML)
• ui-theme-designer (UI/UX)

CORRECT USAGE:
Task(task="Fix the cache implementation", subagent_type="senior-coder")
Task(task="Add error handling", subagent_type="coder")

❌ NEVER use Edit, Write, MultiEdit, or NotebookEdit directly!
❌ NEVER confuse with request_agent_work.py (that's for Jarvis @agent system)!
✅ ALWAYS use Task tool with Claude Code agents!
🔴 VIOLATIONS ARE BEING MONITORED AND LOGGED!

Claude Code agent location: /Users/kianwoonwong/Downloads/jarvis/.claude/agents/
Monitoring: python /Users/kianwoonwong/Downloads/jarvis/enforcement_monitor.py --report"""
        
        # Output JSON with hookSpecificOutput.additionalContext for Claude Code
        hook_output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": reminder_context
            }
        }
        
        # Print JSON output that Claude Code expects
        print(json.dumps(hook_output))
        
        log_event(f"Successfully appended STRONG Claude Code agent enforcement reminder with system separation", "SUCCESS")
        
    except Exception as e:
        log_event(f"Unexpected error: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()