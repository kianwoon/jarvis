#!/usr/bin/env python3
"""
UserPromptSubmit hook script to automatically append Claude Code agent enforcement reminder
AND quality instructions to every message the user sends in Claude Code.

CRITICAL SYSTEM SEPARATION:
- Claude Code agents: .claude/agents/*.md files - FOR CLAUDE'S INTERNAL USE
- Jarvis agents: PostgreSQL database - FOR END USER @agent FEATURE

This script enforces Claude Code agent usage ONLY, not Jarvis agent usage.
Additionally appends quality development instructions for better code quality.

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


def should_add_quality_instructions(user_prompt):
    """Check if quality instructions should be added based on prompt content."""
    if not user_prompt:
        return False
    
    # Check if quality instructions are already present
    quality_indicators = [
        "no hardcode",
        "no superficial", 
        "no assumption",
        "perform web search",
        "latest implementation",
        "latest design",
        "web search when unsure"
    ]
    
    prompt_lower = user_prompt.lower()
    already_has_quality = any(indicator in prompt_lower for indicator in quality_indicators)
    
    if already_has_quality:
        return False
    
    # Check if this is a development/coding related prompt
    dev_keywords = [
        "code", "implement", "build", "create", "develop", "fix", "bug", 
        "function", "class", "method", "api", "database", "query", "script",
        "debug", "error", "exception", "test", "deploy", "config", "setup",
        "install", "package", "library", "framework", "architecture", "design",
        "algorithm", "data structure", "optimization", "performance", "security",
        "authentication", "authorization", "oauth", "jwt", "ssl", "tls",
        "docker", "kubernetes", "ci/cd", "pipeline", "automation", "workflow",
        "file", "edit", "write", "task", "agent", "coder", "programming"
    ]
    
    return any(keyword in prompt_lower for keyword in dev_keywords)


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
        
        # Check if the Claude Code agent reminder is already present to avoid duplication
        reminder_text = "Task tool"
        claude_agents_text = ".claude/agents"
        
        # Check if Claude Code agent enforcement is already mentioned
        needs_agent_reminder = not (claude_agents_text in user_prompt.lower() or reminder_text in user_prompt.lower())
        
        # Check if quality instructions should be added
        needs_quality_instructions = should_add_quality_instructions(user_prompt)
        
        # If neither reminder is needed, skip
        if not needs_agent_reminder and not needs_quality_instructions:
            log_event(f"Both Claude Code agent enforcement and quality instructions already present or not needed, skipping", "INFO")
            print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit"}}))
            sys.exit(0)
        
        # Build the combined reminder context
        reminder_sections = []
        
        # Add Claude Code agent enforcement reminder if needed
        if needs_agent_reminder:
            agent_reminder = f"""🚨 CRITICAL CLAUDE CODE AGENT ENFORCEMENT 🚨

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
            
            reminder_sections.append(agent_reminder)
        
        # Add quality instructions if needed
        if needs_quality_instructions:
            quality_reminder = f"""🎯 DEVELOPMENT QUALITY STANDARDS 🎯

CRITICAL DEVELOPMENT INSTRUCTIONS:
• no hardcode - use configuration, environment variables, or parameters
• no superficial - provide deep, thorough implementation details
• no assumption - verify requirements and validate all inputs/outputs
• perform web search when unsure - always get latest information
• confirm the implementation design is the latest - check current best practices

🔍 RESEARCH REQUIREMENTS:
• Search for latest documentation and examples
• Verify current API versions and syntax
• Check for deprecated methods or security issues
• Validate against current industry standards
• Ensure compatibility with latest framework versions

✅ QUALITY CHECKLIST:
• Configuration-driven implementation
• Comprehensive error handling
• Proper validation and sanitization
• Current best practices and patterns
• Security considerations included"""
            
            reminder_sections.append(quality_reminder)
        
        # Combine all reminder sections
        reminder_context = "\n\n" + "\n\n".join(reminder_sections)
        
        # Output JSON with hookSpecificOutput.additionalContext for Claude Code
        hook_output = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": reminder_context
            }
        }
        
        # Print JSON output that Claude Code expects
        print(json.dumps(hook_output))
        
        # Log what was added
        added_components = []
        if needs_agent_reminder:
            added_components.append("Claude Code agent enforcement")
        if needs_quality_instructions:
            added_components.append("quality instructions")
        
        log_event(f"Successfully appended: {', '.join(added_components)}", "SUCCESS")
        
    except Exception as e:
        log_event(f"Unexpected error: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()