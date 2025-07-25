#!/bin/bash
# Enhanced Claude Code hook manager
# Handles different hook events with appropriate sounds and notifications

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/hook-activity.log"

# Function to log hook activity
log_activity() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Function to handle hook events
handle_hook() {
    local event_type="$1"
    local tool_name="$2"
    local message="$3"
    
    log_activity "Hook triggered: $event_type for $tool_name - $message"
    
    case "$event_type" in
        "PreToolUse")
            "$SCRIPT_DIR/alert-sound.sh" info "Starting $tool_name"
            ;;
        "PostToolUse")
            case "$tool_name" in
                "Write"|"Edit"|"MultiEdit")
                    "$SCRIPT_DIR/alert-sound.sh" success "File operation completed"
                    ;;
                "Bash")
                    "$SCRIPT_DIR/alert-sound.sh" info "Command executed"
                    ;;
                "TodoWrite")
                    "$SCRIPT_DIR/alert-sound.sh" info "Todo list updated"
                    ;;
                "Read"|"Glob"|"Grep"|"LS")
                    # Quiet operations - no sound for file reading
                    log_activity "Quiet operation: $tool_name"
                    ;;
                *)
                    "$SCRIPT_DIR/alert-sound.sh" info "$tool_name completed"
                    ;;
            esac
            ;;
        "Stop")
            "$SCRIPT_DIR/alert-sound.sh" success "Claude task completed"
            ;;
        "UserPromptSubmit")
            "$SCRIPT_DIR/alert-sound.sh" info "Processing request"
            ;;
        "Error")
            "$SCRIPT_DIR/alert-sound.sh" error "Error occurred: $message"
            ;;
        *)
            "$SCRIPT_DIR/alert-sound.sh" info "$message"
            ;;
    esac
}

# Read input from Claude Code hook system
if [ -t 0 ]; then
    # Called directly with arguments
    handle_hook "$1" "$2" "$3"
else
    # Read JSON input from stdin (Claude hook format)
    input=$(cat)
    
    # Parse the JSON input (basic parsing for hook data)
    # In a real implementation, you might want to use jq for proper JSON parsing
    event_type=$(echo "$input" | grep -o '"event":"[^"]*"' | cut -d'"' -f4)
    tool_name=$(echo "$input" | grep -o '"tool":"[^"]*"' | cut -d'"' -f4)
    
    if [ -z "$event_type" ]; then
        event_type="Unknown"
    fi
    if [ -z "$tool_name" ]; then
        tool_name="Unknown"
    fi
    
    handle_hook "$event_type" "$tool_name" "Hook event received"
fi