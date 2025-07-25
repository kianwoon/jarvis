#!/bin/bash
# Error handler for Claude Code hooks - processes command failures and provides feedback
# Usage: error-handler.sh [hook_input_via_stdin] or error-handler.sh <exit_code> <command> [error_output]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/error-handling.log"
ERROR_STATS_FILE="$SCRIPT_DIR/error-stats.json"

# Function to log error handling events
log_error_event() {
    local exit_code="$1"
    local command="$2"
    local error_type="$3"
    local action_taken="$4"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] EXIT_CODE:$exit_code TYPE:$error_type ACTION:$action_taken" >> "$LOG_FILE"
    echo "[$timestamp] COMMAND: $command" >> "$LOG_FILE"
    echo "[$timestamp] ---" >> "$LOG_FILE"
}

# Function to update error statistics
update_error_stats() {
    local error_type="$1"
    local exit_code="$2"
    
    # Create stats file if it doesn't exist
    if [ ! -f "$ERROR_STATS_FILE" ]; then
        echo '{"total_errors": 0, "error_types": {}}' > "$ERROR_STATS_FILE"
    fi
    
    # This is a simple stats updater - in production you'd use jq for proper JSON handling
    local current_total=$(grep -o '"total_errors": *[0-9]*' "$ERROR_STATS_FILE" | grep -o '[0-9]*')
    local new_total=$((current_total + 1))
    
    # Update total count (simplified - real implementation would use jq)
    sed -i.bak "s/\"total_errors\": *[0-9]*/\"total_errors\": $new_total/" "$ERROR_STATS_FILE"
    rm -f "$ERROR_STATS_FILE.bak"
}

# Function to determine error severity
get_error_severity() {
    local exit_code="$1"
    local error_output="$2"
    
    case $exit_code in
        1)
            if echo "$error_output" | grep -qi "permission denied"; then
                echo "high"
            elif echo "$error_output" | grep -qi "no such file"; then
                echo "medium"
            else
                echo "medium"
            fi
            ;;
        2)
            echo "high"  # Syntax errors
            ;;
        124)
            echo "high"  # Timeout
            ;;
        126)
            echo "high"  # Permission denied
            ;;
        127)
            echo "low"   # Command not found
            ;;
        130)
            echo "low"   # Interrupted
            ;;
        *)
            echo "medium"
            ;;
    esac
}

# Function to suggest fixes for common errors
suggest_fix() {
    local exit_code="$1"
    local command="$2"
    local error_output="$3"
    
    case $exit_code in
        127)
            echo "💡 Fix: Install the missing command or check if it's in your PATH"
            ;;
        126)
            echo "💡 Fix: Check file permissions or run with appropriate privileges"
            ;;
        1)
            if echo "$error_output" | grep -qi "permission denied"; then
                echo "💡 Fix: Check file/directory permissions or use sudo if appropriate"
            elif echo "$error_output" | grep -qi "no such file"; then
                echo "💡 Fix: Verify the file path exists or create the missing file/directory"
            elif echo "$error_output" | grep -qi "command not found"; then
                echo "💡 Fix: Install the missing command or check PATH"
            else
                echo "💡 Fix: Review the command syntax and arguments"
            fi
            ;;
        2)
            echo "💡 Fix: Check command syntax - there may be a syntax error"
            ;;
        124)
            echo "💡 Fix: Command timed out - consider optimizing or increasing timeout"
            ;;
        130)
            echo "💡 Fix: Command was interrupted - this may be intentional"
            ;;
        *)
            echo "💡 Fix: Check command documentation for exit code $exit_code meaning"
            ;;
    esac
}

# Function to handle different error types with appropriate responses
handle_error() {
    local exit_code="$1"
    local command="$2"
    local error_output="$3"
    
    # Determine error type and severity
    local error_type="unknown"
    local severity=$(get_error_severity "$exit_code" "$error_output")
    
    # Categorize the error
    case $exit_code in
        1)
            if echo "$error_output" | grep -qi "permission denied"; then
                error_type="permission"
            elif echo "$error_output" | grep -qi "no such file"; then
                error_type="filenotfound"
            elif echo "$error_output" | grep -qi "command not found"; then
                error_type="commandnotfound"
            else
                error_type="general"
            fi
            ;;
        2) error_type="syntax" ;;
        124) error_type="timeout" ;;
        126) error_type="permission" ;;
        127) error_type="commandnotfound" ;;
        130) error_type="interrupted" ;;
        *) error_type="unknown" ;;
    esac
    
    # Take appropriate action based on severity
    case $severity in
        "high")
            "$SCRIPT_DIR/alert-sound.sh" error "Critical error: $error_type"
            echo "🚨 CRITICAL ERROR (Exit Code: $exit_code)" >&2
            ;;
        "medium")
            "$SCRIPT_DIR/alert-sound.sh" error "Error: $error_type"
            echo "❌ ERROR (Exit Code: $exit_code)" >&2
            ;;
        "low")
            "$SCRIPT_DIR/alert-sound.sh" info "Minor issue: $error_type"
            echo "⚠️  WARNING (Exit Code: $exit_code)" >&2
            ;;
    esac
    
    # Display error details
    echo "Command: $command" >&2
    if [ -n "$error_output" ]; then
        echo "Error Output:" >&2
        echo "$error_output" >&2
    fi
    
    # Provide fix suggestions
    local suggestion=$(suggest_fix "$exit_code" "$command" "$error_output")
    echo "$suggestion" >&2
    
    # Log the error event
    log_error_event "$exit_code" "$command" "$error_type" "notification_sent"
    
    # Update statistics
    update_error_stats "$error_type" "$exit_code"
    
    # For Claude Code hooks - return appropriate exit code
    case $severity in
        "high")
            exit 2  # Block further execution for critical errors
            ;;
        *)
            exit 0  # Allow execution to continue for non-critical errors
            ;;
    esac
}

# Function to parse hook input from Claude Code
parse_hook_input() {
    local input="$1"
    
    # Extract information from the hook JSON
    # This is simplified parsing - production code would use jq
    local tool_name=""
    local exit_code=""
    local command=""
    local error_output=""
    
    if echo "$input" | grep -q '"tool":"Bash"'; then
        tool_name="Bash"
        # Try to extract exit code if present
        exit_code=$(echo "$input" | grep -o '"exitCode":[0-9]*' | grep -o '[0-9]*' || echo "")
        # Try to extract command
        command=$(echo "$input" | grep -o '"command":"[^"]*"' | sed 's/"command":"//; s/"$//' || echo "")
        # Try to extract error output
        error_output=$(echo "$input" | grep -o '"stderr":"[^"]*"' | sed 's/"stderr":"//; s/"$//' || echo "")
        
        # If no exit code in hook, assume it's a general error (exit code 1)
        if [ -z "$exit_code" ]; then
            exit_code="1"
        fi
        
        handle_error "$exit_code" "$command" "$error_output"
    else
        # Not a bash command error - log and continue
        log_error_event "0" "$tool_name" "non-bash" "ignored"
        echo "Non-bash tool error - no action taken" >&2
        exit 0
    fi
}

# Main execution logic
main() {
    if [ ! -t 0 ]; then
        # Reading from stdin (hook input)
        local input=$(cat)
        parse_hook_input "$input"
    else
        # Called directly with arguments
        if [ $# -lt 2 ]; then
            echo "Usage: $0 <exit_code> <command> [error_output]" >&2
            echo "   or: echo '<hook_json>' | $0" >&2
            exit 1
        fi
        
        local exit_code="$1"
        local command="$2"
        local error_output="${3:-}"
        
        handle_error "$exit_code" "$command" "$error_output"
    fi
}

# Execute main function
main "$@"