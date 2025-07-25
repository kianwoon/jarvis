#!/bin/bash
# Error detector for Claude Code hooks - monitors bash command exit codes
# Usage: error-detector.sh <command_to_monitor> [args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/command-errors.log"
TIMEOUT_SECONDS=${CLAUDE_HOOK_TIMEOUT:-30}

# Function to log errors with context
log_error() {
    local exit_code="$1"
    local command="$2"
    local stderr_output="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] EXIT_CODE:$exit_code COMMAND:$command" >> "$LOG_FILE"
    if [ -n "$stderr_output" ]; then
        echo "[$timestamp] STDERR: $stderr_output" >> "$LOG_FILE"
    fi
    echo "[$timestamp] ---" >> "$LOG_FILE"
}

# Function to categorize error types
categorize_error() {
    local exit_code="$1"
    local stderr_output="$2"
    
    case $exit_code in
        1)
            if echo "$stderr_output" | grep -qi "permission denied"; then
                echo "permission"
            elif echo "$stderr_output" | grep -qi "command not found"; then
                echo "notfound"
            elif echo "$stderr_output" | grep -qi "no such file"; then
                echo "filenotfound"
            else
                echo "general"
            fi
            ;;
        2)
            echo "syntax"
            ;;
        124)
            echo "timeout"
            ;;
        126)
            echo "permission"
            ;;
        127)
            echo "notfound"
            ;;
        130)
            echo "interrupted"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Main error detection logic
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <command> [args...]" >&2
        exit 1
    fi
    
    local command="$*"
    local temp_stderr=$(mktemp)
    local exit_code=0
    
    # Execute command with timeout and capture stderr
    if timeout "$TIMEOUT_SECONDS" bash -c "$command" 2>"$temp_stderr"; then
        exit_code=0
    else
        exit_code=$?
    fi
    
    local stderr_output=$(cat "$temp_stderr")
    rm -f "$temp_stderr"
    
    # Handle different exit codes
    if [ $exit_code -ne 0 ]; then
        local error_type=$(categorize_error "$exit_code" "$stderr_output")
        
        # Log the error
        log_error "$exit_code" "$command" "$stderr_output"
        
        # Play appropriate error sound
        case $error_type in
            "permission")
                "$SCRIPT_DIR/alert-sound.sh" error "Permission denied: $command"
                ;;
            "notfound"|"filenotfound")
                "$SCRIPT_DIR/alert-sound.sh" error "Not found: $command"
                ;;
            "timeout")
                "$SCRIPT_DIR/alert-sound.sh" error "Command timeout: $command"
                ;;
            "syntax")
                "$SCRIPT_DIR/alert-sound.sh" error "Syntax error: $command"
                ;;
            "interrupted")
                "$SCRIPT_DIR/alert-sound.sh" info "Command interrupted: $command"
                ;;
            *)
                "$SCRIPT_DIR/alert-sound.sh" error "Command failed (exit $exit_code): $command"
                ;;
        esac
        
        # For Claude Code hook system - return exit code 2 to block and show stderr
        if [ -n "$stderr_output" ]; then
            echo "Command failed with exit code $exit_code:" >&2
            echo "$stderr_output" >&2
        else
            echo "Command failed with exit code $exit_code: $command" >&2
        fi
        
        exit 2
    else
        # Success - play success sound for long-running commands only
        if [ $TIMEOUT_SECONDS -gt 5 ]; then
            "$SCRIPT_DIR/alert-sound.sh" success "Command completed: $command"
        fi
        exit 0
    fi
}

# Handle hook input from Claude Code (JSON via stdin)
if [ ! -t 0 ]; then
    # Read JSON input from Claude Code hook system
    input=$(cat)
    
    # Parse basic information from hook input
    # In practice, you might want to use jq for proper JSON parsing
    if echo "$input" | grep -q '"tool":"Bash"'; then
        # Extract command from hook input if possible
        # This is a simplified parser - real implementation would use jq
        command=$(echo "$input" | sed -n 's/.*"command":"\([^"]*\)".*/\1/p')
        if [ -n "$command" ]; then
            main "$command"
        else
            echo "Could not extract command from hook input" >&2
            exit 1
        fi
    else
        # Not a bash command - just log and continue
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Non-bash hook detected" >> "$LOG_FILE"
        exit 0
    fi
else
    # Called directly with command arguments
    main "$@"
fi