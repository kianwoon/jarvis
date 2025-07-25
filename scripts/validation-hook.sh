#!/bin/bash
# PreToolUse validation hook for Claude Code - validates bash commands before execution
# Returns exit code 2 to block dangerous/invalid commands

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/validation.log"

# Dangerous command patterns to block
DANGEROUS_PATTERNS=(
    "rm -rf /"
    "rm -rf ~"
    "rm -rf \*"
    "chmod 777"
    "sudo rm"
    "dd if="
    ":(){ :|:& };:"  # Fork bomb
    "mkfs\."
    "fdisk"
    "format"
    "> /dev/sd"
    "curl.*| *sh"
    "wget.*| *sh"
    "curl.*| *bash"
    "wget.*| *bash"
)

# Suspicious patterns that require confirmation
SUSPICIOUS_PATTERNS=(
    "rm -rf"
    "sudo"
    "su -"
    "chown -R"
    "chmod -R"
    "find.*-delete"
    "find.*-exec rm"
    "> /dev/"
    "cat /etc/passwd"
    "cat /etc/shadow"
    "history -c"
    "killall"
    "pkill"
)

# Function to log validation events
log_validation() {
    local level="$1"
    local command="$2"
    local reason="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] $level: $reason" >> "$LOG_FILE"
    echo "[$timestamp] COMMAND: $command" >> "$LOG_FILE"
    echo "[$timestamp] ---" >> "$LOG_FILE"
}

# Function to check for dangerous patterns
check_dangerous_patterns() {
    local command="$1"
    
    for pattern in "${DANGEROUS_PATTERNS[@]}"; do
        if echo "$command" | grep -qE "$pattern"; then
            return 0  # Found dangerous pattern
        fi
    done
    return 1  # No dangerous patterns found
}

# Function to check for suspicious patterns
check_suspicious_patterns() {
    local command="$1"
    
    for pattern in "${SUSPICIOUS_PATTERNS[@]}"; do
        if echo "$command" | grep -qE "$pattern"; then
            return 0  # Found suspicious pattern
        fi
    done
    return 1  # No suspicious patterns found
}

# Function to validate command syntax
validate_syntax() {
    local command="$1"
    
    # Basic syntax validation using bash's built-in parser
    if bash -n -c "$command" 2>/dev/null; then
        return 0  # Valid syntax
    else
        return 1  # Invalid syntax
    fi
}

# Function to check if command exists
check_command_exists() {
    local command="$1"
    local first_word=$(echo "$command" | awk '{print $1}')
    
    # Remove common redirections and pipes to get base command
    first_word=$(echo "$first_word" | sed 's/.*|//; s/.*&&//; s/.*;//')
    
    if command -v "$first_word" >/dev/null 2>&1; then
        return 0  # Command exists
    else
        return 1  # Command not found
    fi
}

# Main validation function
validate_command() {
    local command="$1"
    
    # Check for dangerous patterns first (highest priority)
    if check_dangerous_patterns "$command"; then
        log_validation "BLOCKED" "$command" "Dangerous pattern detected"
        echo "⚠️  DANGEROUS COMMAND BLOCKED" >&2
        echo "Command contains potentially destructive patterns." >&2
        echo "Command: $command" >&2
        "$SCRIPT_DIR/alert-sound.sh" error "Dangerous command blocked"
        exit 2  # Block the command
    fi
    
    # Check syntax
    if ! validate_syntax "$command"; then
        log_validation "BLOCKED" "$command" "Invalid syntax"
        echo "❌ SYNTAX ERROR" >&2
        echo "Command has invalid bash syntax." >&2
        echo "Command: $command" >&2
        "$SCRIPT_DIR/alert-sound.sh" error "Syntax error in command"
        exit 2  # Block the command
    fi
    
    # Check if command exists
    if ! check_command_exists "$command"; then
        log_validation "WARNING" "$command" "Command not found"
        echo "⚠️  WARNING: Command may not exist" >&2
        echo "Command: $command" >&2
        "$SCRIPT_DIR/alert-sound.sh" info "Warning: Command may not exist"
        # Don't block - let it run and fail naturally
    fi
    
    # Check for suspicious patterns
    if check_suspicious_patterns "$command"; then
        log_validation "SUSPICIOUS" "$command" "Suspicious pattern detected"
        echo "⚠️  SUSPICIOUS COMMAND DETECTED" >&2
        echo "Command contains potentially risky operations." >&2
        echo "Command: $command" >&2
        "$SCRIPT_DIR/alert-sound.sh" info "Suspicious command detected"
        # Don't block suspicious commands - just warn
    fi
    
    # If we reach here, command is considered safe
    log_validation "ALLOWED" "$command" "Command validated successfully"
    "$SCRIPT_DIR/alert-sound.sh" info "Command validated"
    exit 0  # Allow the command
}

# Parse hook input from Claude Code
if [ ! -t 0 ]; then
    # Read JSON input from Claude Code hook system
    input=$(cat)
    
    # Try to extract the bash command from the hook input
    # This is a simplified JSON parser - in production you'd use jq
    if echo "$input" | grep -q '"tool":"Bash"'; then
        # Look for command field in the JSON
        command=$(echo "$input" | grep -o '"command":"[^"]*"' | sed 's/"command":"//; s/"$//')
        
        if [ -n "$command" ]; then
            # Decode any escaped characters
            command=$(echo "$command" | sed 's/\\"/"/g; s/\\n/\n/g; s/\\t/\t/g')
            validate_command "$command"
        else
            log_validation "ERROR" "" "Could not extract command from hook input"
            echo "Could not extract command from hook input" >&2
            exit 1
        fi
    else
        # Not a bash command - allow it to proceed
        log_validation "ALLOWED" "non-bash" "Non-bash tool detected"
        exit 0
    fi
else
    # Called directly with command as argument
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <command_to_validate>" >&2
        echo "   or: echo '<hook_json>' | $0" >&2
        exit 1
    fi
    
    validate_command "$*"
fi