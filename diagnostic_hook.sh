#!/bin/bash

# Diagnostic hook to understand what's happening

echo "=== DIAGNOSTIC HOOK ===" >> /tmp/claude_hook_debug.log
echo "Date: $(date)" >> /tmp/claude_hook_debug.log
echo "Args: $@" >> /tmp/claude_hook_debug.log
echo "PWD: $(pwd)" >> /tmp/claude_hook_debug.log
echo "ENV PATH: $PATH" >> /tmp/claude_hook_debug.log

# Check if stdin has data
if [ -t 0 ]; then
    echo "STDIN: No data (terminal)" >> /tmp/claude_hook_debug.log
else
    stdin_data=$(cat)
    echo "STDIN: $stdin_data" >> /tmp/claude_hook_debug.log
fi

echo "Exit code will be: 0" >> /tmp/claude_hook_debug.log
echo "===================" >> /tmp/claude_hook_debug.log

# ALWAYS return success
exit 0