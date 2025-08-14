#!/usr/bin/env python3
"""
Test script to verify the reminder-hook.py works correctly with various inputs.
"""

import json
import subprocess
import sys

def test_hook(test_name, input_data):
    """Test the hook with given input data."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    # Convert input to JSON string
    input_json = json.dumps(input_data) if input_data else ""
    
    # Run the hook
    result = subprocess.run(
        ["/Users/kianwoonwong/Downloads/jarvis/reminder-hook.py"],
        input=input_json,
        capture_output=True,
        text=True
    )
    
    print(f"Input: {input_json[:100]}..." if len(input_json) > 100 else f"Input: {input_json}")
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    
    # Try to parse the output
    if result.stdout:
        try:
            output_json = json.loads(result.stdout)
            print(f"Parsed output: {json.dumps(output_json, indent=2)}")
            
            # Check if it has the expected structure
            if "hookSpecificOutput" in output_json:
                if "additionalContext" in output_json["hookSpecificOutput"]:
                    print("✅ Hook added reminder context")
                else:
                    print("✅ Hook returned valid response (no reminder needed)")
            else:
                print("❌ Invalid hook output structure")
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse output as JSON: {e}")
    else:
        print("❌ No output from hook")
    
    return result.returncode == 0

# Test cases
tests = [
    # Normal case - should add reminder
    ("Normal prompt", {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Help me fix a bug",
        "session_id": "test-session-1"
    }),
    
    # Prompt already contains reminder - should skip
    ("Prompt with reminder", {
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Help me fix a bug. get agent from agents list to work on this",
        "session_id": "test-session-2"
    }),
    
    # Wrong event type - should return empty response
    ("Wrong event type", {
        "hook_event_name": "PreToolUse",
        "prompt": "Help me fix a bug",
        "session_id": "test-session-3"
    }),
    
    # Empty input - should return empty response
    ("Empty input", None),
    
    # Invalid JSON - should exit with error
    ("Invalid JSON", "not valid json"),
    
    # Missing prompt field - should still work
    ("Missing prompt", {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "test-session-4"
    })
]

# Run all tests
results = []
for test_name, test_input in tests:
    if isinstance(test_input, str):
        # For invalid JSON test, run subprocess directly with string
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"{'='*60}")
        
        result = subprocess.run(
            ["/Users/kianwoonwong/Downloads/jarvis/reminder-hook.py"],
            input=test_input,
            capture_output=True,
            text=True
        )
        print(f"Input: {test_input}")
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        results.append((test_name, result.returncode == 1))  # Expect error
    else:
        success = test_hook(test_name, test_input)
        results.append((test_name, success))

# Summary
print(f"\n{'='*60}")
print("TEST SUMMARY")
print(f"{'='*60}")
for test_name, success in results:
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status}: {test_name}")