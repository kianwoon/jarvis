#!/usr/bin/env python3
"""
Test script for the reminder hook to verify all functionality.
"""

import json
import subprocess
import sys
import os

def run_hook_test(test_name, input_data, expected_output_contains=None, expected_exit_code=0):
    """Run a test case for the reminder hook."""
    print(f"🧪 Running test: {test_name}")
    
    try:
        # Run the hook script with test input
        process = subprocess.run(
            ['/Users/kianwoonwong/Downloads/jarvis/reminder-hook.py'],
            input=input_data,
            text=True,
            capture_output=True
        )
        
        # Check exit code
        if process.returncode != expected_exit_code:
            print(f"❌ FAIL: Expected exit code {expected_exit_code}, got {process.returncode}")
            return False
        
        # Check output if expected
        if expected_output_contains:
            if expected_output_contains.lower() in process.stdout.lower():
                print(f"✅ PASS: Output contains expected text")
            else:
                print(f"❌ FAIL: Expected output to contain '{expected_output_contains}', got: {process.stdout}")
                return False
        elif expected_exit_code == 0 and process.stdout.strip():
            print(f"✅ PASS: Got expected output: {process.stdout.strip()[:50]}...")
        elif expected_exit_code == 0:
            print(f"✅ PASS: No output as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception occurred: {e}")
        return False

def main():
    """Run all test cases."""
    print("🚀 Testing UserPromptSubmit reminder hook")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Valid UserPromptSubmit without existing reminder
    total_tests += 1
    test_data = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "test123",
        "prompt": "Please help me create a new feature",
        "cwd": "/Users/kianwoonwong/Downloads/jarvis"
    }
    if run_hook_test(
        "Valid UserPromptSubmit - append reminder",
        json.dumps(test_data),
        "get agent to work on",
        0
    ):
        tests_passed += 1
    
    print()
    
    # Test 2: UserPromptSubmit with existing reminder
    total_tests += 1
    test_data = {
        "hook_event_name": "UserPromptSubmit", 
        "session_id": "test456",
        "prompt": "Please get agent to work on this feature implementation",
        "cwd": "/Users/kianwoonwong/Downloads/jarvis"
    }
    if run_hook_test(
        "UserPromptSubmit with existing reminder - skip",
        json.dumps(test_data),
        None,  # Expect no output
        0
    ):
        tests_passed += 1
        
    print()
    
    # Test 3: Non-UserPromptSubmit event
    total_tests += 1
    test_data = {
        "hook_event_name": "SomeOtherEvent",
        "prompt": "test prompt"
    }
    if run_hook_test(
        "Non-UserPromptSubmit event - ignore",
        json.dumps(test_data),
        None,  # Expect no output
        0
    ):
        tests_passed += 1
        
    print()
    
    # Test 4: Invalid JSON
    total_tests += 1
    if run_hook_test(
        "Invalid JSON - handle gracefully",
        "invalid json",
        None,
        1  # Expect exit code 1
    ):
        tests_passed += 1
        
    print()
    
    # Test 5: Empty input
    total_tests += 1
    if run_hook_test(
        "Empty input - handle gracefully",
        "",
        None,
        0  # Expect exit code 0
    ):
        tests_passed += 1
    
    print()
    print("=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Hook is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())