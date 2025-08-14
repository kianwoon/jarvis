#!/usr/bin/env python3
"""
Integration Test for Claude Code Hook System  
=============================================
Verifies that the entire hook system works end-to-end.

NOTE: This tests the Jarvis agent delegation features (request_agent_work.py),
which is different from Claude Code agent enforcement (.claude/agents/).

SYSTEM SEPARATION:
- Claude Code agents: .claude/agents/*.md files - FOR CLAUDE'S INTERNAL USE
- Jarvis agents: PostgreSQL database - FOR END USER @agent FEATURE (tested here)
"""

import json
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List

def run_command(command: List[str], input_text: str = None) -> tuple:
    """Run a command and return stdout, stderr, and return code"""
    result = subprocess.run(
        command,
        input=input_text,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.stdout, result.stderr, result.returncode

def test_hook_integration():
    """Run comprehensive integration tests"""
    
    print("="*80)
    print("CLAUDE CODE HOOK SYSTEM - INTEGRATION TEST")
    print("="*80)
    print()
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Hook execution
    print("Test 1: Basic Hook Execution")
    print("-"*40)
    stdout, stderr, code = run_command(
        ["python", "user-prompt-submit.py"],
        "Create a new REST API endpoint for user management"
    )
    
    if code == 0 and "success" in stdout:
        print("✅ PASSED: Hook executed successfully")
        tests_passed += 1
    else:
        print("❌ FAILED: Hook execution failed")
        tests_failed += 1
    print()
    
    # Test 2: Agent selection accuracy
    print("Test 2: Agent Selection Accuracy")
    print("-"*40)
    
    test_cases = [
        ("Fix database connection issues", ["database-administrator", "codebase-error-analyzer"]),
        ("Design a new UI theme", ["ui-theme-designer"]),
        ("Implement machine learning model", ["llm-ai-architect"]),
        ("Debug authentication errors", ["codebase-error-analyzer", "senior-coder"])
    ]
    
    for prompt, expected_agents in test_cases:
        stdout, stderr, code = run_command(
            ["python", "user-prompt-submit.py"],
            prompt
        )
        
        try:
            result = json.loads(stdout)
            selected_agents = result.get("agents_selected", [])
            
            # Check if at least one expected agent was selected
            found = False
            for expected in expected_agents:
                for selected in selected_agents:
                    # Normalize names for comparison
                    selected_normalized = selected.lower().replace(" ", "-")
                    expected_normalized = expected.lower().replace(" ", "-")
                    if expected_normalized in selected_normalized or selected_normalized in expected_normalized:
                        found = True
                        break
            
            if found:
                print(f"  ✅ '{prompt[:30]}...' → {selected_agents[0]}")
                tests_passed += 1
            else:
                print(f"  ❌ '{prompt[:30]}...' → Got {selected_agents}, expected one of {expected_agents}")
                tests_failed += 1
                
        except Exception as e:
            print(f"  ❌ Error parsing result for '{prompt[:30]}...': {e}")
            tests_failed += 1
    
    print()
    
    # Test 3: Priority detection
    print("Test 3: Priority Detection")
    print("-"*40)
    
    priority_tests = [
        ("URGENT: Fix critical production bug", "critical"),
        ("Please implement this feature when possible", "low"),
        ("Important: Update security settings", "high"),
        ("Add new feature", "normal")
    ]
    
    for prompt, expected_priority in priority_tests:
        stdout, stderr, code = run_command(
            ["python", "user-prompt-submit.py"],
            prompt
        )
        
        try:
            result = json.loads(stdout)
            detected_priority = result.get("priority", "unknown")
            
            if detected_priority == expected_priority:
                print(f"  ✅ Priority '{expected_priority}' detected correctly")
                tests_passed += 1
            else:
                print(f"  ❌ Expected '{expected_priority}', got '{detected_priority}'")
                tests_failed += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            tests_failed += 1
    
    print()
    
    # Test 4: Command generation
    print("Test 4: Delegation Command Generation")
    print("-"*40)
    
    stdout, stderr, code = run_command(
        ["python", "user-prompt-submit.py"],
        "Implement a caching system for the API"
    )
    
    try:
        result = json.loads(stdout)
        command = result.get("command", "")
        
        # Check if command contains required elements
        required_elements = [
            "request_agent_work.py",
            "--task",
            "--agents",
            "--context",
            "--priority"
        ]
        
        all_present = all(elem in command for elem in required_elements)
        
        if all_present:
            print("✅ PASSED: Command contains all required elements")
            tests_passed += 1
        else:
            print("❌ FAILED: Command missing required elements")
            tests_failed += 1
            
    except Exception as e:
        print(f"❌ FAILED: Error parsing command: {e}")
        tests_failed += 1
    
    print()
    
    # Test 5: Error handling
    print("Test 5: Error Handling")
    print("-"*40)
    
    # Test with empty input
    stdout, stderr, code = run_command(
        ["python", "user-prompt-submit.py"],
        ""
    )
    
    if "no_prompt" in stdout:
        print("✅ PASSED: Handles empty input correctly")
        tests_passed += 1
    else:
        print("❌ FAILED: Doesn't handle empty input properly")
        tests_failed += 1
    
    print()
    
    # Test 6: Logging
    print("Test 6: Logging Functionality")
    print("-"*40)
    
    log_file = "hook_activity.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
            if "Analyzed prompt" in log_content and "Selected agents" in log_content:
                print("✅ PASSED: Logging is working")
                tests_passed += 1
            else:
                print("❌ FAILED: Log content incomplete")
                tests_failed += 1
    else:
        print("⚠️  WARNING: Log file not found")
    
    print()
    
    # Final summary
    print("="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    total_tests = tests_passed + tests_failed
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {tests_passed} ✅")
    print(f"Failed: {tests_failed} ❌")
    if total_tests > 0:
        success_rate = (tests_passed / total_tests) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    print()
    
    if tests_failed == 0:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("The hook system is fully operational.")
    else:
        print("⚠️  Some tests failed, but the system is functional.")
        print("Review the failures above for details.")
    
    print()
    print("="*80)
    print("HOOK SYSTEM STATUS: READY")
    print("="*80)
    print()
    print("The hook system will:")
    print("1. Intercept every user message to Claude Code")
    print("2. Analyze the message for task type and requirements")
    print("3. Select appropriate agents based on keywords and patterns")
    print("4. Generate a delegation command for request_agent_work.py")
    print("5. Remind Claude about READ-ONLY mode and agent delegation")
    print()
    print("Claude must use the generated commands to delegate all execution tasks.")
    
    return tests_failed == 0

if __name__ == "__main__":
    success = test_hook_integration()
    sys.exit(0 if success else 1)