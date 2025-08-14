#!/usr/bin/env python3
"""
Test Script for Claude Code Hook System
========================================
Tests various user prompts to ensure correct agent selection.
"""

import json
import sys
import os
from typing import List, Dict

# Add hooks directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from user_prompt_analyzer import UserPromptAnalyzer

def run_tests():
    """Run comprehensive tests on the hook system"""
    
    analyzer = UserPromptAnalyzer()
    
    # Test cases with expected agent selections
    test_cases = [
        {
            "prompt": "Fix the database connection error in the PostgreSQL service",
            "expected_agents": ["codebase-error-analyzer", "database-administrator"],
            "expected_type": "debugging"
        },
        {
            "prompt": "Implement a new user authentication system with JWT tokens",
            "expected_agents": ["senior-coder", "coder"],
            "expected_type": "implementation"
        },
        {
            "prompt": "Create a beautiful dark theme for the dashboard UI",
            "expected_agents": ["ui-theme-designer"],
            "expected_type": "implementation"
        },
        {
            "prompt": "Optimize the LLM inference pipeline for better performance",
            "expected_agents": ["llm-ai-architect"],
            "expected_type": "optimization"
        },
        {
            "prompt": "Debug why the API returns 500 errors randomly",
            "expected_agents": ["codebase-error-analyzer"],
            "expected_type": "debugging"
        },
        {
            "prompt": "Design a scalable microservices architecture",
            "expected_agents": ["senior-coder"],
            "expected_type": "implementation"
        },
        {
            "prompt": "Help me understand how the system works",
            "expected_agents": ["general-purpose"],
            "expected_type": "general"
        },
        {
            "prompt": "URGENT: Production database is down, need immediate fix",
            "expected_agents": ["database-administrator", "codebase-error-analyzer"],
            "expected_priority": "critical"
        },
        {
            "prompt": "Create a RAG system with vector database integration",
            "expected_agents": ["llm-ai-architect", "database-administrator"],
            "expected_type": "implementation"
        },
        {
            "prompt": "Write unit tests for the payment processing module",
            "expected_agents": ["coder"],
            "expected_type": "testing"
        }
    ]
    
    print("="*80)
    print("CLAUDE CODE HOOK SYSTEM TEST")
    print("="*80)
    print()
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['prompt'][:50]}...")
        
        # Analyze the prompt
        result = analyzer.analyze_prompt(test_case["prompt"])
        
        # Get selected agent keys
        selected_agent_keys = [a["key"] for a in result["selected_agents"]]
        
        # Check task type
        detected_type = result["analysis"]["task_type"]
        
        # Check priority if specified
        detected_priority = result["analysis"]["priority"]
        
        # Validate results
        test_passed = True
        errors = []
        
        # Check if expected agents are in selected agents
        if "expected_agents" in test_case:
            for expected_agent in test_case["expected_agents"]:
                if expected_agent not in selected_agent_keys:
                    errors.append(f"  ❌ Missing expected agent: {expected_agent}")
                    test_passed = False
        
        # Check task type
        if "expected_type" in test_case:
            if detected_type != test_case["expected_type"]:
                errors.append(f"  ❌ Wrong task type: got '{detected_type}', expected '{test_case['expected_type']}'")
                test_passed = False
        
        # Check priority
        if "expected_priority" in test_case:
            if detected_priority != test_case["expected_priority"]:
                errors.append(f"  ❌ Wrong priority: got '{detected_priority}', expected '{test_case['expected_priority']}'")
                test_passed = False
        
        if test_passed:
            print(f"  ✅ PASSED")
            print(f"     Selected: {', '.join(selected_agent_keys)}")
            print(f"     Type: {detected_type}, Priority: {detected_priority}")
            passed += 1
        else:
            print(f"  ❌ FAILED")
            for error in errors:
                print(error)
            print(f"     Selected: {', '.join(selected_agent_keys)}")
            failed += 1
        
        print()
    
    # Print summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print()
    
    # Test the delegation command generation
    print("="*80)
    print("DELEGATION COMMAND TEST")
    print("="*80)
    test_prompt = "Implement a new caching system for the API"
    result = analyzer.analyze_prompt(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Generated Command:")
    print(result["delegation_command"])
    print()
    
    return passed, failed


def test_hook_execution():
    """Test the actual hook execution"""
    import subprocess
    
    print("="*80)
    print("HOOK EXECUTION TEST")
    print("="*80)
    print()
    
    test_prompts = [
        "Fix the memory leak in the user service",
        "Create a new REST API endpoint for user profiles",
        "Optimize database queries for better performance"
    ]
    
    for prompt in test_prompts:
        print(f"Testing: {prompt}")
        print("-"*40)
        
        # Run the hook script
        result = subprocess.run(
            ["python", "user-prompt-submit.py"],
            input=prompt,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Parse the output
        try:
            output = json.loads(result.stdout)
            print(f"Status: {output.get('status')}")
            print(f"Agents: {', '.join(output.get('agents_selected', []))}")
            print(f"Task Type: {output.get('task_type')}")
            print(f"Priority: {output.get('priority')}")
        except json.JSONDecodeError:
            print(f"Error parsing output: {result.stdout}")
        
        if result.stderr:
            print(f"Hook Output (stderr):")
            print(result.stderr[:500])  # Limit output for readability
        
        print()


if __name__ == "__main__":
    # Run unit tests
    passed, failed = run_tests()
    
    # Run execution tests
    test_hook_execution()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)