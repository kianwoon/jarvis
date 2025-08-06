#!/usr/bin/env python3
"""
Test script to verify the workflow prompt fix is working correctly.
This script tests that the "Int Knowledge Base" workflow now uses rag_knowledge_search
instead of the hardcoded find_email/read_email tools.
"""

import sys
import os
import asyncio
import logging

# Add project root to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.automation.core.workflow_prompt_generator import generate_workflow_agent_prompt
from app.core.langgraph_agents_cache import get_agent_by_name

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_workflow_prompt_generation():
    """Test the dynamic prompt generation for workflow agents"""
    
    logger.info("=" * 60)
    logger.info("TESTING WORKFLOW PROMPT GENERATION FIX")
    logger.info("=" * 60)
    
    # Test case 1: Int Knowledge Base workflow configuration
    print("\n🧪 TEST 1: Int Knowledge Base Workflow")
    print("-" * 40)
    
    # Simulate the workflow configuration from database
    workflow_agent_name = "customer service"
    workflow_tools = ["rag_knowledge_search"]  # From workflow config
    custom_prompt = """# Takes complex reports, analyses, or strategic decisions and distills them into executive summaries, investor briefings, internal memos, or external statements, ensuring clarity, appropriate tone, and strategic messaging. FOR BACKEND ONLY Agent

## Role Overview
You are **customer service**, serving as the **Takes complex reports, analyses, or strategic decisions and distills them into executive summaries, investor briefings, internal memos, or external statements, ensuring clarity, appropriate tone, and strategic messaging. FOR BACKEND ONLY** in this multi-agent collaboration system."""
    
    # Generate dynamic prompt
    dynamic_prompt = generate_workflow_agent_prompt(
        agent_name=workflow_agent_name,
        workflow_tools=workflow_tools,
        base_system_prompt=custom_prompt,
        role="customer service",
        custom_prompt=""
    )
    
    print(f"Agent Name: {workflow_agent_name}")
    print(f"Workflow Tools: {workflow_tools}")
    print(f"\nGenerated Prompt Preview (first 500 chars):")
    print("-" * 50)
    print(dynamic_prompt[:500] + "...")
    print("-" * 50)
    
    # Check if the prompt contains the correct tools
    has_rag_tool = "rag_knowledge_search" in dynamic_prompt
    has_email_tools = "find_email" in dynamic_prompt or "read_email" in dynamic_prompt
    
    print(f"\n✅ Results:")
    print(f"   Contains rag_knowledge_search: {has_rag_tool}")
    print(f"   Contains find_email/read_email: {has_email_tools}")
    
    if has_rag_tool and not has_email_tools:
        print("   ✅ SUCCESS: Dynamic prompt generation working correctly!")
    else:
        print("   ❌ FAILURE: Still showing hardcoded email tools!")
        return False
    
    # Test case 2: Different workflow with different tools
    print("\n🧪 TEST 2: Email Workflow Simulation")
    print("-" * 40)
    
    email_tools = ["find_email", "read_email", "gmail_send"]
    email_prompt = generate_workflow_agent_prompt(
        agent_name="email_agent",
        workflow_tools=email_tools,
        base_system_prompt="You are an email management assistant.",
        role="email_assistant",
        custom_prompt=""
    )
    
    has_email_in_email_workflow = any(tool in email_prompt for tool in email_tools)
    has_rag_in_email_workflow = "rag_knowledge_search" in email_prompt
    
    print(f"Email Tools: {email_tools}")
    print(f"Contains email tools: {has_email_in_email_workflow}")
    print(f"Contains RAG tool: {has_rag_in_email_workflow}")
    
    if has_email_in_email_workflow and not has_rag_in_email_workflow:
        print("   ✅ SUCCESS: Email workflow shows correct tools!")
    else:
        print("   ❌ FAILURE: Email workflow showing wrong tools!")
        return False
    
    # Test case 3: No tools workflow
    print("\n🧪 TEST 3: No Tools Workflow")
    print("-" * 40)
    
    no_tools_prompt = generate_workflow_agent_prompt(
        agent_name="basic_agent",
        workflow_tools=[],
        base_system_prompt="You are a basic assistant.",
        role="assistant",
        custom_prompt=""
    )
    
    has_no_tools_message = "You have no specific tools available" in no_tools_prompt
    
    print(f"Tools: []")
    print(f"Contains 'no tools' message: {has_no_tools_message}")
    
    if has_no_tools_message:
        print("   ✅ SUCCESS: No tools workflow handled correctly!")
    else:
        print("   ❌ FAILURE: No tools workflow not handled properly!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! The workflow prompt fix is working correctly.")
    print("=" * 60)
    
    return True

def test_agent_loading():
    """Test that we can load the customer service agent"""
    print("\n🧪 BONUS TEST: Agent Loading")
    print("-" * 40)
    
    try:
        agent = get_agent_by_name("customer service")
        if agent:
            print(f"✅ Successfully loaded agent: {agent.get('name')}")
            print(f"   Role: {agent.get('role')}")
            print(f"   Tools: {agent.get('tools')}")
            print(f"   System prompt length: {len(agent.get('system_prompt', ''))}")
            
            # Check if the database agent still has hardcoded email tools
            system_prompt = agent.get('system_prompt', '')
            has_hardcoded_email = 'find_email' in system_prompt or 'read_email' in system_prompt
            
            if has_hardcoded_email:
                print("   ⚠️  WARNING: Database agent still has hardcoded email tools")
                print("   ℹ️  This is expected - our fix generates dynamic prompts at runtime")
            else:
                print("   ℹ️  Database agent does not contain hardcoded email tools")
                
            return True
        else:
            print("❌ Failed to load customer service agent")
            return False
            
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Workflow Prompt Fix Tests...")
    
    # Run tests
    prompt_test_passed = test_workflow_prompt_generation()
    agent_test_passed = test_agent_loading()
    
    if prompt_test_passed and agent_test_passed:
        print("\n🎯 SUMMARY: All tests passed! The fix is working correctly.")
        sys.exit(0)
    else:
        print("\n💥 SUMMARY: Some tests failed. Check the output above.")
        sys.exit(1)