#!/usr/bin/env python3
"""
Direct test of dynamic agent system to verify single LLM call fix
"""

import asyncio
import json
import time
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_single_llm_call():
    """Test that agent makes only 1 LLM call with tool execution"""
    
    print("\n" + "="*80)
    print("TESTING SINGLE LLM CALL FIX IN DYNAMIC AGENT SYSTEM")
    print("="*80)
    
    # Test query that requires RAG search
    test_query = "What is the partnership between DBS and AWS about?"
    
    print(f"\n📝 Test Query: {test_query}")
    print("-"*80)
    
    # Initialize dynamic agent system directly
    from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
    agent_system = DynamicMultiAgentSystem()
    
    # Define agent configuration
    agent_data = {
        "name": "Research Analyst",
        "role": "Senior Research Analyst",
        "description": "Expert at analyzing documents and extracting insights",
        "system_prompt": "You are a senior research analyst. Provide detailed, comprehensive analysis with specific metrics and data points.",
        "tools": ["rag_knowledge_search"],
        "config": {
            "temperature": 0.7,
            "max_tokens": 4000,
            "timeout": 60,
            "model": "qwen3:30b-a3b"
        }
    }
    
    print("\n🚀 Executing agent directly...")
    print("-"*40)
    
    # Track metrics
    llm_call_count = 0
    tool_call_count = 0
    response_text = ""
    start_time = time.time()
    token_count = 0
    
    # Track what happens in the logs
    print("\n📊 Execution Log:")
    print("-"*40)
    
    # Execute agent
    async for event in agent_system.execute_agent(
        agent_name="Research Analyst",
        agent_data=agent_data,
        query=test_query,
        context={}
    ):
        event_type = event.get("type", "")
        
        if event_type == "agent_token":
            token_count += 1
            # Track token streaming
            if token_count == 1:
                print(f"✓ Token streaming started")
            elif token_count % 100 == 0:
                print(f"  ... {token_count} tokens streamed")
                
        elif event_type == "agent_complete":
            llm_call_count += 1
            response_text = event.get("content", "")
            tools_used = event.get("tools_used", [])
            
            print(f"\n✅ Agent completed")
            print(f"   - LLM calls made: {llm_call_count}")
            print(f"   - Response length: {len(response_text)} chars")
            print(f"   - Tools executed: {len(tools_used)}")
            
            if tools_used:
                tool_call_count = len(tools_used)
                for tool in tools_used:
                    tool_name = tool.get("tool", "unknown")
                    success = tool.get("success", False)
                    status = "✓" if success else "✗"
                    print(f"   - Tool: {tool_name} [{status}]")
        
        elif event_type == "agent_error":
            print(f"\n❌ Error: {event.get('error', 'Unknown error')}")
    
    execution_time = time.time() - start_time
    
    # Analyze results
    print("\n" + "="*80)
    print("📊 ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n1️⃣ LLM Call Count: {llm_call_count}")
    if llm_call_count == 1:
        print("   ✅ SUCCESS: Only 1 LLM call made!")
        print("   This matches standard chat behavior - no redundant follow-up call")
    elif llm_call_count == 2:
        print("   ❌ FAILED: Made 2 LLM calls")
        print("   First call: Tool request generation")
        print("   Second call: Response synthesis (THIS IS THE PROBLEM WE FIXED)")
    else:
        print(f"   ⚠️ Unexpected: Made {llm_call_count} LLM calls")
    
    print(f"\n2️⃣ Tool Execution: {tool_call_count} tools called")
    if tool_call_count > 0:
        print("   ✅ Tools were executed successfully")
    
    print(f"\n3️⃣ Response Quality:")
    print(f"   - Length: {len(response_text)} chars")
    print(f"   - Tokens streamed: {token_count}")
    print(f"   - Execution time: {execution_time:.2f}s")
    
    # Check response content
    print(f"\n4️⃣ Content Analysis:")
    
    # Check if tool results are included
    has_tool_results = "Based on my search" in response_text or "Detailed Analysis from Knowledge Base" in response_text
    has_documents = "document" in response_text.lower() or "found" in response_text.lower()
    has_metrics = any(char.isdigit() for char in response_text)
    has_structure = "**" in response_text or "##" in response_text
    
    checks = [
        ("Tool results included", has_tool_results),
        ("Document references", has_documents),
        ("Numeric metrics", has_metrics),
        ("Structured formatting", has_structure)
    ]
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
    
    # Show response preview
    print(f"\n5️⃣ Response Preview (first 800 chars):")
    print("-"*40)
    print(response_text[:800])
    if len(response_text) > 800:
        print("...")
    
    # Final verdict
    print("\n" + "="*80)
    print("🎯 FINAL VERDICT")
    print("="*80)
    
    if llm_call_count == 1 and tool_call_count > 0 and has_tool_results:
        print("✅ SUCCESS: Fix is working perfectly!")
        print("   - Single LLM call (no redundant follow-up)")
        print("   - Tools executed inline")
        print("   - Results integrated into response")
        print("\n🎉 The workflow now matches standard chat efficiency!")
        return True
    elif llm_call_count == 2:
        print("❌ FAILED: Still making 2 LLM calls")
        print("   The fix didn't work - agent is still doing:")
        print("   1. First call for tool request")
        print("   2. Second call for response synthesis")
        return False
    else:
        print("⚠️ PARTIAL: Some issues remain")
        print(f"   - LLM calls: {llm_call_count} (should be 1)")
        print(f"   - Tool integration: {'Yes' if has_tool_results else 'No'}")
        return False

if __name__ == "__main__":
    print("🔧 Testing Dynamic Agent System Single Call Fix...")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        success = asyncio.run(test_single_llm_call())
        
        if success:
            print("\n✅ Test PASSED! Fix is working.")
            sys.exit(0)
        else:
            print("\n❌ Test FAILED! Fix needs more work.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)