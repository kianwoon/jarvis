#!/usr/bin/env python3
"""
Test script to verify the GPT OSS thinking model fix
Tests that thinking models properly extract content from <think> tags
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.fixed_multi_agent_streaming import FixedMultiAgentStreamingService

async def test_gpt_oss_thinking_fix():
    """Test that GPT OSS thinking model now properly extracts content from thinking tags"""
    
    print("🧠 Testing GPT OSS Thinking Model Fix...")
    print("🎯 This test verifies that content wrapped in <think> tags gets extracted properly\n")
    
    # Initialize the service
    service = FixedMultiAgentStreamingService()
    
    # Test query that should trigger thinking responses
    test_query = "Compare traditional RPA tools like UIPath vs modern AI workflow tools like n8n. What are the key advantages and limitations?"
    
    print(f"📝 Query: {test_query}")
    print("\n🤖 Starting multi-agent processing with GPT OSS models...\n")
    
    response_count = 0
    successful_agents = []
    failed_agents = []
    thinking_model_results = {}
    
    try:
        async for event in service.stream_multi_agent_response(
            query=test_query,
            conversation_id="test-gpt-oss-thinking-fix"
        ):
            if event.strip():
                import json
                try:
                    data = json.loads(event)
                    
                    # Track thinking model detection
                    if data.get("type") == "agent_thinking_start":
                        agent_name = data.get("agent", "Unknown")
                        print(f"🧠 THINKING DETECTED: {agent_name} is using thinking mode")
                        thinking_model_results[agent_name] = {"thinking_detected": True}
                    
                    # Track final responses
                    elif data.get("type") == "agent_complete":
                        agent_name = data.get("agent", "Unknown")
                        content = data.get("content", "")
                        content_length = len(content.strip())
                        
                        print(f"✅ {agent_name}: Final response length = {content_length}")
                        
                        if content_length > 0:
                            print(f"   📄 Preview: {repr(content[:100])}")
                            
                            # Check if content contains thinking tags (should be extracted by now)
                            has_think_tags = "<think>" in content and "</think>" in content
                            if has_think_tags:
                                print(f"   ⚠️  STILL HAS THINK TAGS: Content not properly extracted")
                                failed_agents.append(agent_name)
                            else:
                                print(f"   ✅ CLEAN CONTENT: Thinking tags properly extracted")
                                successful_agents.append(agent_name)
                                
                            response_count += 1
                            
                            if agent_name in thinking_model_results:
                                thinking_model_results[agent_name]["final_response"] = content
                                thinking_model_results[agent_name]["success"] = not has_think_tags
                        else:
                            print(f"   ❌ EMPTY RESPONSE: Fix didn't work!")
                            failed_agents.append(agent_name)
                            
                except json.JSONDecodeError:
                    # Skip non-JSON streaming content
                    pass
                    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    print(f"\n📊 GPT OSS Thinking Model Fix Test Results:")
    print(f"   Total agents with responses: {response_count}")
    print(f"   Successful agents (clean content): {len(successful_agents)}")
    print(f"   Failed agents (still have issues): {len(failed_agents)}")
    
    print(f"\n🧠 Thinking Model Analysis:")
    for agent_name, results in thinking_model_results.items():
        thinking_detected = results.get("thinking_detected", False)
        has_response = "final_response" in results
        success = results.get("success", False)
        
        print(f"   {agent_name}:")
        print(f"     - Thinking detected: {thinking_detected}")
        print(f"     - Final response: {has_response}")
        print(f"     - Content extracted properly: {success}")
    
    # Determine overall success
    thinking_models_working = len([r for r in thinking_model_results.values() if r.get("success", False)])
    total_thinking_models = len(thinking_model_results)
    
    if response_count >= 2 and thinking_models_working > 0:
        print(f"\n🎉 SUCCESS: GPT OSS thinking model fix working!")
        print(f"   {thinking_models_working}/{total_thinking_models} thinking models extracting content properly")
        return True
    else:
        print(f"\n❌ FAILURE: GPT OSS thinking model issues not fully resolved")
        print(f"   Only {thinking_models_working}/{total_thinking_models} thinking models working")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_gpt_oss_thinking_fix())
    sys.exit(0 if result else 1)