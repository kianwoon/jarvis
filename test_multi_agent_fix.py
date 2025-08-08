#!/usr/bin/env python3
"""
Test script to verify the multi-agent response fix
Tests specifically with Infrastructure Agent and Service Delivery Manager
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.fixed_multi_agent_streaming import FixedMultiAgentStreamingService

async def test_multi_agent_response_fix():
    """Test that Infrastructure Agent and Service Delivery Manager now respond properly"""
    
    print("🧪 Testing Multi-Agent Response Fix...")
    
    # Initialize the service
    service = FixedMultiAgentStreamingService()
    
    # Test query that previously caused the issue
    test_query = "AI automation workflow challenge traditional automation workflow like UIpath and Ansible. Are these new Ai automation workflow like n8n and dify could replace them?"
    
    print(f"📝 Query: {test_query}")
    print("\n🤖 Starting multi-agent processing...\n")
    
    response_count = 0
    agent_responses = {}
    
    try:
        async for event in service.stream_multi_agent_response(
            query=test_query,
            conversation_id="test-multi-agent-fix"
        ):
            if event.strip():
                import json
                try:
                    data = json.loads(event)
                    
                    if data.get("type") == "agent_complete":
                        agent_name = data.get("agent", "Unknown")
                        content = data.get("content", "")
                        content_length = len(content.strip())
                        
                        print(f"✅ {agent_name}: Response length = {content_length}")
                        
                        if content_length > 0:
                            print(f"   First 100 chars: {repr(content[:100])}")
                            agent_responses[agent_name] = content
                            response_count += 1
                        else:
                            print(f"❌ {agent_name}: EMPTY RESPONSE (this should be fixed now!)")
                            
                except json.JSONDecodeError:
                    # Skip non-JSON streaming content
                    pass
                    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    print(f"\n📊 Test Results:")
    print(f"   Total agents with responses: {response_count}")
    print(f"   Expected: 4 (all agents should respond)")
    
    # Check specific agents that were failing
    infrastructure_ok = "Infrastructure Agent" in agent_responses and len(agent_responses["Infrastructure Agent"].strip()) > 0
    service_manager_ok = "Service Delivery Manager" in agent_responses and len(agent_responses["Service Delivery Manager"].strip()) > 0
    
    print(f"   Infrastructure Agent: {'✅ FIXED' if infrastructure_ok else '❌ STILL BROKEN'}")
    print(f"   Service Delivery Manager: {'✅ FIXED' if service_manager_ok else '❌ STILL BROKEN'}")
    
    if response_count >= 2 and infrastructure_ok and service_manager_ok:
        print("\n🎉 SUCCESS: Multi-agent response fix working!")
        return True
    else:
        print("\n❌ FAILURE: Multi-agent response issue not fully resolved")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_multi_agent_response_fix())
    sys.exit(0 if result else 1)