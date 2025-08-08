#!/usr/bin/env python3
"""
Simple test for Service Delivery Manager GPT-OSS fix
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_service_delivery_manager_fix():
    """Simple test to see if the fix works"""
    
    print("🧪 Testing Service Delivery Manager GPT-OSS Fix")
    print("=" * 60)
    
    try:
        from app.langchain.fixed_multi_agent_streaming import FixedMultiAgentStreamingService
        
        service = FixedMultiAgentStreamingService()
        test_query = "What are the key metrics for measuring service delivery effectiveness?"
        
        print(f"📋 Test Query: {test_query}")
        print(f"🎯 Expected: Service Delivery Manager should now work with GPT-OSS")
        print()
        
        agent_responses = {}
        response_count = 0
        
        try:
            async for event in service.stream_multi_agent_response(
                query=test_query,
                conversation_id="test-sdm-fix"
            ):
                if event.strip():
                    import json
                    try:
                        data = json.loads(event)
                        
                        # Look for agent completion
                        if data.get("type") == "agent_complete":
                            agent_name = data.get("agent", "Unknown")
                            content = data.get("content", "")
                            content_length = len(content.strip())
                            
                            print(f"✅ {agent_name}: Response length = {content_length}")
                            
                            if content_length > 0:
                                agent_responses[agent_name] = content
                                response_count += 1
                                
                                # Show preview for Service Delivery Manager
                                if "Service Delivery Manager" in agent_name:
                                    print(f"   🎯 SERVICE DELIVERY MANAGER PREVIEW:")
                                    print(f"      {repr(content[:150])}...")
                            else:
                                print(f"❌ {agent_name}: EMPTY RESPONSE")
                                
                    except json.JSONDecodeError:
                        # Skip non-JSON streaming content
                        pass
                        
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
        
        print(f"\n📊 Test Results:")
        print(f"   Total agents with responses: {response_count}")
        print(f"   Agents that responded: {list(agent_responses.keys())}")
        
        # Check specifically for Service Delivery Manager
        service_mgr_ok = "Service Delivery Manager" in agent_responses and len(agent_responses["Service Delivery Manager"].strip()) > 0
        
        print(f"\n🎯 Service Delivery Manager Status:")
        print(f"   {'✅ FIXED' if service_mgr_ok else '❌ STILL BROKEN'}")
        
        if service_mgr_ok:
            print(f"\n🎉 SUCCESS: Service Delivery Manager fix is working!")
            print(f"   Response length: {len(agent_responses['Service Delivery Manager'])}")
            return True
        else:
            print(f"\n❌ FAILURE: Service Delivery Manager still not working")
            return False
            
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_service_delivery_manager_fix())
    print(f"\n🏁 Final Result: {'SUCCESS' if result else 'FAILURE'}")