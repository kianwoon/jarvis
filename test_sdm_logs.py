#!/usr/bin/env python3
"""
Test Service Delivery Manager with focus on empty chunk logging
"""
import asyncio
import sys
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def main():
    print("🔍 Monitoring Service Delivery Manager GPT-OSS Behavior")
    print("=" * 60)
    
    from app.langchain.fixed_multi_agent_streaming import FixedMultiAgentStreamingService
    
    service = FixedMultiAgentStreamingService()
    
    # Simple query that should trigger Service Delivery Manager
    query = "What are service delivery best practices?"
    
    print(f"Query: {query}")
    print("Looking for Service Delivery Manager logs...")
    print()
    
    try:
        async for event in service.stream_multi_agent_response(
            query=query,
            conversation_id="test-sdm-logs"
        ):
            if event.strip():
                import json
                try:
                    data = json.loads(event)
                    
                    if data.get("type") == "agent_complete":
                        agent = data.get("agent")
                        content = data.get("content", "")
                        
                        print(f"\n📊 AGENT COMPLETE: {agent}")
                        print(f"   Content length: {len(content)}")
                        
                        if "Service Delivery Manager" in agent:
                            print(f"\n🎯 SERVICE DELIVERY MANAGER RESULT:")
                            print(f"   Success: {len(content) > 0}")
                            if len(content) > 0:
                                print(f"   Preview: {content[:200]}...")
                                print(f"\n✅ FIX SUCCESSFUL!")
                                return True
                            else:
                                print(f"   ❌ Still getting empty response")
                                return False
                                
                except json.JSONDecodeError:
                    pass
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    print("⚠️ Service Delivery Manager not found in responses")
    return False

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\n🏁 Final Result: {'SUCCESS - Service Delivery Manager Fixed' if result else 'FAILURE - Still broken'}")