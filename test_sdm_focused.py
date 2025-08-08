#!/usr/bin/env python3
"""
Focused test for Service Delivery Manager only
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def main():
    from app.langchain.fixed_multi_agent_streaming import FixedMultiAgentStreamingService
    
    service = FixedMultiAgentStreamingService()
    
    print("🎯 Testing Service Delivery Manager GPT-OSS Fix")
    print("=" * 50)
    
    try:
        async for event in service.stream_multi_agent_response(
            query="What are the key service delivery metrics?",
            conversation_id="test-sdm-focused"
        ):
            if event.strip():
                import json
                try:
                    data = json.loads(event)
                    event_type = data.get("type")
                    
                    if event_type == "agent_start":
                        agent = data.get("agent")
                        print(f"🚀 {agent} starting...")
                        
                    elif event_type == "agent_complete":
                        agent = data.get("agent")
                        content = data.get("content", "")
                        print(f"✅ {agent}: {len(content)} characters")
                        
                        if "Service Delivery Manager" in agent:
                            if len(content) > 0:
                                print(f"🎉 SERVICE DELIVERY MANAGER SUCCESS!")
                                print(f"   Content: {content[:200]}...")
                                return True
                            else:
                                print(f"❌ Service Delivery Manager still empty")
                                return False
                                
                except json.JSONDecodeError:
                    pass
                    
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    return False

if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nResult: {'SUCCESS' if result else 'FAILED'}")