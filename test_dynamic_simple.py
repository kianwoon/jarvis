"""
Simple test for dynamic agents with a managed services query
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.langchain.multi_agent_system_simple import MultiAgentSystem

async def test_managed_services():
    """Test with the original managed services query"""
    
    print("Testing multi-agent system with managed services query...")
    
    system = MultiAgentSystem()
    
    # Use the original test query
    query = "Client OCBC bank is requesting 3 x L1 system engineers. Currently on T&M model. How to propose managed services instead?"
    
    print(f"\nQuery: {query}\n")
    
    agent_responses = {}
    
    try:
        async for event in system.stream_events(query):
            event_type = event.get("type")
            agent = event.get("agent")
            
            if event_type == "agent_start":
                print(f"🚀 {agent} starting...")
                
            elif event_type == "agent_complete":
                content = event.get("content", "")
                if agent and content:
                    agent_responses[agent] = content
                    print(f"✅ {agent} completed ({len(content)} chars)")
                    
            elif event_type == "collaboration_pattern":
                print(f"\n🔄 Pattern: {event.get('pattern')}")
                print(f"📋 Agents: {event.get('order', [])}\n")
                
            elif event_type == "final_response":
                response = event.get("response", "")
                print(f"\n{'='*60}")
                print("FINAL SYNTHESIZED RESPONSE:")
                print("="*60)
                print(response[:1000] + "..." if len(response) > 1000 else response)
                
            elif event_type == "error":
                print(f"❌ Error in {agent}: {event.get('error')}")
                
        print(f"\n\n📊 SUMMARY:")
        print(f"Total agents responded: {len(agent_responses)}")
        for agent, response in agent_responses.items():
            print(f"\n{agent}:")
            print(f"  Preview: {response[:150]}...")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_managed_services())