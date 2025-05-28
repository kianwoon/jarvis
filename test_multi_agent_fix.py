"""
Test script to verify multi-agent chat functionality after fixes
"""

import asyncio
import json
from app.langchain.multi_agent_system_simple import MultiAgentSystem

async def test_multi_agent_streaming():
    """Test multi-agent streaming with concurrent agent execution"""
    system = MultiAgentSystem()
    
    test_queries = [
        "What are the benefits of using a multi-agent system for complex queries?",
        "Client OCBC bank is requesting 3 x L1 system engineers. Currently on T&M model. How to propose managed services instead?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing query: {query}")
        print(f"{'='*80}\n")
        
        agent_outputs = {}
        streaming_agents = set()
        completed_agents = set()
        
        try:
            async for event in system.stream_events(query):
                event_type = event.get("type")
                agent = event.get("agent")
                
                if event_type == "agent_start":
                    print(f"🚀 Starting agent: {agent}")
                    streaming_agents.add(agent)
                    
                elif event_type == "agent_streaming":
                    # Don't print every token, just track that agent is streaming
                    if agent not in agent_outputs:
                        agent_outputs[agent] = ""
                        print(f"📝 {agent} is streaming response...")
                    agent_outputs[agent] = event.get("partial_content", "")
                    
                elif event_type == "agent_complete":
                    completed_agents.add(agent)
                    streaming_agents.discard(agent)
                    content = event.get("content", "")
                    print(f"✅ {agent} completed (response length: {len(content)} chars)")
                    
                elif event_type == "final_response":
                    print(f"\n📊 Final Response:")
                    print("-" * 40)
                    print(event.get("response", "No response"))
                    
                elif event_type == "error":
                    print(f"❌ Error in {agent}: {event.get('error')}")
                    
            print(f"\n📈 Summary:")
            print(f"- Total agents involved: {len(completed_agents)}")
            print(f"- Agents: {', '.join(completed_agents)}")
            print(f"- All agents completed successfully: {'Yes' if len(streaming_agents) == 0 else 'No'}")
            
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

async def test_concurrent_execution():
    """Test that multiple agents execute concurrently"""
    system = MultiAgentSystem()
    query = "Analyze the pros and cons of microservices architecture"
    
    print(f"\n{'='*80}")
    print("Testing concurrent agent execution")
    print(f"{'='*80}\n")
    
    agent_start_times = {}
    agent_end_times = {}
    
    import time
    
    async for event in system.stream_events(query):
        current_time = time.time()
        event_type = event.get("type")
        agent = event.get("agent")
        
        if event_type == "agent_start" and agent:
            agent_start_times[agent] = current_time
            print(f"⏰ {agent} started at {current_time:.2f}")
            
        elif event_type == "agent_complete" and agent:
            agent_end_times[agent] = current_time
            duration = current_time - agent_start_times.get(agent, current_time)
            print(f"⏰ {agent} completed at {current_time:.2f} (duration: {duration:.2f}s)")
    
    # Check for overlapping execution times
    agents = list(agent_start_times.keys())
    overlapping_pairs = []
    
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            agent1, agent2 = agents[i], agents[j]
            if agent1 in agent_start_times and agent2 in agent_start_times:
                # Check if execution times overlap
                start1, end1 = agent_start_times[agent1], agent_end_times.get(agent1, float('inf'))
                start2, end2 = agent_start_times[agent2], agent_end_times.get(agent2, float('inf'))
                
                if not (end1 < start2 or end2 < start1):
                    overlapping_pairs.append((agent1, agent2))
    
    print(f"\n🔄 Concurrent execution analysis:")
    print(f"- Total agents: {len(agents)}")
    print(f"- Overlapping agent pairs: {len(overlapping_pairs)}")
    if overlapping_pairs:
        print("- Concurrent pairs:")
        for pair in overlapping_pairs:
            print(f"  • {pair[0]} ↔️ {pair[1]}")

if __name__ == "__main__":
    print("Starting multi-agent system tests...\n")
    asyncio.run(test_multi_agent_streaming())
    asyncio.run(test_concurrent_execution())
    print("\n✅ All tests completed!")