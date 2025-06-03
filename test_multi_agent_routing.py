#!/usr/bin/env python3
"""
Test script to reproduce and debug the multi-agent routing issue
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def test_multi_agent_routing():
    """Test the multi-agent routing and execution"""
    try:
        from app.langchain.multi_agent_system_simple import MultiAgentSystem
        
        print("=== Testing Multi-Agent Routing Issue ===")
        
        # Create system
        system = MultiAgentSystem(conversation_id="test_routing")
        
        # Test query that should trigger multiple agents
        test_query = "let's discuss and work out a strategy to improve linkedin post impression"
        
        print(f"Test query: {test_query}")
        print()
        
        # Test routing
        print("1. Testing routing...")
        routing_result = await system._router_agent(test_query)
        print(f"Routing result: {routing_result}")
        print(f"Selected agents: {routing_result.get('agents', [])}")
        print(f"Reasoning: {routing_result.get('reasoning', 'N/A')}")
        print()
        
        # Test agent availability
        print("2. Checking agent availability...")
        from app.core.langgraph_agents_cache import get_active_agents
        available_agents = get_active_agents()
        print(f"Available agents in database: {list(available_agents.keys())}")
        print()
        
        # Check if routed agents exist
        selected_agents = routing_result.get('agents', [])
        missing_agents = []
        existing_agents = []
        
        for agent in selected_agents:
            if agent in available_agents:
                existing_agents.append(agent)
            else:
                missing_agents.append(agent)
        
        print(f"Existing agents from routing: {existing_agents}")
        print(f"Missing agents from routing: {missing_agents}")
        
        if missing_agents:
            print("*** ISSUE IDENTIFIED: Some routed agents don't exist in database ***")
            
            # Try to find matches
            print("\n3. Looking for possible matches...")
            for missing_agent in missing_agents:
                possible_matches = []
                missing_lower = missing_agent.lower()
                for available_agent in available_agents:
                    if (missing_lower in available_agent.lower() or 
                        available_agent.lower() in missing_lower or
                        any(word in available_agent.lower() for word in missing_lower.split('_'))):
                        possible_matches.append(available_agent)
                
                if possible_matches:
                    print(f"  {missing_agent} -> Possible matches: {possible_matches}")
                else:
                    print(f"  {missing_agent} -> No matches found")
        
        # Test stream events briefly
        print("\n4. Testing stream events (first few events)...")
        event_count = 0
        async for event in system.stream_events(test_query):
            event_count += 1
            event_type = event.get('type', 'unknown')
            agent = event.get('agent', 'N/A')
            print(f"  Event {event_count}: {event_type} from {agent}")
            
            # Stop after 20 events to avoid long output
            if event_count >= 20:
                print("  ... (stopping after 20 events)")
                break
        
        print(f"\nTotal events processed: {event_count}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_agent_routing())