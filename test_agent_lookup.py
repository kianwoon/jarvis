#!/usr/bin/env python3
"""
Test script to verify agent lookup is working correctly
"""
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_agent_lookup():
    """Test that all agents can be found correctly"""
    try:
        from app.core.langgraph_agents_cache import get_active_agents, get_agent_by_name
        
        print("=== Testing Agent Lookup ===")
        
        # Get all available agents
        available_agents = get_active_agents()
        print(f"Total available agents: {len(available_agents)}")
        
        # Test the specific agents mentioned in routing
        test_agents = ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]
        
        print("\nTesting specific agents from routing:")
        for agent_name in test_agents:
            agent_data = get_agent_by_name(agent_name)
            if agent_data:
                print(f"✓ {agent_name}: Found - {agent_data.get('description', 'No description')[:50]}...")
            else:
                print(f"✗ {agent_name}: NOT FOUND")
                
                # Try to find similar names
                similar = []
                for available_agent in available_agents:
                    if agent_name.lower() in available_agent.lower() or available_agent.lower() in agent_name.lower():
                        similar.append(available_agent)
                
                if similar:
                    print(f"  Possible matches: {similar}")
        
        print("\nAll available agents:")
        for agent_name, agent_data in available_agents.items():
            status = "ACTIVE" if agent_data.get('is_active', True) else "INACTIVE"
            print(f"  {agent_name} ({status}): {agent_data.get('description', 'No description')[:60]}...")
        
        # Test Corporate_Strategist specifically (the one that's executing)
        print("\nTesting Corporate_Strategist (the agent that's currently executing):")
        corp_strategist = get_agent_by_name("Corporate_Strategist")
        if corp_strategist:
            print(f"✓ Corporate_Strategist: Found - {corp_strategist.get('description', 'No description')[:50]}...")
        else:
            print("✗ Corporate_Strategist: NOT FOUND")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent_lookup()