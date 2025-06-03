#!/usr/bin/env python3
"""
Test script to verify Redis cache is working properly
"""
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def test_redis_cache():
    """Test Redis cache functionality"""
    print("=== Testing Redis Cache Integration ===")
    
    try:
        from app.core.langgraph_agents_cache import (
            get_cache_status, 
            validate_and_warm_cache, 
            get_langgraph_agents,
            get_active_agents,
            get_agent_by_name
        )
        
        # 1. Check initial cache status
        print("1. Checking initial cache status...")
        status = get_cache_status()
        print(f"Cache status: {status}")
        
        # 2. Warm cache if needed
        print("\n2. Warming cache if needed...")
        success = validate_and_warm_cache()
        print(f"Cache warming success: {success}")
        
        # 3. Check cache status after warming
        print("\n3. Checking cache status after warming...")
        status = get_cache_status()
        print(f"Cache status after warming: {status}")
        
        # 4. Test agent retrieval
        print("\n4. Testing agent retrieval...")
        all_agents = get_langgraph_agents()
        print(f"Total agents from cache: {len(all_agents)}")
        
        active_agents = get_active_agents()
        print(f"Active agents: {len(active_agents)}")
        
        if active_agents:
            print(f"Active agent names: {list(active_agents.keys())}")
            
            # 5. Test specific agent lookup
            print("\n5. Testing specific agent lookups...")
            test_agents = ["sales_strategist", "technical_architect", "financial_analyst", "service_delivery_manager"]
            
            found_agents = []
            for agent_name in test_agents:
                agent_data = get_agent_by_name(agent_name)
                if agent_data:
                    found_agents.append(agent_name)
                    print(f"✓ Found {agent_name}: {agent_data.get('description', 'No description')[:50]}...")
                else:
                    print(f"✗ NOT found: {agent_name}")
            
            print(f"\nFound {len(found_agents)}/{len(test_agents)} target agents")
            
            # 6. Test Corporate_Strategist (the one currently executing)
            print("\n6. Testing Corporate_Strategist...")
            corp_strategist = get_agent_by_name("Corporate_Strategist")
            if corp_strategist:
                print(f"✓ Corporate_Strategist found: {corp_strategist.get('description', 'No description')[:50]}...")
            else:
                print("✗ Corporate_Strategist NOT found")
            
        else:
            print("No active agents found in cache!")
            
    except Exception as e:
        print(f"Error during cache testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_redis_cache()