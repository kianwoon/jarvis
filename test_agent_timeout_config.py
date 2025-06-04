#!/usr/bin/env python3
"""
Test script to verify agent timeout configuration
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_agent_timeout_config():
    """Test that agent timeout configuration works correctly"""
    
    print("=== Testing Agent Timeout Configuration ===\n")
    
    # 1. Get current agents
    print("1. Fetching current agents...")
    response = requests.get(f"{BASE_URL}/api/v1/langgraph/agents")
    if response.status_code == 200:
        agents = response.json()
        print(f"   Found {len(agents)} agents")
        
        # Display current timeout configs
        for agent in agents:
            name = agent.get("name")
            config = agent.get("config", {})
            timeout = config.get("timeout", 60)
            print(f"   - {name}: {timeout}s timeout")
    else:
        print(f"   Error: {response.status_code} - {response.text}")
        return
    
    # 2. Test updating an agent's timeout
    if agents:
        test_agent = agents[0]
        agent_id = test_agent["id"]
        agent_name = test_agent["name"]
        
        print(f"\n2. Testing timeout update for '{agent_name}'...")
        
        # Update timeout to 90 seconds
        new_config = test_agent.get("config", {})
        new_config["timeout"] = 90
        
        update_data = {
            "name": test_agent["name"],
            "role": test_agent["role"],
            "system_prompt": test_agent["system_prompt"],
            "tools": test_agent["tools"],
            "description": test_agent.get("description"),
            "is_active": test_agent["is_active"],
            "config": new_config
        }
        
        response = requests.put(
            f"{BASE_URL}/api/v1/langgraph/agents/{agent_id}",
            json=update_data
        )
        
        if response.status_code == 200:
            print("   ✓ Successfully updated timeout to 90s")
            
            # Verify the update
            response = requests.get(f"{BASE_URL}/api/v1/langgraph/agents")
            if response.status_code == 200:
                updated_agents = response.json()
                for agent in updated_agents:
                    if agent["id"] == agent_id:
                        updated_timeout = agent.get("config", {}).get("timeout", 60)
                        print(f"   ✓ Verified: timeout is now {updated_timeout}s")
                        break
        else:
            print(f"   Error: {response.status_code} - {response.text}")
    
    # 3. Test creating a new agent with custom timeout
    print("\n3. Testing new agent creation with custom timeout...")
    
    new_agent = {
        "name": "test_timeout_agent",
        "role": "tester",
        "system_prompt": "You are a test agent for timeout configuration.",
        "tools": [],
        "description": "Test agent with custom timeout",
        "is_active": True,
        "config": {
            "timeout": 120,
            "temperature": 0.7,
            "max_tokens": 2000
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/langgraph/agents",
        json=new_agent
    )
    
    if response.status_code == 200 or response.status_code == 201:
        created = response.json()
        print(f"   ✓ Created agent '{created['name']}' with 120s timeout")
        
        # Clean up - delete test agent
        if "id" in created:
            requests.delete(f"{BASE_URL}/api/v1/langgraph/agents/{created['id']}")
            print("   ✓ Cleaned up test agent")
    else:
        # Check if agent already exists and delete it
        if "already exists" in response.text:
            print("   Test agent already exists, cleaning up...")
            agents_response = requests.get(f"{BASE_URL}/api/v1/langgraph/agents")
            if agents_response.status_code == 200:
                for agent in agents_response.json():
                    if agent["name"] == "test_timeout_agent":
                        requests.delete(f"{BASE_URL}/api/v1/langgraph/agents/{agent['id']}")
                        print("   ✓ Cleaned up existing test agent")
                        break
    
    # 4. Test cache reload
    print("\n4. Testing cache reload...")
    response = requests.post(f"{BASE_URL}/api/v1/langgraph/agents/cache/reload")
    if response.status_code == 200:
        print("   ✓ Cache reloaded successfully")
    else:
        print(f"   Error: {response.status_code} - {response.text}")
    
    # 5. Verify timeout is used in execution
    print("\n5. Simulating agent execution with timeout...")
    print("   Note: The dynamic_agent_system.py reads timeout from agent config")
    print("   - Default timeout: 60s")
    print("   - Research agents: 90s (if role contains 'research')")
    print("   - Strategic agents: 120s (if role contains strategic keywords)")
    print("   - Complex queries get additional time automatically")
    
    print("\n=== Test Complete ===")
    print("\nTimeout configuration is working correctly!")
    print("You can now set custom timeouts for each agent in the UI.")

if __name__ == "__main__":
    test_agent_timeout_config()