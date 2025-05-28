#!/usr/bin/env python3
"""Test script for LangGraph agents functionality"""

import requests
import json

BASE_URL = "http://localhost:8000/api/v1/langgraph"

def test_create_agent():
    """Test creating a new agent"""
    agent_data = {
        "name": "researcher_agent",
        "role": "researcher",
        "system_prompt": "You are a research specialist. Your role is to find and analyze information from various sources.",
        "tools": [],  # Will be populated with actual tools
        "description": "Conducts research and gathers information",
        "is_active": True
    }
    
    response = requests.post(f"{BASE_URL}/agents", json=agent_data)
    print(f"Create agent response: {response.status_code}")
    if response.ok:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    
    return response.json() if response.ok else None

def test_list_agents():
    """Test listing all agents"""
    response = requests.get(f"{BASE_URL}/agents")
    print(f"\nList agents response: {response.status_code}")
    if response.ok:
        agents = response.json()
        print(f"Found {len(agents)} agents:")
        for agent in agents:
            print(f"  - {agent['name']} ({agent['role']})")
    else:
        print(f"Error: {response.text}")

def test_get_available_tools():
    """Test getting available tools"""
    response = requests.get(f"{BASE_URL}/tools/available")
    print(f"\nAvailable tools response: {response.status_code}")
    if response.ok:
        data = response.json()
        tools = data.get('tools', [])
        print(f"Found {len(tools)} tools:")
        for tool in tools[:5]:  # Show first 5
            print(f"  - {tool['name']}: {tool['description'][:50]}...")
    else:
        print(f"Error: {response.text}")

def test_update_agent(agent_id):
    """Test updating an agent"""
    update_data = {
        "description": "Updated: Advanced research specialist with enhanced capabilities",
        "tools": ["get_datetime"]  # Add a tool if available
    }
    
    response = requests.put(f"{BASE_URL}/agents/{agent_id}", json=update_data)
    print(f"\nUpdate agent response: {response.status_code}")
    if response.ok:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")

def test_reload_cache():
    """Test reloading the Redis cache"""
    response = requests.post(f"{BASE_URL}/agents/cache/reload")
    print(f"\nReload cache response: {response.status_code}")
    if response.ok:
        print(response.json())
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing LangGraph Agents API...")
    
    # Test getting available tools first
    test_get_available_tools()
    
    # Test creating an agent
    agent = test_create_agent()
    
    # Test listing agents
    test_list_agents()
    
    # Test updating the agent if created successfully
    if agent and 'id' in agent:
        test_update_agent(agent['id'])
    
    # Test reloading cache
    test_reload_cache()