#!/usr/bin/env python3
"""
Test script to check customer service agent system_prompt loading
and debug the multi-agent system configuration.
"""

import os
import sys
import asyncio
import json
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

async def test_customer_service_agent_config():
    """Test the customer service agent configuration loading"""
    
    print("=" * 60)
    print("CUSTOMER SERVICE AGENT DEBUG TEST")
    print("=" * 60)
    
    try:
        # Test 1: Check langgraph_agents cache
        print("\n1. Testing LangGraph Agents Cache...")
        from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_name
        
        agents = get_langgraph_agents()
        print(f"   Found {len(agents)} agents in cache:")
        
        customer_service_agents = []
        for name, agent in agents.items():
            print(f"   - {name}: {agent.get('role', 'Unknown role')}")
            if any(keyword in name.lower() or keyword in agent.get('role', '').lower() 
                   for keyword in ['customer', 'service', 'support']):
                customer_service_agents.append((name, agent))
        
        print(f"\n   Found {len(customer_service_agents)} potential customer service agents:")
        for name, agent in customer_service_agents:
            print(f"   - {name}: role='{agent.get('role')}', tools={agent.get('tools')}")
            print(f"     system_prompt length: {len(agent.get('system_prompt', ''))} chars")
            if len(agent.get('system_prompt', '')) > 0:
                print(f"     system_prompt preview: {agent.get('system_prompt', '')[:200]}...")
        
        # Test 2: Check specific agent retrieval
        print("\n2. Testing specific agent retrieval...")
        test_names = [
            "Customer Service Agent",
            "Customer Support Agent", 
            "Support Agent",
            "Service Agent",
            "customer service agent",  # case variation
        ]
        
        for test_name in test_names:
            agent = get_agent_by_name(test_name)
            if agent:
                print(f"   ✓ Found '{test_name}':")
                print(f"     Role: {agent.get('role')}")
                print(f"     Tools: {agent.get('tools')}")
                print(f"     System prompt length: {len(agent.get('system_prompt', ''))} chars")
                print(f"     Is active: {agent.get('is_active')}")
                print(f"     Config: {agent.get('config', {})}")
                
                # Show first 300 chars of system prompt
                system_prompt = agent.get('system_prompt', '')
                if system_prompt:
                    print(f"     System prompt (first 300 chars):")
                    print(f"     '{system_prompt[:300]}{'...' if len(system_prompt) > 300 else ''}'")
                break
            else:
                print(f"   ✗ Agent '{test_name}' not found")
        
        # Test 3: Test the multi-agent system loading
        print("\n3. Testing Multi-Agent System...")
        from app.langchain.multi_agent_system_simple import MultiAgentSystem
        
        multi_agent = MultiAgentSystem()
        
        # Create a test state
        test_state = {
            'query': 'I need help with my account',
            'conversation_id': 'test-123',
            'messages': [],
            'routing_decision': {},
            'agent_outputs': {},
            'tools_used': [],
            'documents_retrieved': [],
            'final_response': '',
            'metadata': {},
            'error': None,
            'agent_messages': [],
            'pending_requests': {},
            'agent_conversations': [],
            'execution_pattern': 'sequential',
            'agent_dependencies': {},
            'execution_order': [],
            'agent_metrics': {}
        }
        
        # Test agent loading for each potential customer service agent
        for name, agent in customer_service_agents:
            print(f"\n   Testing agent execution setup for: {name}")
            try:
                # This would normally call _dynamic_agent, but we'll just test the loading part
                print(f"   - Agent name: {name}")
                print(f"   - Agent role: {agent.get('role')}")
                print(f"   - Agent tools: {agent.get('tools')}")
                print(f"   - System prompt length: {len(agent.get('system_prompt', ''))}")
                
                if len(agent.get('system_prompt', '')) > 0:
                    print(f"   - System prompt content starts with:")
                    print(f"     '{agent.get('system_prompt', '')[:100]}...'")
                    
            except Exception as e:
                print(f"   ✗ Error testing {name}: {e}")
        
        # Test 4: Check MCP tools for customer service
        print("\n4. Testing MCP Tools for Customer Service...")
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        
        mcp_tools = get_enabled_mcp_tools()
        print(f"   Total MCP tools available: {len(mcp_tools)}")
        
        # Look for email-related tools (common for customer service)
        email_tools = [name for name in mcp_tools.keys() if 'email' in name.lower() or 'gmail' in name.lower()]
        if email_tools:
            print(f"   Email-related tools: {email_tools}")
            for tool in email_tools[:2]:  # Show first 2
                tool_info = mcp_tools[tool]
                print(f"   - {tool}: {tool_info.get('description', 'No description')}")
        
        # Look for tools configured in customer service agents
        for name, agent in customer_service_agents:
            agent_tools = agent.get('tools', [])
            if agent_tools:
                print(f"   Tools configured for {name}: {agent_tools}")
                missing_tools = [t for t in agent_tools if t not in mcp_tools]
                if missing_tools:
                    print(f"   ⚠️  Missing tools for {name}: {missing_tools}")
                else:
                    print(f"   ✓ All tools available for {name}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_customer_service_agent_config())