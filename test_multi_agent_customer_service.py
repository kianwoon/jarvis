#!/usr/bin/env python3
"""
Test script to simulate a multi-agent query that would trigger 
the customer service agent and show the system_prompt being used.
"""

import os
import sys
import asyncio
import json
from typing import Dict, Any

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

async def test_multi_agent_customer_service():
    """Test multi-agent system with customer service query"""
    
    print("=" * 60)
    print("MULTI-AGENT CUSTOMER SERVICE TEST")
    print("=" * 60)
    
    try:
        from app.langchain.multi_agent_system_simple import MultiAgentSystem
        
        # Initialize the multi-agent system
        multi_agent = MultiAgentSystem()
        
        # Test query that should trigger customer service
        test_queries = [
            "I need help with my account login issues",
            "Can you help me with a billing question?",
            "I want to report a problem with my service",
            "Customer service inquiry about my recent order"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: '{query}'")
            print("-" * 50)
            
            # Create conversation ID
            conversation_id = f"test-customer-service-{i}"
            
            try:
                # This would normally be called by the API endpoint
                # We'll try to trigger the agent selection/routing logic
                
                # First, let's see what agents would be selected
                print(f"   Query: {query}")
                print(f"   Conversation ID: {conversation_id}")
                
                # Test the routing logic if available
                from app.core.langgraph_agents_cache import get_langgraph_agents
                agents = get_langgraph_agents()
                
                # Look for customer service related agents
                customer_agents = []
                for name, agent in agents.items():
                    role = agent.get('role', '').lower()
                    name_lower = name.lower()
                    if any(keyword in role or keyword in name_lower 
                           for keyword in ['customer', 'service', 'support']):
                        customer_agents.append((name, agent))
                
                if customer_agents:
                    print(f"   Found {len(customer_agents)} customer service agents:")
                    for name, agent in customer_agents:
                        print(f"   - {name}: {agent.get('role')}")
                        tools = agent.get('tools', [])
                        if tools:
                            print(f"     Tools: {tools}")
                        system_prompt = agent.get('system_prompt', '')
                        if system_prompt:
                            print(f"     System prompt length: {len(system_prompt)} chars")
                            print(f"     System prompt preview: {system_prompt[:150]}...")
                
                # Try to simulate the _dynamic_agent call
                if customer_agents:
                    agent_name, agent_info = customer_agents[0]  # Use first customer service agent
                    print(f"\n   Simulating execution of: {agent_name}")
                    
                    # Create test state
                    test_state = {
                        'query': query,
                        'conversation_id': conversation_id,
                        'messages': [],
                        'routing_decision': {'selected_agent': agent_name},
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
                        'execution_order': [agent_name],
                        'agent_metrics': {}
                    }
                    
                    print(f"   State prepared for agent: {agent_name}")
                    print(f"   Agent system prompt preview:")
                    system_prompt = agent_info.get('system_prompt', '')
                    if system_prompt:
                        print(f"   '{system_prompt[:300]}...'")
                    else:
                        print("   No system prompt found!")
                    
                    # Note: We won't actually call _dynamic_agent here as it requires
                    # full infrastructure (LLM, MCP tools, etc.)
                    print(f"   ✓ Agent configuration loaded successfully")
                    
                else:
                    print("   ⚠️  No customer service agents found!")
                    
            except Exception as e:
                print(f"   ✗ Error testing query: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("MULTI-AGENT TEST COMPLETED")
        print("=" * 60)
        
        # Summary
        print("\nSUMMARY:")
        print("- This test shows how the customer service agent would be loaded")
        print("- The actual system_prompt from the database would be used")
        print("- To see the full execution, run with the actual infrastructure")
        print("- Check logs when running real multi-agent queries for detailed output")
        
    except Exception as e:
        print(f"Error during multi-agent testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_agent_customer_service())