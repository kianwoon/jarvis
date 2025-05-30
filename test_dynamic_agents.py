#!/usr/bin/env python
"""Test dynamic agent execution directly"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem

async def test_dynamic_agent():
    system = DynamicMultiAgentSystem()
    
    # Test PreSalesArchitect directly
    agent_name = "PreSalesArchitect"
    query = "What are the key differences between MariaDB and OceanBase?"
    
    print(f"Testing {agent_name} with query: {query}")
    
    try:
        response = await system.execute_agent(agent_name, query)
        print(f"\nResponse from {agent_name}:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_dynamic_agent())