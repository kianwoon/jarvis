#!/usr/bin/env python3
"""
Test script for the hybrid routing system
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.multi_agent_system_simple import MultiAgentSystem

async def test_hybrid_routing():
    """Test the hybrid routing system"""
    print("🧪 Testing Hybrid Agent Routing System\n")
    
    # Initialize the system
    system = MultiAgentSystem()
    
    # Test queries
    test_queries = [
        "We are responding to a bank client request on providing 3 x system engineer. We want to propose a managed service model instead of T&M model. Let's discuss",
        "Can you find documents about API security best practices?",
        "Calculate the ROI for this project and execute a cost analysis",
        "What did we discuss in our previous conversation about microservices?",
        "How do I implement OAuth2 authentication?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query[:60]}...")
        
        try:
            # Test the routing
            routing_result = await system._router_agent(query)
            
            print(f"  ✅ Selected agents: {routing_result['agents']}")
            print(f"  📝 Reasoning: {routing_result['reasoning']}")
            print(f"  🔍 Method: {'LLM-based' if 'LLM-based' in routing_result.get('reasoning', '') else 'Keyword-based'}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
        
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_hybrid_routing())