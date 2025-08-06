#!/usr/bin/env python3
"""
Test script to verify workflow agent improvements
Tests that workflow agents now provide the same quality responses as standard chat
"""

import asyncio
import json
from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem

async def test_workflow_agent_response():
    """Test that workflow agents provide comprehensive responses with RAG content"""
    
    print("="*80)
    print("TESTING WORKFLOW AGENT IMPROVEMENTS")
    print("="*80)
    
    # Initialize the dynamic agent system
    system = DynamicMultiAgentSystem()
    
    # Test query that should trigger RAG search
    test_query = "What is the partnership between Alibaba and Salesforce about?"
    
    # Create a test agent with RAG tool
    test_agent = {
        "name": "research_analyst",
        "role": "Research Analyst",
        "system_prompt": "You are a research analyst who provides detailed analysis based on available information.",
        "tools": ["rag_knowledge_search"],
        "config": {
            "model": "qwen2.5:14b",
            "temperature": 0.7,
            "max_tokens": 8192
        }
    }
    
    print(f"\nTest Query: {test_query}")
    print(f"Test Agent: {test_agent['name']}")
    print(f"Available Tools: {test_agent['tools']}")
    print("\n" + "-"*40)
    
    # Execute the agent
    response = ""
    tool_results = []
    
    async for event in system.execute_agent(
        agent_name=test_agent['name'],
        agent_data=test_agent,
        query=test_query,
        context={}
    ):
        if event.get("type") == "agent_complete":
            response = event.get("content", "")
            tool_results = event.get("tools_used", [])
            break
        elif event.get("type") == "agent_token":
            # Could stream tokens here if needed
            pass
    
    # Analyze the response
    print("\nRESPONSE ANALYSIS:")
    print("-"*40)
    
    # Check response length
    response_length = len(response)
    print(f"✓ Response Length: {response_length} characters")
    
    # Check if tools were used
    tools_used = len(tool_results) > 0
    print(f"✓ Tools Used: {'Yes' if tools_used else 'No'}")
    
    if tool_results:
        for result in tool_results:
            tool_name = result.get("tool", "Unknown")
            success = result.get("success", False)
            print(f"  - {tool_name}: {'Success' if success else 'Failed'}")
    
    # Check for specific content indicators
    quality_indicators = {
        "Has specific years/dates": any(year in response for year in ["2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]),
        "Mentions team sizes": any(term in response.lower() for term in ["engineer", "team", "staff", "employee"]),
        "Has partnership details": any(term in response.lower() for term in ["partnership", "collaboration", "joint", "together"]),
        "References documents": any(term in response.lower() for term in ["according to", "based on", "document", "shows", "indicates"]),
        "Has business domains": any(term in response.lower() for term in ["cloud", "commerce", "taobao", "tmall", "ant"]),
        "Comprehensive (>1000 chars)": response_length > 1000
    }
    
    print("\nQUALITY INDICATORS:")
    for indicator, present in quality_indicators.items():
        status = "✓" if present else "✗"
        print(f"{status} {indicator}")
    
    # Calculate quality score
    quality_score = sum(quality_indicators.values()) / len(quality_indicators) * 100
    print(f"\nOverall Quality Score: {quality_score:.1f}%")
    
    # Show response preview
    print("\nRESPONSE PREVIEW (first 500 chars):")
    print("-"*40)
    print(response[:500] + "..." if len(response) > 500 else response)
    
    # Final verdict
    print("\n" + "="*80)
    if quality_score >= 70 and tools_used and response_length > 500:
        print("✅ TEST PASSED: Workflow agent provides comprehensive response with tool integration")
    else:
        print("❌ TEST FAILED: Response quality needs improvement")
        print(f"   - Quality Score: {quality_score:.1f}% (need >=70%)")
        print(f"   - Tools Used: {tools_used} (need True)")
        print(f"   - Response Length: {response_length} (need >500)")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(test_workflow_agent_response())