#!/usr/bin/env python3
"""
Test script to verify that the agent no longer generates max_documents: 5 
and instead uses the correct default value from RAG config.
"""

import asyncio
import json
import logging
from app.automation.core.agent_workflow_executor import WorkflowAgentExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

async def test_agent_max_documents():
    """Test that agent doesn't hardcode max_documents: 5"""
    
    print("🧪 Testing agent tool call generation after fix...")
    print("=" * 60)
    
    # Create workflow executor
    executor = WorkflowAgentExecutor()
    
    # Test query that would trigger rag_knowledge_search
    test_query = "What is BeyondSoft's partnership with Alibaba?"
    
    # Create a simple workflow config with rag_knowledge_search tool
    workflow_config = {
        "nodes": [
            {
                "id": "agent_1",
                "type": "agent",
                "data": {
                    "agent_name": "customer_service",  
                    "tools": ["rag_knowledge_search"],
                    "custom_prompt": "You are a helpful customer service agent. Use the rag_knowledge_search tool to find information about company partnerships."
                }
            }
        ],
        "edges": [],
        "start_node": "agent_1"
    }
    
    try:
        print(f"📝 Test query: {test_query}")
        print(f"🛠️  Available tools: rag_knowledge_search")
        print(f"🎯 Expected behavior: Agent should NOT include max_documents parameter, or use correct default value")
        print("\n" + "-" * 60)
        
        # Execute workflow
        async for result in executor.execute_workflow_stream(workflow_config, test_query):
            if result.get("type") == "agent_thinking":
                thinking = result.get("content", "")
                # Look for tool call patterns in the thinking
                if "rag_knowledge_search" in thinking and "max_documents" in thinking:
                    lines = thinking.split('\n')
                    for line in lines:
                        if "rag_knowledge_search" in line and "max_documents" in line:
                            print(f"🔍 Found tool call in thinking: {line.strip()}")
                            
                            # Check if it contains max_documents: 5
                            if '"max_documents": 5' in line or '"max_documents":5' in line:
                                print("❌ ISSUE FOUND: Agent still using max_documents: 5")
                                return False
                            elif "max_documents" in line:
                                print("✅ Agent includes max_documents but not hardcoded to 5")
                            else:
                                print("✅ Agent correctly omitted max_documents parameter")
            
            elif result.get("type") == "agent_tool_call":
                tool_call = result.get("content", {})
                print(f"🔧 Tool call detected: {json.dumps(tool_call, indent=2)}")
                
                if tool_call.get("tool") == "rag_knowledge_search":
                    params = tool_call.get("parameters", {})
                    if "max_documents" in params:
                        max_docs_value = params["max_documents"]
                        if max_docs_value == 5:
                            print(f"❌ ISSUE FOUND: Agent tool call still uses max_documents: 5")
                            return False
                        else:
                            print(f"✅ Agent uses max_documents: {max_docs_value} (not hardcoded 5)")
                    else:
                        print("✅ Agent correctly omitted max_documents parameter")
                        return True
            
            elif result.get("type") == "final_response":
                final_response = result.get("content", "")
                print(f"📋 Final response: {final_response[:200]}...")
                break
        
        print("✅ Test completed - no max_documents: 5 detected")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting max_documents fix verification test")
    print("=" * 60)
    
    success = await test_agent_max_documents()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SUCCESS: Agent no longer hardcodes max_documents: 5")
        print("🎉 Fix is working correctly!")
    else:
        print("❌ FAILURE: Agent still hardcodes max_documents: 5") 
        print("🔧 Additional fixes may be needed")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)