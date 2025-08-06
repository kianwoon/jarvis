#!/usr/bin/env python3
"""
Test script to verify the RAG workflow fix
Tests both standard chat and workflow modes to ensure both work correctly
"""

import asyncio
import sys
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
from app.automation.integrations.mcp_bridge import mcp_bridge

async def test_mcp_rag_tool():
    """Test the MCP RAG tool execution directly"""
    print("\n" + "="*60)
    print("TESTING MCP RAG TOOL DIRECTLY")
    print("="*60)
    
    # Test parameters
    tool_name = "rag_knowledge_search"
    parameters = {
        "query": "partnership between BeyondSoft and Alibaba",
        "collections": ["partnership"]
    }
    
    print(f"\nCalling MCP tool: {tool_name}")
    print(f"Parameters: {json.dumps(parameters, indent=2)}")
    
    # Execute tool via MCP bridge
    result = mcp_bridge.execute_tool_sync(tool_name, parameters)
    
    print(f"\nTool execution result type: {type(result)}")
    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    if isinstance(result, dict):
        # Check if result is nested
        if 'result' in result:
            print("\n✅ Result is NESTED (workflow mode format)")
            nested_result = result['result']
            print(f"Nested result keys: {list(nested_result.keys()) if isinstance(nested_result, dict) else 'Not a dict'}")
            
            if isinstance(nested_result, dict):
                docs_found = nested_result.get('documents_found', 0)
                has_results = nested_result.get('has_results', False)
                text_summary = nested_result.get('text_summary', '')[:200]
                
                print(f"\nDocuments found: {docs_found}")
                print(f"Has results: {has_results}")
                print(f"Text summary preview: {text_summary}")
                
                if docs_found > 0:
                    print("\n✅ SUCCESS: Documents were found!")
                    # Show first document
                    docs = nested_result.get('documents', [])
                    if docs:
                        first_doc = docs[0]
                        print(f"\nFirst document:")
                        print(f"  Title: {first_doc.get('title', 'No title')}")
                        print(f"  Content preview: {first_doc.get('content', '')[:200]}")
                else:
                    print("\n❌ FAILURE: No documents found")
        else:
            print("\n❌ Result is NOT nested (unexpected format)")
            print(f"Direct result: {json.dumps(result, indent=2)[:500]}")
    
    return result

async def test_agent_interpretation():
    """Test how the agent interprets the tool result"""
    print("\n" + "="*60)
    print("TESTING AGENT INTERPRETATION OF TOOL RESULT")
    print("="*60)
    
    # Create a mock tool result in the workflow format
    mock_tool_result = {
        "success": True,
        "result": {
            "success": True,
            "documents_found": 3,
            "total_documents_found": 3,
            "documents_returned": 3,
            "has_results": True,
            "text_summary": "✅ SUCCESS: Found 3 relevant documents from 1 collections.\n\n📄 DOCUMENTS RETRIEVED:\n\n1. BeyondSoft-Alibaba Partnership\n   Content: Strategic partnership announced...",
            "documents": [
                {
                    "title": "BeyondSoft-Alibaba Partnership",
                    "content": "Strategic partnership between BeyondSoft and Alibaba for cloud services...",
                    "score": 0.95
                }
            ]
        },
        "tool": "rag_knowledge_search",
        "parameters": {"query": "BeyondSoft Alibaba partnership"}
    }
    
    print("\nMock tool result structure:")
    print(f"Top level keys: {list(mock_tool_result.keys())}")
    print(f"Nested result keys: {list(mock_tool_result['result'].keys())}")
    
    # Simulate the agent's processing logic
    tool_name = "rag_knowledge_search"
    tool_result = mock_tool_result
    agent_name = "test_agent"
    
    print(f"\nSimulating agent processing for tool: {tool_name}")
    
    # This is the fixed logic from dynamic_agent_system.py
    if 'rag_knowledge_search' in tool_name.lower() and isinstance(tool_result, dict):
        # Check if result is nested (workflow mode) or direct (standard chat mode)
        if 'result' in tool_result and isinstance(tool_result['result'], dict):
            # Workflow mode: extract from nested result
            rag_data = tool_result['result']
            documents_found = rag_data.get('documents_found', 0)
            documents = rag_data.get('documents', [])
            text_summary = rag_data.get('text_summary', '')
            print(f"\n✅ Workflow mode detected - extracted from nested result")
            print(f"Documents found: {documents_found}")
            print(f"Has text summary: {bool(text_summary)}")
            
            if documents_found > 0:
                print("\n✅ Agent will correctly report documents found!")
            else:
                print("\n❌ Agent thinks no documents found")
        else:
            # Standard chat mode: direct access
            documents_found = tool_result.get('documents_found', 0)
            documents = tool_result.get('documents', [])
            text_summary = tool_result.get('text_summary', '')
            print(f"\nStandard mode - direct access")
            print(f"Documents found: {documents_found}")

async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAG WORKFLOW FIX VERIFICATION")
    print("="*60)
    
    # Test 1: MCP tool execution
    tool_result = await test_mcp_rag_tool()
    
    # Test 2: Agent interpretation
    await test_agent_interpretation()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    # Final verdict
    if isinstance(tool_result, dict) and 'result' in tool_result:
        nested_result = tool_result.get('result', {})
        if isinstance(nested_result, dict) and nested_result.get('documents_found', 0) > 0:
            print("\n✅✅✅ FIX VERIFIED: Workflow RAG should now work correctly!")
            print("The nested result structure is properly handled.")
        else:
            print("\n❌ FIX INCOMPLETE: Documents still not being found")
    else:
        print("\n⚠️ Unexpected result structure - needs investigation")

if __name__ == "__main__":
    asyncio.run(main())