#!/usr/bin/env python3
"""
Test RAG Knowledge Search Tool in Workflow Mode
===============================================

Test the rag_knowledge_search MCP tool fix by calling it through the MCP bridge.
This simulates how it would be called in workflow mode.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.automation.integrations.mcp_bridge import MCPToolsBridge

def test_rag_search_via_mcp_bridge():
    """Test RAG search through MCP bridge (workflow mode)"""
    print("=" * 60)
    print("Testing RAG Knowledge Search via MCP Bridge")
    print("=" * 60)
    
    # Initialize MCP bridge
    bridge = MCPToolsBridge()
    
    # Test parameters
    test_query = "What is artificial intelligence?"
    parameters = {
        "query": test_query,
        "max_documents": 5,
        "include_content": True
    }
    
    print(f"Query: {test_query}")
    print(f"Parameters: {parameters}")
    print("\n" + "-" * 40)
    
    try:
        # Execute via MCP bridge (same path as workflow)
        result = bridge.execute_tool_sync("rag_knowledge_search", parameters)
        
        print("RESULT:")
        print("-" * 20)
        
        if result.get("success"):
            print("✅ SUCCESS: Tool executed successfully")
            
            tool_result = result.get("result", {})
            if isinstance(tool_result, dict):
                if "result" in tool_result:
                    # JSON-RPC response format
                    rag_result = tool_result["result"]
                    if isinstance(rag_result, dict):
                        print(f"Documents found: {rag_result.get('total_documents_found', 0)}")
                        print(f"Documents returned: {rag_result.get('documents_returned', 0)}")
                        print(f"Collections searched: {rag_result.get('collections_searched', [])}")
                        print(f"Execution time: {rag_result.get('execution_time_ms', 0)}ms")
                        
                        documents = rag_result.get('documents', [])
                        if documents:
                            print(f"\nFirst document preview:")
                            first_doc = documents[0]
                            print(f"  Title: {first_doc.get('title', 'N/A')}")
                            print(f"  Score: {first_doc.get('score', 'N/A')}")
                            content = first_doc.get('content', '')
                            if content:
                                print(f"  Content: {content[:200]}...")
                    else:
                        print(f"RAG result: {rag_result}")
                else:
                    print(f"Tool result: {tool_result}")
            else:
                print(f"Result: {tool_result}")
                
        else:
            print("❌ FAILURE: Tool execution failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_rag_search_via_mcp_bridge()