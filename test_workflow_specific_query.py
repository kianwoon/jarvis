#!/usr/bin/env python3
"""
Test RAG Knowledge Search with the Exact Query that Failed in Workflow
======================================================================

Test the same "beyondsoft alibaba partnership" query through MCP bridge
to see what the workflow is actually receiving.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.automation.integrations.mcp_bridge import MCPToolsBridge

def test_workflow_specific_query():
    """Test the exact query that failed in workflow"""
    print("=" * 60)
    print("Testing Workflow-Specific Query via MCP Bridge")
    print("=" * 60)
    
    # Initialize MCP bridge
    bridge = MCPToolsBridge()
    
    # Test parameters - exact same as the workflow that failed
    test_query = "beyondsoft alibaba partnership"
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
                        docs_found = rag_result.get('total_documents_found', 0)
                        docs_returned = rag_result.get('documents_returned', 0)
                        collections = rag_result.get('collections_searched', [])
                        exec_time = rag_result.get('execution_time_ms', 0)
                        
                        print(f"Documents found: {docs_found}")
                        print(f"Documents returned: {docs_returned}")
                        print(f"Collections searched: {collections}")
                        print(f"Execution time: {exec_time}ms")
                        
                        documents = rag_result.get('documents', [])
                        if documents:
                            print(f"\n📄 DOCUMENTS FOUND ({len(documents)}):")
                            for i, doc in enumerate(documents):
                                title = doc.get('title', 'N/A')
                                score = doc.get('score', 'N/A')
                                content = doc.get('content', '')
                                print(f"\n  Document {i+1}:")
                                print(f"    Title: {title}")
                                print(f"    Score: {score}")
                                if content:
                                    print(f"    Content Preview: {content[:300]}...")
                                else:
                                    print(f"    Content: [Empty or not available]")
                        else:
                            print("\n❌ NO DOCUMENTS RETURNED")
                            print("This explains why workflow reports 'zero results'!")
                    else:
                        print(f"RAG result: {rag_result}")
                else:
                    print(f"Tool result: {tool_result}")
            else:
                print(f"Result: {tool_result}")
                
        else:
            print("❌ FAILURE: Tool execution failed")
            error = result.get('error', 'Unknown error')
            print(f"Error: {error}")
            print("This explains why workflow reports 'zero results'!")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        print("This explains why workflow reports 'zero results'!")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_workflow_specific_query()