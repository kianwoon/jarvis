#!/usr/bin/env python3
"""
Debug Workflow RAG Response Format
================================

Deep analysis of the response format mismatch between:
1. Direct API call (works: finds 8 documents)
2. MCP bridge tool call (works: finds 5 documents)  
3. Workflow context (fails: reports "zero results")

The issue is likely in how the workflow interprets the MCP tool response.
"""

import sys
import os
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.automation.integrations.mcp_bridge import MCPToolsBridge

def analyze_response_structure():
    """Analyze the exact response structure that workflows receive"""
    print("=" * 80)
    print("DEBUG: Workflow RAG Response Format Analysis")
    print("=" * 80)
    
    # Initialize MCP bridge (same as workflow)
    bridge = MCPToolsBridge()
    
    # Test with the exact query that failed in workflow
    test_query = "beyondsoft alibaba partnership"
    parameters = {
        "query": test_query,
        "max_documents": 5,
        "include_content": True
    }
    
    print(f"Query: '{test_query}'")
    print(f"Parameters: {parameters}")
    print("\n" + "-" * 80)
    
    try:
        # Execute via MCP bridge (exact same path as workflow)
        result = bridge.execute_tool_sync("rag_knowledge_search", parameters)
        
        print("FULL RAW RESPONSE STRUCTURE:")
        print("=" * 40)
        print(f"Type: {type(result)}")
        print(f"Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print()
        
        # Analyze the response structure step by step
        if isinstance(result, dict):
            success = result.get("success")
            print(f"✅ Success field: {success}")
            
            if success:
                tool_result = result.get("result")
                print(f"📦 Tool result type: {type(tool_result)}")
                print(f"📦 Tool result keys: {list(tool_result.keys()) if isinstance(tool_result, dict) else 'Not a dict'}")
                
                # Check JSON-RPC format
                if isinstance(tool_result, dict):
                    if "result" in tool_result:
                        print("🔍 Found JSON-RPC 'result' field")
                        rag_result = tool_result["result"]
                        print(f"🔍 RAG result type: {type(rag_result)}")
                        print(f"🔍 RAG result keys: {list(rag_result.keys()) if isinstance(rag_result, dict) else 'Not a dict'}")
                        
                        if isinstance(rag_result, dict):
                            # Extract key fields that workflow might be checking
                            docs_found = rag_result.get('total_documents_found', 'MISSING')
                            docs_returned = rag_result.get('documents_returned', 'MISSING')
                            documents = rag_result.get('documents', 'MISSING')
                            collections = rag_result.get('collections_searched', 'MISSING')
                            success_flag = rag_result.get('success', 'MISSING')
                            
                            print(f"\n📊 CRITICAL FIELDS WORKFLOW MIGHT CHECK:")
                            print(f"   • success: {success_flag}")
                            print(f"   • total_documents_found: {docs_found}")
                            print(f"   • documents_returned: {docs_returned}")
                            print(f"   • documents count: {len(documents) if isinstance(documents, list) else 'Not a list'}")
                            print(f"   • collections_searched: {collections}")
                            
                            # Check if documents actually exist
                            if isinstance(documents, list) and len(documents) > 0:
                                print(f"\n🎯 DOCUMENTS ARE PRESENT!")
                                print(f"   First document title: {documents[0].get('title', 'No title')}")
                                print(f"   First document score: {documents[0].get('score', 'No score')}")
                                content = documents[0].get('content', '')
                                print(f"   First document content preview: {content[:100]}...")
                                
                                # This proves documents exist - so why does workflow report "zero results"?
                                print(f"\n❓ MYSTERY: Documents exist but workflow reports 'zero results'")
                                print(f"   This suggests the workflow is checking different fields")
                                print(f"   or interpreting the response incorrectly.")
                                
                            else:
                                print(f"\n❌ NO DOCUMENTS IN RESPONSE")
                                print(f"   This would explain the 'zero results' issue")
                    else:
                        print("❌ No 'result' field in JSON-RPC response")
                        print(f"   Available fields: {list(tool_result.keys())}")
                        
            else:
                error = result.get("error", "No error message")
                print(f"❌ Tool execution failed: {error}")
                
        print(f"\n" + "=" * 80)
        print("FULL RESPONSE DUMP (for debugging):")
        print("=" * 40)
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ EXCEPTION during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def compare_with_direct_api():
    """Compare MCP bridge result with direct API call result"""
    print(f"\n" + "=" * 80)
    print("COMPARISON: MCP Bridge vs Direct API Response")
    print("=" * 80)
    
    print("🔍 We know from testing that:")
    print("   • Direct API call: SUCCESS - finds 8 documents, generates response")
    print("   • MCP bridge call: SUCCESS - finds 5 documents in partnership collection")
    print("   • Workflow execution: FAILURE - reports 'zero results'")
    print()
    print("💡 This suggests the issue is in how the WORKFLOW INTERPRETS the MCP response,")
    print("   not in the MCP tool execution itself.")
    print()
    print("🎯 Possible causes:")
    print("   1. Workflow checks different response fields than expected")
    print("   2. Workflow expects different data structure format")
    print("   3. Workflow has incorrect success/failure logic")
    print("   4. Response gets corrupted during workflow processing")
    print("   5. Collections parameter differs between direct API and workflow")

if __name__ == "__main__":
    analyze_response_structure()
    compare_with_direct_api()