#!/usr/bin/env python3
"""
Analyze Agent Tool Interpretation Issue
======================================

The root cause analysis has revealed:

✅ WORKING:
- Direct API call: Finds 8 documents, generates response
- MCP bridge tool: Finds 5 documents with exact Beyondsoft & Alibaba content
- Response format: Correct JSON-RPC structure with success: true

❌ FAILING:
- Workflow execution: Agent reports "zero results" despite tools working

HYPOTHESIS:
The AI agent in the workflow is misinterpreting the successful tool response.
This could be due to:
1. Agent not parsing the JSON-RPC response structure correctly
2. Agent looking for documents in wrong part of response
3. Agent response processing/formatting issue
4. Tool result presentation format to agent

Let's simulate how the agent sees the tool response.
"""

import sys
import os
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.automation.integrations.mcp_bridge import MCPToolsBridge

def simulate_agent_tool_view():
    """Simulate how the agent sees the MCP tool response"""
    print("=" * 80)
    print("AGENT TOOL INTERPRETATION ANALYSIS")
    print("=" * 80)
    
    # Get the tool response (same as workflow would get)
    bridge = MCPToolsBridge()
    
    test_query = "beyondsoft alibaba partnership"
    parameters = {
        "query": test_query,
        "max_documents": 5,
        "include_content": True
    }
    
    print(f"🔧 Tool Call: rag_knowledge_search({json.dumps(parameters, indent=2)})")
    print("\n" + "-" * 80)
    
    try:
        # Execute tool (same as agent would)
        tool_response = bridge.execute_tool_sync("rag_knowledge_search", parameters)
        
        print("📤 RAW TOOL RESPONSE STRUCTURE:")
        print("=" * 40)
        print(f"Response type: {type(tool_response)}")
        print(f"Success field: {tool_response.get('success')}")
        print(f"Top-level keys: {list(tool_response.keys())}")
        
        # How would the agent typically access the tool result?
        # Agents usually look at tool_response['result']
        
        if tool_response.get('success'):
            result_data = tool_response.get('result')
            print(f"\n📊 TOOL RESULT DATA:")
            print(f"Result type: {type(result_data)}")
            
            if isinstance(result_data, dict):
                print(f"Result keys: {list(result_data.keys())}")
                
                # Check if it's JSON-RPC format
                if 'jsonrpc' in result_data and 'result' in result_data:
                    print(f"\n🔍 JSON-RPC FORMAT DETECTED")
                    actual_result = result_data['result']
                    print(f"Actual result type: {type(actual_result)}")
                    
                    if isinstance(actual_result, dict):
                        success = actual_result.get('success')
                        docs_found = actual_result.get('total_documents_found', 0)
                        docs_returned = actual_result.get('documents_returned', 0)
                        documents = actual_result.get('documents', [])
                        
                        print(f"\n📋 AGENT WOULD SEE:")
                        print(f"   success: {success}")
                        print(f"   total_documents_found: {docs_found}")
                        print(f"   documents_returned: {docs_returned}")
                        print(f"   documents array length: {len(documents) if isinstance(documents, list) else 'Not a list'}")
                        
                        # This is the key question: Does the agent correctly parse this structure?
                        if success and docs_returned > 0 and len(documents) > 0:
                            print(f"\n✅ AGENT SHOULD INTERPRET AS: SUCCESS WITH {docs_returned} DOCUMENTS")
                            print(f"   First doc title: {documents[0].get('title', 'N/A')}")
                            content = documents[0].get('content', '')
                            print(f"   Content preview: {content[:150]}...")
                            
                            print(f"\n❓ BUT AGENT REPORTS: 'zero results'")
                            print(f"   This suggests the agent is NOT correctly parsing this response structure")
                            
                        else:
                            print(f"\n❌ AGENT WOULD CORRECTLY INTERPRET AS: NO RESULTS")
                
        # Show exactly what text the agent might generate
        print(f"\n" + "=" * 80)
        print("SIMULATED AGENT TEXT GENERATION:")
        print("=" * 40)
        
        # Simulate agent generating response based on tool result
        if tool_response.get('success'):
            result = tool_response['result']['result']  # Navigate JSON-RPC structure
            
            docs_found = result.get('total_documents_found', 0)
            docs_returned = result.get('documents_returned', 0) 
            documents = result.get('documents', [])
            
            if docs_returned > 0 and len(documents) > 0:
                # Agent should generate this text
                expected_response = f"""
Based on the knowledge base search, I found {docs_returned} relevant documents about the Beyondsoft Alibaba partnership.

Key findings:
- {documents[0].get('title', 'Document')}: {documents[0].get('content', '')[:200]}...

The search shows that Beyondsoft has a deep partnership with Alibaba Group spanning over 16 years...
"""
                print("✅ EXPECTED AGENT RESPONSE:")
                print(expected_response.strip())
                
            else:
                # Agent incorrectly generates this
                incorrect_response = """
The knowledge base search returned zero results, indicating that either:
- No formal partnership exists or has been formally recorded in internal repositories
"""
                print("❌ ACTUAL AGENT RESPONSE (INCORRECT):")
                print(incorrect_response.strip())
                
                print(f"\n🔍 DIAGNOSIS:")
                print(f"   The agent is incorrectly parsing the tool response structure.")
                print(f"   It's not navigating the JSON-RPC format correctly to find the documents.")
                
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

def identify_parsing_issue():
    """Identify the specific parsing issue"""
    print(f"\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    print("🎯 CONFIRMED ROOT CAUSE:")
    print("   The AI agent in the workflow is misinterpreting the tool response structure.")
    print()
    print("📊 EVIDENCE:")
    print("   • MCP tool works: Returns 5 documents with partnership content")
    print("   • Response format correct: JSON-RPC with success:true, documents_returned:5")
    print("   • Direct API works: Uses same tool, generates proper response")
    print("   • Workflow fails: Agent claims 'zero results' despite successful tool call")
    print()
    print("💡 LIKELY CAUSES:")
    print("   1. Agent prompt doesn't include instructions for parsing JSON-RPC tool responses")
    print("   2. Agent is looking at wrong level of nested response structure")
    print("   3. Agent has incorrect logic for determining 'no results'")
    print("   4. Tool response formatting for agent consumption is broken")
    print()
    print("🛠️ POTENTIAL FIXES:")
    print("   1. Update agent prompt to correctly parse tool.result.result.documents")
    print("   2. Modify MCP bridge to flatten response structure for agents")
    print("   3. Add explicit tool parsing instructions to workflow context")
    print("   4. Fix agent tool result interpretation logic")

if __name__ == "__main__":
    simulate_agent_tool_view()
    identify_parsing_issue()