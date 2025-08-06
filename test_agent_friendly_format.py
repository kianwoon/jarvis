#!/usr/bin/env python3
"""
Test Agent-Friendly Format Fix
==============================

Test the fix for the agent interpretation issue:
- MCP bridge now provides flattened, agent-friendly response format
- Agents should now correctly see documents_found, summary, etc.
"""

import sys
import os
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from app.automation.integrations.mcp_bridge import MCPToolsBridge

def test_agent_friendly_format():
    """Test the new agent-friendly response format"""
    print("=" * 80)
    print("TESTING AGENT-FRIENDLY FORMAT FIX")
    print("=" * 80)
    
    # Initialize MCP bridge
    bridge = MCPToolsBridge()
    
    # Test with the exact query that failed before
    test_query = "beyondsoft alibaba partnership"
    parameters = {
        "query": test_query,
        "max_documents": 5,
        "include_content": True
    }
    
    print(f"🔧 Tool Call: rag_knowledge_search")
    print(f"📋 Parameters: {json.dumps(parameters, indent=2)}")
    print("\n" + "-" * 80)
    
    try:
        # Execute tool with updated MCP bridge
        result = bridge.execute_tool_sync("rag_knowledge_search", parameters)
        
        print("📤 NEW AGENT-FRIENDLY RESPONSE:")
        print("=" * 40)
        print(f"✅ Success: {result.get('success')}")
        
        if result.get('success'):
            tool_result = result.get('result')
            print(f"📊 Result type: {type(tool_result)}")
            print(f"📊 Result keys: {list(tool_result.keys()) if isinstance(tool_result, dict) else 'Not a dict'}")
            
            if isinstance(tool_result, dict):
                # Test the new agent-friendly fields
                success = tool_result.get('success')
                docs_found = tool_result.get('documents_found', 0)
                docs_returned = tool_result.get('documents_returned', 0) 
                summary = tool_result.get('summary', 'No summary')
                documents = tool_result.get('documents', [])
                collections = tool_result.get('collections_searched', [])
                
                print(f"\n🎯 AGENT-FRIENDLY FIELDS:")
                print(f"   success: {success}")
                print(f"   documents_found: {docs_found}")
                print(f"   documents_returned: {docs_returned}")
                print(f"   summary: {summary}")
                print(f"   collections_searched: {collections}")
                print(f"   documents array length: {len(documents)}")
                
                if documents and len(documents) > 0:
                    print(f"\n📄 FIRST DOCUMENT:")
                    first_doc = documents[0]
                    print(f"   Title: {first_doc.get('title', 'N/A')}")
                    print(f"   Score: {first_doc.get('score', 'N/A')}")
                    content = first_doc.get('content', '')
                    print(f"   Content: {content[:200]}...")
                    
                # Simulate how agent would now interpret this
                print(f"\n🤖 AGENT INTERPRETATION:")
                if success and docs_found > 0:
                    print(f"   ✅ Agent should now see: SUCCESS with {docs_found} documents")
                    print(f"   ✅ Agent should generate proper response about partnership")
                    print(f"   ✅ Summary field helps: '{summary}'")
                else:
                    print(f"   ❌ Agent would see: No results")
                    
        else:
            print(f"❌ Tool failed: {result.get('error', 'Unknown error')}")
            
        print(f"\n" + "=" * 80)
        print("COMPARISON: OLD vs NEW FORMAT")
        print("=" * 40)
        print("❌ OLD FORMAT (caused confusion):")
        print("   • Nested JSON-RPC: result.result.documents")
        print("   • Agent had to navigate complex structure")
        print("   • No clear success indicators")
        print()
        print("✅ NEW FORMAT (agent-friendly):")
        print("   • Flat structure: result.documents_found")
        print("   • Clear success field: result.success")
        print("   • Summary field: result.summary")
        print("   • Direct document access: result.documents")
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

def simulate_workflow_success():
    """Simulate what the workflow should now produce"""
    print(f"\n" + "=" * 80)
    print("EXPECTED WORKFLOW BEHAVIOR AFTER FIX")
    print("=" * 80)
    
    print("🎯 BEFORE FIX:")
    print("   Workflow agent: 'zero results, indicating that either: No formal partnership exists'")
    print()
    print("✅ AFTER FIX:")
    print("   Workflow agent should now say:")
    print("   'Based on the knowledge base search, I found 5 relevant documents about")
    print("   the Beyondsoft Alibaba partnership. The search shows that Beyondsoft has")
    print("   a deep partnership with Alibaba Group spanning over 16 years...'")
    print()
    print("🔧 ROOT CAUSE FIXED:")
    print("   • Agent now gets clear documents_found: 5")
    print("   • Agent now gets helpful summary: 'Found 5 relevant documents'")
    print("   • Agent now has direct access to documents array")
    print("   • No more nested JSON-RPC parsing required")

if __name__ == "__main__":
    test_agent_friendly_format()
    simulate_workflow_success()