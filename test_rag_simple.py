#!/usr/bin/env python3
"""
Simple test to verify RAG workflow fix
"""

import requests
import json
import time

def test_standard_rag():
    """Test standard RAG endpoint"""
    print("\n" + "="*60)
    print("TESTING STANDARD RAG QUERY")
    print("="*60)
    
    url = 'http://localhost:8000/api/v1/langchain/rag'
    payload = {
        'question': 'Tell me about the partnership between BeyondSoft and Alibaba',
        'thinking': False,
        'collections': ['partnership']
    }
    
    print(f"\nEndpoint: {url}")
    print(f"Query: {payload['question']}")
    print(f"Collections: {payload['collections']}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            # Handle streaming response
            lines = response.text.strip().split('\n')
            final_answer = None
            
            for line in lines:
                try:
                    data = json.loads(line)
                    if 'answer' in data:
                        final_answer = data['answer']
                except:
                    pass
            
            if final_answer:
                print("\n✅ Response received!")
                print(f"\nAnswer preview (first 500 chars):")
                print(final_answer[:500])
                
                # Check if documents were found
                if 'beyondsoft' in final_answer.lower() and 'alibaba' in final_answer.lower():
                    print("\n✅✅✅ SUCCESS: Standard RAG found the BeyondSoft-Alibaba partnership!")
                    return True
                elif 'no documents' in final_answer.lower() or 'not found' in final_answer.lower():
                    print("\n❌ FAILURE: Standard RAG claims no documents found")
                    return False
                else:
                    print("\n⚠️ Unclear response - check manually")
                    return None
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text[:500])
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def test_mcp_tool_directly():
    """Test MCP tool execution directly via internal endpoint"""
    print("\n" + "="*60)
    print("TESTING MCP TOOL EXECUTION")
    print("="*60)
    
    url = 'http://localhost:8000/api/v1/mcp-tools/execute'
    payload = {
        'tool_name': 'rag_knowledge_search',
        'parameters': {
            'query': 'partnership between BeyondSoft and Alibaba',
            'collections': ['partnership']
        }
    }
    
    print(f"\nEndpoint: {url}")
    print(f"Tool: {payload['tool_name']}")
    print(f"Parameters: {json.dumps(payload['parameters'], indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Tool executed successfully!")
            
            # Check result structure
            print(f"\nResult type: {type(result)}")
            print(f"Top-level keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Check for nested structure
            if isinstance(result, dict) and 'result' in result:
                print("\n✅ Result has nested structure (workflow mode)")
                nested = result['result']
                if isinstance(nested, dict):
                    docs_found = nested.get('documents_found', 0)
                    has_results = nested.get('has_results', False)
                    print(f"Documents found: {docs_found}")
                    print(f"Has results: {has_results}")
                    
                    if docs_found > 0:
                        print("\n✅✅✅ SUCCESS: MCP tool found documents!")
                        # Show first document
                        docs = nested.get('documents', [])
                        if docs:
                            print(f"\nFirst document title: {docs[0].get('title', 'No title')}")
                            print(f"Content preview: {docs[0].get('content', '')[:200]}")
                        return True
                    else:
                        print("\n❌ FAILURE: MCP tool found no documents")
                        return False
            else:
                print("\n⚠️ Unexpected result structure")
                print(json.dumps(result, indent=2)[:500])
                return None
                
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text[:500])
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAG WORKFLOW FIX VERIFICATION")
    print("="*60)
    
    # Test 1: Standard RAG
    standard_result = test_standard_rag()
    
    # Wait a bit between tests
    time.sleep(2)
    
    # Test 2: MCP Tool
    mcp_result = test_mcp_tool_directly()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if standard_result and mcp_result:
        print("\n✅✅✅ ALL TESTS PASSED!")
        print("Both standard RAG and MCP tool execution work correctly.")
        print("The workflow should now properly retrieve documents.")
    elif standard_result and not mcp_result:
        print("\n⚠️ PARTIAL SUCCESS")
        print("Standard RAG works but MCP tool has issues.")
        print("Workflow may still have problems.")
    elif not standard_result and mcp_result:
        print("\n⚠️ UNEXPECTED RESULT")
        print("MCP tool works but standard RAG doesn't.")
        print("This is unusual - check configuration.")
    else:
        print("\n❌ TESTS FAILED")
        print("Neither standard RAG nor MCP tool are working properly.")
        print("Check if services are running and data is indexed.")

if __name__ == "__main__":
    main()