#!/usr/bin/env python3
"""
Test Agent Response Interpretation
==================================

Simulate how an agent would interpret the RAG tool response
to understand why it's claiming "zero results".
"""

import json
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_agent_interpretation():
    """Test how agent would interpret the RAG response"""
    
    # This is the actual response from the MCP bridge
    rag_response = {
        "success": True,
        "result": {
            "success": True,
            "documents_found": 5,
            "total_documents_found": 10,
            "documents_returned": 5,
            "collections_searched": ["partnership", "training_materials", "audit_reports"],
            "execution_time_ms": 9046,
            "documents": [
                {
                    "title": "bys & alibaba partnership.pdf",
                    "score": 0.749,
                    "metadata": {"source": "bys & alibaba partnership.pdf", "collection": "partnership"},
                    "content": "Beyondsoft's Deep Partnership with Alibaba and ANT Group..."
                }
                # More documents...
            ],
            "summary": "Found 5 relevant documents",
            "search_metadata": {"fallback": True, "search_strategy": "simple_vector_search"}
        },
        "tool": "rag_knowledge_search",
        "parameters": {"query": "beyondsoft alibaba partnership"}
    }
    
    print("=" * 60)
    print("Testing Agent Response Interpretation")
    print("=" * 60)
    
    # Check different ways an agent might interpret this
    print("\n1. CHECKING TOP-LEVEL SUCCESS:")
    if rag_response.get("success"):
        print("   ✅ Top-level success = True")
    else:
        print("   ❌ Top-level success = False")
    
    print("\n2. CHECKING RESULT FIELD:")
    result = rag_response.get("result", {})
    if result:
        print(f"   ✅ Result field exists: {type(result)}")
    else:
        print("   ❌ No result field")
    
    print("\n3. CHECKING NESTED SUCCESS:")
    if result.get("success"):
        print("   ✅ Nested success = True")
    else:
        print("   ❌ Nested success = False")
    
    print("\n4. CHECKING DOCUMENTS:")
    documents = result.get("documents", [])
    if documents:
        print(f"   ✅ Documents found: {len(documents)}")
    else:
        print("   ❌ No documents in result")
    
    print("\n5. CHECKING DOCUMENT COUNT FIELDS:")
    print(f"   - documents_found: {result.get('documents_found', 0)}")
    print(f"   - total_documents_found: {result.get('total_documents_found', 0)}")
    print(f"   - documents_returned: {result.get('documents_returned', 0)}")
    
    print("\n6. CHECKING SUMMARY:")
    summary = result.get("summary", "")
    print(f"   Summary: '{summary}'")
    
    # Simulate what an LLM agent might conclude
    print("\n" + "=" * 60)
    print("AGENT INTERPRETATION LOGIC:")
    print("=" * 60)
    
    # Common ways agents might misinterpret
    print("\n❌ POTENTIAL MISINTERPRETATION 1:")
    print("   If agent looks for 'content' field at top level:")
    if "content" in rag_response:
        print("   - Would find content")
    else:
        print("   - Would NOT find content (might think no results)")
    
    print("\n❌ POTENTIAL MISINTERPRETATION 2:")
    print("   If agent looks for 'text' field:")
    if "text" in result:
        print("   - Would find text")
    else:
        print("   - Would NOT find text (might think no results)")
    
    print("\n❌ POTENTIAL MISINTERPRETATION 3:")
    print("   If agent expects string response instead of structured:")
    if isinstance(result, str):
        print("   - Result is string")
    else:
        print("   - Result is NOT string (agent might not parse it)")
    
    print("\n✅ CORRECT INTERPRETATION:")
    print("   Agent should check:")
    print("   1. result.success = True")
    print("   2. result.documents_found > 0")
    print("   3. result.documents is a non-empty list")
    print(f"   All checks pass: {result.get('success') and result.get('documents_found', 0) > 0 and len(result.get('documents', [])) > 0}")
    
    # Generate a prompt that would help the agent interpret correctly
    print("\n" + "=" * 60)
    print("RECOMMENDED AGENT PROMPT ADDITION:")
    print("=" * 60)
    print("""
When using the rag_knowledge_search tool:
1. The tool returns a structured response with 'success' and 'result' fields
2. Check result.success to verify the search succeeded
3. Check result.documents_found for the number of documents
4. Access the actual documents in result.documents array
5. Each document has 'title', 'content', 'score', and 'metadata'
6. If result.documents_found > 0, documents were successfully retrieved
7. Never claim "zero results" if result.documents_found > 0
""")
    
    return rag_response

if __name__ == "__main__":
    response = test_agent_interpretation()
    
    # Save the response for debugging
    with open("agent_interpretation_test.json", "w") as f:
        json.dump(response, f, indent=2)
    
    print("\n✅ Test complete. Response saved to agent_interpretation_test.json")