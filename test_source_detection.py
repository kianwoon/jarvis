#!/usr/bin/env python3
"""
Test the source detection logic for temporary documents
"""

def test_source_detection():
    """Test various scenarios for detecting temporary document sources"""
    
    print("Testing source detection logic for temporary documents...\n")
    
    # Test cases
    test_cases = [
        {
            "name": "Empty sources",
            "rag_context": "",
            "rag_sources": [],
            "expected_has_temp": False,
            "expected_source": "LLM"
        },
        {
            "name": "Regular RAG source",
            "rag_context": "Some context",
            "rag_sources": [
                {"file": "regular.pdf", "collection": "default_knowledge"}
            ],
            "expected_has_temp": False,
            "expected_source": "RAG+LLM"
        },
        {
            "name": "Temp doc with [TEMP] prefix",
            "rag_context": "Some context",
            "rag_sources": [
                {"file": "[TEMP] uploaded.pdf", "collection": "default"}
            ],
            "expected_has_temp": True,
            "expected_source": "RAG_TEMP+LLM"
        },
        {
            "name": "Temp doc with is_temporary flag",
            "rag_context": "Some context",
            "rag_sources": [
                {"file": "document.pdf", "is_temporary": True}
            ],
            "expected_has_temp": True,
            "expected_source": "RAG_TEMP+LLM"
        },
        {
            "name": "Temp doc with temp_documents collection",
            "rag_context": "Some context",
            "rag_sources": [
                {"file": "document.pdf", "collection": "temp_documents"}
            ],
            "expected_has_temp": True,
            "expected_source": "RAG_TEMP+LLM"
        },
        {
            "name": "Empty rag_context but with temp sources",
            "rag_context": "",
            "rag_sources": [
                {"file": "document.pdf", "collection": "temp_documents", "is_temporary": True}
            ],
            "expected_has_temp": True,
            "expected_source": "RAG_TEMP+LLM"
        },
        {
            "name": "Mixed sources (temp and regular)",
            "rag_context": "Some context",
            "rag_sources": [
                {"file": "regular.pdf", "collection": "default_knowledge"},
                {"file": "temp.pdf", "collection": "temp_collection", "is_temporary": True}
            ],
            "expected_has_temp": True,
            "expected_source": "RAG_TEMP+LLM"
        }
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        print(f"Test: {test['name']}")
        
        # Simulate the detection logic from unified_llm_synthesis
        sources = []
        rag_context = test['rag_context']
        rag_sources = test['rag_sources']
        
        # The fixed logic that checks both rag_context OR rag_sources
        has_temp_docs = False
        if rag_context or rag_sources:
            if rag_sources:
                for source in rag_sources:
                    if (source.get("file", "").startswith("[TEMP]") or 
                        source.get("is_temporary", False) or
                        source.get("collection", "").lower() == "temp_documents" or
                        "temp" in source.get("collection", "").lower()):
                        has_temp_docs = True
                        break
            
            if has_temp_docs:
                sources.append("RAG_TEMP")
            elif rag_context or rag_sources:
                sources.append("RAG")
        
        sources.append("LLM")
        source_label = "+".join(sources)
        
        # Check results
        if has_temp_docs == test['expected_has_temp'] and source_label == test['expected_source']:
            print(f"  ✅ Passed - Source: {source_label}")
            passed += 1
        else:
            print(f"  ❌ Failed - Expected: {test['expected_source']}, Got: {source_label}")
            failed += 1
        print()
    
    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("✨ All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    test_source_detection()