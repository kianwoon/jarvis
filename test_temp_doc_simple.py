#!/usr/bin/env python3
"""
Simple test to check if temporary document source tagging works
"""
import json

# Test data that simulates what hybrid_context would contain
test_hybrid_context = {
    "strategy": "temp_priority",
    "sources": [
        {
            "content": "This is from a temporary document",
            "filename": "test_doc.pdf",
            "page": 1,
            "metadata": {"temp_doc_id": "temp_123"}
        }
    ]
}

# Simulate the logic from unified_llm_synthesis
def check_temp_doc_detection(rag_sources):
    """Check if any sources are from temporary documents"""
    has_temp_docs = False
    for source in rag_sources:
        # Check various ways temporary documents might be marked
        if (source.get("file", "").startswith("[TEMP]") or 
            source.get("is_temporary", False) or
            source.get("collection", "").lower() == "temp_documents" or
            "temp" in source.get("collection", "").lower()):
            has_temp_docs = True
            break
    return has_temp_docs

# Test 1: Sources extracted from hybrid_context (simulating the code we added)
print("Test 1: Extracting sources from hybrid_context")
rag_sources = []
if test_hybrid_context and 'sources' in test_hybrid_context:
    for source in test_hybrid_context.get('sources', []):
        source_info = {
            "content": source.get('content', ''),
            "file": source.get('filename', source.get('source', 'Unknown')),
            "page": source.get('page', 0),
            "collection": "temp_documents",  # Mark as temp documents
            "is_temporary": True,  # Explicit flag
            "metadata": source.get('metadata', {})
        }
        rag_sources.append(source_info)
    print(f"Extracted {len(rag_sources)} sources from hybrid_context")

print("RAG sources:", json.dumps(rag_sources, indent=2))

# Test detection
has_temp = check_temp_doc_detection(rag_sources)
print(f"\nDetected temporary documents: {has_temp}")

# Build source label
sources = []
if rag_sources:  # Simulating rag_context being present
    if has_temp:
        sources.append("RAG_TEMP")
    else:
        sources.append("RAG")
sources.append("LLM")
source_label = "+".join(sources)

print(f"Source label: {source_label}")
print(f"Expected: RAG_TEMP+LLM")
print(f"Success: {source_label == 'RAG_TEMP+LLM'}")