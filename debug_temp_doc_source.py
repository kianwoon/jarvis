#!/usr/bin/env python3
"""
Debug script to trace temporary document source tagging
"""
import json

print("=== Debug: Temporary Document Source Tagging ===\n")

# Simulate what happens in the rag endpoint when hybrid_context is used
print("1. Hybrid context from orchestrator:")
hybrid_context = {
    'sources': [
        {
            'content': 'Sample content from temp doc',
            'filename': 'my_uploaded_doc.pdf',
            'page': 1,
            'metadata': {'temp_doc_id': 'temp_123'}
        }
    ],
    'strategy_used': 'temp_priority'
}
print(json.dumps(hybrid_context, indent=2))

print("\n2. Processing in rag_answer:")
# This simulates the code we added to rag_answer
rag_sources = []
if hybrid_context and 'sources' in hybrid_context:
    print("   - Extracting sources from hybrid_context")
    for source in hybrid_context.get('sources', []):
        source_info = {
            "content": source.get('content', ''),
            "file": source.get('filename', source.get('source', 'Unknown')),
            "page": source.get('page', 0),
            "collection": "temp_documents",
            "is_temporary": True,
            "metadata": source.get('metadata', {})
        }
        rag_sources.append(source_info)
    print(f"   - Extracted {len(rag_sources)} sources")

print("\n3. RAG sources passed to unified_llm_synthesis:")
print(json.dumps(rag_sources, indent=2))

print("\n4. In unified_llm_synthesis:")
# Simulate empty rag_context but with rag_sources
rag_context = ""  # Empty because context is in the enhanced question
print(f"   - rag_context: '{rag_context}' (empty)")
print(f"   - rag_sources: {len(rag_sources)} sources")

# Source detection logic (after our fix)
sources = []
if rag_context or rag_sources:
    print("   - Checking for temp docs...")
    has_temp_docs = False
    if rag_sources:
        for source in rag_sources:
            if (source.get("file", "").startswith("[TEMP]") or 
                source.get("is_temporary", False) or
                source.get("collection", "").lower() == "temp_documents" or
                "temp" in source.get("collection", "").lower()):
                has_temp_docs = True
                print(f"     ✓ Found temp doc: {source.get('file')}")
                break
    
    if has_temp_docs:
        sources.append("RAG_TEMP")
        print("   - Added RAG_TEMP to sources")
    elif rag_context or rag_sources:
        sources.append("RAG")
        print("   - Added RAG to sources")

sources.append("LLM")
source_label = "+".join(sources)

print(f"\n5. Final source label: {source_label}")
print(f"   Expected: RAG_TEMP+LLM")
print(f"   Success: {'✅' if source_label == 'RAG_TEMP+LLM' else '❌'}")

print("\n=== End Debug ===")