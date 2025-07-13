#!/usr/bin/env python3
"""
Fix the critical similarity threshold issue that's preventing document retrieval
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.core.rag_settings_cache import get_rag_settings, update_rag_setting
    
    print("🔧 Fixing Critical Similarity Threshold Issue")
    print("=" * 60)
    
    # Get current settings
    settings = get_rag_settings()
    current_threshold = settings.get('document_retrieval', {}).get('similarity_threshold', 'Unknown')
    
    print(f"Current similarity threshold: {current_threshold}")
    print(f"❌ PROBLEM: Threshold of 1.0 means only perfect matches (impossible in vector search)")
    print(f"✅ SOLUTION: Set to 0.7 (70% similarity) for effective retrieval")
    
    # Fix the similarity threshold
    success = update_rag_setting('document_retrieval', 'similarity_threshold', 0.7)
    
    if success:
        print(f"\n✅ Successfully updated similarity threshold to 0.7")
        
        # Verify the change
        updated_settings = get_rag_settings()
        new_threshold = updated_settings.get('document_retrieval', {}).get('similarity_threshold')
        print(f"✅ Verified new threshold: {new_threshold}")
        
        # Also check other critical settings
        doc_retrieval = updated_settings.get('document_retrieval', {})
        print(f"\n📊 Updated Document Retrieval Settings:")
        print(f"  similarity_threshold: {doc_retrieval.get('similarity_threshold', 'Not set')}")
        print(f"  max_documents_per_collection: {doc_retrieval.get('max_documents_per_collection', 'Not set')}")
        print(f"  max_documents_mcp: {doc_retrieval.get('max_documents_mcp', 'Not set')}")
        print(f"  num_docs_retrieve: {doc_retrieval.get('num_docs_retrieve', 'Not set')}")
        
    else:
        print(f"❌ Failed to update similarity threshold")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()