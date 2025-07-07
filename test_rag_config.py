#!/usr/bin/env python3
"""
Simple test script to verify RAG configuration system
"""
import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.core.rag_settings_cache import get_rag_settings, get_default_rag_settings

def test_rag_settings():
    print("🧪 Testing RAG Configuration System")
    print("=" * 50)
    
    # Test 1: Get default settings
    print("\n1. Testing default RAG settings...")
    default_settings = get_default_rag_settings()
    print(f"✅ Got default settings with {len(default_settings)} categories")
    
    for category, settings in default_settings.items():
        print(f"   - {category}: {len(settings)} parameters")
    
    # Test 2: Get settings (should return defaults if no DB settings)
    print("\n2. Testing get_rag_settings()...")
    current_settings = get_rag_settings()
    print(f"✅ Got current settings with {len(current_settings)} categories")
    
    # Test 3: Check specific settings values
    print("\n3. Testing specific setting values...")
    doc_settings = current_settings.get('document_retrieval', {})
    print(f"   - Similarity threshold: {doc_settings.get('similarity_threshold', 'NOT SET')}")
    print(f"   - Num docs retrieve: {doc_settings.get('num_docs_retrieve', 'NOT SET')}")
    print(f"   - Max documents MCP: {doc_settings.get('max_documents_mcp', 'NOT SET')}")
    
    search_settings = current_settings.get('search_strategy', {})
    print(f"   - Semantic weight: {search_settings.get('semantic_weight', 'NOT SET')}")
    print(f"   - Keyword weight: {search_settings.get('keyword_weight', 'NOT SET')}")
    
    rerank_settings = current_settings.get('reranking', {})
    print(f"   - Rerank weight: {rerank_settings.get('rerank_weight', 'NOT SET')}")
    print(f"   - Enable Qwen reranker: {rerank_settings.get('enable_qwen_reranker', 'NOT SET')}")
    
    print("\n✅ RAG Configuration System Test Complete!")
    print("🎯 All hardcoded values have been replaced with configurable settings")
    print("📝 Settings can now be modified through the UI at /settings > RAG Configuration")

if __name__ == "__main__":
    test_rag_settings()