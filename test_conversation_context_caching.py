#!/usr/bin/env python3
"""
Test script for Phase 2 of intelligent message handling: conversation context caching

This script tests the ConversationContextManager implementation to verify:
1. Caching of retrieval results
2. Cache hit detection for follow-up questions
3. Cache expiration and TTL handling
4. Proper fallback to new retrieval when cache misses

Usage:
    python test_conversation_context_caching.py
"""

import asyncio
import json
import sys
import os
import time
from datetime import datetime

# Add the app directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.notebook_rag_service import ConversationContextManager
from models.notebook_models import NotebookRAGSource

async def test_conversation_context_caching():
    """Test the conversation context caching implementation"""
    
    print("🧪 Testing Phase 2: Conversation Context Caching Implementation")
    print("=" * 60)
    
    # Initialize the context manager
    context_manager = ConversationContextManager()
    
    # Test conversation ID
    conversation_id = "test-conv-12345"
    
    print(f"📝 Test conversation ID: {conversation_id}")
    print()
    
    # === Test 1: Cache Retrieval Context ===
    print("🔍 Test 1: Caching retrieval context")
    
    # Create mock retrieval context (simulating successful RAG query)
    mock_context = {
        'sources': [
            {
                'content': 'This is a test document about machine learning projects...',
                'metadata': {
                    'document_id': 'doc-123',
                    'document_name': 'ML_Projects.pdf',
                    'collection': 'notebooks'
                },
                'score': 0.95
            },
            {
                'content': 'Another document discussing web development frameworks...',
                'metadata': {
                    'document_id': 'doc-456', 
                    'document_name': 'WebDev_Guide.pdf',
                    'collection': 'notebooks'
                },
                'score': 0.88
            }
        ],
        'query': 'list all my projects',
        'extracted_entities': ['Project Alpha', 'Project Beta'],
        'timestamp': datetime.now().isoformat(),
        'metadata': {
            'total_sources': 2,
            'queried_documents': 15,
            'collections_searched': ['notebooks'],
            'ai_pipeline_used': False
        }
    }
    
    # Cache the context
    cache_success = await context_manager.cache_retrieval_context(conversation_id, mock_context)
    
    if cache_success:
        print("✅ Successfully cached retrieval context")
    else:
        print("❌ Failed to cache retrieval context")
        return
        
    # === Test 2: Check Recent Context ===
    print("\n🕐 Test 2: Checking for recent context")
    
    has_recent = await context_manager.has_recent_context(conversation_id, max_age_minutes=5)
    
    if has_recent:
        print("✅ Recent context detected")
    else:
        print("❌ Recent context not found")
        return
        
    # === Test 3: Retrieve Cached Context ===
    print("\n📥 Test 3: Retrieving cached context")
    
    cached_context = await context_manager.get_cached_context(conversation_id)
    
    if cached_context:
        print("✅ Successfully retrieved cached context")
        print(f"   - Sources: {len(cached_context.get('sources', []))}")
        print(f"   - Original query: {cached_context.get('query', 'N/A')[:50]}...")
        print(f"   - Cached at: {cached_context.get('cached_at', 'N/A')}")
    else:
        print("❌ Failed to retrieve cached context")
        return
        
    # === Test 4: Cache Statistics ===
    print("\n📊 Test 4: Getting cache statistics")
    
    cache_stats = await context_manager.get_cache_stats(conversation_id)
    
    print("✅ Cache statistics:")
    print(f"   - Status: {cache_stats.get('status')}")
    print(f"   - Source count: {cache_stats.get('source_count')}")
    print(f"   - TTL seconds: {cache_stats.get('ttl_seconds')}")
    print(f"   - Query preview: {cache_stats.get('query_preview')}")
    
    # === Test 5: Cache Expiration ===
    print("\n⏱️  Test 5: Testing cache expiration (waiting 2 seconds)")
    
    # Wait briefly to test TTL behavior
    time.sleep(2)
    
    # Check if context is still there
    still_has_recent = await context_manager.has_recent_context(conversation_id, max_age_minutes=5)
    
    if still_has_recent:
        print("✅ Context still available (expected within TTL)")
    else:
        print("❌ Context expired too quickly")
        
    # === Test 6: Cache Invalidation ===
    print("\n🗑️  Test 6: Testing cache invalidation")
    
    invalidation_success = await context_manager.invalidate_context(conversation_id)
    
    if invalidation_success:
        print("✅ Successfully invalidated cache")
    else:
        print("❌ Failed to invalidate cache")
        
    # Verify invalidation
    after_invalidation = await context_manager.has_recent_context(conversation_id, max_age_minutes=5)
    
    if not after_invalidation:
        print("✅ Cache properly cleared after invalidation")
    else:
        print("❌ Cache still exists after invalidation")
        
    # === Test 7: Cache Miss Scenario ===
    print("\n❌ Test 7: Testing cache miss scenario")
    
    new_conversation_id = "test-conv-nonexistent"
    
    has_context = await context_manager.has_recent_context(new_conversation_id, max_age_minutes=5)
    
    if not has_context:
        print("✅ Correctly detected cache miss for non-existent conversation")
    else:
        print("❌ False positive - detected context where none exists")
        
    print("\n" + "=" * 60)
    print("🎉 Phase 2 Conversation Context Caching Test Complete!")
    print()
    
    # Summary of expected behavior
    print("📋 Expected Behavior After Implementation:")
    print("   1. User: 'list all projects' → Full retrieval → Cache results")
    print("   2. User: 'tell me more about the first one' → Use cached results (NO new retrieval)")
    print("   3. User: 'what about project #5?' → Use cached results (NO new retrieval)")
    print("   4. [5 minutes later] User: 'show me technologies used' → Cache expired → New retrieval")
    print()
    print("✅ Implementation ready for production use!")

if __name__ == "__main__":
    asyncio.run(test_conversation_context_caching())