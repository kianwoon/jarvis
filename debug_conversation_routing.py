#!/usr/bin/env python3

"""
Emergency debug script to trace why conversation intelligence is not working
for the failing conversation ID: notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd
"""

import asyncio
import json
import sys
import os

# Add the app directory to the Python path
sys.path.append(os.path.abspath('./app'))

from services.notebook_rag_service import ConversationContextManager

async def debug_conversation_routing():
    """Debug the actual conversation routing logic."""
    
    test_conversation_id = "notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd"
    test_message = "order by end year, give me the list again"
    
    print(f"🔍 DEBUGGING CONVERSATION ROUTING FOR: {test_conversation_id}")
    print(f"📨 Message: '{test_message}'")
    print("=" * 80)
    
    # Test the conversation context manager directly
    context_manager = ConversationContextManager()
    
    print("\n1️⃣ TESTING CONVERSATION CONTEXT RETRIEVAL:")
    try:
        conversation_context = await context_manager.get_conversation_context(test_conversation_id)
        if conversation_context:
            print(f"✅ Found conversation context: {json.dumps(conversation_context, indent=2)}")
        else:
            print("❌ NO conversation context found")
    except Exception as e:
        print(f"💥 Error getting conversation context: {e}")
    
    print("\n2️⃣ TESTING CACHED CONTEXT RETRIEVAL:")
    try:
        cached_context = await context_manager.get_cached_context(test_conversation_id)
        if cached_context:
            sources_count = len(cached_context.get('sources', []))
            print(f"✅ Found cached context with {sources_count} sources")
            print(f"   Cached keys: {list(cached_context.keys())}")
        else:
            print("❌ NO cached context found")
    except Exception as e:
        print(f"💥 Error getting cached context: {e}")
    
    print("\n3️⃣ TESTING REDIS CONNECTION:")
    try:
        redis_client = context_manager._get_redis_client()
        if redis_client:
            print("✅ Redis client connected successfully")
            
            # Check all keys related to this conversation
            cache_key = context_manager._get_cache_key(test_conversation_id)
            response_key = f"conversation_response:{test_conversation_id}"
            
            print(f"   Checking cache key: {cache_key}")
            cache_exists = redis_client.exists(cache_key)
            print(f"   Cache exists: {bool(cache_exists)}")
            
            print(f"   Checking response key: {response_key}")
            response_exists = redis_client.exists(response_key)
            print(f"   Response exists: {bool(response_exists)}")
            
            # List all keys that might be related
            all_keys = redis_client.keys("*144c3ac3*")
            print(f"   All related keys: {all_keys}")
            
        else:
            print("❌ Failed to get Redis client")
    except Exception as e:
        print(f"💥 Error testing Redis: {e}")
    
    print("\n4️⃣ ROOT CAUSE ANALYSIS:")
    print("The issue is likely:")
    print("- Either no cached context exists for this conversation")  
    print("- Or conversation context is not being found")
    print("- Or the respond_with_llm_intelligence function is failing silently")
    print("- Or the routing logic has a bug that bypasses the intelligence check")
    
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDATION:")
    print("If no context is found, the system will skip conversation intelligence")
    print("and proceed directly to expensive retrieval planning.")

if __name__ == "__main__":
    asyncio.run(debug_conversation_routing())