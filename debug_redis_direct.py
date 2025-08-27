#!/usr/bin/env python3

"""
Direct Redis debugging for conversation intelligence failure
"""

import redis
import json
import os

def debug_redis_conversation_data():
    """Debug Redis data directly."""
    
    test_conversation_id = "notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd"
    
    print(f"🔍 DEBUGGING REDIS DATA FOR: {test_conversation_id}")
    print("=" * 80)
    
    try:
        # Connect to Redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        print("✅ Connected to Redis")
        
        # Check for cached context
        cache_key = f"notebook_retrieval_context:{test_conversation_id}"
        cached_data = redis_client.get(cache_key)
        
        print(f"\n1️⃣ CACHED CONTEXT ({cache_key}):")
        if cached_data:
            try:
                context = json.loads(cached_data)
                sources_count = len(context.get('sources', []))
                print(f"✅ Found cached context with {sources_count} sources")
                print(f"   Keys: {list(context.keys())}")
                if 'sources' in context:
                    print(f"   First source preview: {str(context['sources'][0])[:200]}..." if context['sources'] else "   No sources")
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"   Raw data: {cached_data[:200]}...")
        else:
            print("❌ NO cached context found")
        
        # Check for conversation context
        response_key = f"conversation_response:{test_conversation_id}"
        response_data = redis_client.get(response_key)
        
        print(f"\n2️⃣ CONVERSATION CONTEXT ({response_key}):")
        if response_data:
            try:
                conv_context = json.loads(response_data)
                print(f"✅ Found conversation context")
                print(f"   Keys: {list(conv_context.keys())}")
                print(f"   User message: {conv_context.get('user_message', 'N/A')}")
                print(f"   AI response preview: {str(conv_context.get('ai_response', ''))[:100]}...")
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
        else:
            print("❌ NO conversation context found")
        
        # Check all related keys
        print(f"\n3️⃣ ALL RELATED KEYS:")
        all_related = redis_client.keys(f"*{test_conversation_id.split('-')[-1]}*")
        print(f"   Found {len(all_related)} related keys: {all_related}")
        
        # Check broader patterns
        notebook_keys = redis_client.keys("*notebook*")
        print(f"   Found {len(notebook_keys)} notebook keys (showing first 10):")
        for key in notebook_keys[:10]:
            print(f"     {key}")
        
    except Exception as e:
        print(f"💥 Redis connection failed: {e}")
    
    print("\n" + "=" * 80)
    print("🎯 ROOT CAUSE ANALYSIS:")
    print("If BOTH cached context AND conversation context are missing:")
    print("1. The conversation intelligence will be SKIPPED")
    print("2. System will proceed directly to retrieval planning")
    print("3. This explains why the 45-second timeout still occurs")

if __name__ == "__main__":
    debug_redis_conversation_data()