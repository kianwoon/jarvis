#!/usr/bin/env python3
"""
Critical test to debug why conversation memory detection is COMPLETELY BROKEN.

The logs show:
"[LLM_INTELLIGENCE] No conversation or cached context available, proceeding to retrieval"

But also show:
"[CONVERSATION_MEMORY] Stored response for conversation notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd"

This test will verify:
1. Redis connectivity
2. Key storage vs retrieval consistency  
3. Data serialization/deserialization
4. TTL behavior
"""

import asyncio
import json
import redis
from datetime import datetime

async def debug_conversation_memory():
    """Debug why conversation context detection is failing completely"""
    
    test_conversation_id = "notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd"
    
    print("🔍 DEBUGGING CONVERSATION MEMORY DETECTION FAILURE")
    print("=" * 60)
    
    # Test 1: Redis Connection
    print("\n1️⃣ Testing Redis Connection")
    try:
        from app.core.redis_client import get_redis_client
        redis_client = get_redis_client(decode_responses=True)
        
        if not redis_client:
            print("❌ CRITICAL: Redis client is None")
            return
        
        # Test basic operation
        test_key = "debug_test"
        redis_client.set(test_key, "test_value", ex=60)
        test_result = redis_client.get(test_key)
        
        print(f"   Raw result: {repr(test_result)}")
        print(f"   Type: {type(test_result)}")
        
        # Handle both string and bytes
        if test_result == "test_value" or (isinstance(test_result, bytes) and test_result == b"test_value"):
            print("✅ Redis connection working (handling encoding)")
            redis_client.delete(test_key)  # Clean up
        else:
            print(f"❌ Redis connection failed: expected 'test_value', got '{test_result}'")
            return
            
    except Exception as e:
        print(f"❌ CRITICAL: Redis connection failed: {e}")
        return
    
    # Test 2: Check if conversation response exists
    print("\n2️⃣ Checking Existing Conversation Response")
    response_key = f"conversation_response:{test_conversation_id}"
    
    try:
        existing_data = redis_client.get(response_key)
        
        if existing_data:
            print(f"✅ Found existing conversation response")
            print(f"   Key: {response_key}")
            print(f"   Data length: {len(existing_data)} chars")
            
            # Try to parse it
            try:
                parsed_data = json.loads(existing_data)
                print(f"   Timestamp: {parsed_data.get('timestamp', 'MISSING')}")
                print(f"   TTL: {parsed_data.get('ttl', 'MISSING')} seconds")
                print(f"   User message: {parsed_data.get('user_message', 'MISSING')[:50]}...")
                print(f"   AI response: {parsed_data.get('ai_response', 'MISSING')[:50]}...")
                print(f"   Sources count: {parsed_data.get('sources_count', 'MISSING')}")
            except json.JSONDecodeError as e:
                print(f"❌ CRITICAL: JSON decode failed: {e}")
                print(f"   Raw data: {existing_data[:100]}...")
        else:
            print(f"❌ No conversation response found for key: {response_key}")
            
    except Exception as e:
        print(f"❌ Error checking existing response: {e}")
    
    # Test 3: Check Redis TTL
    print("\n3️⃣ Checking Redis TTL")
    try:
        ttl = redis_client.ttl(response_key)
        print(f"TTL for key '{response_key}': {ttl}")
        
        if ttl == -2:
            print("❌ CRITICAL: Key does not exist")
        elif ttl == -1:
            print("❌ Key exists but has no TTL (never expires)")
        elif ttl > 0:
            print(f"✅ Key expires in {ttl} seconds")
        else:
            print(f"❌ Unexpected TTL value: {ttl}")
            
    except Exception as e:
        print(f"❌ Error checking TTL: {e}")
    
    # Test 4: Test the actual get_conversation_context method
    print("\n4️⃣ Testing get_conversation_context Method")
    try:
        from app.services.notebook_rag_service import conversation_context_manager
        
        context_result = await conversation_context_manager.get_conversation_context(test_conversation_id)
        
        if context_result:
            print("✅ get_conversation_context returned data:")
            print(f"   Type: {type(context_result)}")
            print(f"   Keys: {list(context_result.keys()) if isinstance(context_result, dict) else 'Not a dict'}")
        else:
            print("❌ CRITICAL: get_conversation_context returned None")
            
    except Exception as e:
        print(f"❌ Error testing get_conversation_context: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Test cached context
    print("\n5️⃣ Testing get_cached_context Method")
    try:
        cached_result = await conversation_context_manager.get_cached_context(test_conversation_id)
        
        if cached_result:
            print("✅ get_cached_context returned data:")
            print(f"   Type: {type(cached_result)}")
            print(f"   Keys: {list(cached_result.keys()) if isinstance(cached_result, dict) else 'Not a dict'}")
        else:
            print("❌ get_cached_context returned None (this may be normal if no cache)")
            
    except Exception as e:
        print(f"❌ Error testing get_cached_context: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: List all conversation-related keys
    print("\n6️⃣ Scanning All Related Redis Keys")
    try:
        # Scan for conversation response keys
        response_keys = []
        for key in redis_client.scan_iter(match="conversation_response:*"):
            response_keys.append(key)
        
        print(f"Found {len(response_keys)} conversation response keys:")
        for key in response_keys:
            ttl = redis_client.ttl(key)
            print(f"   {key} (TTL: {ttl})")
        
        # Scan for context cache keys
        cache_keys = []
        for key in redis_client.scan_iter(match="conversation_context:*"):
            cache_keys.append(key)
            
        print(f"Found {len(cache_keys)} conversation context keys:")
        for key in cache_keys:
            ttl = redis_client.ttl(key)
            print(f"   {key} (TTL: {ttl})")
            
        # Scan for metadata keys
        meta_keys = []
        for key in redis_client.scan_iter(match="conversation_meta:*"):
            meta_keys.append(key)
            
        print(f"Found {len(meta_keys)} conversation metadata keys:")
        for key in meta_keys:
            ttl = redis_client.ttl(key)
            print(f"   {key} (TTL: {ttl})")
            
    except Exception as e:
        print(f"❌ Error scanning Redis keys: {e}")
    
    # Test 7: Simulate storage and retrieval
    print("\n7️⃣ Testing Storage and Retrieval Simulation")
    try:
        test_conv_id = "debug-test-conversation"
        test_user_msg = "What are all the projects?"
        test_ai_response = "Here are the projects from your notebook..."
        
        # Store using the same method
        store_success = await conversation_context_manager.store_conversation_response(
            test_conv_id,
            test_user_msg, 
            test_ai_response,
            []
        )
        
        if store_success:
            print("✅ Test storage succeeded")
            
            # Immediately try to retrieve
            retrieved = await conversation_context_manager.get_conversation_context(test_conv_id)
            
            if retrieved:
                print("✅ Test retrieval succeeded immediately after storage")
                print(f"   Retrieved data type: {type(retrieved)}")
            else:
                print("❌ CRITICAL: Test retrieval failed immediately after storage")
                
            # Clean up
            cleanup_key = f"conversation_response:{test_conv_id}"
            redis_client.delete(cleanup_key)
            
        else:
            print("❌ Test storage failed")
            
    except Exception as e:
        print(f"❌ Error in storage/retrieval test: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUGGING COMPLETE")
    print("\nKey Questions to Answer:")
    print("1. Is Redis working? (Should be ✅)")
    print("2. Is the conversation response stored? (Should be ✅)")
    print("3. Is get_conversation_context finding it? (Should be ✅ but probably ❌)")
    print("4. Are keys consistent between storage/retrieval? (Should be ✅)")
    print("5. Is TTL too short causing premature expiry?")

if __name__ == "__main__":
    asyncio.run(debug_conversation_memory())