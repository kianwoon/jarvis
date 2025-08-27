#!/usr/bin/env python3
"""
Test to verify the conversation memory detection fix works.

This test will:
1. Simulate the exact scenario that was failing
2. Test both conversation_context and cached_context retrieval
3. Verify the bytes encoding fix resolves the issue
"""

import asyncio
import json
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

async def test_conversation_memory_fix():
    """Test the fix for conversation memory detection"""
    
    print("🧪 TESTING CONVERSATION MEMORY DETECTION FIX")
    print("=" * 60)
    
    # Test conversation ID that we know has data
    test_conversation_id = "notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd"
    
    # Import Redis client directly to check data 
    try:
        from app.core.redis_client import get_redis_client
        redis_client = get_redis_client(decode_responses=False)  # Use False to get bytes
        
        # Check raw data
        response_key = f"conversation_response:{test_conversation_id}"
        raw_data = redis_client.get(response_key)
        
        if raw_data:
            print(f"✅ Raw Redis data found:")
            print(f"   Type: {type(raw_data)}")
            print(f"   Length: {len(raw_data)} bytes")
            
            # This is what the issue was - bytes data
            if isinstance(raw_data, bytes):
                decoded_data = raw_data.decode('utf-8')
                parsed_data = json.loads(decoded_data)
                print(f"   Successfully decoded and parsed JSON")
                print(f"   Keys: {list(parsed_data.keys())}")
            else:
                print(f"   Data is not bytes: {type(raw_data)}")
        else:
            print(f"❌ No raw data found in Redis (likely expired due to short TTL)")
            print(f"   Creating test data to verify fix...")
            
            # Create test data to verify our fix works
            test_data = {
                'user_message': 'What are all the projects?',
                'ai_response': 'Here are all the projects from the notebook...',
                'sources_count': 5,
                'timestamp': '2025-08-26T19:00:00.000000',
                'ttl': 300
            }
            
            # Store it as bytes (simulating the original issue)
            redis_client.setex(
                response_key, 
                300,  # 5 minute TTL
                json.dumps(test_data).encode('utf-8')  # Store as bytes
            )
            print(f"✅ Test data created in Redis as bytes")
            
            # Verify it's stored as bytes
            stored_data = redis_client.get(response_key)
            print(f"   Stored type: {type(stored_data)}")
            print(f"   This is exactly the scenario our fix addresses")
            
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return
    
    # Now test our fixed ConversationContextManager
    print(f"\n🔧 Testing Fixed ConversationContextManager")
    
    try:
        # Import the fixed service
        from app.services.notebook_rag_service import ConversationContextManager
        
        # Create instance
        context_manager = ConversationContextManager()
        
        # Test get_conversation_context (the method that was failing)
        print(f"\n1️⃣ Testing get_conversation_context")
        conversation_result = await context_manager.get_conversation_context(test_conversation_id)
        
        if conversation_result:
            print(f"✅ get_conversation_context SUCCESS!")
            print(f"   Type: {type(conversation_result)}")
            print(f"   Keys: {list(conversation_result.keys())}")
            print(f"   User message: {conversation_result.get('user_message', 'MISSING')[:50]}...")
            print(f"   AI response: {conversation_result.get('ai_response', 'MISSING')[:50]}...")
            print(f"   Sources count: {conversation_result.get('sources_count', 'MISSING')}")
        else:
            print(f"❌ get_conversation_context still returning None")
            return
            
        # Test get_cached_context 
        print(f"\n2️⃣ Testing get_cached_context")
        cached_result = await context_manager.get_cached_context(test_conversation_id)
        
        if cached_result:
            print(f"✅ get_cached_context SUCCESS!")
            print(f"   Type: {type(cached_result)}")
            print(f"   Keys: {list(cached_result.keys())}")
            print(f"   Sources: {len(cached_result.get('sources', []))}")
        else:
            print(f"ℹ️  get_cached_context returned None (may be normal)")
            
        # Test has_recent_context
        print(f"\n3️⃣ Testing has_recent_context")
        has_recent = await context_manager.has_recent_context(test_conversation_id, max_age_minutes=10)
        
        if has_recent:
            print(f"✅ has_recent_context SUCCESS - found recent context")
        else:
            print(f"❌ has_recent_context failed - no recent context detected")
        
        print(f"\n" + "=" * 60)
        print(f"🎉 CONVERSATION MEMORY FIX VERIFICATION COMPLETE")
        
        if conversation_result:
            print(f"✅ PRIMARY FIX SUCCESSFUL")
            print(f"   The LLM intelligence system should now work correctly!")
            print(f"   Future queries will properly detect conversation context")
        else:
            print(f"❌ PRIMARY FIX FAILED") 
            print(f"   Additional debugging needed")
            
    except Exception as e:
        print(f"❌ ConversationContextManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    asyncio.run(test_conversation_memory_fix())