#!/usr/bin/env python3
"""
Test script to verify Redis cache cleanup when conversation history is cleared.
This tests the new functionality added to the conversation deletion endpoint.
"""

import asyncio
import requests
import json
import time
import uuid
from app.core.redis_client import get_redis_client
from app.core.database import get_db, engine
from sqlalchemy import text
from sqlalchemy.orm import Session

# Configuration
BASE_URL = "http://localhost:8000"  
CONVERSATION_ID = f"test_conv_{int(time.time())}"
NOTEBOOK_ID = f"test_notebook_{int(time.time())}"

async def test_cache_cleanup():
    """Test that conversation deletion clears associated notebook caches."""
    
    print(f"🧪 Testing Redis cache cleanup for conversation deletion")
    print(f"Conversation ID: {CONVERSATION_ID}")
    print(f"Notebook ID: {NOTEBOOK_ID}")
    
    try:
        # 1. Set up test data in Redis
        redis_client = get_redis_client()
        if not redis_client:
            print("❌ Redis client not available")
            return False
        
        # Create some test cache entries that should be cleared
        test_cache_keys = [
            f"notebook_content_count:{NOTEBOOK_ID}:test_query",
            f"task_plan_{NOTEBOOK_ID}_abc123",
            f"llm_settings_notebook_{NOTEBOOK_ID}",
            f"notebook_rag_cache:{NOTEBOOK_ID}:test_data",
        ]
        
        print(f"📝 Setting up {len(test_cache_keys)} test cache entries...")
        for key in test_cache_keys:
            redis_client.set(key, f"test_value_{key}", ex=3600)
        
        # Verify cache keys exist
        existing_keys = [key for key in test_cache_keys if redis_client.exists(key)]
        print(f"✅ Created {len(existing_keys)} cache entries")
        
        # 2. Set up database relationship between conversation and notebook
        print(f"📊 Setting up database relationship...")
        with Session(engine) as db:
            try:
                # Insert test notebook conversation relationship
                insert_query = text("""
                    INSERT INTO notebook_conversations (id, notebook_id, conversation_id, started_at, last_activity)
                    VALUES (:id, :notebook_id, :conversation_id, NOW(), NOW())
                    ON CONFLICT (notebook_id, conversation_id) DO NOTHING
                """)
                db.execute(insert_query, {
                    "id": str(uuid.uuid4()),
                    "notebook_id": NOTEBOOK_ID,
                    "conversation_id": CONVERSATION_ID
                })
                db.commit()
                print(f"✅ Database relationship created")
            except Exception as e:
                print(f"⚠️ Database setup failed (might already exist): {e}")
        
        # 3. Call the conversation deletion endpoint
        print(f"🗑️ Calling conversation deletion endpoint...")
        response = requests.delete(f"{BASE_URL}/langchain/conversation/{CONVERSATION_ID}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Conversation deleted successfully: {result['message']}")
        else:
            print(f"❌ Failed to delete conversation: {response.status_code} - {response.text}")
            return False
        
        # 4. Verify cache keys were cleared
        print(f"🔍 Checking if cache keys were cleared...")
        remaining_keys = [key for key in test_cache_keys if redis_client.exists(key)]
        cleared_keys = len(test_cache_keys) - len(remaining_keys)
        
        print(f"📊 Results:")
        print(f"   - Initial cache keys: {len(test_cache_keys)}")
        print(f"   - Keys cleared: {cleared_keys}")
        print(f"   - Keys remaining: {len(remaining_keys)}")
        
        if remaining_keys:
            print(f"   - Remaining keys: {remaining_keys}")
        
        # 5. Cleanup remaining test data
        if remaining_keys:
            redis_client.delete(*remaining_keys)
            print(f"🧹 Cleaned up remaining test keys")
        
        # Success criteria
        success = cleared_keys >= len(test_cache_keys) * 0.8  # At least 80% cleared
        
        if success:
            print(f"✅ TEST PASSED: Cache cleanup working correctly!")
        else:
            print(f"❌ TEST FAILED: Expected more cache keys to be cleared")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_endpoint_accessibility():
    """Test that the endpoint is accessible and returns expected response format."""
    print(f"🌐 Testing endpoint accessibility...")
    
    try:
        # Test with non-existent conversation
        test_conv_id = f"nonexistent_{int(time.time())}"
        response = requests.delete(f"{BASE_URL}/langchain/conversation/{test_conv_id}")
        
        if response.status_code == 200:
            result = response.json()
            if "success" in result and "message" in result:
                print(f"✅ Endpoint accessible and returns correct format")
                return True
            else:
                print(f"❌ Endpoint returns unexpected format: {result}")
                return False
        else:
            print(f"⚠️ Endpoint returned {response.status_code}: {response.text}")
            # This might still be ok if it's a validation error
            return response.status_code in [400, 404, 422]  # These are acceptable
            
    except requests.ConnectionError:
        print(f"❌ Cannot connect to {BASE_URL} - is the server running?")
        return False
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 Starting Redis Cache Cleanup Test")
    print(f"=" * 60)
    
    # Test endpoint accessibility first
    if not test_endpoint_accessibility():
        print(f"❌ Endpoint accessibility test failed")
        exit(1)
    
    print()
    
    # Run main cache cleanup test
    success = asyncio.run(test_cache_cleanup())
    
    print(f"=" * 60)
    if success:
        print(f"🎉 ALL TESTS PASSED! Redis cache cleanup is working correctly.")
    else:
        print(f"💥 TESTS FAILED! Redis cache cleanup needs attention.")
    
    exit(0 if success else 1)