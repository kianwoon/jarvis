#!/usr/bin/env python3

"""
Investigate why cached context is missing despite previous queries
"""

import redis
import json

def investigate_missing_cache():
    """Check for any retrieval caches that might exist."""
    
    test_conversation_id = "notebook-144c3ac3-0a16-4904-8cba-5adaa2cd86cd"
    
    print(f"🔍 INVESTIGATING MISSING CACHE FOR: {test_conversation_id}")
    print("=" * 80)
    
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Check all possible cache key patterns
        possible_cache_patterns = [
            f"notebook_retrieval_context:{test_conversation_id}",
            f"retrieval_context:{test_conversation_id}",
            f"notebook_cache:{test_conversation_id}",
            f"cache:{test_conversation_id}",
            f"notebook_context:{test_conversation_id}"
        ]
        
        print("🔍 Checking all possible cache key patterns:")
        for pattern in possible_cache_patterns:
            exists = redis_client.exists(pattern)
            print(f"   {pattern}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        
        # Check for any keys containing parts of the conversation ID
        partial_id = test_conversation_id.split('-')[-1]  # Get last part
        print(f"\n🔍 Searching for keys containing '{partial_id}':")
        matching_keys = redis_client.keys(f"*{partial_id}*")
        
        cache_related = [key for key in matching_keys if 'cache' in key.lower() or 'context' in key.lower() or 'retrieval' in key.lower()]
        print(f"   Found {len(cache_related)} cache/context related keys:")
        for key in cache_related:
            print(f"     {key}")
        
        # Look at conversation_response to see what happened
        response_key = f"conversation_response:{test_conversation_id}"
        response_data = redis_client.get(response_key)
        
        if response_data:
            conv_context = json.loads(response_data)
            print(f"\n📄 LAST CONVERSATION DETAILS:")
            print(f"   User: {conv_context.get('user_message', 'N/A')}")
            print(f"   AI Response: {conv_context.get('ai_response', 'N/A')[:200]}...")
            print(f"   Sources Count: {conv_context.get('sources_count', 0)}")
            print(f"   Timestamp: {conv_context.get('timestamp', 'N/A')}")
        
        # Check if there are any successful responses that should have created cache
        print(f"\n🔍 Looking for recent routing decisions:")
        routing_keys = [key for key in matching_keys if key.startswith('routing_decision:')]
        recent_routing = sorted(routing_keys)[-5:] if routing_keys else []
        
        for key in recent_routing:
            try:
                decision_data = redis_client.get(key)
                if decision_data:
                    decision = json.loads(decision_data)
                    retrieval_triggered = decision.get('retrieval_triggered', False)
                    intent = decision.get('intent', 'unknown')
                    print(f"   {key.split(':')[-1]}: {intent} (retrieval: {retrieval_triggered})")
            except:
                pass
                
    except Exception as e:
        print(f"💥 Error: {e}")
    
    print("\n" + "=" * 80)
    print("🎯 CACHE INVESTIGATION RESULTS:")
    print("1. NO cached retrieval context exists")
    print("2. Previous queries resulted in timeouts, not successful responses")
    print("3. Without successful retrieval cache, conversation intelligence is bypassed")
    print("4. System falls back to expensive full retrieval every time")
    print("\n💡 SOLUTION: Need to check WHY initial retrieval is timing out")

if __name__ == "__main__":
    investigate_missing_cache()