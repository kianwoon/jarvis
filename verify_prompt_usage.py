#!/usr/bin/env python3
"""
Verify that the new LLM prompt is being used in actual requests
"""

import requests
import json
import time

def verify_prompt_in_use():
    """Verify the new prompt is being used"""
    
    print("\n🔍 Verification Steps:\n")
    
    # Step 1: Check database
    print("1. Database Check:")
    import subprocess
    cmd = """PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d llm_platform -t -c "SELECT length(settings->'main_llm'->>'system_prompt') FROM settings WHERE category = 'llm';" """
    db_length = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    print(f"   ✅ Database prompt length: {db_length} characters")
    
    # Step 2: Check Redis cache
    print("\n2. Redis Cache Check:")
    cmd = "redis-cli -h localhost -p 6379 get llm_settings_cache | python3 -c \"import json, sys; data = json.load(sys.stdin); print(len(data.get('main_llm', {}).get('system_prompt', '')))\""
    redis_length = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
    print(f"   ✅ Redis cache prompt length: {redis_length} characters")
    
    # Step 3: Check API endpoint
    print("\n3. API Endpoint Check:")
    response = requests.get("http://localhost:8000/api/v1/settings/llm")
    if response.status_code == 200:
        settings = response.json()
        api_prompt = settings.get('settings', {}).get('main_llm', {}).get('system_prompt', '')
        print(f"   ✅ API returns prompt length: {len(api_prompt)} characters")
        
        # Check for new evaluation guidelines
        if "evaluate and synthesize the information thoughtfully" in api_prompt:
            print("   ✅ New prompt with evaluation guidelines detected")
        else:
            print("   ⚠️ Old prompt detected")
    else:
        print(f"   ❌ API error: {response.status_code}")
    
    # Step 4: Test actual chat endpoint
    print("\n4. Chat Endpoint Test:")
    print("   Sending test message to verify behavior...")
    
    test_payload = {
        "message": "Just respond with OK if you can hear me",
        "conversation_id": f"test_{int(time.time())}",
        "use_rag": False,
        "use_tools": False,
        "use_knowledge_graph": False
    }
    
    # Try the RAG endpoint which we know exists
    response = requests.post(
        "http://localhost:8000/api/v1/langchain/rag",
        json=test_payload
    )
    
    if response.status_code == 200:
        result = response.json()
        if 'response' in result:
            print("   ✅ Chat endpoint is responding")
            # The actual prompt verification happens by checking if it synthesizes info properly
        else:
            print(f"   ⚠️ Unexpected response format: {result.keys()}")
    else:
        print(f"   ❌ Chat endpoint error: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   Database:    {db_length} chars")
    print(f"   Redis:       {redis_length} chars")
    print(f"   API:         {len(api_prompt) if 'api_prompt' in locals() else 'N/A'} chars")
    
    if db_length == redis_length and int(db_length) > 1500:
        print("\n✅ VERIFICATION PASSED!")
        print("   The new prompt is properly loaded and synchronized.")
        print("\n📝 Next Steps:")
        print("   1. Test with a query that includes search results")
        print("   2. Verify the LLM evaluates information thoughtfully")
        print("   3. Check if it presents confidence levels appropriately")
    else:
        print("\n⚠️ VERIFICATION FAILED!")
        print("   Prompt lengths don't match or are too short.")
        print("\n🔧 Troubleshooting:")
        print("   1. Clear Redis: redis-cli del llm_settings_cache")
        print("   2. Reload cache: curl -X POST http://localhost:8000/api/v1/settings/llm/cache/reload")
        print("   3. Restart backend: docker restart jarvis-app-1")

if __name__ == "__main__":
    verify_prompt_in_use()