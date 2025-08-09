#!/usr/bin/env python3
"""
Test that the LLM is using the updated system prompt
"""

import requests
import json
import sys

def test_llm_prompt():
    """Test the LLM endpoint to see if new prompt is active"""
    
    print("\n🧪 Testing LLM with new prompt...")
    
    # Test message that should trigger the new prompt behavior
    test_message = "What is Claude Code and when was it released?"
    
    url = "http://localhost:8000/api/v1/langchain/chat"
    
    payload = {
        "message": test_message,
        "use_knowledge_graph": False,
        "use_rag": False,
        "use_tools": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"\n📤 Sending test message: {test_message}")
    print("   (This tests if the LLM accepts new information beyond training data)")
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '')
            
            print(f"\n📥 Response received (first 500 chars):")
            print(f"   {answer[:500]}...")
            
            # Check for signs of the new prompt behavior
            denial_phrases = [
                "i don't have information",
                "i'm not aware",
                "i cannot confirm",
                "beyond my training",
                "knowledge cutoff",
                "don't have data",
                "cannot provide information about"
            ]
            
            has_denial = any(phrase in answer.lower() for phrase in denial_phrases)
            
            if has_denial:
                print("\n⚠️ WARNING: Response contains denial phrases")
                print("   The LLM might still be using the old prompt")
                print("   Consider restarting the backend: docker restart jarvis-app-1")
                return False
            else:
                print("\n✅ SUCCESS: No denial phrases detected")
                print("   The LLM appears to be using the new prompt")
                return True
                
        else:
            print(f"\n❌ Error: Status code {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing LLM: {e}")
        return False

def check_settings_in_memory():
    """Check if settings are properly loaded in memory"""
    
    print("\n🔍 Checking in-memory settings...")
    
    # Call the settings endpoint to see what's loaded
    url = "http://localhost:8000/api/v1/settings/llm"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            settings = response.json()
            prompt = settings.get('main_llm', {}).get('system_prompt', '')
            
            print(f"   Prompt length from API: {len(prompt)} characters")
            
            # Check if it has the new evaluation guidelines
            if "evaluate and synthesize the information thoughtfully" in prompt:
                print("   ✅ New prompt with evaluation guidelines is loaded")
                return True
            elif "accept and use that information as current and accurate" in prompt:
                print("   ⚠️ Old prompt without evaluation guidelines is loaded")
                return False
            else:
                print("   ❓ Unknown prompt version")
                return False
        else:
            print(f"   ❌ Failed to get settings: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error checking settings: {e}")
        return False

def main():
    """Main test execution"""
    
    print("=" * 60)
    print("🧪 LLM Prompt Update Test")
    print("=" * 60)
    
    # Check settings first
    settings_ok = check_settings_in_memory()
    
    # Test actual LLM behavior
    llm_ok = test_llm_prompt()
    
    print("\n" + "=" * 60)
    print("📊 Test Results:")
    print(f"   Settings check: {'✅ PASS' if settings_ok else '❌ FAIL'}")
    print(f"   LLM behavior:   {'✅ PASS' if llm_ok else '❌ FAIL'}")
    
    if settings_ok and llm_ok:
        print("\n🎉 All tests passed! The new prompt is active.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Troubleshooting steps:")
        print("   1. Clear Redis cache: redis-cli del llm_settings_cache")
        print("   2. Reload cache: curl -X POST http://localhost:8000/api/v1/settings/llm/cache/reload")
        print("   3. Restart backend: docker restart jarvis-app-1")
        print("   4. Wait 30 seconds and run this test again")
        return 1

if __name__ == "__main__":
    sys.exit(main())