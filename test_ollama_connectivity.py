#!/usr/bin/env python3
"""
Test Ollama connectivity and model availability
"""

import asyncio
import httpx
import json
from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

async def test_ollama_connectivity():
    """Test if Ollama server is responding and model is available."""
    
    # Get configuration
    llm_settings = get_llm_settings()
    config = get_main_llm_full_config(llm_settings)
    
    model_name = config.get("model", "qwen3:30b")
    base_url = config.get("model_server", "http://localhost:11434")
    
    print(f"Testing Ollama connectivity...")
    print(f"Model: {model_name}")
    print(f"Server: {base_url}")
    print("-" * 50)
    
    try:
        # Test 1: Server availability
        async with httpx.AsyncClient(timeout=10.0) as client:
            print("1. Testing server availability...")
            response = await client.get(f"{base_url}/api/version")
            if response.status_code == 200:
                version_data = response.json()
                print(f"   ✅ Ollama server responding - version: {version_data.get('version', 'unknown')}")
            else:
                print(f"   ❌ Server error: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    try:
        # Test 2: List available models
        async with httpx.AsyncClient(timeout=10.0) as client:
            print("2. Checking available models...")
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m["name"] for m in models_data.get("models", [])]
                print(f"   Available models: {available_models}")
                
                if model_name in available_models:
                    print(f"   ✅ Target model '{model_name}' is available")
                else:
                    print(f"   ❌ Target model '{model_name}' NOT found in available models")
                    print(f"   💡 Try: ollama pull {model_name}")
                    return False
            else:
                print(f"   ❌ Error listing models: {response.status_code}")
                return False
    except Exception as e:
        print(f"   ❌ Error checking models: {e}")
        return False
    
    try:
        # Test 3: Simple generation test
        async with httpx.AsyncClient(timeout=60.0) as client:
            print("3. Testing model generation...")
            
            test_payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Be concise."},
                    {"role": "user", "content": "Say hello in exactly 3 words."}
                ],
                "stream": False,
                "options": {
                    "num_predict": 10,
                    "temperature": 0.1
                }
            }
            
            response = await client.post(f"{base_url}/api/chat", json=test_payload)
            if response.status_code == 200:
                chat_data = response.json()
                message = chat_data.get("message", {})
                content = message.get("content", "").strip()
                print(f"   ✅ Generation successful: '{content}'")
                
                if len(content) > 0:
                    print(f"   ✅ Model is responding properly")
                    return True
                else:
                    print(f"   ❌ Empty response from model")
                    return False
            else:
                error_text = await response.aread()
                print(f"   ❌ Generation failed: {response.status_code} - {error_text.decode()}")
                return False
    except Exception as e:
        print(f"   ❌ Generation test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_ollama_connectivity())
    if success:
        print("\n🎉 All tests passed! Ollama is working correctly.")
    else:
        print("\n💥 Tests failed! Check the issues above.")