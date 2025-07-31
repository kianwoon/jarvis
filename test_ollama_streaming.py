#!/usr/bin/env python3
"""
Test script to diagnose Ollama streaming response issues.
This script tests the exact parameters and request format used by the synthesis system.
"""

import asyncio
import httpx
import json

async def test_ollama_streaming():
    """Test Ollama streaming with exact synthesis parameters"""
    
    # Simulate the exact request that would be sent during synthesis
    payload = {
        "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
        "messages": [
            {
                "role": "system", 
                "content": "You are Jarvis, an AI assistant. Based on the search results below, provide a comprehensive answer to the user's question.\n\nSearch Results:\n🔍 google_search: Google Trends data shows photosynthesis-related searches peaked in 2023. Recent studies highlight chlorophyll's role in capturing light energy for glucose production. The process involves light-dependent reactions in thylakoids and the Calvin cycle in stroma.\n\nPlease provide a detailed, helpful response based on the information found."
            },
            {
                "role": "user",
                "content": "Based on the search results provided, explain how photosynthesis works in plants."
            }
        ],
        "stream": True,
        "options": {
            "temperature": 0.6,
            "top_p": 0.9,
            "num_predict": 4000,  # This should be sufficient for comprehensive response
            "num_ctx": 128000,
        }
    }
    
    print(f"Testing Ollama streaming with payload:")
    print(f"- Model: {payload['model']}")
    print(f"- Temperature: {payload['options']['temperature']}")
    print(f"- num_predict (max_tokens): {payload['options']['num_predict']}")
    print(f"- num_ctx (context): {payload['options']['num_ctx']}")
    print(f"- System prompt length: {len(payload['messages'][0]['content'])} chars")
    print(f"- User prompt length: {len(payload['messages'][1]['content'])} chars")
    print()
    
    base_url = "http://localhost:11434"
    
    token_count = 0
    response_text = ""
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            print("Sending streaming request to Ollama...")
            async with client.stream("POST", f"{base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                print(f"Response status: {response.status_code}")
                print("Streaming response:")
                print("-" * 50)
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data_json = json.loads(line)
                        message = data_json.get("message", {})
                        
                        if message and "content" in message:
                            content = message["content"]
                            response_text += content
                            token_count += 1
                            
                            # Print token with count
                            print(f"[{token_count:3d}] {repr(content)}")
                            
                            # If we're hitting the truncation issue, we should see it stop around 22 tokens
                            if token_count >= 25:
                                print(f"... (continuing, received {token_count} tokens so far)")
                                break
                        
                        # Check if done
                        if data_json.get("done", False):
                            print(f"\n[DONE] Stream completed.")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} - Line: {line}")
                        continue
    
    except Exception as e:
        print(f"Error during streaming test: {e}")
        return False
    
    print("-" * 50)
    print(f"Total tokens received: {token_count}")
    print(f"Total response length: {len(response_text)} characters")
    print(f"First 200 chars: {response_text[:200]}...")
    
    # Check if we hit the 22-token issue
    if token_count <= 25 and len(response_text) < 200:
        print(f"\n❌ ISSUE DETECTED: Only received {token_count} tokens (~{len(response_text)} chars)")
        print("This matches the reported 22-token truncation issue!")
        return False
    else:
        print(f"\n✅ Streaming appears to work correctly - received {token_count} tokens")
        return True

async def test_different_parameters():
    """Test with different parameter combinations to isolate the issue"""
    
    test_cases = [
        {
            "name": "High max_tokens",
            "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": 16384, "num_ctx": 128000}
        },
        {
            "name": "Low max_tokens", 
            "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": 100, "num_ctx": 128000}
        },
        {
            "name": "Very low max_tokens",
            "options": {"temperature": 0.6, "top_p": 0.9, "num_predict": 25, "num_ctx": 128000}
        },
        {
            "name": "Default values",
            "options": {"temperature": 0.7, "top_p": 1.0, "num_predict": 4000, "num_ctx": 128000}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"Parameters: {test_case['options']}")
        print('='*60)
        
        payload = {
            "model": "qwen3:30b-a3b-instruct-2507-q4_K_M",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain photosynthesis in plants. Provide a detailed explanation."}
            ],
            "stream": True,
            "options": test_case['options']
        }
        
        token_count = 0
        response_text = ""
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", "http://localhost:11434/api/chat", json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        
                        try:
                            data_json = json.loads(line)
                            message = data_json.get("message", {})
                            
                            if message and "content" in message:
                                content = message["content"]
                                response_text += content
                                token_count += 1
                            
                            if data_json.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            print(f"Result: {token_count} tokens, {len(response_text)} characters")
            
            if token_count <= 25:
                print(f"❌ TRUNCATED - Only {token_count} tokens received")
            else:
                print(f"✅ Good response - {token_count} tokens received")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Ollama Streaming Diagnosis Tool")
    print("="*50)
    
    # First test with synthesis-like parameters
    print("1. Testing with synthesis-like parameters...")
    result = asyncio.run(test_ollama_streaming())
    
    # Then test with different parameter combinations
    print("\n2. Testing different parameter combinations...")
    asyncio.run(test_different_parameters())
    
    print("\nDiagnosis complete.")