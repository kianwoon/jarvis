#!/usr/bin/env python3
"""
Test both Qwen3 model variants to understand their response formats
and determine how to make the frontend compatible with both.
"""

import asyncio
import httpx
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

async def test_model_response_format(model_name: str, question: str):
    """Test a specific model's response format"""
    
    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print('='*80)
    
    # Create test payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are Jarvis, an AI assistant. Provide comprehensive and helpful responses."
            },
            {
                "role": "user", 
                "content": question
            }
        ],
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 500,  # Limited for testing
            "num_ctx": 4096,
        }
    }
    
    print(f"Test question: {question}")
    print(f"Expected tokens: {payload['options']['num_predict']}")
    
    token_count = 0
    response_text = ""
    json_responses = []
    
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
                            
                            # Show first few tokens to see pattern
                            if token_count <= 10:
                                print(f"[{token_count:2d}] {repr(content)}")
                        
                        # Check if response is done
                        if data_json.get("done", False):
                            print(f"\n[DONE] Stream completed. Total tokens: {token_count}")
                            break
                            
                        # Store entire response objects for analysis
                        json_responses.append(data_json)
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue
    
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return None
    
    # Analyze the response
    print(f"\nResponse Analysis:")
    print(f"- Total tokens: {token_count}")
    print(f"- Total characters: {len(response_text)}")
    print(f"- Average token length: {len(response_text)/token_count if token_count > 0 else 0:.1f} chars")
    
    # Check if response looks like JSON
    try:
        # Try to parse the entire response as JSON
        parsed_response = json.loads(response_text)
        print(f"- Response is valid JSON!")
        print(f"- JSON keys: {list(parsed_response.keys()) if isinstance(parsed_response, dict) else 'Not a dict'}")
        
        # Check for known instruct-2507 format fields
        if isinstance(parsed_response, dict):
            if 'useful' in parsed_response:
                print(f"- Contains 'useful' field: {parsed_response['useful']}")
            if 'comment' in parsed_response:
                print(f"- Contains 'comment' field: {parsed_response['comment'][:100]}...")
            if 'jarvis_opinion' in parsed_response:
                print(f"- Contains 'jarvis_opinion' field: {parsed_response['jarvis_opinion'][:100]}...")
        
        return {
            'model': model_name,
            'format': 'json',
            'tokens': token_count,
            'content': parsed_response,
            'raw_text': response_text
        }
        
    except json.JSONDecodeError:
        print(f"- Response is plain text (not JSON)")
        print(f"- First 200 chars: {response_text[:200]}...")
        
        return {
            'model': model_name,
            'format': 'text',
            'tokens': token_count,
            'content': response_text,
            'raw_text': response_text
        }

async def test_both_models():
    """Test both Qwen3 model variants"""
    
    question = "How does photosynthesis work in plants? Please provide a detailed explanation."
    
    models_to_test = [
        "qwen3:30b-a3b-q4_K_M",
        "qwen3:30b-a3b-instruct-2507-q4_K_M"
    ]
    
    results = []
    
    for model in models_to_test:
        result = await test_model_response_format(model, question)
        if result:
            results.append(result)
        
        # Wait between tests
        await asyncio.sleep(1)
    
    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON ANALYSIS")
    print('='*80)
    
    for i, result in enumerate(results):
        print(f"\nModel {i+1}: {result['model']}")
        print(f"- Format: {result['format']}")
        print(f"- Tokens: {result['tokens']}")
        
        if result['format'] == 'json':
            print(f"- JSON structure: {type(result['content'])}")
            if isinstance(result['content'], dict):
                print(f"- JSON keys: {list(result['content'].keys())}")
        else:
            print(f"- Text length: {len(result['content'])}")
    
    # Provide recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR FRONTEND COMPATIBILITY")
    print('='*80)
    
    json_models = [r for r in results if r['format'] == 'json']
    text_models = [r for r in results if r['format'] == 'text']
    
    if json_models and text_models:
        print("✅ Mixed response formats detected!")
        print("\nFrontend needs to handle both:")
        print("1. JSON responses with structured fields (instruct-2507)")
        print("2. Plain text responses (standard model)")
        
        if json_models:
            json_result = json_models[0]
            if isinstance(json_result['content'], dict):
                print(f"\nJSON format fields to extract:")
                for key in json_result['content'].keys():
                    print(f"- {key}: {type(json_result['content'][key])}")
        
        print(f"\nRequired frontend changes:")
        print("1. Detect response format (JSON vs text)")
        print("2. Extract appropriate content field from JSON responses")
        print("3. Fall back to original text handling for non-JSON")
        
    elif json_models:
        print("All models return JSON - frontend needs JSON parsing")
    elif text_models:
        print("All models return text - current frontend should work")

if __name__ == "__main__":
    print("Model Response Format Compatibility Test")
    print("="*80)
    
    asyncio.run(test_both_models())
    
    print("\nTest complete.")