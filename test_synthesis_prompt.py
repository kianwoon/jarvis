#!/usr/bin/env python3
"""
Test with the exact synthesis system prompt to reproduce the JSON response issue.
"""

import asyncio
import httpx
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

async def test_with_synthesis_prompt(model_name: str):
    """Test model with the exact synthesis system prompt"""
    
    print(f"\n{'='*80}")
    print(f"Testing Model with Synthesis Prompt: {model_name}")
    print('='*80)
    
    # Get the actual system prompt used by synthesis
    settings = get_llm_settings()
    mode_config = get_main_llm_full_config(settings)
    synthesis_system_prompt = mode_config.get('system_prompt', 'You are Jarvis, an AI assistant.')
    
    print(f"System prompt length: {len(synthesis_system_prompt)} characters")
    print(f"System prompt preview: {synthesis_system_prompt[:200]}...")
    
    # Create payload matching synthesis conditions
    user_content = """Based on the search results below, provide a comprehensive answer to the user's question.

🔍 google_search: Recent studies in 2025 show that photosynthesis efficiency has improved through genetic modifications. The process involves chloroplasts capturing light energy through chlorophyll molecules, converting CO2 and water into glucose and oxygen. New research indicates enhanced light-harvesting complexes increase energy conversion rates by 15%.

User question: How does photosynthesis work in plants? Please provide a detailed explanation."""

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": synthesis_system_prompt
            },
            {
                "role": "user", 
                "content": user_content
            }
        ],
        "stream": True,
        "options": {
            "temperature": float(mode_config.get('temperature', 0.7)),
            "top_p": float(mode_config.get('top_p', 1.0)),
            "num_predict": 1000,  # Enough to see the full response pattern
            "num_ctx": int(mode_config.get('context_length', 40960)),
        }
    }
    
    print(f"Request parameters:")
    print(f"- temperature: {payload['options']['temperature']}")
    print(f"- top_p: {payload['options']['top_p']}")
    print(f"- num_predict: {payload['options']['num_predict']}")
    print(f"- num_ctx: {payload['options']['num_ctx']}")
    
    token_count = 0
    response_text = ""
    first_50_tokens = []
    
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
                            
                            # Capture first 50 tokens for analysis
                            if token_count <= 50:
                                first_50_tokens.append(content)
                                print(f"[{token_count:2d}] {repr(content)}")
                        
                        # Check if response is done
                        if data_json.get("done", False):
                            print(f"\n[DONE] Stream completed. Total tokens: {token_count}")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue
    
    except Exception as e:
        print(f"Error testing {model_name}: {e}")
        return None
    
    # Analyze the response format
    print(f"\nResponse Analysis:")
    print(f"- Total tokens: {token_count}")
    print(f"- Total characters: {len(response_text)}")
    print(f"- First 100 chars: {response_text[:100]}...")
    
    # Check if it looks like JSON
    response_clean = response_text.strip()
    is_json = False
    parsed_json = None
    
    if response_clean.startswith('{') and response_clean.endswith('}'):
        try:
            parsed_json = json.loads(response_clean)
            is_json = True
            print(f"✅ Response is valid JSON!")
            print(f"- JSON keys: {list(parsed_json.keys())}")
            
            # Check for specific fields
            if 'useful' in parsed_json:
                print(f"- 'useful' field: {parsed_json['useful']}")
            if 'comment' in parsed_json:
                print(f"- 'comment' field: {parsed_json['comment'][:100]}...")
            if 'jarvis_opinion' in parsed_json:
                print(f"- 'jarvis_opinion' field: {parsed_json['jarvis_opinion'][:100]}...")
            if 'answer' in parsed_json:
                print(f"- 'answer' field: {parsed_json['answer'][:100]}...")
                
        except json.JSONDecodeError:
            print(f"❌ Looks like JSON but invalid format")
    else:
        print(f"📝 Response is plain text")
    
    return {
        'model': model_name,
        'tokens': token_count,
        'is_json': is_json,
        'parsed_json': parsed_json,
        'raw_response': response_text,
        'first_tokens': first_50_tokens[:10]  # First 10 tokens
    }

async def main():
    """Test both models with synthesis prompt"""
    
    models = [
        "qwen3:30b-a3b-q4_K_M",
        "qwen3:30b-a3b-instruct-2507-q4_K_M"
    ]
    
    results = []
    
    for model in models:
        result = await test_with_synthesis_prompt(model)
        if result:
            results.append(result)
        await asyncio.sleep(2)  # Wait between tests
    
    # Compare results
    print(f"\n{'='*80}")
    print("SYNTHESIS PROMPT COMPARISON")
    print('='*80)
    
    for result in results:
        print(f"\nModel: {result['model']}")
        print(f"- Tokens: {result['tokens']}")
        print(f"- Is JSON: {result['is_json']}")
        print(f"- First tokens: {result['first_tokens']}")
        
        if result['is_json'] and result['parsed_json']:
            print(f"- JSON structure detected with keys: {list(result['parsed_json'].keys())}")
    
    # Provide frontend compatibility plan
    print(f"\n{'='*80}")
    print("FRONTEND COMPATIBILITY PLAN")
    print('='*80)
    
    json_models = [r for r in results if r['is_json']]
    text_models = [r for r in results if not r['is_json']]
    
    if json_models and text_models:
        print("🔍 Different response formats detected!")
        print("\nRequired changes:")
        print("1. Detect response format in frontend (JSON vs text)")
        print("2. Extract content from appropriate field for JSON responses")
        print("3. Maintain backward compatibility for text responses")
        
        if json_models:
            json_result = json_models[0]
            if json_result['parsed_json']:
                print(f"\nJSON fields to check for content extraction:")
                for key, value in json_result['parsed_json'].items():
                    if isinstance(value, str) and len(value) > 50:
                        print(f"- {key}: (contains {len(value)} chars)")
    else:
        if json_models:
            print("All models return JSON - implement JSON parsing")
        else:
            print("All models return text - current handling should work")

if __name__ == "__main__":
    print("Synthesis Prompt Response Format Test")
    print("="*80)
    
    asyncio.run(main())
    
    print("\nTest complete.")