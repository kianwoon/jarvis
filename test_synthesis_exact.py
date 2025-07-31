#!/usr/bin/env python3
"""
Test script to reproduce the exact synthesis conditions that cause 22-token truncation.
This simulates the exact code path taken during synthesis with tool results.
"""

import asyncio
import httpx
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

async def test_synthesis_exact_conditions():
    """Test with the exact parameters calculated by the synthesis code"""
    
    print("Testing Synthesis Exact Conditions")
    print("=" * 60)
    
    # Get the actual settings used by synthesis
    settings = get_llm_settings()
    mode_config = get_main_llm_full_config(settings)
    
    print("Current LLM configuration:")
    print(f"- Model: {mode_config.get('model')}")
    print(f"- max_tokens (raw): {mode_config.get('max_tokens')} (type: {type(mode_config.get('max_tokens'))})")
    print(f"- context_length: {mode_config.get('context_length', 'NOT_SET')}")
    print(f"- Temperature: {mode_config.get('temperature')}")
    print(f"- Top-p: {mode_config.get('top_p')}")
    
    # Apply the EXACT same logic as the service.py code
    max_tokens_raw = mode_config.get("max_tokens", 16384)
    context_length = mode_config.get("context_length", 40960)  # This is the potential issue!
    
    try:
        max_tokens = int(max_tokens_raw)
        print(f"\nApplying synthesis logic:")
        print(f"- max_tokens_raw: {max_tokens_raw}")
        print(f"- context_length from config: {context_length}")
        
        # This is the problematic logic from service.py line 3830-3835
        max_output_ratio = 0.4  # Use max 40% of context for output
        max_safe_tokens = int(context_length * max_output_ratio)
        
        print(f"- max_output_ratio: {max_output_ratio}")
        print(f"- calculated max_safe_tokens: {max_safe_tokens}")
        
        if max_tokens > max_safe_tokens:
            print(f"🚨 SAFETY CAP APPLIED: {max_tokens} > {max_safe_tokens}")
            print(f"   Original max_tokens: {max_tokens}")
            print(f"   Limited to: {max_safe_tokens}")
            max_tokens = max_safe_tokens
        else:
            print(f"✅ No safety cap needed: {max_tokens} <= {max_safe_tokens}")
            
    except (ValueError, TypeError):
        # Use fallback based on context length instead of hardcoded value
        fallback_tokens = int(context_length * 0.3)  # Conservative 30% for fallback
        print(f"🚨 FALLBACK APPLIED: Invalid max_tokens, using {fallback_tokens}")
        max_tokens = fallback_tokens
    
    print(f"\nFinal calculated max_tokens: {max_tokens}")
    
    # Now test with this exact value
    print(f"\n{'='*60}")
    print("Testing with calculated max_tokens...")
    
    # Create the exact payload that would be sent during synthesis
    payload = {
        "model": mode_config.get('model'),
        "messages": [
            {
                "role": "system",
                "content": mode_config.get('system_prompt', 'You are Jarvis, an AI assistant.')
            },
            {
                "role": "user", 
                "content": "Based on the search results below, provide a comprehensive answer to the user's question.\n\n🔍 google_search: Recent studies in 2025 show that photosynthesis efficiency has improved through genetic modifications. The process involves chloroplasts capturing light energy through chlorophyll molecules, converting CO2 and water into glucose and oxygen. New research indicates enhanced light-harvesting complexes increase energy conversion rates by 15%.\n\nUser question: How does photosynthesis work in plants? Please provide a detailed explanation."
            }
        ],
        "stream": True,
        "options": {
            "temperature": float(mode_config.get('temperature', 0.7)),
            "top_p": float(mode_config.get('top_p', 1.0)),
            "num_predict": int(max_tokens),  # This is the key parameter!
            "num_ctx": int(context_length),
        }
    }
    
    print(f"Payload parameters:")
    print(f"- model: {payload['model']}")
    print(f"- temperature: {payload['options']['temperature']}")
    print(f"- top_p: {payload['options']['top_p']}")
    print(f"- num_predict: {payload['options']['num_predict']}")
    print(f"- num_ctx: {payload['options']['num_ctx']}")
    print(f"- System prompt length: {len(payload['messages'][0]['content'])} chars")
    print(f"- User prompt length: {len(payload['messages'][1]['content'])} chars")
    
    # Test the streaming
    base_url = mode_config.get('model_server', 'http://localhost:11434')
    if "localhost" in base_url:
        base_url = base_url.replace("localhost", "localhost")  # Keep localhost for local test
    
    print(f"- Ollama URL: {base_url}")
    
    token_count = 0
    response_text = ""
    truncated = False
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            print(f"\nSending request to {base_url}/api/chat...")
            async with client.stream("POST", f"{base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                print(f"Response status: {response.status_code}")
                print("Streaming tokens:")
                print("-" * 40)
                
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
                            
                            # Show first 30 tokens to see the pattern
                            if token_count <= 30:
                                print(f"[{token_count:2d}] {repr(content)}")
                            elif token_count == 31:
                                print("... (continuing)")
                        
                        # Check if done
                        if data_json.get("done", False):
                            print(f"\n[DONE] Stream completed naturally.")
                            break
                            
                        # Check if we're getting cut off around 22 tokens
                        if token_count >= 100:  # Continue longer to see full response
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        continue
    
    except Exception as e:
        print(f"Error during test: {e}")
        return False
    
    print("-" * 40)
    print(f"Results:")
    print(f"- Total tokens: {token_count}")
    print(f"- Total characters: {len(response_text)}")
    print(f"- First 300 chars: {response_text[:300]}...")
    
    # Analyze the results
    if token_count <= 30 and len(response_text) < 200:
        print(f"\n❌ ISSUE REPRODUCED: Only {token_count} tokens received")
        print("This matches the reported truncation issue!")
        
        # Check if our calculated max_tokens could be the cause
        if max_tokens < 100:
            print(f"🎯 ROOT CAUSE FOUND: max_tokens is only {max_tokens}")
            print("This is likely why responses are truncated!")
        
        return False
    else:
        print(f"\n✅ No truncation with these parameters")
        return True

async def test_different_context_lengths():
    """Test how different context_length values affect max_tokens calculation"""
    
    print(f"\n{'='*60}")
    print("Testing Different Context Length Scenarios")
    print('='*60)
    
    settings = get_llm_settings()
    mode_config = get_main_llm_full_config(settings)
    base_max_tokens = int(mode_config.get("max_tokens", 196608))
    
    test_scenarios = [
        {"context_length": 40960, "description": "Default fallback (40960)"},
        {"context_length": 262144, "description": "Current config value"},
        {"context_length": 128000, "description": "Common model limit"},
        {"context_length": 4096, "description": "Very small context"},
        {"context_length": None, "description": "Missing context_length"},
    ]
    
    for scenario in test_scenarios:
        context_length = scenario["context_length"] or 40960  # Apply same fallback as code
        max_output_ratio = 0.4
        max_safe_tokens = int(context_length * max_output_ratio)
        
        if base_max_tokens > max_safe_tokens:
            final_max_tokens = max_safe_tokens
            capped = True
        else:
            final_max_tokens = base_max_tokens
            capped = False
        
        print(f"\nScenario: {scenario['description']}")
        print(f"  context_length: {context_length}")
        print(f"  max_safe_tokens (40%): {max_safe_tokens}")
        print(f"  final_max_tokens: {final_max_tokens}")
        print(f"  capped: {'YES' if capped else 'NO'}")
        
        if final_max_tokens < 100:
            print(f"  ⚠️  This would cause severe truncation!")

if __name__ == "__main__":
    print("Synthesis Exact Conditions Test")
    print("="*60)
    
    # Test the exact synthesis conditions
    result = asyncio.run(test_synthesis_exact_conditions())
    
    # Test different context length scenarios
    asyncio.run(test_different_context_lengths())
    
    print("\nTest complete.")