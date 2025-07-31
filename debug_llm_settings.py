#!/usr/bin/env python3
"""
Debug script to check the actual LLM settings being used during synthesis.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_settings_cache import get_llm_settings, get_main_llm_full_config

def debug_llm_settings():
    """Debug the actual LLM settings being used"""
    
    print("LLM Settings Debug")
    print("=" * 50)
    
    try:
        # Get the raw settings
        settings = get_llm_settings()
        print("Raw LLM settings structure:")
        print(json.dumps(settings, indent=2, default=str))
        
        print("\n" + "="*50)
        
        # Get the main LLM full config that would be used during synthesis
        main_llm_config = get_main_llm_full_config(settings)
        print("Main LLM full config (used during synthesis):")
        print(json.dumps(main_llm_config, indent=2, default=str))
        
        print("\n" + "="*50)
        
        # Check specific parameters that could cause truncation
        max_tokens = main_llm_config.get('max_tokens', 'NOT_FOUND')
        temperature = main_llm_config.get('temperature', 'NOT_FOUND')
        top_p = main_llm_config.get('top_p', 'NOT_FOUND')
        model = main_llm_config.get('model', 'NOT_FOUND')
        
        print("Key synthesis parameters:")
        print(f"- Model: {model}")
        print(f"- Max tokens: {max_tokens} (type: {type(max_tokens)})")
        print(f"- Temperature: {temperature} (type: {type(temperature)})")
        print(f"- Top-p: {top_p} (type: {type(top_p)})")
        
        # Check if max_tokens could be causing the issue
        if isinstance(max_tokens, (int, str)):
            try:
                max_tokens_int = int(max_tokens)
                if max_tokens_int < 100:
                    print(f"⚠️  WARNING: max_tokens ({max_tokens_int}) is very low - this could cause truncation!")
                elif max_tokens_int > 50000:
                    print(f"ℹ️  INFO: max_tokens ({max_tokens_int}) is very high - should not cause truncation")
                else:
                    print(f"ℹ️  INFO: max_tokens ({max_tokens_int}) appears reasonable")
            except (ValueError, TypeError):
                print(f"❌ ERROR: max_tokens value '{max_tokens}' is not a valid integer")
        
        print("\n" + "="*50)
        print("Context length and other settings:")
        context_length = settings.get('context_length', 'NOT_FOUND')
        print(f"- Context length: {context_length}")
        
        # Check thinking/non-thinking mode params
        thinking_params = settings.get('thinking_mode_params', {})
        non_thinking_params = settings.get('non_thinking_mode_params', {})
        
        print(f"- Thinking mode params: {json.dumps(thinking_params, indent=2)}")
        print(f"- Non-thinking mode params: {json.dumps(non_thinking_params, indent=2)}")
        
        # Check which mode is being used
        main_llm = settings.get('main_llm', {})
        current_mode = main_llm.get('mode', 'unknown')
        print(f"- Current mode: {current_mode}")
        
    except Exception as e:
        print(f"Error loading LLM settings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_llm_settings()