#!/usr/bin/env python3
"""
Check what max_tokens is actually being used
"""

import sys
sys.path.insert(0, '.')

from app.core.llm_settings_cache import get_llm_settings

try:
    settings = get_llm_settings()
    print("Current LLM Settings:")
    print(f"  Model: {settings.get('model', 'not set')}")
    print(f"  Max Tokens: {settings.get('max_tokens', 'not set')}")
    print(f"  Context Length: {settings.get('context_length', 'not set (will default to 128000)')}")
    
    # Check if max_tokens is a string that needs conversion
    max_tokens = settings.get('max_tokens')
    if isinstance(max_tokens, str):
        print(f"\n⚠️  WARNING: max_tokens is a string: '{max_tokens}'")
        print(f"  This should be a number!")
        
    print("\nTo fix short responses:")
    print("1. Ensure max_tokens is set to 16384 (as a number, not string)")
    print("2. The system should now use this value")
    print("3. Check the debug logs for actual values being sent")
    
except Exception as e:
    print(f"Error getting settings: {e}")
    print("\nMake sure the database is accessible and settings are configured.")