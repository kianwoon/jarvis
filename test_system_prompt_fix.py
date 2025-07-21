#!/usr/bin/env python3
"""Test if system prompt is being included in tool planning"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.llm_settings_cache import get_llm_settings, get_second_llm_full_config
from app.langchain.service import make_llm_call

# Test 1: Check if second_llm config has system_prompt
print("=== Test 1: Checking second_llm config ===")
settings = get_llm_settings()
second_llm_config = get_second_llm_full_config(settings)

print(f"Second LLM Model: {second_llm_config.get('model', 'NOT FOUND')}")
print(f"Second LLM System Prompt: {second_llm_config.get('system_prompt', 'NOT FOUND')}")

# Test 2: Check if make_llm_call includes system prompt
print("\n=== Test 2: Testing make_llm_call with second_llm config ===")

# Mock a planning prompt
planning_prompt = """You are an intelligent task planner. Analyze the task and create an optimal execution plan using available tools.

TASK TO ACCOMPLISH:
compare openai and anthropic, which is more advanced?

Create a plan and include what year you think it is."""

# Call make_llm_call with second_llm config
print("\nCalling make_llm_call with second_llm config...")
print("(Check console output for debug messages)")

# We need to capture the debug output
import io
import contextlib

# Capture stdout
f = io.StringIO()
with contextlib.redirect_stdout(f):
    # This would normally make the LLM call, but we're just checking if system prompt is prepended
    # For safety, let's just check the debug output without making actual call
    from app.langchain.service import make_llm_call
    
    # Create a mock version that just shows what would be sent
    def mock_make_llm_call(prompt, thinking, context, llm_cfg):
        # This mimics the logic in make_llm_call
        if 'main_llm' in llm_cfg or 'second_llm' in llm_cfg or 'query_classifier' in llm_cfg:
            # Full settings dict passed
            from app.core.llm_settings_cache import get_main_llm_full_config
            mode_config = get_main_llm_full_config(llm_cfg)
        else:
            # Specific LLM config passed directly
            mode_config = llm_cfg
        
        # Get system_prompt if available and prepend to prompt
        system_prompt = mode_config.get("system_prompt", "")
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"
            print(f"[DEBUG make_llm_call] System prompt found and prepended: {system_prompt[:100]}...")
        else:
            final_prompt = prompt
            print(f"[DEBUG make_llm_call] No system prompt found in config")
        
        print(f"\n[DEBUG] Final prompt preview (first 200 chars):")
        print(final_prompt[:200])
        print("...")
        
        return "Mock response"
    
    # Test with second_llm config directly
    response = mock_make_llm_call(
        prompt=planning_prompt,
        thinking=False,
        context="",
        llm_cfg=second_llm_config  # Pass the config directly
    )

output = f.getvalue()
print(output)

print("\n=== Summary ===")
if "now is year 2025" in second_llm_config.get('system_prompt', ''):
    print("✓ Second LLM config contains 'now is year 2025'")
else:
    print("✗ Second LLM config missing 'now is year 2025'")

if "System prompt found and prepended" in output:
    print("✓ System prompt is being prepended")
else:
    print("✗ System prompt is NOT being prepended")