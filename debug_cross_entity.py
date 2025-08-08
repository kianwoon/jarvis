#!/usr/bin/env python3
"""Debug the cross-entity semantic matching issue"""

import re

def debug_cross_entity_matching():
    tool_context = "ChatGPT-5 launched with significant improvements and new capabilities."
    history = "OpenAI has not released any new models beyond GPT-4."
    
    tool_lower = tool_context.lower()
    history_lower = history.lower()
    
    print(f"Tool context: {tool_context}")
    print(f"History: {history}")
    print(f"Tool lower: {tool_lower}")
    print(f"History lower: {history_lower}")
    
    # Check ChatGPT-5 existence indicators
    chatgpt5_existence_indicators = ['chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5']
    existence_positive = ['is out', 'released', 'available', 'exists', 'launched']
    existence_negative = ['does not exist', 'persistent myth', 'myth', 'no official release', 'there is no', 'has not released']
    
    print(f"\n🔍 Checking ChatGPT-5 indicators:")
    for indicator in chatgpt5_existence_indicators:
        tool_match = re.search(indicator, tool_lower)
        history_match = re.search(indicator, history_lower)
        print(f"  {indicator}: tool={bool(tool_match)}, history={bool(history_match)}")
    
    print(f"\n🔍 Checking existence positive:")
    for pos in existence_positive:
        match = re.search(pos, tool_lower)
        print(f"  {pos}: {bool(match)}")
    
    print(f"\n🔍 Checking existence negative:")
    for neg in existence_negative:
        match = re.search(neg, history_lower)
        print(f"  {neg}: {bool(match)}")
    
    has_chatgpt5_reference = any(re.search(indicator, tool_lower) or re.search(indicator, history_lower) 
                               for indicator in chatgpt5_existence_indicators)
    has_existence_positive = any(re.search(pos, tool_lower) for pos in existence_positive)
    has_existence_negative = any(re.search(neg, history_lower) for neg in existence_negative)
    
    print(f"\n📊 Final checks:")
    print(f"has_chatgpt5_reference: {has_chatgpt5_reference}")
    print(f"has_existence_positive: {has_existence_positive}")
    print(f"has_existence_negative: {has_existence_negative}")
    
    should_be_critical = has_chatgpt5_reference and has_existence_positive and has_existence_negative
    print(f"Should be critical: {should_be_critical}")

if __name__ == "__main__":
    debug_cross_entity_matching()