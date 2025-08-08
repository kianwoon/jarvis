#!/usr/bin/env python3
"""Debug pattern matching in detail"""

import re

def debug_pattern_matching():
    tool_context = "ChatGPT-5 launched with significant improvements and new capabilities."
    history = "OpenAI has not released any new models beyond GPT-4."
    
    tool_lower = tool_context.lower()
    history_lower = history.lower()
    
    print("🔍 DEBUGGING PATTERN MATCHING")
    print("=" * 50)
    print(f"Tool: {tool_context}")
    print(f"History: {history}")
    
    # The first pattern from our contradiction patterns
    pattern = {
        'entity': ['openai', 'open ai', 'chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5', 'chatgpt 5', 'gpt 5'],
        'positive_actions': ['released', 'launches', 'introduces', 'announces', 'open-source', 'open source', 'is out', 'available', 'exists', 'launched'],
        'negative_phrases': ['has not released', 'no.*open-source', 'no.*open source', 'not released any', 'does not exist', 'no official release', 'persistent myth', 'myth', 'not exist', 'there is no'],
        'topic': ['models', 'gpt', 'language model', 'chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5', 'chatgpt 5', 'gpt 5']
    }
    
    print(f"\n1️⃣ ENTITY MATCHING:")
    entity_found = False
    for entity in pattern['entity']:
        if entity == '.*':
            entity_found = True
            break
            
        # Direct entity match in both contexts
        tool_match = re.search(entity, tool_lower)
        history_match = re.search(entity, history_lower) 
        direct_match = tool_match and history_match
        
        print(f"  Entity '{entity}': tool={bool(tool_match)}, history={bool(history_match)}, direct={direct_match}")
        
        if direct_match:
            entity_found = True
            break
        
        # Semantic matching
        chatgpt_patterns = ['chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5', 'chatgpt 5', 'gpt 5']
        openai_patterns = ['openai', 'open ai']
        
        if entity in chatgpt_patterns:
            tool_has_chatgpt = re.search(entity, tool_lower)
            history_has_openai = any(re.search(openai_pattern, history_lower) for openai_pattern in openai_patterns)
            history_has_chatgpt = any(re.search(chatgpt_pattern, history_lower) for chatgpt_pattern in chatgpt_patterns)
            
            semantic_match = tool_has_chatgpt and (history_has_openai or history_has_chatgpt)
            print(f"    Semantic (ChatGPT->OpenAI): tool_chatgpt={bool(tool_has_chatgpt)}, history_openai={history_has_openai}, history_chatgpt={history_has_chatgpt}, match={semantic_match}")
            
            if semantic_match:
                entity_found = True
                break
        
        if entity in openai_patterns:
            tool_has_openai = re.search(entity, tool_lower)
            history_has_chatgpt = any(re.search(chatgpt_pattern, history_lower) for chatgpt_pattern in chatgpt_patterns)
            
            semantic_match = tool_has_openai and history_has_chatgpt
            print(f"    Semantic (OpenAI->ChatGPT): tool_openai={bool(tool_has_openai)}, history_chatgpt={history_has_chatgpt}, match={semantic_match}")
            
            if semantic_match:
                entity_found = True
                break
    
    print(f"  📊 Entity found: {entity_found}")
    
    print(f"\n2️⃣ POSITIVE ACTIONS:")
    for action in pattern['positive_actions']:
        match = re.search(action, tool_lower)
        print(f"  '{action}': {bool(match)}")
    positive_in_tool = any(re.search(action, tool_lower) for action in pattern['positive_actions'])
    print(f"  📊 Positive in tool: {positive_in_tool}")
    
    print(f"\n3️⃣ NEGATIVE PHRASES:")
    for phrase in pattern['negative_phrases']:
        match = re.search(phrase, history_lower)
        print(f"  '{phrase}': {bool(match)}")
    negative_in_history = any(re.search(phrase, history_lower) for phrase in pattern['negative_phrases'])
    print(f"  📊 Negative in history: {negative_in_history}")
    
    print(f"\n4️⃣ TOPIC RELEVANCE:")
    for topic in pattern['topic']:
        tool_match = re.search(topic, tool_lower)
        history_match = re.search(topic, history_lower)
        print(f"  '{topic}': tool={bool(tool_match)}, history={bool(history_match)}")
    topic_relevant = any(re.search(topic, tool_lower) or re.search(topic, history_lower) for topic in pattern['topic'])
    print(f"  📊 Topic relevant: {topic_relevant}")
    
    print(f"\n5️⃣ CONFIDENCE DETERMINATION:")
    chatgpt5_existence_indicators = ['chatgpt.*5', 'gpt.*5', 'chatgpt-5', 'gpt-5']
    existence_positive = ['is out', 'released', 'available', 'exists', 'launched']
    existence_negative = ['does not exist', 'persistent myth', 'myth', 'no official release', 'there is no', 'has not released']
    
    has_chatgpt5_reference = any(re.search(indicator, tool_lower) or re.search(indicator, history_lower) 
                               for indicator in chatgpt5_existence_indicators)
    has_existence_positive = any(re.search(pos, tool_lower) for pos in existence_positive)
    has_existence_negative = any(re.search(neg, history_lower) for neg in existence_negative)
    
    print(f"  ChatGPT-5 reference: {has_chatgpt5_reference}")
    print(f"  Existence positive: {has_existence_positive}")  
    print(f"  Existence negative: {has_existence_negative}")
    
    should_be_critical = has_chatgpt5_reference and has_existence_positive and has_existence_negative
    print(f"  📊 Should be CRITICAL: {should_be_critical}")
    
    print(f"\n6️⃣ FINAL CONTRADICTION CHECK:")
    print(f"  Entity found: {entity_found}")
    print(f"  Positive in tool: {positive_in_tool}")
    print(f"  Negative in history: {negative_in_history}")
    print(f"  Topic relevant: {topic_relevant}")
    
    has_contradiction = entity_found and positive_in_tool and negative_in_history and topic_relevant
    print(f"  📊 HAS CONTRADICTION: {has_contradiction}")
    
    if has_contradiction and should_be_critical:
        print(f"  🎯 RESULT: CRITICAL contradiction detected!")
    elif has_contradiction:
        print(f"  ⚠️ RESULT: HIGH contradiction detected")
    else:
        print(f"  ❌ RESULT: No contradiction detected")

if __name__ == "__main__":
    debug_pattern_matching()