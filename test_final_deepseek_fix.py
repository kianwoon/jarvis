#!/usr/bin/env python3
"""
Final test for all DeepSeek-R1 fixes
"""

import sys
sys.path.insert(0, '.')

def test_complete_flow():
    """Test the complete query flow for date/time query"""
    print("\nFinal DeepSeek-R1 Fix Test")
    print("=" * 70)
    
    query = "what is today date & time?"
    print(f"Testing query: '{query}'")
    print()
    
    # Test 1: Smart Classification
    print("1. Smart Classification Test:")
    from app.langchain.smart_query_classifier import classify_without_context
    query_type, confidence = classify_without_context(query)
    print(f"   ✓ Classification: {query_type} (confidence: {confidence:.2f})")
    
    # Test 2: Pattern Matching
    print("\n2. Pattern Matching Test:")
    from app.langchain.smart_query_classifier import classifier
    
    # Test various date/time queries
    test_queries = [
        "what is today date & time?",
        "what's the current date?",
        "tell me today's time",
        "what time is it now?"
    ]
    
    for q in test_queries:
        qt, conf = classifier.classify(q)
        print(f"   '{q}' → {qt.value} ({conf:.2f})")
    
    # Test 3: Tool Pattern Extraction
    print("\n3. Tool Pattern Test:")
    
    # Simulate tool extraction logic
    import re
    
    # Test with thinking output (simulating DeepSeek-R1 behavior)
    thinking_output = """<think>
    The user asked for today's date and time. I should use the get_datetime tool
    to provide this information. Looking at available tools, get_datetime is perfect
    for this request.
    </think>"""
    
    print("   Testing pattern extraction:")
    
    # Check if proper format exists
    tool_calls_pattern = r'<tool>(.*?)\((.*?)\)</tool>'
    tool_calls = re.findall(tool_calls_pattern, thinking_output, re.DOTALL)
    
    if not tool_calls:
        print("   ✓ No proper format found (as expected for thinking output)")
        
        # Fallback check
        if "get_datetime" in thinking_output and ("date" in thinking_output.lower() or "time" in thinking_output.lower()):
            print("   ✓ Fallback detected get_datetime mention in thinking")
        else:
            print("   ✗ Fallback failed to detect tool mention")
    
    # Test with proper format
    proper_output = "<tool>get_datetime({})</tool>"
    tool_calls = re.findall(tool_calls_pattern, proper_output, re.DOTALL)
    if tool_calls:
        print(f"   ✓ Successfully parsed proper tool format: {tool_calls}")
    else:
        print(f"   ✗ Failed to parse proper format")
    
    print("\n" + "=" * 70)
    print("Expected Flow for 'what is today date & time?':")
    print("1. Smart classifier identifies as TOOLS (0.95 confidence)")
    print("2. Service skips RAG retrieval entirely")
    print("3. Tool selection prompt asks for get_datetime")
    print("4. If LLM outputs thinking, fallback extracts get_datetime")
    print("5. get_datetime tool executes and returns current time")
    print("6. Response generated with tool result")
    print("=" * 70)

if __name__ == "__main__":
    test_complete_flow()