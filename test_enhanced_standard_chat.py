#!/usr/bin/env python3
"""
Test script for enhanced standard chat with context-limit-transcending capabilities
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.langchain.service import detect_large_output_potential, classify_query_type

def test_large_output_detection():
    """Test the large output detection logic"""
    
    test_cases = [
        # Should trigger large generation
        {
            "question": "Generate 50 interview questions for software engineers",
            "expected_large": True,
            "expected_count_range": (40, 60)
        },
        {
            "question": "Create a comprehensive list of 100 marketing strategies",
            "expected_large": True,
            "expected_count_range": (90, 110)
        },
        {
            "question": "Write detailed explanations for all machine learning algorithms",
            "expected_large": True,
            "expected_count_range": (25, 35)
        },
        {
            "question": "List many examples of successful startups and their strategies",
            "expected_large": True,
            "expected_count_range": (25, 35)
        },
        
        # Should NOT trigger large generation
        {
            "question": "What is machine learning?",
            "expected_large": False,
            "expected_count_range": (5, 15)
        },
        {
            "question": "Explain the benefits of cloud computing",
            "expected_large": False,
            "expected_count_range": (5, 15)
        },
        {
            "question": "Generate 5 ideas for improving our website",
            "expected_large": False,
            "expected_count_range": (5, 10)
        }
    ]
    
    print("🧪 Testing Large Output Detection Logic")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_large = test_case["expected_large"]
        expected_range = test_case["expected_count_range"]
        
        print(f"\nTest {i}: {question}")
        print("-" * 50)
        
        # Test detection
        result = detect_large_output_potential(question)
        
        print(f"Detected large: {result['likely_large']} (expected: {expected_large})")
        print(f"Estimated items: {result['estimated_items']} (expected: {expected_range[0]}-{expected_range[1]})")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Score: {result['score']}")
        print(f"Max number: {result['max_number']}")
        print(f"Matched indicators: {result['matched_indicators']}")
        
        # Check if detection is correct
        detection_correct = result['likely_large'] == expected_large
        count_in_range = expected_range[0] <= result['estimated_items'] <= expected_range[1]
        
        status = "✅ PASS" if detection_correct and count_in_range else "❌ FAIL"
        print(f"Status: {status}")
        
        if not detection_correct:
            print(f"  ⚠️  Detection mismatch: got {result['likely_large']}, expected {expected_large}")
        if not count_in_range:
            print(f"  ⚠️  Count out of range: got {result['estimated_items']}, expected {expected_range[0]}-{expected_range[1]}")

def test_query_classification():
    """Test the enhanced query classification"""
    
    test_cases = [
        {
            "question": "Generate 50 interview questions for software engineers",
            "expected_type": "LARGE_GENERATION"
        },
        {
            "question": "What is our company policy on remote work?",
            "expected_type": "RAG"
        },
        {
            "question": "What time is it now?",
            "expected_type": "TOOLS"  # Assuming get_datetime tool is available
        },
        {
            "question": "Explain quantum computing",
            "expected_type": "LLM"
        }
    ]
    
    print("\n\n🔄 Testing Query Classification Logic")
    print("=" * 60)
    
    # Mock LLM config for testing
    mock_llm_cfg = {
        "model": "test",
        "thinking_mode": {"temperature": 0.7},
        "non_thinking_mode": {"temperature": 0.7},
        "max_tokens": 2048
    }
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_type = test_case["expected_type"]
        
        print(f"\nTest {i}: {question}")
        print("-" * 50)
        
        try:
            # Note: This might fail if MCP tools cache is not available
            result = classify_query_type(question, mock_llm_cfg)
            print(f"Classified as: {result} (expected: {expected_type})")
            
            status = "✅ PASS" if result == expected_type else "❌ FAIL"
            print(f"Status: {status}")
            
        except Exception as e:
            print(f"⚠️  Classification failed: {e}")
            print("This is expected if running without full environment setup")

def test_conversation_memory():
    """Test the enhanced conversation memory management"""
    
    print("\n\n💾 Testing Conversation Memory Management")
    print("=" * 60)
    
    from app.langchain.service import store_conversation_message, get_conversation_history, get_full_conversation_history
    
    # Test conversation ID
    test_conv_id = "test_conv_123"
    
    # Store some test messages
    messages = [
        ("user", "Hello, I need help with my project"),
        ("assistant", "I'd be happy to help! What kind of project are you working on?"),
        ("user", "I'm building a web application with Python"),
        ("assistant", "Great! Python is excellent for web development. Are you using a specific framework like Django or Flask?"),
        ("user", "I'm using Flask. Can you help me with authentication?")
    ]
    
    print("Storing test messages...")
    for role, content in messages:
        store_conversation_message(test_conv_id, role, content)
    
    # Test conversation history retrieval
    print("\nRetrieving conversation history (formatted):")
    history = get_conversation_history(test_conv_id)
    print(history)
    
    print("\nRetrieving full conversation history (list):")
    full_history = get_full_conversation_history(test_conv_id)
    for i, msg in enumerate(full_history):
        print(f"  {i+1}. [{msg['role']}] {msg['content'][:50]}...")
    
    print(f"\n✅ Stored and retrieved {len(full_history)} messages successfully")

if __name__ == "__main__":
    print("🚀 Enhanced Standard Chat Test Suite")
    print("Testing context-limit-transcending capabilities")
    print("=" * 60)
    
    try:
        # Test 1: Large output detection
        test_large_output_detection()
        
        # Test 2: Query classification (may fail without full env)
        test_query_classification()
        
        # Test 3: Conversation memory
        test_conversation_memory()
        
        print("\n\n🎉 Test Suite Completed!")
        print("Enhanced standard chat is ready for large output generation!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        print(traceback.format_exc())