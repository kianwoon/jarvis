#!/usr/bin/env python3
"""
Test the classifier fix to ensure simple queries don't trigger RAG
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_classifier_fix():
    """Test that the fix properly classifies simple queries"""
    
    # Import both classifiers for comparison
    from app.langchain.smart_query_classifier import SmartQueryClassifier, integrate_smart_classifier
    from app.langchain.query_classifier import QueryClassifier
    
    print("=" * 80)
    print("Testing Query Classifier Fix")
    print("=" * 80)
    
    # Test cases that were problematic
    test_cases = [
        # (query, expected_type, should_use_rag)
        ("what is today date & time?", "TOOLS", False),
        ("what is today's date?", "TOOLS", False),
        ("what time is it?", "TOOLS", False),
        ("hello", "LLM", False),
        ("hi there", "LLM", False),
        ("how are you?", "LLM", False),
        ("what is 2 + 2?", "LLM", False),
        ("tell me a joke", "LLM", False),
        ("write a haiku", "LLM", False),
        ("what do you think about AI?", "LLM", False),
        ("explain what water is", "LLM", False),
        ("define happiness", "LLM", False),
        ("what is the capital of France?", "LLM", False),
        
        # These SHOULD use RAG
        ("what does the document say about security?", "RAG", True),
        ("find information in the uploaded file", "RAG", True),
        ("according to the report, what are the findings?", "RAG", True),
        ("search the documents for customer data", "RAG", True),
    ]
    
    smart_classifier = SmartQueryClassifier()
    basic_classifier = QueryClassifier()
    
    print("\nComparing classifiers:")
    print("-" * 80)
    print(f"{'Query':<50} {'Basic':<15} {'Smart':<15} {'Correct?':<10}")
    print("-" * 80)
    
    correct_count = 0
    for query, expected_type, expected_rag in test_cases:
        # Test basic classifier
        basic_type, basic_conf, _ = basic_classifier.classify(query)
        basic_rag = basic_classifier.should_use_rag(basic_type, basic_conf)
        
        # Test smart classifier
        smart_result = integrate_smart_classifier(query)
        smart_type = smart_result['query_type']
        smart_rag = smart_result['should_use_rag']
        
        # Check if smart classifier is correct
        is_correct = (smart_type == expected_type and smart_rag == expected_rag)
        if is_correct:
            correct_count += 1
        
        # Format output
        query_short = query[:47] + "..." if len(query) > 50 else query
        basic_str = f"{basic_type.value}({basic_rag})"
        smart_str = f"{smart_type}({smart_rag})"
        correct_str = "✓" if is_correct else "✗"
        
        print(f"{query_short:<50} {basic_str:<15} {smart_str:<15} {correct_str:<10}")
    
    print("-" * 80)
    print(f"Smart classifier accuracy: {correct_count}/{len(test_cases)} "
          f"({correct_count/len(test_cases)*100:.1f}%)")
    
    # Show detailed analysis for date/time queries
    print("\n" + "=" * 80)
    print("Detailed Analysis: Date/Time Queries")
    print("=" * 80)
    
    date_time_queries = [
        "what is today date & time?",
        "what is today's date?",
        "what time is it?",
        "what's the current time?",
        "tell me today's date",
    ]
    
    for query in date_time_queries:
        print(f"\nQuery: '{query}'")
        
        # Smart classifier analysis
        result = smart_classifier.classify(query)
        should_rag, reason = smart_classifier.should_use_rag(query)
        
        print(f"  Classification: {result.query_type.value}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Should use RAG: {should_rag}")
        print(f"  RAG reason: {reason}")

def test_integration():
    """Test the integration function directly"""
    
    print("\n" + "=" * 80)
    print("Testing Integration Function")
    print("=" * 80)
    
    from app.langchain.smart_query_classifier import integrate_smart_classifier
    
    test_queries = [
        "what is today date & time?",
        "what does the document say about AI?",
        "hello, how are you?",
        "search the web for news",
        "write a Python function",
    ]
    
    for query in test_queries:
        result = integrate_smart_classifier(query)
        print(f"\nQuery: '{query}'")
        print(f"  Result: {result}")

if __name__ == "__main__":
    test_classifier_fix()
    test_integration()