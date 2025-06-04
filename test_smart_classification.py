#!/usr/bin/env python3
"""
Test the enhanced smart query classification for DeepSeek-R1
"""

import sys
sys.path.insert(0, '.')

from app.langchain.smart_query_classifier import classify_query, get_classification_explanation

def test_classification():
    """Test various query types to ensure proper classification"""
    
    test_queries = [
        # TOOLS queries - should NOT trigger RAG
        ("what is today date & time?", "tools"),
        ("what's the current date?", "tools"),
        ("tell me today's date", "tools"),
        ("what time is it now?", "tools"),
        ("calculate 15 * 23", "tools"),
        ("what's the weather like?", "tools"),
        
        # LLM queries - simple questions that don't need RAG
        ("hello", "llm"),
        ("how are you?", "llm"),
        ("what is photosynthesis?", "llm"),
        ("explain quantum physics", "llm"),
        ("what is a cat?", "llm"),
        ("tell me a joke", "llm"),
        ("translate hello to spanish", "llm"),
        
        # RAG queries - need document search
        ("what does the report say about Q4 revenue?", "rag"),
        ("find documents about AI strategy", "rag"),
        ("search for information about customer feedback", "rag"),
        ("what's in the latest financial report?", "rag"),
        ("show me documents about product roadmap", "rag"),
        ("retrieve information from the knowledge base", "rag"),
        
        # Edge cases that were problematic
        ("what is the migration timeline?", "rag"),  # Should search for migration docs
        ("how long does the migration take?", "rag"),  # Should search for migration info
        ("what are the benefits of our product?", "rag"),  # Company-specific, needs RAG
    ]
    
    print("Testing Smart Query Classification")
    print("=" * 60)
    
    correct = 0
    total = len(test_queries)
    
    for query, expected_type in test_queries:
        result_type, confidence = classify_query(query)
        is_correct = result_type == expected_type
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Query: '{query}'")
        print(f"  Expected: {expected_type}, Got: {result_type} (confidence: {confidence:.2f})")
        if not is_correct:
            print(f"  Explanation: {get_classification_explanation(query, result_type)}")
        print()
    
    accuracy = (correct / total) * 100
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Test specific problematic case
    print("\n" + "=" * 60)
    print("Testing problematic DeepSeek-R1 case:")
    query = "what is today date & time?"
    result_type, confidence = classify_query(query)
    print(f"Query: '{query}'")
    print(f"Classification: {result_type} (confidence: {confidence:.2f})")
    print(f"Explanation: {get_classification_explanation(query, result_type)}")
    print("\nThis should be classified as 'tools' with high confidence, NOT triggering RAG search.")

if __name__ == "__main__":
    test_classification()