#!/usr/bin/env python3
"""Test the smart query classifier"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.langchain.smart_query_classifier import SmartQueryClassifier, integrate_smart_classifier

def test_smart_classifier():
    """Test the smart classifier with various queries"""
    
    classifier = SmartQueryClassifier()
    
    # Test queries organized by expected type
    test_queries = {
        "Simple LLM queries (should NOT use RAG)": [
            "what is today date & time?",  # Actually TOOL, not RAG
            "what is today's date?",       # TOOL
            "what time is it?",            # TOOL
            "hello",                       # LLM
            "hi there",                    # LLM
            "how are you?",                # LLM
            "what is 2 + 2?",              # LLM
            "tell me a joke",              # LLM
            "write a haiku",               # LLM
            "what do you think about AI?", # LLM
            "explain what water is",       # LLM
            "define happiness",            # LLM
            "what is the capital of France?", # LLM
        ],
        "RAG queries (should use RAG)": [
            "what does the document say about security?",
            "find information about policies in the uploaded file",
            "according to the report, what are the requirements?",
            "search for customer data in the documents",
            "what is mentioned in the manual about installation?",
            "based on the uploaded PDF, what are the key findings?",
        ],
        "Tool queries": [
            "search the web for latest news",
            "what's the weather today?",
            "get current stock prices",
            "browse to google.com",
        ],
        "Code queries": [
            "write a Python function to sort a list",
            "debug this JavaScript code",
            "implement a binary search algorithm",
            "fix the error in my code",
        ]
    }
    
    print("=" * 80)
    print("Smart Query Classifier Test Results")
    print("=" * 80)
    
    for category, queries in test_queries.items():
        print(f"\n{category}")
        print("-" * len(category))
        
        for query in queries:
            result = classifier.classify(query)
            should_rag, rag_reason = classifier.should_use_rag(query)
            
            print(f"\nQuery: '{query}'")
            print(f"  Classification: {result.query_type.value} (confidence: {result.confidence:.2%})")
            print(f"  Reasoning: {result.reasoning}")
            print(f"  Should use RAG: {should_rag} - {rag_reason}")
            
            # Also test the integration function
            integrated = integrate_smart_classifier(query)
            print(f"  Legacy mapping: {integrated['query_type']}")

def compare_classifiers():
    """Compare smart classifier with the basic one for problematic queries"""
    
    from app.langchain.query_classifier import QueryClassifier
    
    print("\n" + "=" * 80)
    print("Classifier Comparison for Problematic Queries")
    print("=" * 80)
    
    basic_classifier = QueryClassifier()
    smart_classifier = SmartQueryClassifier()
    
    problematic_queries = [
        "what is today date & time?",
        "what is 2 + 2?",
        "how are you?",
        "what do you think about AI?",
        "explain what water is",
        "what is the capital of France?",
    ]
    
    for query in problematic_queries:
        print(f"\nQuery: '{query}'")
        
        # Basic classifier
        basic_type, basic_conf, _ = basic_classifier.classify(query)
        basic_should_rag = basic_classifier.should_use_rag(basic_type, basic_conf)
        
        # Smart classifier
        smart_result = smart_classifier.classify(query)
        smart_should_rag, smart_reason = smart_classifier.should_use_rag(query)
        
        print(f"  Basic: {basic_type.value} ({basic_conf:.2%}) - Use RAG: {basic_should_rag}")
        print(f"  Smart: {smart_result.query_type.value} ({smart_result.confidence:.2%}) - Use RAG: {smart_should_rag}")
        print(f"  Smart reasoning: {smart_reason}")

if __name__ == "__main__":
    test_smart_classifier()
    compare_classifiers()