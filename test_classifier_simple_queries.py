#!/usr/bin/env python3
"""Test the query classifier with simple queries to understand the issue"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier, QueryType
from app.langchain.query_classifier import QueryClassifier
import json

def test_simple_queries():
    """Test classifier with simple queries that should NOT trigger RAG"""
    
    # Initialize classifiers
    enhanced_classifier = EnhancedQueryClassifier()
    basic_classifier = QueryClassifier()
    
    # Test queries that should be DIRECT_LLM
    simple_queries = [
        "what is today date & time?",
        "what is today's date?",
        "what time is it?",
        "hello",
        "hi there",
        "how are you?",
        "what is 2 + 2?",
        "tell me a joke",
        "write a haiku",
        "what do you think about AI?",
        "explain what water is",
        "define happiness",
        "what is the capital of France?",
    ]
    
    print("=" * 80)
    print("Testing Simple Queries Classification")
    print("=" * 80)
    
    for query in simple_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        # Test enhanced classifier
        print("Enhanced Classifier:")
        results = enhanced_classifier.classify(query)
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.query_type.value}: {result.confidence:.2%}")
            if result.matched_patterns:
                print(f"     Matched patterns: {[p[0] for p in result.matched_patterns]}")
        
        # Test basic classifier
        print("\nBasic Classifier:")
        query_type, confidence, metadata = basic_classifier.classify(query)
        print(f"  Type: {query_type.value}, Confidence: {confidence:.2%}")
        if metadata.get("matched_patterns"):
            print(f"  Patterns: {[p[0] for p in metadata['matched_patterns']]}")
        
        # Get routing recommendation
        print("\nRouting Recommendation:")
        recommendation = enhanced_classifier.get_routing_recommendation(query)
        print(f"  Primary: {recommendation['primary_type']}")
        print(f"  Confidence: {recommendation['confidence']:.2%}")
        print(f"  Is Hybrid: {recommendation['is_hybrid']}")
        
        # Check if RAG would be used
        should_use_rag = basic_classifier.should_use_rag(query_type, confidence)
        print(f"\nShould use RAG (basic): {should_use_rag}")
        
        print("=" * 40)

def analyze_patterns():
    """Analyze which patterns are matching for simple queries"""
    
    print("\n" + "=" * 80)
    print("Pattern Analysis for 'what is today date & time?'")
    print("=" * 80)
    
    query = "what is today date & time?"
    classifier = EnhancedQueryClassifier()
    
    # Check each pattern category
    for category, patterns in classifier.compiled_patterns.items():
        if category == "hybrid":
            continue
            
        print(f"\nCategory: {category}")
        for pattern, config in patterns:
            if pattern.search(query.lower()):
                print(f"  ✓ MATCHED: {pattern.pattern}")
                print(f"    Group: {config['group']}")
                print(f"    Boost: {config['confidence_boost']}")

def test_config_patterns():
    """Test specific patterns from the config"""
    
    print("\n" + "=" * 80)
    print("Testing Specific Pattern Matches")
    print("=" * 80)
    
    import re
    
    test_cases = [
        ("what is today date & time?", [
            (r'\b(what|who|when|where|why|how)\b.*\b(is|are|was|were|does|do|did)\b', "question_answering"),
            (r'\b(current|today|now|latest|real-time|live)\b', "realtime_data"),
            (r'\b(date|time)\b', "custom_pattern"),
        ])
    ]
    
    for query, patterns in test_cases:
        print(f"\nQuery: '{query}'")
        for pattern_str, pattern_type in patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(query.lower())
            print(f"  Pattern '{pattern_str}' ({pattern_type}): {'MATCH' if match else 'NO MATCH'}")
            if match:
                print(f"    Matched text: '{match.group()}'")

if __name__ == "__main__":
    test_simple_queries()
    analyze_patterns()
    test_config_patterns()