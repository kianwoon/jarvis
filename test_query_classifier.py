#!/usr/bin/env python3
"""
Test script for the query classifier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.langchain.query_classifier import QueryClassifier, QueryType

def test_query_classifier():
    classifier = QueryClassifier()
    
    # Test queries
    test_queries = [
        # RAG queries
        "What is the company's vacation policy?",
        "Tell me about the Q3 financial results",
        "According to the documentation, how do I configure the API?",
        "Summarize the project requirements document",
        
        # Tool use queries
        "What's the weather like today in New York?",
        "Search the web for latest AI news",
        "Calculate the compound interest on $10,000 at 5% for 10 years",
        "Send an email to john@example.com",
        
        # Direct LLM queries
        "Hello, how are you?",
        "Can you explain quantum computing in simple terms?",
        "What do you think about artificial intelligence?",
        "Write a haiku about spring",
        
        # Code generation queries
        "Write a Python function to sort a list",
        "Debug this JavaScript code: function add(a,b) { return a + b }",
        "Implement a binary search algorithm in Java",
        "Create a REST API endpoint in FastAPI",
        
        # Multi-agent queries
        "Analyze the market data and then create a comprehensive report with visualizations",
        "Research the latest ML techniques and implement a demo application",
        "Complete analysis of our system architecture and suggest improvements",
        "Step by step guide to building a web application with authentication",
    ]
    
    print("Query Classification Test Results")
    print("=" * 80)
    
    for query in test_queries:
        recommendation = classifier.get_routing_recommendation(query)
        
        print(f"\nQuery: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"  Type: {recommendation['query_type']}")
        print(f"  Confidence: {recommendation['confidence']:.2%}")
        print(f"  Handler: {recommendation['routing']['primary_handler']}")
        print(f"  Use RAG: {recommendation['use_rag']}")
        
        if recommendation['routing'].get('suggested_tools'):
            print(f"  Tools: {', '.join(recommendation['routing']['suggested_tools'])}")
        if recommendation['routing'].get('suggested_agents'):
            print(f"  Agents: {', '.join(recommendation['routing']['suggested_agents'])}")
        
        # Show matched patterns for debugging
        if recommendation['metadata']['matched_patterns']:
            patterns = [f"{cat}:{pat}" for cat, pat in recommendation['metadata']['matched_patterns'][:3]]
            print(f"  Patterns: {', '.join(patterns)}")
    
    print("\n" + "=" * 80)
    print("Classification Summary:")
    print("- RAG queries: For knowledge retrieval from documents")
    print("- Tool queries: For external API/tool usage")
    print("- Direct LLM: For general conversation")
    print("- Code generation: For programming tasks")
    print("- Multi-agent: For complex multi-step tasks")

if __name__ == "__main__":
    test_query_classifier()