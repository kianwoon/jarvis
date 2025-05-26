#!/usr/bin/env python3
"""
Test script to verify RAG functionality after fixes
"""

import requests
import json
import sys

def test_rag_search(query: str):
    """Test the RAG search functionality"""
    
    print(f"\n{'='*60}")
    print(f"Testing RAG with query: '{query}'")
    print(f"{'='*60}\n")
    
    # Test the debug endpoint first
    debug_url = f"http://localhost:8000/api/v1/document/debug_search/{query}"
    
    try:
        print(f"1. Testing debug search endpoint...")
        response = requests.get(debug_url, params={"limit": 5, "min_score": 0.5})
        if response.status_code == 200:
            data = response.json()
            print(f"   - Found {data['results_count']} results")
            print(f"   - Collection: {data['collection']}")
            
            for i, result in enumerate(data['results'][:3]):
                print(f"\n   Result {i+1}:")
                print(f"   - Score: {result['score']:.3f} (original: {result['original_score']:.3f})")
                print(f"   - Similarity: {result.get('similarity', 'N/A')}")
                print(f"   - Content preview: {result['content_preview'][:150]}...")
                print(f"   - Metadata: {result['metadata']}")
        else:
            print(f"   ERROR: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ERROR: {str(e)}")
    
    # Test the main RAG endpoint
    print(f"\n2. Testing main RAG endpoint...")
    rag_url = "http://localhost:8000/api/v1/langchain/rag_answer"
    
    try:
        response = requests.post(
            rag_url,
            json={"question": query, "thinking": False, "stream": False}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   - Query type: {data.get('query_type', 'Unknown')}")
            print(f"   - Source: {data.get('source', 'Unknown')}")
            
            if data.get('context'):
                print(f"   - Context found: {len(data['context'])} characters")
                print(f"   - Context preview: {data['context'][:200]}...")
            else:
                print(f"   - No context found")
            
            if data.get('answer'):
                print(f"\n   Answer preview: {data['answer'][:300]}...")
        else:
            print(f"   ERROR: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"   ERROR: {str(e)}")

def main():
    """Run RAG tests with various queries"""
    
    test_queries = [
        # Add your test queries here based on documents in your database
        "What is machine learning?",
        "How does neural network work?",
        "What are the company policies?",
        "Tell me about our products",
        "What is the revenue for last quarter?"
    ]
    
    if len(sys.argv) > 1:
        # Use command line argument as query
        test_queries = [' '.join(sys.argv[1:])]
    
    print("RAG Functionality Test")
    print("=" * 60)
    print("This script tests the RAG search functionality after fixes.")
    print("Make sure you have uploaded some documents first!")
    print("=" * 60)
    
    for query in test_queries:
        test_rag_search(query)
        print("\n")

if __name__ == "__main__":
    main()