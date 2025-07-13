#!/usr/bin/env python3
"""
Simple test to check if RAG is working for partnership queries
"""

import requests
import json

def test_rag_query():
    """Test RAG query via API"""
    
    # Test query
    query = "partnership between beyondsoft and tencent"
    
    print(f"Testing query: '{query}'")
    print("=" * 50)
    
    try:
        # Test via langchain endpoint
        url = "http://localhost:8010/api/v1/langchain/chat"
        
        payload = {
            "message": query,
            "thinking": False,
            "stream": False
        }
        
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print("Response received:")
            print(f"Status: {response.status_code}")
            print(f"Content length: {len(result.get('response', ''))}")
            print(f"Response preview: {result.get('response', '')[:500]}...")
            
            # Check if sources are included
            if 'sources' in result:
                sources = result['sources']
                print(f"\nSources found: {len(sources)}")
                for i, source in enumerate(sources[:3]):
                    print(f"  Source {i+1}:")
                    print(f"    File: {source.get('file', 'Unknown')}")
                    print(f"    Score: {source.get('score', 'N/A')}")
                    print(f"    Collection: {source.get('collection', 'Unknown')}")
                    print()
            else:
                print("No sources in response")
                
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

def test_direct_search():
    """Test direct search endpoint"""
    
    query = "tencent partnership"
    
    print(f"\nTesting direct search for: '{query}'")
    print("=" * 50)
    
    try:
        url = "http://localhost:8010/api/v1/documents/search"
        
        payload = {
            "query": query,
            "collections": ["partnership"],
            "k": 10
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Direct search results: {len(result)} documents")
            for i, doc in enumerate(result[:3]):
                print(f"  Doc {i+1}:")
                print(f"    Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
                print(f"    Score: {doc.get('score', 'N/A')}")
                print(f"    Content: {doc.get('page_content', '')[:200]}...")
                print()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_rag_query()
    test_direct_search()