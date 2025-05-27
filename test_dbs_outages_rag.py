#!/usr/bin/env python3
"""
Test script to verify RAG retrieval for DBS outages query
"""
import requests
import json

def test_rag_query(question: str):
    """Test a RAG query"""
    print(f"\n🔍 Testing query: '{question}'")
    print("=" * 80)
    
    url = "http://localhost:8000/api/v1/langchain/rag"
    payload = {
        "question": question,
        "thinking": False
    }
    
    try:
        # Make request
        response = requests.post(url, json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"❌ Error: HTTP {response.status_code}")
            return
        
        # Process streaming response
        full_answer = ""
        context = ""
        source = ""
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'token' in data:
                            print(data['token'], end='', flush=True)
                        elif 'answer' in data:
                            full_answer = data.get('answer', '')
                            context = data.get('context', '')
                            source = data.get('source', '')
                    except json.JSONDecodeError:
                        pass
        
        print("\n\n" + "=" * 80)
        print("📊 RESULTS ANALYSIS:")
        print(f"Source: {source}")
        print(f"Context length: {len(context)} characters")
        
        # Check if the answer contains expected information
        expected_terms = ['outage', 'november', '2021', '2023', 'mas', 'monetary authority', 
                         'unacceptable', 'resilience', 'disruption', 'incident']
        
        found_terms = []
        for term in expected_terms:
            if term.lower() in full_answer.lower() or term.lower() in context.lower():
                found_terms.append(term)
        
        print(f"\n✅ Found terms: {', '.join(found_terms)}")
        print(f"📈 Coverage: {len(found_terms)}/{len(expected_terms)} expected terms")
        
        # Check context for the specific passage
        if "november 2021" in context.lower() and "mas" in context.lower():
            print("\n🎯 SUCCESS: Found the DBS outage information in context!")
        else:
            print("\n⚠️  WARNING: DBS outage information not found in context")
            
        # Print a snippet of the context to verify
        if context:
            print("\n📄 Context snippet:")
            # Find the most relevant part of context
            context_lower = context.lower()
            for keyword in ['november 2021', 'outage', 'mas to term']:
                idx = context_lower.find(keyword)
                if idx != -1:
                    start = max(0, idx - 100)
                    end = min(len(context), idx + 300)
                    print(f"...{context[start:end]}...")
                    break
                    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_debug_search(query: str):
    """Test the debug search endpoint"""
    print(f"\n🔍 Debug search for: '{query}'")
    print("=" * 80)
    
    url = f"http://localhost:8000/api/v1/documents/debug_search/{query}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Found {data['results_count']} results")
            
            # Look for the outage content
            for i, result in enumerate(data['results']):
                content = result['content_preview'].lower()
                if 'november 2021' in content or 'outage' in content or 'mas' in content:
                    print(f"\n✅ Result {i}: Score={result['score']:.3f}")
                    print(f"Source: {result['metadata']['source']}")
                    print(f"Content: {result['content_preview']}")
                    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Testing RAG retrieval for DBS outages")
    
    # Test different query variations
    queries = [
        "dbs outages",
        "DBS outages",
        "DBS bank outages",
        "DBS November 2021 outage",
        "DBS technology incidents MAS",
        "What happened with DBS outages?",
        "Tell me about DBS bank disruptions and regulatory issues"
    ]
    
    # First test debug search to see what's in the index
    test_debug_search("DBS outage November 2021")
    
    # Then test RAG queries
    for query in queries:
        test_rag_query(query)
        print("\n" + "=" * 80 + "\n")