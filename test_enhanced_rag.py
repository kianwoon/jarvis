#!/usr/bin/env python3
"""
Test script for enhanced RAG with query expansion, hybrid search, and re-ranking
"""
import requests
import json
import time

def test_rag_query(question: str, expected_terms=None):
    """Test a single RAG query and analyze results"""
    print(f"\n{'='*80}")
    print(f"🔍 Query: '{question}'")
    print(f"{'='*80}")
    
    url = "http://localhost:8000/api/v1/langchain/rag"
    payload = {
        "question": question,
        "thinking": False
    }
    
    start_time = time.time()
    
    try:
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
        
        elapsed_time = time.time() - start_time
        
        print(f"\n\n📊 METRICS:")
        print(f"  - Response time: {elapsed_time:.2f}s")
        print(f"  - Source: {source}")
        print(f"  - Context length: {len(context)} chars")
        print(f"  - Answer length: {len(full_answer)} chars")
        
        # Check if expected terms are found
        if expected_terms:
            found_terms = []
            missing_terms = []
            
            for term in expected_terms:
                if term.lower() in full_answer.lower() or term.lower() in context.lower():
                    found_terms.append(term)
                else:
                    missing_terms.append(term)
            
            print(f"\n✅ Found terms ({len(found_terms)}/{len(expected_terms)}): {', '.join(found_terms)}")
            if missing_terms:
                print(f"❌ Missing terms: {', '.join(missing_terms)}")
        
        # Show a snippet of context to verify retrieval
        if context:
            print(f"\n📄 Context preview (first 300 chars):")
            print(f"  {context[:300]}...")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_query_expansion():
    """Test the query expansion feature directly"""
    print("\n" + "="*80)
    print("🔬 Testing Query Expansion")
    print("="*80)
    
    from app.langchain.service import llm_expand_query
    from app.core.llm_settings_cache import get_llm_settings
    
    test_queries = [
        "dbs outages",
        "What is machine learning?",
        "How to implement authentication in Python?",
        "Company financial results Q4 2023"
    ]
    
    llm_cfg = get_llm_settings()
    
    for query in test_queries:
        print(f"\n📝 Original: {query}")
        expanded = llm_expand_query(query, llm_cfg)
        print(f"📈 Expanded ({len(expanded)} variants):")
        for i, eq in enumerate(expanded):
            print(f"   {i+1}. {eq}")

def main():
    """Run comprehensive RAG tests"""
    print("🚀 Enhanced RAG System Test Suite")
    print("   Testing: Query Expansion + Hybrid Search + Re-ranking")
    
    # Test 1: Short technical query
    test_rag_query(
        "dbs outages",
        expected_terms=["outage", "incident", "disruption", "2021", "2023"]
    )
    
    # Test 2: Natural language question
    test_rag_query(
        "What happened with DBS bank technology incidents?",
        expected_terms=["DBS", "incident", "technology", "outage"]
    )
    
    # Test 3: Specific date query
    test_rag_query(
        "Tell me about November 2021 DBS disruption",
        expected_terms=["November", "2021", "DBS", "MAS"]
    )
    
    # Test 4: Generic technical query
    test_rag_query(
        "machine learning algorithms",
        expected_terms=["algorithm", "learning", "model", "data"]
    )
    
    # Test 5: Complex multi-part query
    test_rag_query(
        "How does distributed database architecture help with resilience and regulatory compliance?",
        expected_terms=["distributed", "database", "resilience", "compliance"]
    )
    
    # Test query expansion separately
    try:
        test_query_expansion()
    except Exception as e:
        print(f"\n⚠️  Query expansion test failed: {str(e)}")
    
    print("\n\n✅ Test suite completed!")
    print("\n💡 Key improvements tested:")
    print("  1. LLM-powered query expansion for better recall")
    print("  2. Hybrid search combining vector similarity and TF-IDF-like keyword matching")
    print("  3. LLM re-ranking of top candidates for better precision")
    print("  4. Multi-query retrieval with deduplication")

if __name__ == "__main__":
    main()