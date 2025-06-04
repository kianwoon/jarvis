#\!/usr/bin/env python3
"""Test hybrid query classification and handling"""

import requests
import json

# Test queries that should trigger hybrid handling
test_queries = [
    # Tool + RAG queries
    "Search online for DBS bank's current status and compare with our outage documents",
    "Find the latest news about bank outages and check if it matches our knowledge base",
    
    # Tool + LLM queries
    "What's the current weather and explain how it affects banking operations",
    "Calculate the exchange rate and analyze its impact on our business",
    
    # RAG + LLM queries
    "Based on our policy documents, create a summary report",
    "Using our knowledge base, generate a comprehensive guide",
    
    # Tool + RAG + LLM queries (most comprehensive)
    "Search online for recent DBS outages, compare with our documents, and create a detailed analysis",
    "Find current banking trends online, integrate with our knowledge base, and generate insights",
    "What does the internet say about bank outages versus our internal data, and summarize the differences",
    
    # Regular queries for comparison
    "What is the weather today",  # Tool only
    "Find information about DBS bank in our documents",  # RAG only
    "Hello, how are you"  # LLM only
]

url = "http://localhost:8000/api/v1/langchain/rag"

print("Testing Hybrid Query Classification and Handling\n" + "="*60)

for query in test_queries:
    print(f"\n🔍 Query: {query}")
    print("-" * 60)
    
    payload = {
        "question": query,
        "thinking": False,
        "skip_classification": False
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        
        if response.status_code == 200:
            # Read classification and status messages
            lines_read = 0
            for line in response.iter_lines():
                if line and lines_read < 10:  # Read more lines to see status messages
                    try:
                        data = json.loads(line)
                        
                        # Show classification
                        if data.get("type") == "classification":
                            routing = data.get("routing", {})
                            print(f"  📊 Classification: {routing.get('primary_type', 'unknown')}")
                            print(f"  📊 Confidence: {routing.get('confidence', 0):.2f}")
                            print(f"  📊 Is Hybrid: {routing.get('is_hybrid', False)}")
                            print(f"  📊 Handler: {data.get('handler', 'unknown')}")
                            
                            # Show all classifications if available
                            if routing.get('classifications'):
                                print(f"  📊 All Classifications:")
                                for i, cls in enumerate(routing['classifications'][:3]):
                                    print(f"     {i+1}. {cls['type']} (confidence: {cls['confidence']:.2f})")
                        
                        # Show status messages
                        elif data.get("type") == "status":
                            print(f"  ⚙️  Status: {data.get('message', '')}")
                        
                        # Show tool results
                        elif data.get("type") == "tool_result":
                            print(f"  🔧 Tool Result: {data.get('data', {})}")
                        
                        # Show RAG results
                        elif data.get("type") == "rag_result":
                            rag_data = data.get('data', {})
                            print(f"  📚 RAG Result: Found context: {rag_data.get('context_found', False)}, Sources: {rag_data.get('num_sources', 0)}")
                            
                    except json.JSONDecodeError:
                        pass
                    lines_read += 1
        else:
            print(f"  ❌ Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"  ❌ Exception: {str(e)}")

print("\n" + "="*60)
print("✅ Hybrid queries should show:")
print("   - is_hybrid: True")
print("   - Multiple status messages (🔧 Tool search, 📚 Knowledge base, 🤖 Synthesis)")
print("   - Handler: 'hybrid'")
print("\n💡 This enables combining real-time data + knowledge base + LLM synthesis\!")
EOF < /dev/null