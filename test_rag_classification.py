#!/usr/bin/env python3
"""Test RAG classification for queries"""

import requests
import json

# Test queries
test_queries = [
    "find out dbs bank outages info",
    "what is the weather today",
    "tell me about company policies",
    "calculate 2 + 2",
    "hello how are you"
]

url = "http://localhost:8000/api/v1/langchain/rag"

print("Testing Query Classification\n" + "="*50)

for query in test_queries:
    print(f"\nQuery: {query}")
    
    payload = {
        "question": query,
        "thinking": False,
        "skip_classification": False
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        
        if response.status_code == 200:
            # Read first few lines to get classification
            lines_read = 0
            for line in response.iter_lines():
                if line and lines_read < 3:
                    try:
                        data = json.loads(line)
                        if data.get("type") == "classification":
                            routing = data.get("routing", {})
                            print(f"  Classification: {routing.get('primary_type', 'unknown')}")
                            print(f"  Confidence: {routing.get('confidence', 0):.2f}")
                            print(f"  Handler: {data.get('handler', 'unknown')}")
                            break
                    except json.JSONDecodeError:
                        pass
                    lines_read += 1
        else:
            print(f"  Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
    
    except Exception as e:
        print(f"  Exception: {str(e)}")

print("\n" + "="*50)
print("If 'find out dbs bank outages info' shows 'rag' classification, the fix is working!")