#!/usr/bin/env python3
"""Test hybrid query classification"""

import requests
import json

# Simple test for hybrid queries
test_queries = [
    "Search online for DBS bank status and compare with our documents",
    "Find current weather and explain its impact",
    "What is the weather today",
    "Find info about DBS in our documents"
]

url = "http://localhost:8000/api/v1/langchain/rag"

for query in test_queries:
    print(f"\nQuery: {query}")
    
    response = requests.post(url, json={"question": query}, stream=True)
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if data.get("type") == "classification":
                        routing = data.get("routing", {})
                        print(f"  Type: {routing.get('primary_type')}")
                        print(f"  Hybrid: {routing.get('is_hybrid')}")
                        print(f"  Handler: {data.get('handler')}")
                        break
                except:
                    pass