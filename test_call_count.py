#!/usr/bin/env python3
import requests
import json

print("Testing to see HTTP call count...")

try:
    response = requests.post(
        "http://localhost:8000/api/v1/langchain/rag",
        json={"question": "Hi", "thinking": False},
        stream=True,
        timeout=3  # Short timeout to fail fast
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("Request started successfully")
        # Don't actually process the stream, just see if we get a response
        for i, line in enumerate(response.iter_lines()):
            if i < 2:  # Just get first couple chunks
                print(f"Got chunk: {line[:50] if line else 'empty'}")
            else:
                break
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Expected error due to timeout: {e}")
    
print("\nCheck the backend logs above to see how many HTTP calls were made")