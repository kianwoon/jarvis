#!/usr/bin/env python3
import requests
import json
import time

try:
    print("Testing minimal streaming...")
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8000/api/v1/langchain/rag",
        json={"question": "Hi", "thinking": False, "skip_classification": True},
        stream=True,
        timeout=5
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Error response: {response.text}")
        exit(1)
    
    chunk_count = 0
    for line in response.iter_lines():
        if line:
            chunk_count += 1
            print(f"Chunk {chunk_count}: {line[:100]}")
            
            if chunk_count > 5:  # Just get first few chunks
                print("Got initial chunks, looks good!")
                break
                
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()