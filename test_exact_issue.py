#!/usr/bin/env python3
"""Show exactly what frontend receives"""

import requests
import time

url = "http://localhost:8000/api/v1/langchain/rag"
payload = {
    "question": "Say hello in 5 words",
    "thinking": False,
    "conversation_id": "test-exact"
}

print("Sending request...")
start_time = time.time()

response = requests.post(url, json=payload, stream=True)
print(f"Status: {response.status_code}")
print(f"Headers: {dict(response.headers)}")

print("\n--- RAW RESPONSE ---")
buffer = ""
chunk_count = 0
for chunk in response.iter_content(chunk_size=1):  # Read byte by byte
    if chunk:
        chunk_count += 1
        char = chunk.decode('utf-8', errors='ignore')
        buffer += char
        
        # Print each character as it arrives
        if char == '\n':
            print(f"[Line {chunk_count}]: {repr(buffer.strip())}")
            buffer = ""
        
        # Stop after some data
        if chunk_count > 500:
            print("\n... truncated ...")
            break

elapsed = time.time() - start_time
print(f"\nReceived {chunk_count} bytes in {elapsed:.2f}s")
print(f"Rate: {chunk_count/elapsed:.1f} bytes/second")