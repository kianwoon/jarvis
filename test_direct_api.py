#!/usr/bin/env python3
"""Test API directly to see what's being returned"""

import requests
import json

url = "http://localhost:8000/api/v1/langchain/rag"
payload = {
    "question": "What is Python programming language?",
    "thinking": False,
    "conversation_id": "test-direct",
    "use_langgraph": False,
    "collections": None,
    "collection_strategy": "auto",
    "skip_classification": False
}

print("[TEST] Sending request...")
response = requests.post(url, json=payload, stream=True)

print(f"[TEST] Status: {response.status_code}")
print(f"[TEST] Headers: {dict(response.headers)}")

print("\n[RESPONSE CHUNKS]:")
for i, line in enumerate(response.iter_lines()):
    if line:
        print(f"Chunk {i}: {line.decode('utf-8')[:200]}...")
        try:
            data = json.loads(line)
            if "answer" in data:
                print(f"  -> Answer: '{data['answer']}'")
                print(f"  -> Length: {len(data['answer'])}")
        except:
            pass
    if i > 10:  # Limit output
        print("... (truncated)")
        break