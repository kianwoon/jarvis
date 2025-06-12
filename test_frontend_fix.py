#!/usr/bin/env python3
"""Test that tokens are now streaming properly"""

import requests
import json

url = "http://localhost:8000/api/v1/langchain/rag"
payload = {
    "question": "Count from 1 to 5",
    "thinking": False,
    "conversation_id": "test-fix"
}

print("Testing standard chat streaming after frontend fix...")
response = requests.post(url, json=payload, stream=True)

if response.status_code == 200:
    print("✓ Connected successfully")
    token_count = 0
    
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode('utf-8'))
                if 'token' in data:
                    print(f"Token {token_count}: {repr(data['token'])}", end='', flush=True)
                    token_count += 1
                elif 'answer' in data:
                    print(f"\n✓ Received completion event")
                    print(f"✓ Total tokens streamed: {token_count}")
                    break
            except json.JSONDecodeError:
                pass
else:
    print(f"✗ Failed with status: {response.status_code}")