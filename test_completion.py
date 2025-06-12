#!/usr/bin/env python3
import requests
import json

try:
    print("Testing completion event...")
    response = requests.post(
        "http://localhost:8000/api/v1/langchain/rag",
        json={"question": "Count from 1 to 5", "thinking": False},
        stream=True,
        timeout=30
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        token_count = 0
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'token' in data:
                        token_count += 1
                        if token_count <= 5:  # Show first few tokens
                            print(f"Token {token_count}: '{data['token']}'")
                    elif 'answer' in data:
                        print(f"\n✅ COMPLETION EVENT RECEIVED!")
                        print(f"Final answer: {data['answer']}")
                        print(f"Source: {data.get('source', 'N/A')}")
                        print(f"Total tokens streamed: {token_count}")
                        break
                except json.JSONDecodeError:
                    pass
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")