#!/usr/bin/env python3
import requests
import json

try:
    print("Testing standard chat streaming...")
    response = requests.post(
        "http://localhost:8000/api/v1/langchain/rag",
        json={"question": "Hi", "thinking": False},
        stream=True,
        timeout=10
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        for i, line in enumerate(response.iter_lines()):
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    if 'token' in data:
                        print(f"Token: {data['token']}", end='', flush=True)
                    elif 'answer' in data:
                        print(f"\nFinal answer: {data['answer'][:50]}...")
                        break
                except:
                    pass
            if i > 20:  # Limit output
                break
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")