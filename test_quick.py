#!/usr/bin/env python3
import requests
import json
import time

try:
    print("Testing quick response...")
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8000/api/v1/langchain/rag",
        json={"question": "Just say OK", "thinking": False},
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
                        print(f"T: {data['token']}", end='', flush=True)
                    elif 'answer' in data:
                        elapsed = time.time() - start_time
                        print(f"\n✅ Complete in {elapsed:.1f}s: {data['answer']}")
                        break
                except:
                    pass
            if i > 100:  # Safety limit
                print("\nReached limit")
                break
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Error: {e}")