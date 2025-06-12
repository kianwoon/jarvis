#!/usr/bin/env python3
"""Simulate exactly what the frontend does"""

import asyncio
import aiohttp
import json
import time

async def test_frontend_simulation():
    """Test streaming exactly like frontend"""
    url = "http://localhost:8000/api/v1/langchain/rag"
    
    payload = {
        "question": "What is Python programming language?",
        "thinking": False,
        "conversation_id": "test-frontend",
        "use_langgraph": False,
        "collections": None,
        "collection_strategy": "auto",
        "skip_classification": False
    }
    
    print("[TEST] Simulating frontend request...")
    print(f"[TEST] URL: {url}")
    print(f"[TEST] Question: {payload['question']}")
    print("\n[STREAMING]:")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            print(f"[TEST] Response status: {response.status}")
            print(f"[TEST] Response headers: {dict(response.headers)}")
            
            buffer = ""
            token_count = 0
            chunk_count = 0
            start_time = time.time()
            
            async for data in response.content:
                chunk_count += 1
                buffer += data.decode('utf-8')
                
                # Process complete lines
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        if "token" in data:
                            token_count += 1
                            print(data["token"], end='', flush=True)
                        
                        elif "answer" in data:
                            elapsed = time.time() - start_time
                            print(f"\n\n[WARNING] Full answer received!")
                            print(f"  - Answer length: {len(data['answer'])}")
                            print(f"  - Time to receive: {elapsed:.2f}s")
                            print(f"  - Tokens before answer: {token_count}")
                            
                    except json.JSONDecodeError:
                        pass
            
            elapsed = time.time() - start_time
            print(f"\n\n[COMPLETE]")
            print(f"  - Total chunks: {chunk_count}")
            print(f"  - Total tokens: {token_count}")
            print(f"  - Total time: {elapsed:.2f}s")
            print(f"  - Chunks/second: {chunk_count/elapsed:.1f}")

if __name__ == "__main__":
    asyncio.run(test_frontend_simulation())