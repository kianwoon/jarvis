#!/usr/bin/env python3
"""Test with simplest possible query"""

import asyncio
import aiohttp
import json
import time

async def test_simple_streaming():
    """Test with a simple query that should stream quickly"""
    
    # Test both endpoints
    tests = [
        {
            "name": "Standard Chat",
            "url": "http://localhost:8000/api/v1/langchain/rag",
            "payload": {
                "question": "Say hello",
                "thinking": False,
                "skip_classification": False
            }
        },
        {
            "name": "Multi-Agent",
            "url": "http://localhost:8000/api/v1/langchain/multi-agent",
            "payload": {
                "question": "Say hello",
                "conversation_id": "test-multi"
            }
        }
    ]
    
    for test in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test['name']}")
        print(f"URL: {test['url']}")
        print(f"Question: {test['payload']['question']}")
        print('='*50)
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            first_token_time = None
            token_count = 0
            
            async with session.post(test['url'], json=test['payload']) as response:
                print(f"Status: {response.status}")
                
                buffer = ""
                async for chunk in response.content:
                    buffer += chunk.decode('utf-8')
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if not line.strip():
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            # Look for tokens
                            if "token" in data:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    print(f"\nFirst token received after: {first_token_time - start_time:.2f}s")
                                    print("Tokens: ", end='', flush=True)
                                
                                token_count += 1
                                print(data["token"], end='', flush=True)
                            
                            # Look for agent tokens (multi-agent)
                            elif data.get("type") == "agent_token":
                                if first_token_time is None:
                                    first_token_time = time.time()
                                    print(f"\nFirst token received after: {first_token_time - start_time:.2f}s")
                                    print("Tokens: ", end='', flush=True)
                                
                                token_count += 1
                                print(data["token"], end='', flush=True)
                            
                            # Check for completion
                            elif "answer" in data:
                                print(f"\n\n[COMPLETE] Full answer received")
                                print(f"Answer length: {len(data['answer'])}")
                            
                        except json.JSONDecodeError:
                            pass
                
                total_time = time.time() - start_time
                print(f"\n\nSummary:")
                print(f"  Total time: {total_time:.2f}s")
                print(f"  Total tokens: {token_count}")
                if first_token_time:
                    print(f"  Time to first token: {first_token_time - start_time:.2f}s")
                    print(f"  Tokens/second: {token_count/(total_time):.1f}")
                else:
                    print(f"  NO TOKENS RECEIVED!")

if __name__ == "__main__":
    asyncio.run(test_simple_streaming())