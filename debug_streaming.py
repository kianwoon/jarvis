#!/usr/bin/env python3
"""Debug streaming issue in standard chat mode"""

import httpx
import json
import asyncio

async def test_streaming():
    """Test the streaming endpoint directly"""
    url = "http://localhost:8000/api/v1/langchain/rag"
    
    # Test query that should trigger tool execution
    payload = {
        "question": "apple wwdc 2025 keynotes, search internet",
        "thinking": False,
        "conversation_id": "test-123",
        "use_langgraph": False,
        "collections": None,
        "collection_strategy": "auto",
        "skip_classification": False
    }
    
    print(f"[TEST] Sending request to {url}")
    print(f"[TEST] Payload: {json.dumps(payload, indent=2)}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream('POST', url, json=payload) as response:
            print(f"[TEST] Response status: {response.status_code}")
            print(f"[TEST] Response headers: {dict(response.headers)}")
            
            chunk_count = 0
            token_count = 0
            error_count = 0
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                    
                chunk_count += 1
                print(f"\n[TEST] Chunk {chunk_count}: {line[:100]}...")
                
                try:
                    data = json.loads(line)
                    
                    if "token" in data:
                        token_count += 1
                        print(f"  -> Token {token_count}: {repr(data['token'][:50])}")
                    elif "error" in data:
                        error_count += 1
                        print(f"  -> ERROR: {data['error']}")
                    elif "type" in data:
                        print(f"  -> Event type: {data['type']}")
                        if data['type'] == 'classification':
                            print(f"     Routing: {data.get('routing', {}).get('primary_type')}")
                    elif "answer" in data:
                        print(f"  -> Final answer: {data['answer'][:100]}...")
                    else:
                        print(f"  -> Other data: {list(data.keys())}")
                        
                except json.JSONDecodeError as e:
                    print(f"  -> JSON decode error: {e}")
                except Exception as e:
                    print(f"  -> Unexpected error: {e}")
            
            print(f"\n[TEST] Summary:")
            print(f"  - Total chunks: {chunk_count}")
            print(f"  - Total tokens: {token_count}")
            print(f"  - Total errors: {error_count}")

if __name__ == "__main__":
    asyncio.run(test_streaming())