#!/usr/bin/env python3
"""Visual test of streaming - shows tokens as they arrive"""

import httpx
import json
import time

def test_streaming_visual():
    """Test streaming with visual output"""
    url = "http://localhost:8000/api/v1/langchain/rag"
    
    payload = {
        "question": "What is Python? Give me a brief explanation.",
        "thinking": False,
        "conversation_id": "test-streaming",
        "use_langgraph": False,
        "collections": None,
        "collection_strategy": "auto",
        "skip_classification": False
    }
    
    print("[TEST] Starting streaming test...")
    print("[TEST] Question:", payload["question"])
    print("\n[STREAMING OUTPUT]:")
    print("-" * 50)
    
    with httpx.Client(timeout=60.0) as client:
        with client.stream('POST', url, json=payload) as response:
            token_count = 0
            start_time = time.time()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    if "token" in data:
                        token_count += 1
                        # Print token immediately without newline
                        print(data["token"], end='', flush=True)
                    
                    elif "answer" in data:
                        # This should NOT happen during proper streaming
                        print(f"\n\n[WARNING] Received full answer chunk! Length: {len(data['answer'])}")
                        print("[WARNING] This means streaming is NOT working properly!")
                    
                    elif "streaming_complete" in data:
                        elapsed = time.time() - start_time
                        print(f"\n\n[COMPLETE] Streaming finished")
                        print(f"  - Total tokens: {token_count}")
                        print(f"  - Time elapsed: {elapsed:.2f}s")
                        print(f"  - Tokens/second: {token_count/elapsed:.1f}")
                        
                except Exception as e:
                    print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    test_streaming_visual()