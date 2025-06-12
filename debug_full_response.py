#!/usr/bin/env python3
"""Get full response to check if tools were executed"""

import httpx
import json

def test_full_response():
    """Get the complete response"""
    url = "http://localhost:8000/api/v1/langchain/rag"
    
    payload = {
        "question": "apple wwdc 2025 keynotes, search internet",
        "thinking": False,
        "conversation_id": "test-123",
        "use_langgraph": False,
        "collections": None,
        "collection_strategy": "auto",
        "skip_classification": False
    }
    
    print(f"[TEST] Sending request...")
    
    with httpx.Client(timeout=60.0) as client:
        with client.stream('POST', url, json=payload) as response:
            full_text = ""
            tool_results_found = False
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    if "token" in data:
                        full_text += data["token"]
                    
                    if "tool_result" in data or "Tool Results:" in full_text:
                        tool_results_found = True
                        
                except:
                    pass
            
            print(f"\n[TEST] Tool results found: {tool_results_found}")
            print(f"\n[TEST] Full response preview (first 1000 chars):")
            print(full_text[:1000])
            
            # Check if response mentions actual search results
            if "search results" in full_text.lower() or "according to" in full_text.lower():
                print("\n[TEST] Response appears to reference search results")
            else:
                print("\n[TEST] Response does NOT appear to reference actual search results")

if __name__ == "__main__":
    test_full_response()