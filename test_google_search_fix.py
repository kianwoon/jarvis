#!/usr/bin/env python3
"""
Test script to verify google_search tool bypass is working
"""

import sys
import os
import json

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_google_search_via_api():
    """Test the google_search tool via the API endpoint"""
    
    import requests
    
    # API endpoint
    url = "http://localhost:8000/api/v1/intelligent-chat"
    
    # Test query
    payload = {
        "question": "Use google_search to find information about Python FastAPI framework",
        "thinking": False
    }
    
    print("Testing google_search via API...")
    print(f"Query: {payload['question']}")
    
    try:
        # Make the request
        response = requests.post(url, json=payload, stream=True)
        
        if response.status_code != 200:
            print(f"❌ API Error: Status {response.status_code}")
            return False
        
        # Process streaming response
        tool_executed = False
        has_result = False
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    # Check for tool execution
                    if data.get("type") == "tool_result":
                        tool_result = data.get("tool_result", {})
                        if tool_result.get("tool") == "google_search":
                            tool_executed = True
                            if tool_result.get("success"):
                                print("✅ google_search executed successfully via bypass!")
                                has_result = True
                            else:
                                print(f"❌ google_search failed: {tool_result.get('error')}")
                    
                    # Check final answer
                    if "answer" in data and data.get("source"):
                        if has_result:
                            print(f"✅ Final answer generated from source: {data['source']}")
                        
                except json.JSONDecodeError:
                    continue
        
        return tool_executed and has_result
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Start the server with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return False

def test_direct_function():
    """Test the direct google_search function"""
    
    import asyncio
    
    async def run_test():
        # Import after event loop is created
        from app.core.unified_mcp_service import UnifiedMCPService
        
        service = UnifiedMCPService()
        
        parameters = {
            "query": "Python FastAPI",
            "num_results": 2
        }
        
        print("\nTesting _direct_google_search function...")
        print(f"Parameters: {json.dumps(parameters, indent=2)}")
        
        try:
            result = await service._direct_google_search(parameters)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return False
            else:
                print("✅ Direct function works! Results received:")
                if "content" in result:
                    for content in result["content"]:
                        if content.get("type") == "text":
                            # Just show first 200 chars
                            text = content.get("text", "")[:200]
                            print(f"   {text}...")
                return True
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
            return False
        finally:
            await service.close()
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    print("=" * 60)
    print("Google Search Bypass Test")
    print("=" * 60)
    
    # Test 1: Direct function
    print("\n1. Testing direct function...")
    direct_success = test_direct_function()
    
    # Test 2: Via API
    print("\n2. Testing via API endpoint...")
    api_success = test_google_search_via_api()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Direct function: {'✅ PASS' if direct_success else '❌ FAIL'}")
    print(f"  API endpoint:    {'✅ PASS' if api_success else '❌ FAIL'}")
    print("=" * 60)
    
    exit(0 if (direct_success or api_success) else 1)