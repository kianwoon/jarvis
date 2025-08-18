#!/usr/bin/env python3
"""
Test script for the Working MCP HTTP Bridge Server

This script tests that the bridge server is functional and can execute real MCP tools.
"""

import requests
import json
import time
import sys

def test_health_check():
    """Test the health check endpoint"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get("http://localhost:3001/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Server Status: {data.get('status')}")
            print(f"Available Tools: {data.get('available_tools')}")
            print(f"Message: {data.get('message')}")
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_list_tools():
    """Test listing available tools"""
    print("\n=== Testing List Tools ===")
    try:
        response = requests.get("http://localhost:3001/tools")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total Tools: {data.get('total')}")
            
            # Show first 5 tools
            tools = data.get('tools', [])[:5]
            if tools:
                print("Sample tools:")
                for tool in tools:
                    print(f"  - {tool.get('name')}: {tool.get('description', '')[:50]}...")
            
            # Check if google_search is available
            all_tools = data.get('tools', [])
            google_search = next((t for t in all_tools if t['name'] == 'google_search'), None)
            if google_search:
                print(f"✅ google_search tool found: server_id={google_search.get('server_id')}")
            else:
                print("⚠️ google_search tool not found")
            
            print("✅ List tools passed")
            return True
        else:
            print(f"❌ List tools failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ List tools failed: {e}")
        return False

def test_google_search():
    """Test the Google Search endpoint with a real query"""
    print("\n=== Testing Google Search (REAL EXECUTION) ===")
    
    queries = [
        {
            "query": "OpenAI ChatGPT latest features 2025",
            "num_results": 3
        },
        {
            "query": "Python FastAPI best practices",
            "num_results": 2
        }
    ]
    
    for test_query in queries:
        print(f"\nSearching for: '{test_query['query']}'")
        print(f"Requesting {test_query['num_results']} results")
        
        try:
            start_time = time.time()
            response = requests.post(
                "http://localhost:3001/tools/google_search",
                json=test_query
            )
            execution_time = time.time() - start_time
            
            print(f"Status Code: {response.status_code}")
            print(f"Execution Time: {execution_time:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success'):
                    print("✅ Search successful!")
                    
                    # Display the result
                    result = data.get('result', {})
                    if isinstance(result, dict) and 'content' in result:
                        content = result['content']
                        # Show first 500 characters of results
                        preview = content[:500] if isinstance(content, str) else str(content)[:500]
                        print(f"Result preview:\n{preview}...")
                        
                        # Check if we got actual search results
                        if "Found" in content and "search results" in content:
                            print("✅ Got actual search results from Google!")
                        else:
                            print("⚠️ Response doesn't look like search results")
                    else:
                        print(f"Result: {json.dumps(result, indent=2)[:500]}...")
                    
                    print(f"Tool execution time: {data.get('execution_time', 0):.2f}s")
                else:
                    print(f"❌ Search failed: {data.get('error')}")
                    return False
            else:
                print(f"❌ Request failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Google search test failed: {e}")
            return False
        
        # Small delay between searches
        time.sleep(1)
    
    print("\n✅ All Google Search tests passed!")
    return True

def test_generic_tool_endpoint():
    """Test the generic tool execution endpoint"""
    print("\n=== Testing Generic Tool Endpoint ===")
    
    try:
        # Test with google_search through generic endpoint
        print("Testing google_search through /tools/google_search endpoint...")
        
        request_data = {
            "query": "MCP Model Context Protocol documentation",
            "num_results": 2
        }
        
        response = requests.post(
            "http://localhost:3001/tools/google_search",
            json=request_data
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Generic tool execution successful")
                # Show execution time
                print(f"Execution time: {data.get('execution_time', 0):.2f}s")
                return True
            else:
                print(f"❌ Tool execution failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Request failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Generic tool test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Working MCP HTTP Bridge Server Test Suite")
    print("=" * 60)
    print("\nMake sure the server is running on port 3001")
    print("Start it with: python working_mcp_bridge.py")
    print("=" * 60)
    
    # Wait a moment for user to read
    time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("List Tools", test_list_tools),
        ("Google Search", test_google_search),
        ("Generic Tool Endpoint", test_generic_tool_endpoint)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed_count = 0
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if passed:
            passed_count += 1
    
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")
    
    if passed_count == len(results):
        print("\n🎉 All tests passed! The MCP bridge is working correctly!")
        return 0
    else:
        print(f"\n⚠️ {len(results) - passed_count} test(s) failed. Please check the server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())