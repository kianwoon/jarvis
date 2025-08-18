#!/usr/bin/env python3
"""
Test script for MCP HTTP Bridge Server

This script tests the MCP HTTP Bridge Server endpoints to ensure
they are working correctly.

Usage:
    # First start the bridge server in one terminal:
    python mcp_http_bridge_server.py
    
    # Then run this test in another terminal:
    python test_mcp_bridge.py
"""

import requests
import json
import time
from typing import Dict, Any

# Bridge server URL
BASE_URL = "http://localhost:3001"

def test_health_check():
    """Test the health check endpoint"""
    print("\n🔍 Testing Health Check Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   - Timestamp: {data['timestamp']}")
            print(f"   - Available Tools: {data['available_tools']}")
            print(f"   - Database Connected: {data['database_connected']}")
            return True
        else:
            print(f"❌ Health Check Failed: Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to bridge server. Is it running on port 3001?")
        return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_list_tools():
    """Test listing available tools"""
    print("\n🔍 Testing List Tools Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/tools")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['total']} tools")
            
            # Show first 5 tools
            for tool in data['tools'][:5]:
                print(f"   - {tool['name']}: {tool.get('description', 'No description')[:50]}...")
            
            if data['total'] > 5:
                print(f"   ... and {data['total'] - 5} more tools")
            
            # Check if google_search is available
            google_search = next((t for t in data['tools'] if t['name'] == 'google_search'), None)
            if google_search:
                print(f"\n   📌 google_search tool found with server_id={google_search.get('server_id')}")
            
            return True
        else:
            print(f"❌ List Tools Failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ List Tools Error: {e}")
        return False

def test_get_tool_info():
    """Test getting info for a specific tool"""
    print("\n🔍 Testing Get Tool Info Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/tools/google_search")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Got info for google_search tool:")
            print(f"   - Name: {data.get('name')}")
            print(f"   - Description: {data.get('description', 'No description')[:100]}...")
            print(f"   - Server ID: {data.get('server_id')}")
            print(f"   - Endpoint: {data.get('endpoint')}")
            return True
        elif response.status_code == 404:
            print("⚠️ google_search tool not found in database")
            print("   You may need to register it first")
            return False
        else:
            print(f"❌ Get Tool Info Failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Get Tool Info Error: {e}")
        return False

def test_google_search():
    """Test executing Google Search"""
    print("\n🔍 Testing Google Search Execution...")
    
    # Test query
    search_params = {
        "query": "OpenAI GPT-4 latest news",
        "num_results": 3
    }
    
    print(f"   Searching for: '{search_params['query']}'")
    
    try:
        # Test the direct google_search endpoint
        print("\n   Testing direct /tools/google_search endpoint...")
        response = requests.post(
            f"{BASE_URL}/tools/google_search",
            json=search_params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Google Search successful!")
                print(f"   - Execution time: {data['execution_time']:.2f} seconds")
                
                # Show results preview
                if data.get('result') and 'content' in data['result']:
                    content = data['result']['content'][0]['text']
                    print(f"\n   Results preview:")
                    # Show first 500 characters
                    preview = content[:500]
                    if len(content) > 500:
                        preview += "..."
                    print(f"   {preview}")
                
                return True
            else:
                print(f"❌ Google Search failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Google Search Failed: Status {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Google Search Error: {e}")
        return False

def test_generic_tool_execution():
    """Test the generic tool execution endpoint"""
    print("\n🔍 Testing Generic Tool Execution Endpoint...")
    
    request_data = {
        "tool_name": "google_search",
        "parameters": {
            "query": "Python FastAPI tutorial",
            "num_results": 2
        },
        "server_id": 9  # Specific server ID for google_search
    }
    
    print(f"   Executing tool: {request_data['tool_name']}")
    print(f"   Query: '{request_data['parameters']['query']}'")
    
    try:
        response = requests.post(
            f"{BASE_URL}/tools/google_search",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Tool execution successful!")
                print(f"   - Tool: {data['tool_name']}")
                print(f"   - Server ID: {data.get('server_id')}")
                print(f"   - Execution time: {data['execution_time']:.2f} seconds")
                return True
            else:
                print(f"❌ Tool execution failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Tool Execution Failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Tool Execution Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MCP HTTP Bridge Server Test Suite")
    print("=" * 60)
    
    # Check if server is running
    if not test_health_check():
        print("\n⚠️ Bridge server is not running!")
        print("Please start it with: python mcp_http_bridge_server.py")
        return
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    # Test listing tools
    tests_total += 1
    if test_list_tools():
        tests_passed += 1
    
    # Test getting tool info
    tests_total += 1
    if test_get_tool_info():
        tests_passed += 1
    
    # Test Google Search
    tests_total += 1
    if test_google_search():
        tests_passed += 1
    
    # Test generic tool execution
    tests_total += 1
    if test_generic_tool_execution():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed}/{tests_total} passed")
    
    if tests_passed == tests_total:
        print("✅ All tests passed!")
    else:
        print(f"⚠️ {tests_total - tests_passed} tests failed")
    
    print("=" * 60)

if __name__ == "__main__":
    main()