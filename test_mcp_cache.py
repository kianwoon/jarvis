#!/usr/bin/env python
"""
Script to test the MCP tools cache with the API key from the manifest table
"""
import os
import sys
import json
import requests

# Add the parent directory to the path so we can import app
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from app.core.mcp_tools_cache import reload_enabled_mcp_tools, get_enabled_mcp_tools

def test_mcp_cache():
    print("Reloading MCP tools cache...")
    tools = reload_enabled_mcp_tools()
    
    # Print information about cached tools
    print(f"\nFound {len(tools)} tools in cache:")
    for tool_name, tool_info in tools.items():
        print(f"\n  Tool: {tool_name}")
        print(f"  Description: {tool_info.get('description', 'N/A')}")
        print(f"  Endpoint: {tool_info.get('endpoint', 'N/A')}")
        print(f"  Method: {tool_info.get('method', 'N/A')}")
        print(f"  Manifest ID: {tool_info.get('manifest_id', 'N/A')}")
        print(f"  Manifest Hostname: {tool_info.get('manifest_hostname', 'N/A')}")
        
        # Check API key
        api_key = tool_info.get('api_key')
        if api_key:
            print(f"  API Key: [REDACTED] (length: {len(api_key)})")
        else:
            print(f"  API Key: None")
    
    print("\nTesting manual API call with the API key...")
    print("This will check if API key is correctly loaded from manifest")
    
    # Get a tool name from the cache
    if tools:
        tool_name = next(iter(tools.keys()))
        tool_info = tools[tool_name]
        
        print(f"\nTrying to call tool: {tool_name}")
        endpoint = tool_info.get("endpoint")
        method = tool_info.get("method", "POST")
        api_key = tool_info.get("api_key")
        
        # Replace localhost with manifest_hostname if needed
        manifest_hostname = tool_info.get("manifest_hostname")
        if manifest_hostname and "localhost" in endpoint:
            endpoint = endpoint.replace("localhost:9000", manifest_hostname)
            print(f"Replaced localhost with manifest hostname: {endpoint}")
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            print(f"Added API key to Authorization header (length: {len(api_key)})")
        
        print(f"Calling {method} {endpoint}")
        try:
            if method.upper() == "GET":
                response = requests.get(endpoint, headers=headers, timeout=5)
            else:
                response = requests.post(endpoint, json={}, headers=headers, timeout=5)
            
            print(f"Response status: {response.status_code}")
            try:
                print(f"Response body: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"Response body: {response.text}")
        except Exception as e:
            print(f"Error calling API: {str(e)}")
    else:
        print("No tools found in cache, skipping API call test")

if __name__ == "__main__":
    test_mcp_cache() 