#!/usr/bin/env python3
"""
Simple test to check if the API routes are working
"""

import requests
import sys

def test_mcp_endpoints():
    """Test the MCP server endpoints"""
    base_url = "http://localhost:8000/api/v1/mcp"
    
    print("🧪 Testing MCP API endpoints...")
    
    try:
        # Test GET /servers/
        print("Testing GET /servers/")
        response = requests.get(f"{base_url}/servers/", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            servers = response.json()
            print(f"✅ Found {len(servers)} servers")
            for server in servers:
                print(f"  - {server.get('name')} ({server.get('config_type')})")
        else:
            print(f"❌ Error: {response.text}")
            
        # Test GET /servers/1 if we have servers
        if response.status_code == 200 and len(response.json()) > 0:
            server_id = response.json()[0]['id']
            print(f"\nTesting GET /servers/{server_id}")
            detail_response = requests.get(f"{base_url}/servers/{server_id}", timeout=5)
            print(f"Status: {detail_response.status_code}")
            if detail_response.status_code == 200:
                print("✅ Server detail retrieved successfully")
            else:
                print(f"❌ Error: {detail_response.text}")
                
            # Test health check
            print(f"\nTesting POST /servers/{server_id}/health")
            health_response = requests.post(f"{base_url}/servers/{server_id}/health", timeout=5)
            print(f"Status: {health_response.status_code}")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Health check: {health_data.get('status')}")
            else:
                print(f"❌ Health check error: {health_response.text}")
                
            # Test tools endpoint
            print(f"\nTesting GET /servers/{server_id}/tools")
            tools_response = requests.get(f"{base_url}/servers/{server_id}/tools", timeout=5)
            print(f"Status: {tools_response.status_code}")
            if tools_response.status_code == 200:
                tools = tools_response.json()
                print(f"✅ Found {len(tools)} tools for server {server_id}")
            else:
                print(f"❌ Tools error: {tools_response.text}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure your FastAPI server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing MCP API endpoints...")
    print("Make sure your FastAPI application is running first!")
    print()
    
    success = test_mcp_endpoints()
    
    if success:
        print("\n🎉 API tests completed!")
    else:
        print("\n❌ API tests failed!")
        sys.exit(1)