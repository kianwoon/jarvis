#!/usr/bin/env python3
"""
Test script for remote HTTP MCP server functionality
"""

import requests
import json
import sys

def test_remote_http_server():
    """Test creating a remote HTTP MCP server"""
    
    # API endpoint
    url = "http://localhost:8000/api/v1/mcp/servers"
    
    # Test data for remote HTTP server
    test_server = {
        "name": "Test Zapier MCP Server",
        "config_type": "remote_http",
        "remote_config": {
            "server_url": "https://mcp.zapier.com/api/mcp/s/test-server-url/sse",
            "transport_type": "sse",
            "auth_type": "none",
            "auth_headers": {},
            "client_info": {
                "name": "jarvis-mcp-client",
                "version": "1.0.0"
            },
            "capabilities": {},
            "connection_timeout": 30
        },
        "is_active": True
    }
    
    try:
        # Create the server
        print("Creating remote HTTP MCP server...")
        response = requests.post(url, json=test_server, timeout=10)
        
        if response.status_code == 201 or response.status_code == 200:
            server_data = response.json()
            print(f"✅ Successfully created remote HTTP server:")
            print(f"   - ID: {server_data.get('id')}")
            print(f"   - Name: {server_data.get('name')}")
            print(f"   - Config Type: {server_data.get('config_type')}")
            print(f"   - Server URL: {server_data.get('remote_config', {}).get('server_url')}")
            print(f"   - Transport: {server_data.get('remote_config', {}).get('transport_type')}")
            
            # Now test getting the server
            server_id = server_data.get('id')
            if server_id:
                print(f"\nTesting retrieval of server {server_id}...")
                get_response = requests.get(f"{url}/{server_id}", timeout=10)
                if get_response.status_code == 200:
                    retrieved_server = get_response.json()
                    print(f"✅ Successfully retrieved server:")
                    print(f"   - Remote config present: {'remote_config' in retrieved_server}")
                    if 'remote_config' in retrieved_server and retrieved_server['remote_config']:
                        print(f"   - Server URL: {retrieved_server['remote_config'].get('server_url')}")
                else:
                    print(f"❌ Failed to retrieve server: {get_response.status_code}")
                    print(f"   Response: {get_response.text}")
            
            return True
        else:
            print(f"❌ Failed to create remote HTTP server: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_list_servers():
    """Test listing all servers including remote HTTP ones"""
    
    try:
        print("\nTesting server list endpoint...")
        response = requests.get("http://localhost:8000/api/v1/mcp/servers", timeout=10)
        
        if response.status_code == 200:
            servers = response.json()
            print(f"✅ Successfully retrieved {len(servers)} servers")
            
            # Look for remote HTTP servers
            remote_servers = [s for s in servers if s.get('config_type') == 'remote_http']
            if remote_servers:
                print(f"   Found {len(remote_servers)} remote HTTP servers:")
                for server in remote_servers:
                    print(f"   - {server['name']} (ID: {server['id']})")
                    if server.get('remote_config'):
                        print(f"     URL: {server['remote_config'].get('server_url')}")
            else:
                print("   No remote HTTP servers found")
            
            return True
        else:
            print(f"❌ Failed to list servers: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ List test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Remote HTTP MCP Server functionality\n")
    
    # Test listing servers first
    list_success = test_list_servers()
    
    # Test creating a remote HTTP server
    create_success = test_remote_http_server()
    
    if list_success and create_success:
        print("\n✅ All tests passed! Remote HTTP MCP server functionality is working.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed.")
        sys.exit(1)