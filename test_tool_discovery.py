#!/usr/bin/env python3
"""
Test tool discovery for the remote HTTP MCP server
"""

import requests
import json
import sys

def test_tool_discovery():
    """Test discovering tools from the remote HTTP server"""
    
    # First, get the remote HTTP server
    try:
        response = requests.get("http://localhost:8000/api/v1/mcp/servers", timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to get servers: {response.status_code}")
            return False
            
        servers = response.json()
        remote_servers = [s for s in servers if s.get('config_type') == 'remote_http']
        
        if not remote_servers:
            print("❌ No remote HTTP servers found")
            return False
            
        server = remote_servers[0]
        server_id = server['id']
        print(f"✅ Found remote HTTP server: {server['name']} (ID: {server_id})")
        print(f"   URL: {server.get('remote_config', {}).get('server_url', 'N/A')}")
        
        # Check current tools
        tools_response = requests.get(f"http://localhost:8000/api/v1/mcp/servers/{server_id}/tools", timeout=10)
        if tools_response.status_code == 200:
            current_tools = tools_response.json()
            print(f"   Current tools: {len(current_tools)}")
            for tool in current_tools:
                print(f"   - {tool['name']}: {tool.get('description', 'No description')}")
        
        # Now test tool discovery
        print(f"\n🔍 Testing tool discovery for server {server_id}...")
        discovery_response = requests.post(
            f"http://localhost:8000/api/v1/mcp/servers/{server_id}/discover-tools",
            timeout=30
        )
        
        if discovery_response.status_code == 200:
            result = discovery_response.json()
            print("✅ Tool discovery successful!")
            print(f"   Status: {result.get('status')}")
            print(f"   Tools discovered: {result.get('tools_discovered', 0)}")
            print(f"   Tools added: {result.get('tools_added', 0)}")
            print(f"   Tools updated: {result.get('tools_updated', 0)}")
            
            if result.get('tools'):
                print("\n📋 Discovered tools:")
                for tool in result['tools']:
                    print(f"   - {tool.get('name')}: {tool.get('description', 'No description')}")
            
            return True
        else:
            print(f"❌ Tool discovery failed: {discovery_response.status_code}")
            print(f"   Response: {discovery_response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Remote HTTP MCP Server Tool Discovery\n")
    
    success = test_tool_discovery()
    
    if success:
        print("\n✅ Tool discovery test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Tool discovery test failed.")
        sys.exit(1)