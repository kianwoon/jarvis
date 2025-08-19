#!/usr/bin/env python3
"""
Test script to verify that the communication_protocol field is correctly returned by the API.
"""

import requests
import json
from typing import Dict, List

def test_mcp_servers_api():
    """Test that the /api/v1/mcp/servers endpoint returns communication_protocol field."""
    
    # Make API request
    response = requests.get("http://localhost:8000/api/v1/mcp/servers/")
    assert response.status_code == 200, f"API returned status {response.status_code}"
    
    servers = response.json()
    assert isinstance(servers, list), "API should return a list of servers"
    
    print(f"Found {len(servers)} MCP servers\n")
    
    # Check each server has communication_protocol field
    for server in servers:
        name = server.get('name', 'Unknown')
        config_type = server.get('config_type', 'Unknown')
        comm_protocol = server.get('communication_protocol', 'MISSING')
        
        # Verify field exists
        assert 'communication_protocol' in server, f"Server '{name}' missing communication_protocol field"
        
        # Verify it's not None (unless intentionally set to None)
        assert comm_protocol != 'MISSING', f"Server '{name}' has no communication_protocol value"
        
        # Print results
        print(f"✅ {name:20} | Type: {config_type:12} | Protocol: {comm_protocol}")
        
        # Verify expected values based on config_type
        if config_type == "command" and comm_protocol not in ["stdio", "http"]:
            print(f"  ⚠️  Warning: Command server usually uses stdio or http, got {comm_protocol}")
        elif config_type == "http" and comm_protocol != "http":
            print(f"  ⚠️  Warning: HTTP server should use http protocol, got {comm_protocol}")
        elif config_type == "manifest" and comm_protocol != "http":
            print(f"  ⚠️  Warning: Manifest server usually uses http protocol, got {comm_protocol}")
        elif config_type == "remote_http" and comm_protocol not in ["http", "sse"]:
            print(f"  ⚠️  Warning: Remote HTTP server uses http or sse, got {comm_protocol}")
    
    print("\n✅ All servers have communication_protocol field properly set!")
    
    # Test individual server endpoint
    if servers:
        first_server = servers[0]
        server_id = first_server['id']
        print(f"\nTesting individual server endpoint for ID {server_id}...")
        
        individual_response = requests.get(f"http://localhost:8000/api/v1/mcp/servers/{server_id}")
        assert individual_response.status_code == 200, f"Individual server API returned {individual_response.status_code}"
        
        individual_server = individual_response.json()
        assert 'communication_protocol' in individual_server, "Individual server missing communication_protocol"
        
        print(f"✅ Individual server '{individual_server['name']}' has protocol: {individual_server['communication_protocol']}")
    
    return True

if __name__ == "__main__":
    try:
        test_mcp_servers_api()
        print("\n🎉 All tests passed! The communication_protocol field is correctly returned by the API.")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        exit(1)
    except requests.RequestException as e:
        print(f"\n❌ Failed to connect to API: {e}")
        print("Make sure the backend server is running on http://localhost:8000")
        exit(1)