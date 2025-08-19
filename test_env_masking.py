#!/usr/bin/env python3
"""
Test script to verify that sensitive environment variables are properly masked in the API response.
"""

import requests
import json
import sys

def test_env_masking():
    """Test that sensitive environment variables are masked properly."""
    
    base_url = "http://localhost:8000/api/v1/mcp-servers"
    
    print("=" * 60)
    print("Testing MCP Server Environment Variable Masking")
    print("=" * 60)
    
    # Test 1: List all servers (should mask sensitive values by default)
    print("\n1. Testing list servers endpoint (should mask sensitive values)...")
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        servers = response.json()
        
        # Find the Local MCP server
        local_mcp = None
        for server in servers:
            if server.get('name') == 'Local MCP':
                local_mcp = server
                break
        
        if local_mcp:
            print(f"   ✓ Found Local MCP server (ID: {local_mcp.get('id')})")
            env_vars = local_mcp.get('env', {})
            
            # Check if sensitive values are masked
            masked_count = 0
            unmasked_sensitive = []
            
            for key, value in env_vars.items():
                if any(sensitive in key.upper() for sensitive in ['TOKEN', 'KEY', 'SECRET', 'PASSWORD']):
                    if '•' in str(value):
                        masked_count += 1
                        print(f"   ✓ {key}: {value[:20]}... (masked)")
                    else:
                        unmasked_sensitive.append(key)
                        print(f"   ✗ {key}: VALUE NOT MASKED!")
            
            if unmasked_sensitive:
                print(f"\n   ⚠️  WARNING: {len(unmasked_sensitive)} sensitive variables are not masked!")
            else:
                print(f"\n   ✓ All {masked_count} sensitive variables are properly masked")
        else:
            print("   ✗ Local MCP server not found")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Get specific server without show_sensitive flag
    print("\n2. Testing get server endpoint without show_sensitive...")
    try:
        response = requests.get(f"{base_url}/9")  # Local MCP has ID 9
        if response.status_code == 200:
            server = response.json()
            env_vars = server.get('env', {})
            
            sensitive_masked = all(
                '•' in str(value) 
                for key, value in env_vars.items() 
                if any(s in key.upper() for s in ['TOKEN', 'KEY', 'SECRET', 'PASSWORD'])
            )
            
            if sensitive_masked:
                print("   ✓ Sensitive values are masked")
            else:
                print("   ✗ Some sensitive values are NOT masked")
        else:
            print(f"   ✗ Server not found or error: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Get specific server WITH show_sensitive flag
    print("\n3. Testing get server endpoint WITH show_sensitive=true...")
    try:
        response = requests.get(f"{base_url}/9?show_sensitive=true")
        if response.status_code == 200:
            server = response.json()
            env_vars = server.get('env', {})
            
            has_unmasked = any(
                '•' not in str(value) 
                for key, value in env_vars.items() 
                if any(s in key.upper() for s in ['TOKEN', 'KEY', 'SECRET', 'PASSWORD']) and value
            )
            
            if has_unmasked:
                print("   ✓ Sensitive values are shown (as requested)")
                # Don't print the actual values for security
                print("   ℹ️  Values visible but not displayed here for security")
            else:
                print("   ✗ Values are still masked (should be unmasked)")
        else:
            print(f"   ✗ Server not found or error: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Get environment variables endpoint (admin endpoint)
    print("\n4. Testing dedicated /env endpoint (admin access)...")
    try:
        response = requests.get(f"{base_url}/9/env")
        if response.status_code == 200:
            data = response.json()
            env_vars = data.get('env', {})
            warning = data.get('warning', '')
            
            if warning:
                print(f"   ℹ️  Warning: {warning}")
            
            has_unmasked = any(
                '•' not in str(value) 
                for key, value in env_vars.items() 
                if any(s in key.upper() for s in ['TOKEN', 'KEY', 'SECRET', 'PASSWORD']) and value
            )
            
            if has_unmasked:
                print("   ✓ Full environment variables returned (admin access)")
                print(f"   ℹ️  Found {len(env_vars)} environment variables")
            else:
                print("   ✗ Values are masked (should be unmasked for admin)")
        else:
            print(f"   ✗ Endpoint not accessible or error: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("- Sensitive environment variables should be masked by default")
    print("- Only admin endpoints or explicit flags should show full values")
    print("- The system is working correctly if tests 1-2 show masked values")
    print("  and tests 3-4 show unmasked values")
    print("=" * 60)

if __name__ == "__main__":
    test_env_masking()