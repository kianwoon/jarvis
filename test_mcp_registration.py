#!/usr/bin/env python3
"""Test script to check MCP tools registration and fix connection issues"""

import asyncio
import json
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mcp_tools_cache():
    """Test if Google Search tool is registered in the MCP tools cache"""
    print("\n" + "="*60)
    print("1. Testing MCP Tools Cache")
    print("="*60)
    
    try:
        from app.core.mcp_tools_cache import get_enabled_mcp_tools, reload_enabled_mcp_tools
        
        # Force reload to get fresh data
        print("\nReloading MCP tools from database...")
        tools = reload_enabled_mcp_tools()
        
        print(f"\nTotal tools registered: {len(tools)}")
        
        # Filter for Google Search related tools
        google_tools = {name: info for name, info in tools.items() 
                       if 'google' in name.lower() or 'search' in name.lower()}
        
        if google_tools:
            print(f"\n✅ Found {len(google_tools)} Google/Search related tools:")
            for tool_name, tool_info in google_tools.items():
                print(f"\n  Tool: {tool_name}")
                print(f"  - Description: {tool_info.get('description', 'N/A')}")
                print(f"  - Endpoint: {tool_info.get('endpoint', 'N/A')}")
                print(f"  - Server ID: {tool_info.get('server_id', 'N/A')}")
                print(f"  - Server Hostname: {tool_info.get('server_hostname', 'N/A')}")
                if tool_info.get('parameters'):
                    print(f"  - Parameters: {json.dumps(tool_info['parameters'], indent=4)}")
        else:
            print("\n❌ No Google Search tools found in cache!")
            
        return tools
    except Exception as e:
        print(f"\n❌ Error testing MCP tools cache: {e}")
        import traceback
        traceback.print_exc()
        return {}

def check_database_configuration():
    """Check the database for MCP tool and server configuration"""
    print("\n" + "="*60)
    print("2. Checking Database Configuration")
    print("="*60)
    
    try:
        from app.core.db import SessionLocal, MCPTool, MCPServer, MCPManifest
        
        db = SessionLocal()
        try:
            # Check for MCP servers
            servers = db.query(MCPServer).all()
            print(f"\n📡 MCP Servers: {len(servers)} found")
            
            for server in servers:
                print(f"\n  Server ID {server.id}: {server.name}")
                print(f"  - Type: {server.config_type}")
                print(f"  - Active: {server.is_active}")
                print(f"  - Hostname: {server.hostname}")
                if server.config_type == "manifest":
                    manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server.id).first()
                    if manifest:
                        print(f"  - Manifest Hostname: {manifest.hostname}")
                        print(f"  - Has API Key: {'Yes' if manifest.api_key else 'No'}")
                elif server.config_type == "command":
                    print(f"  - Command: {server.command}")
                    print(f"  - Args: {server.args}")
                elif server.config_type == "remote":
                    if server.remote_config:
                        print(f"  - Remote URL: {server.remote_config.get('url', 'N/A')}")
            
            # Check for Google Search specific tools
            google_tools = db.query(MCPTool).filter(
                (MCPTool.name.like('%google%')) | 
                (MCPTool.name.like('%search%'))
            ).all()
            
            print(f"\n🔍 Google/Search Tools in DB: {len(google_tools)} found")
            for tool in google_tools:
                print(f"\n  Tool: {tool.name}")
                print(f"  - Active: {tool.is_active}")
                print(f"  - Endpoint: {tool.endpoint}")
                print(f"  - Server ID: {tool.server_id}")
                print(f"  - Method: {tool.method}")
                
                # Check if server exists
                if tool.server_id:
                    server = db.query(MCPServer).filter(MCPServer.id == tool.server_id).first()
                    if server:
                        print(f"  - Server: {server.name} (hostname: {server.hostname})")
                    else:
                        print(f"  - ⚠️  Server ID {tool.server_id} not found!")
            
            # Check all active MCP tools
            all_active_tools = db.query(MCPTool).filter(MCPTool.is_active == True).all()
            print(f"\n📦 Total Active Tools: {len(all_active_tools)}")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"\n❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

async def test_mcp_server_connection():
    """Test direct connection to the MCP server"""
    print("\n" + "="*60)
    print("3. Testing MCP Server Connection")
    print("="*60)
    
    try:
        import aiohttp
        
        # Test both localhost and host.docker.internal
        test_urls = [
            ("http://localhost:3001", "localhost"),
            ("http://host.docker.internal:3001", "host.docker.internal"),
            ("http://127.0.0.1:3001", "127.0.0.1")
        ]
        
        for url, name in test_urls:
            print(f"\n🔗 Testing connection to {name}: {url}")
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    # Try to connect to the server
                    test_endpoint = f"{url}/tools"
                    async with session.get(test_endpoint) as response:
                        if response.status == 200:
                            print(f"  ✅ Successfully connected to {name}")
                            data = await response.json()
                            print(f"  - Response preview: {str(data)[:200]}...")
                        else:
                            print(f"  ❌ Server responded with status {response.status}")
            except aiohttp.ClientConnectorError as e:
                print(f"  ❌ Connection refused: {e}")
            except asyncio.TimeoutError:
                print(f"  ❌ Connection timeout")
            except Exception as e:
                print(f"  ❌ Connection error: {e}")
                
    except Exception as e:
        print(f"\n❌ Error testing server connection: {e}")
        import traceback
        traceback.print_exc()

async def test_unified_mcp_service():
    """Test the unified MCP service with Google Search"""
    print("\n" + "="*60)
    print("4. Testing Unified MCP Service")
    print("="*60)
    
    try:
        from app.core.unified_mcp_service import call_mcp_tool_unified
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        
        # Get tools from cache
        tools = get_enabled_mcp_tools()
        
        # Find Google Search tool
        google_tool = None
        google_tool_name = None
        for name, info in tools.items():
            if 'google' in name.lower() and 'search' in name.lower():
                google_tool = info
                google_tool_name = name
                break
        
        if google_tool:
            print(f"\n🔍 Found Google Search tool: {google_tool_name}")
            print(f"  - Endpoint: {google_tool.get('endpoint', 'N/A')}")
            print(f"  - Server Hostname: {google_tool.get('server_hostname', 'N/A')}")
            
            # Test the tool
            print("\n📤 Testing Google Search with query: 'Python programming'")
            test_params = {
                "query": "Python programming",
                "num_results": 3
            }
            
            result = await call_mcp_tool_unified(google_tool, google_tool_name, test_params)
            
            if "error" in result:
                print(f"  ❌ Tool call failed: {result['error']}")
            else:
                print(f"  ✅ Tool call successful!")
                print(f"  - Result type: {type(result)}")
                print(f"  - Result preview: {str(result)[:500]}...")
        else:
            print("\n❌ No Google Search tool found to test!")
            
    except Exception as e:
        print(f"\n❌ Error testing unified MCP service: {e}")
        import traceback
        traceback.print_exc()

def fix_database_hostname():
    """Fix the hostname configuration in the database if needed"""
    print("\n" + "="*60)
    print("5. Fixing Database Hostname Configuration")
    print("="*60)
    
    try:
        from app.core.db import SessionLocal, MCPServer, MCPManifest, MCPTool
        
        db = SessionLocal()
        try:
            # Find servers with host.docker.internal
            servers_to_fix = db.query(MCPServer).filter(
                MCPServer.hostname.like('%host.docker.internal%')
            ).all()
            
            if servers_to_fix:
                print(f"\n⚠️  Found {len(servers_to_fix)} servers with host.docker.internal")
                
                for server in servers_to_fix:
                    old_hostname = server.hostname
                    # Replace host.docker.internal with localhost
                    new_hostname = old_hostname.replace('host.docker.internal', 'localhost')
                    server.hostname = new_hostname
                    print(f"  📝 Updated Server {server.id} ({server.name}):")
                    print(f"     {old_hostname} → {new_hostname}")
                
                # Also check manifests
                manifests_to_fix = db.query(MCPManifest).filter(
                    MCPManifest.hostname.like('%host.docker.internal%')
                ).all()
                
                for manifest in manifests_to_fix:
                    old_hostname = manifest.hostname
                    new_hostname = old_hostname.replace('host.docker.internal', 'localhost')
                    manifest.hostname = new_hostname
                    print(f"  📝 Updated Manifest for Server {manifest.server_id}:")
                    print(f"     {old_hostname} → {new_hostname}")
                
                # Update tool endpoints if they contain host.docker.internal
                tools_to_fix = db.query(MCPTool).filter(
                    MCPTool.endpoint.like('%host.docker.internal%')
                ).all()
                
                for tool in tools_to_fix:
                    old_endpoint = tool.endpoint
                    new_endpoint = old_endpoint.replace('host.docker.internal', 'localhost')
                    tool.endpoint = new_endpoint
                    print(f"  📝 Updated Tool {tool.name}:")
                    print(f"     {old_endpoint} → {new_endpoint}")
                
                # Commit changes
                db.commit()
                print("\n✅ Database hostname configuration updated successfully!")
                
                # Clear the cache to reload with new configuration
                from app.core.mcp_tools_cache import reload_enabled_mcp_tools
                reload_enabled_mcp_tools()
                print("✅ MCP tools cache reloaded with new configuration")
                
            else:
                print("\n✅ No servers with host.docker.internal found - configuration looks good!")
                
        finally:
            db.close()
            
    except Exception as e:
        print(f"\n❌ Error fixing database configuration: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("\n" + "="*80)
    print(" MCP TOOLS REGISTRATION AND CONNECTION TEST ")
    print("="*80)
    
    # 1. Test MCP tools cache
    tools = test_mcp_tools_cache()
    
    # 2. Check database configuration
    check_database_configuration()
    
    # 3. Test MCP server connection
    await test_mcp_server_connection()
    
    # 4. Test unified MCP service
    await test_unified_mcp_service()
    
    # 5. Ask user if they want to fix the hostname issue
    if tools:
        print("\n" + "="*60)
        print("💡 Hostname Configuration Check")
        print("="*60)
        print("\nIf the tests above showed connection issues with 'host.docker.internal',")
        print("we can automatically fix this by updating the database to use 'localhost' instead.")
        print("\nThis is needed when running the application locally (not in Docker).")
        
        response = input("\n🔧 Do you want to fix the hostname configuration? (y/n): ")
        if response.lower() == 'y':
            fix_database_hostname()
            
            # Re-test after fixing
            print("\n🔄 Re-testing connection after fix...")
            await test_mcp_server_connection()
            await test_unified_mcp_service()
    
    print("\n" + "="*80)
    print(" TEST COMPLETE ")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())