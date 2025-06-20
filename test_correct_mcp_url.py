#!/usr/bin/env python3
"""
Test MCP client with the correct Zapier URL format
"""

import asyncio
import sys
import logging
from app.core.remote_mcp_client import RemoteMCPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_endpoint():
    """Test MCP client with correct endpoint format"""
    
    # According to Zapier docs, the URL should end with /mcp, not /sse
    correct_url = "https://mcp.zapier.com/api/mcp/s/ZjFiNDBjZmUtOGVmMy00NDU0LWE3ODAtZDc0NTk4NDUwNjljOmMzODljYzRiLTAyZDUtNGE3OS1iN2I0LTUxNjI5ZGU0YzUyZQ==/mcp"
    
    print(f"🔍 Testing MCP client with correct URL: {correct_url}")
    
    client = RemoteMCPClient(
        server_url=correct_url,
        transport_type="http",  # Use HTTP instead of SSE for now
        client_info={"name": "jarvis-test-client", "version": "1.0.0"}
    )
    
    try:
        async with client:
            print("✅ MCP client connected successfully")
            
            # Try to get server info
            server_info = await client.get_server_info()
            print(f"📊 Server info: {server_info}")
            
            # Try to list tools
            print("🔍 Attempting to list tools...")
            tools = await client.list_tools()
            print(f"🛠️  Found {len(tools)} tools")
            
            for tool in tools[:3]:  # Show first 3 tools
                print(f"   - {tool.get('name')}: {tool.get('description', 'No description')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        return False

async def main():
    print("🧪 Testing MCP Client with Correct Zapier URL Format\n")
    
    success = await test_mcp_endpoint()
    
    if success:
        print("\n✅ MCP client test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ MCP client test failed.")
        print("📝 Note: This may be expected if the Zapier URL is a demo/example URL")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())