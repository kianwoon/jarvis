#!/usr/bin/env python3
"""
Test the smart endpoint detection with Zapier SSE URL
"""

import asyncio
import sys
import logging
from app.core.remote_mcp_client import RemoteMCPClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_zapier_sse_auto_correction():
    """Test that /sse URLs are automatically corrected to /mcp for Zapier"""
    
    # This is the URL you're using (with /sse)
    sse_url = "https://mcp.zapier.com/api/mcp/s/ZjFiNDBjZmUtOGVmMy00NDU0LWE3ODAtZDc0NTk4NDUwNjljOmMzODljYzRiLTAyZDUtNGE3OS1iN2I0LTUxNjI5ZGU0YzUyZQ==/sse"
    
    print(f"🔍 Testing smart endpoint detection with: {sse_url}")
    print(f"   Transport Type: SSE")
    
    client = RemoteMCPClient(
        server_url=sse_url,
        transport_type="sse",  # This should be auto-corrected
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
    print("🧪 Testing Smart Endpoint Detection for Zapier MCP\n")
    
    success = await test_zapier_sse_auto_correction()
    
    if success:
        print("\n✅ Smart endpoint detection worked! Your existing configuration should now work.")
        sys.exit(0)
    else:
        print("\n❌ Smart endpoint detection failed.")
        print("📝 Check the error message above for details")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())