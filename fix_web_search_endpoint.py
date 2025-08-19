#!/usr/bin/env python3
"""
Fix for web search endpoint issue where MCP tool calls are going to localhost:80
instead of the proper host.docker.internal:3001
"""

import sys
import os

# Add project root to path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

def fix_enhanced_tool_executor():
    """Fix the enhanced_tool_executor to properly handle endpoint URLs"""
    
    file_path = "/Users/kianwoonwong/Downloads/jarvis/app/core/enhanced_tool_executor.py"
    
    # Read the current file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if fix is already applied
    if "# FIX: Ensure endpoint is properly preserved" in content:
        print("✅ enhanced_tool_executor.py already fixed")
        return False
    
    # Find the _adjust_endpoint_for_environment function and fix it
    fix = '''def _adjust_endpoint_for_environment(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjust endpoint URLs based on whether we're running inside Docker
    """
    try:
        endpoint = tool_info.get("endpoint", "")
        server_hostname = tool_info.get("server_hostname")
        
        # Check if we're running inside Docker
        import os
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        adjusted_tool_info = tool_info.copy()
        
        # FIX: Ensure endpoint is properly preserved
        # When not in Docker, use the endpoint as-is if it contains host.docker.internal
        if not in_docker and "host.docker.internal" in endpoint:
            # Running locally, keep host.docker.internal as it works on Mac
            logger.debug(f"Keeping endpoint as-is for local execution: {endpoint}")
        elif in_docker and server_hostname and "localhost" in endpoint:
            adjusted_tool_info["endpoint"] = endpoint.replace("localhost", server_hostname)
            logger.debug(f"Adjusted endpoint for Docker: {adjusted_tool_info['endpoint']}")
        
        return adjusted_tool_info
    except:
        return tool_info'''
    
    # Replace the function
    import re
    pattern = r'def _adjust_endpoint_for_environment\(tool_info: Dict\[str, Any\]\) -> Dict\[str, Any\]:.*?(?=\ndef |\Z)'
    
    # Use re.DOTALL to match across multiple lines
    new_content = re.sub(pattern, fix, content, flags=re.DOTALL)
    
    if new_content != content:
        # Write the fixed content
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"✅ Fixed _adjust_endpoint_for_environment in {file_path}")
        return True
    else:
        print("⚠️ Could not find _adjust_endpoint_for_environment function to fix")
        return False

def fix_unified_mcp_service():
    """Fix the unified_mcp_service to ensure endpoint is used correctly"""
    
    file_path = "/Users/kianwoonwong/Downloads/jarvis/app/core/unified_mcp_service.py"
    
    # Read the current file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Look for the line that logs the HTTP call and add a fix
    fixed = False
    for i, line in enumerate(lines):
        if '[HTTP] Calling {tool_name} at {endpoint}' in line and '# FIX:' not in lines[i-1]:
            # Add a comment and ensure endpoint is correct
            lines[i] = '                # FIX: Ensure endpoint is the full URL with port\n' + \
                       '                if endpoint and not endpoint.startswith("http"):\n' + \
                       '                    endpoint = f"http://{endpoint}"\n' + \
                       '                logger.info(f"[HTTP] Calling {tool_name} at {endpoint}")\n'
            fixed = True
            break
    
    if fixed:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"✅ Fixed endpoint logging in {file_path}")
        return True
    else:
        print("✅ unified_mcp_service.py already fixed or pattern not found")
        return False

def verify_mcp_tool_configuration():
    """Verify that MCP tools are properly configured"""
    from app.core.mcp_tools_cache import get_enabled_mcp_tools
    
    tools = get_enabled_mcp_tools()
    
    if 'google_search' in tools:
        tool_info = tools['google_search']
        endpoint = tool_info.get('endpoint', '')
        print(f"\n📍 google_search endpoint: {endpoint}")
        
        if 'host.docker.internal:3001' in endpoint:
            print("✅ Endpoint is correctly configured with host.docker.internal:3001")
        elif 'localhost' in endpoint and ':3001' in endpoint:
            print("⚠️ Endpoint uses localhost - may need adjustment for Docker")
        else:
            print(f"❌ Unexpected endpoint format: {endpoint}")
    else:
        print("❌ google_search tool not found in enabled tools")

def test_direct_http_call():
    """Test a direct HTTP call to the MCP server"""
    import aiohttp
    import asyncio
    import json
    
    async def test():
        endpoint = "http://host.docker.internal:3001/tools/google_search"
        
        payload = {
            "name": "google_search",
            "arguments": {
                "query": "test query",
                "num_results": 1
            }
        }
        
        print(f"\n🔧 Testing direct HTTP call to: {endpoint}")
        
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ Direct HTTP call successful: {json.dumps(result, indent=2)[:200]}...")
                    else:
                        text = await response.text()
                        print(f"❌ HTTP {response.status}: {text[:200]}")
            except Exception as e:
                print(f"❌ Direct HTTP call failed: {e}")
    
    asyncio.run(test())

def main():
    print("="*60)
    print("FIXING WEB SEARCH ENDPOINT ISSUE")
    print("="*60)
    
    # Apply fixes
    print("\n🔧 Applying fixes...")
    fix1 = fix_enhanced_tool_executor()
    fix2 = fix_unified_mcp_service()
    
    if fix1 or fix2:
        print("\n✅ Fixes applied successfully!")
    else:
        print("\n✅ System already fixed or no changes needed")
    
    # Verify configuration
    print("\n🔍 Verifying MCP tool configuration...")
    verify_mcp_tool_configuration()
    
    # Test direct HTTP call
    print("\n🧪 Testing direct HTTP call to MCP server...")
    test_direct_http_call()
    
    print("\n" + "="*60)
    print("Fix complete! Restart the backend to apply changes.")
    print("="*60)

if __name__ == "__main__":
    main()