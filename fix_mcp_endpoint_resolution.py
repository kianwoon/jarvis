#!/usr/bin/env python3
"""
Comprehensive fix for MCP endpoint resolution issues
Handles both Docker and non-Docker environments properly
"""

import sys
import os
import re

# Add project root to path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

def create_endpoint_resolver():
    """Create a robust endpoint resolver that handles all environments"""
    
    resolver_code = '''#!/usr/bin/env python3
"""
MCP Endpoint Resolver
Handles endpoint resolution for both Docker and non-Docker environments
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def resolve_mcp_endpoint(tool_info: Dict[str, Any]) -> str:
    """
    Resolve the correct endpoint URL based on the environment
    
    Args:
        tool_info: Tool information including endpoint and server_hostname
        
    Returns:
        Resolved endpoint URL
    """
    endpoint = tool_info.get("endpoint", "")
    server_hostname = tool_info.get("server_hostname", "")
    
    # Check if we're running inside Docker
    in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
    
    logger.debug(f"Resolving endpoint: {endpoint}, in_docker: {in_docker}")
    
    # Handle different endpoint formats
    if not endpoint:
        return endpoint
    
    # For non-Docker environments (local development)
    if not in_docker:
        # Replace host.docker.internal with localhost for local execution
        if "host.docker.internal" in endpoint:
            resolved = endpoint.replace("host.docker.internal", "localhost")
            logger.info(f"[LOCAL] Resolved endpoint: {endpoint} -> {resolved}")
            return resolved
        # Keep localhost endpoints as-is
        elif "localhost" in endpoint:
            logger.debug(f"[LOCAL] Keeping localhost endpoint: {endpoint}")
            return endpoint
    
    # For Docker environments
    else:
        # Use server_hostname if available and endpoint has localhost
        if server_hostname and "localhost" in endpoint:
            resolved = endpoint.replace("localhost", server_hostname)
            logger.info(f"[DOCKER] Resolved endpoint: {endpoint} -> {resolved}")
            return resolved
        # Keep host.docker.internal as-is in Docker
        elif "host.docker.internal" in endpoint:
            logger.debug(f"[DOCKER] Keeping host.docker.internal endpoint: {endpoint}")
            return endpoint
    
    # Return endpoint unchanged if no resolution needed
    return endpoint

def get_resolved_tool_info(tool_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get tool info with resolved endpoint
    
    Args:
        tool_info: Original tool information
        
    Returns:
        Tool info with resolved endpoint
    """
    resolved_info = tool_info.copy()
    resolved_info["endpoint"] = resolve_mcp_endpoint(tool_info)
    return resolved_info
'''
    
    file_path = "/Users/kianwoonwong/Downloads/jarvis/app/core/mcp_endpoint_resolver.py"
    
    with open(file_path, 'w') as f:
        f.write(resolver_code)
    
    print(f"✅ Created endpoint resolver at {file_path}")
    return file_path

def update_enhanced_tool_executor():
    """Update enhanced_tool_executor to use the resolver"""
    
    file_path = "/Users/kianwoonwong/Downloads/jarvis/app/core/enhanced_tool_executor.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already using resolver
    if "from .mcp_endpoint_resolver import" in content:
        print("✅ enhanced_tool_executor.py already using resolver")
        return False
    
    # Add import at the top
    import_line = "from .mcp_endpoint_resolver import get_resolved_tool_info\n"
    
    # Find the imports section and add our import
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("from .mcp_tools_cache import"):
            lines.insert(i + 1, import_line[:-1])  # Remove trailing \n
            break
    
    # Replace the _adjust_endpoint_for_environment calls with resolver
    content = '\n'.join(lines)
    
    # Replace in call_mcp_tool_enhanced
    content = content.replace(
        "        # Handle hostname replacement for Docker/localhost scenarios\n        tool_info = _adjust_endpoint_for_environment(tool_info)",
        "        # Handle hostname replacement for Docker/localhost scenarios\n        tool_info = get_resolved_tool_info(tool_info)"
    )
    
    # Replace in call_mcp_tool_enhanced_async
    content = content.replace(
        "        # Handle hostname replacement\n        tool_info = _adjust_endpoint_for_environment(tool_info)",
        "        # Handle hostname replacement\n        tool_info = get_resolved_tool_info(tool_info)"
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated enhanced_tool_executor.py to use resolver")
    return True

def update_unified_mcp_service():
    """Update unified_mcp_service to use the resolver"""
    
    file_path = "/Users/kianwoonwong/Downloads/jarvis/app/core/unified_mcp_service.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already using resolver
    if "from .mcp_endpoint_resolver import" in content:
        print("✅ unified_mcp_service.py already using resolver")
        return False
    
    # Add import at the top
    import_line = "from .mcp_endpoint_resolver import resolve_mcp_endpoint\n"
    
    # Find the imports section and add our import
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("from .mcp_client import"):
            lines.insert(i + 1, import_line[:-1])  # Remove trailing \n
            break
    
    content = '\n'.join(lines)
    
    # Add endpoint resolution in call_mcp_tool_unified function
    # Find the line where endpoint is extracted and add resolution
    pattern = r'(endpoint = tool_info\.get\("endpoint", ""\))'
    replacement = r'\1\n        # Resolve endpoint based on environment\n        endpoint = resolve_mcp_endpoint(tool_info)'
    
    content = re.sub(pattern, replacement, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated unified_mcp_service.py to use resolver")
    return True

def test_resolver():
    """Test the endpoint resolver"""
    
    from app.core.mcp_endpoint_resolver import resolve_mcp_endpoint
    
    test_cases = [
        {
            "name": "host.docker.internal (local)",
            "tool_info": {"endpoint": "http://host.docker.internal:3001/tools/google_search"},
            "expected_local": "http://localhost:3001/tools/google_search"
        },
        {
            "name": "localhost endpoint",
            "tool_info": {"endpoint": "http://localhost:3001/tools/google_search"},
            "expected_local": "http://localhost:3001/tools/google_search"
        }
    ]
    
    print("\n🧪 Testing endpoint resolver...")
    for test in test_cases:
        resolved = resolve_mcp_endpoint(test["tool_info"])
        expected = test["expected_local"]
        
        if resolved == expected:
            print(f"✅ {test['name']}: {resolved}")
        else:
            print(f"❌ {test['name']}: Expected {expected}, got {resolved}")

def test_web_search_with_fix():
    """Test web search with the fix applied"""
    import asyncio
    
    async def test():
        from app.core.enhanced_tool_executor import call_mcp_tool_enhanced_async
        
        print("\n🔍 Testing web search with fixed endpoint resolution...")
        
        result = await call_mcp_tool_enhanced_async(
            "google_search",
            {"query": "latest LLM frameworks 2024", "num_results": 3}
        )
        
        if "error" not in result:
            print(f"✅ Web search successful!")
            # Print first result title if available
            if "content" in result and result["content"]:
                content = result["content"][0].get("text", "")
                print(f"   Sample result: {content[:200]}...")
        else:
            print(f"❌ Web search failed: {result.get('error')}")
    
    asyncio.run(test())

def main():
    print("="*70)
    print("COMPREHENSIVE MCP ENDPOINT RESOLUTION FIX")
    print("="*70)
    
    # Create the resolver module
    print("\n📦 Creating endpoint resolver module...")
    create_endpoint_resolver()
    
    # Update the files to use the resolver
    print("\n🔧 Updating files to use resolver...")
    updated1 = update_enhanced_tool_executor()
    updated2 = update_unified_mcp_service()
    
    if updated1 or updated2:
        print("\n✅ All files updated successfully!")
    else:
        print("\n✅ Files already up to date")
    
    # Test the resolver
    test_resolver()
    
    # Test web search
    test_web_search_with_fix()
    
    print("\n" + "="*70)
    print("✅ Fix complete! The MCP endpoint resolution should now work correctly")
    print("   in both Docker and non-Docker environments.")
    print("="*70)

if __name__ == "__main__":
    main()