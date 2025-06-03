#!/usr/bin/env python3
"""
Debug script to check FastAPI routes
"""

import sys
import os
sys.path.insert(0, '.')

try:
    from app.api.v1.endpoints.mcp_servers import router
    print("✅ Successfully imported mcp_servers router")
    print(f"Router has {len(router.routes)} routes:")
    
    for route in router.routes:
        print(f"  {route.methods} {route.path} -> {route.name}")
        
except Exception as e:
    print(f"❌ Failed to import router: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)

try:
    from app.api.v1 import api_router
    print("✅ Successfully imported main api_router")
    print(f"Main router has {len(api_router.routes)} routes")
    
    # Look for MCP routes specifically
    mcp_routes = [route for route in api_router.routes if 'mcp' in str(route.path)]
    print(f"Found {len(mcp_routes)} MCP routes:")
    
    for route in mcp_routes:
        print(f"  {route.methods} {route.path}")
        
except Exception as e:
    print(f"❌ Failed to import main api_router: {e}")
    import traceback
    traceback.print_exc()