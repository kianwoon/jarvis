#!/usr/bin/env python3
"""Fix MCP tool endpoints to use the correct /tools/ prefix"""

import logging
from app.core.db import SessionLocal, MCPTool
from app.core.mcp_tools_cache import reload_enabled_mcp_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_mcp_endpoints():
    """Update MCP tool endpoints to use /tools/ prefix"""
    print("\n" + "="*60)
    print("Fixing MCP Tool Endpoints")
    print("="*60)
    
    db = SessionLocal()
    try:
        # Find tools that need endpoint fixing
        tools_to_fix = db.query(MCPTool).filter(
            MCPTool.endpoint.like('http://%'),
            ~MCPTool.endpoint.contains('/tools/')
        ).all()
        
        if tools_to_fix:
            print(f"\n⚠️  Found {len(tools_to_fix)} tools with incorrect endpoints")
            
            for tool in tools_to_fix:
                old_endpoint = tool.endpoint
                # Extract the base URL and tool name
                if '/' in old_endpoint.replace('http://', '').replace('https://', ''):
                    # Has a path component
                    parts = old_endpoint.rsplit('/', 1)
                    if len(parts) == 2:
                        base_url = parts[0]
                        tool_path = parts[1]
                        # Check if it already has /tools/ in the path
                        if '/tools/' not in base_url:
                            new_endpoint = f"{base_url}/tools/{tool_path}"
                        else:
                            new_endpoint = old_endpoint
                    else:
                        new_endpoint = old_endpoint
                else:
                    # No path component, shouldn't happen for HTTP tools
                    new_endpoint = old_endpoint
                
                if new_endpoint != old_endpoint:
                    tool.endpoint = new_endpoint
                    print(f"  📝 Updated {tool.name}:")
                    print(f"     FROM: {old_endpoint}")
                    print(f"     TO:   {new_endpoint}")
        
        # Commit changes
        db.commit()
        
        if tools_to_fix:
            print("\n✅ Endpoints updated successfully!")
            
            # Reload the cache
            print("🔄 Reloading MCP tools cache...")
            reload_enabled_mcp_tools()
            print("✅ Cache reloaded!")
        else:
            print("\n✅ All endpoints already correct!")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    fix_mcp_endpoints()