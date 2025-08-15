#!/usr/bin/env python3
"""Test google_search MCP tool with new parameters"""

import asyncio
import json
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_google_search():
    """Test google_search tool with new date filtering and sorting parameters"""
    
    # Import here to avoid initialization issues
    from app.core.unified_mcp_service import call_mcp_tool_unified
    from app.core.mcp_tools_cache import get_enabled_mcp_tools
    
    # Get tool info for google_search
    tools = get_enabled_mcp_tools()
    google_search_tool = None
    
    # tools is a dict, with tool names as keys
    if isinstance(tools, dict) and 'google_search' in tools:
        google_search_tool = tools['google_search']
    else:
        print(f"❌ google_search tool not found. Available tools: {list(tools.keys()) if isinstance(tools, dict) else 'Invalid format'}")
        return
    
    # Helper function to call the tool
    async def call_google_search(params):
        return await call_mcp_tool_unified(google_search_tool, 'google_search', params)
    
    print("\n" + "="*60)
    print("Testing Google Search with New Parameters")
    print("="*60)
    
    # Test 1: Search with date restriction and sorting
    print("\n1. Testing with date_restrict='d1' and sort_by_date=True:")
    print("-" * 40)
    
    try:
        result = await call_google_search({
            'query': 'latest AI breakthroughs',
            'date_restrict': 'd1',  # Last day
            'sort_by_date': True,    # Sort by date
            'num_results': 3
        })
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        elif 'content' in result and result['content']:
            content = result['content'][0].get('text', '')
            # Show first 800 chars for readability
            preview = content[:800] + '...' if len(content) > 800 else content
            print(f"✅ Success! Results preview:\n{preview}")
        else:
            print(f"Result structure: {json.dumps(result, indent=2)[:500]}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Search with week restriction
    print("\n2. Testing with date_restrict='w1' (last week):")
    print("-" * 40)
    
    try:
        result = await call_google_search({
            'query': 'technology news',
            'date_restrict': 'w1',  # Last week
            'num_results': 2
        })
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        elif 'content' in result and result['content']:
            print("✅ Success! Tool responded with results")
        else:
            print(f"Unexpected result: {result}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 3: Regular search without new parameters (backward compatibility)
    print("\n3. Testing backward compatibility (no new parameters):")
    print("-" * 40)
    
    try:
        result = await call_google_search({
            'query': 'machine learning',
            'num_results': 2
        })
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        elif 'content' in result and result['content']:
            print("✅ Success! Backward compatibility maintained")
        else:
            print(f"Unexpected result: {result}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_google_search())