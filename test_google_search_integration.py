#!/usr/bin/env python3
"""Integration test for google_search with fixed temporal parameters"""

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


async def test_google_search_with_temporal_params():
    """Test google_search tool with temporal parameters through the full pipeline"""
    
    # Import here to avoid initialization issues
    from app.core.unified_mcp_service import call_mcp_tool_unified
    from app.core.mcp_tools_cache import get_enabled_mcp_tools
    from app.core.mcp_parameter_injector import inject_mcp_parameters
    
    # Get tool info for google_search
    tools = get_enabled_mcp_tools()
    google_search_tool = None
    
    if isinstance(tools, dict) and 'google_search' in tools:
        google_search_tool = tools['google_search']
    else:
        print(f"❌ google_search tool not found. Available tools: {list(tools.keys()) if isinstance(tools, dict) else 'Invalid format'}")
        return
    
    print("\n" + "="*60)
    print("Testing Google Search with Temporal Parameter Injection")
    print("="*60)
    
    # Test 1: Query with "latest" - should automatically add date_restrict and sort_by_date
    print("\n--- Test 1: 'latest' keyword (automatic injection) ---")
    print("Query: 'latest AI model releases'")
    
    # Start with just the query
    params1 = {'query': 'latest AI model releases', 'num_results': 3}
    
    # The parameter injector should add temporal params
    enhanced_params1 = inject_mcp_parameters('google_search', params1, google_search_tool)
    print(f"Original params: {params1}")
    print(f"Enhanced params: {enhanced_params1}")
    
    # Verify injection happened
    if 'date_restrict' in enhanced_params1:
        print(f"✅ date_restrict injected: {enhanced_params1['date_restrict']}")
    else:
        print("❌ date_restrict NOT injected")
    
    if 'sort_by_date' in enhanced_params1:
        print(f"✅ sort_by_date injected: {enhanced_params1['sort_by_date']}")
    else:
        print("❌ sort_by_date NOT injected")
    
    try:
        result = await call_mcp_tool_unified(google_search_tool, 'google_search', enhanced_params1)
        
        if 'error' in result:
            print(f"⚠️ Error: {result['error']}")
        elif 'content' in result and result['content']:
            content = result['content'][0].get('text', '')
            # Show first 500 chars for readability
            preview = content[:500] + '...' if len(content) > 500 else content
            print(f"✅ Success! Results preview:\n{preview}")
            
            # Check if results mention sorting or recency
            if 'week' in content.lower() or 'recent' in content.lower() or 'day' in content.lower():
                print("✅ Results appear to be temporally filtered")
        else:
            print("⚠️ Unexpected response format")
            
    except Exception as e:
        print(f"❌ Error calling tool: {e}")
    
    # Test 2: Query with "current" - should use more aggressive filtering (d3)
    print("\n--- Test 2: 'current' keyword (aggressive filtering) ---")
    print("Query: 'current weather San Francisco'")
    
    params2 = {'query': 'current weather San Francisco', 'num_results': 3}
    enhanced_params2 = inject_mcp_parameters('google_search', params2, google_search_tool)
    print(f"Original params: {params2}")
    print(f"Enhanced params: {enhanced_params2}")
    
    # Verify more aggressive filtering
    if enhanced_params2.get('date_restrict') == 'd3':
        print("✅ Aggressive filtering applied: d3 (last 3 days)")
    else:
        print(f"⚠️ Expected d3, got: {enhanced_params2.get('date_restrict')}")
    
    try:
        result = await call_mcp_tool_unified(google_search_tool, 'google_search', enhanced_params2)
        
        if 'error' in result:
            print(f"⚠️ Error: {result['error']}")
        elif 'content' in result and result['content']:
            content = result['content'][0].get('text', '')
            preview = content[:500] + '...' if len(content) > 500 else content
            print(f"✅ Success! Results preview:\n{preview}")
        else:
            print("⚠️ Unexpected response format")
            
    except Exception as e:
        print(f"❌ Error calling tool: {e}")
    
    # Test 3: Explicit parameters override injection
    print("\n--- Test 3: Explicit parameters (override injection) ---")
    print("Query: 'AI news' with explicit date_restrict='m1' and sort_by_date=True")
    
    params3 = {
        'query': 'AI news',
        'date_restrict': 'm1',  # Explicitly set to last month
        'sort_by_date': True,    # Explicitly enable date sorting
        'num_results': 3
    }
    
    # Even though "news" is temporal, explicit params should be preserved
    enhanced_params3 = inject_mcp_parameters('google_search', params3, google_search_tool)
    print(f"Original params: {params3}")
    print(f"Enhanced params: {enhanced_params3}")
    
    # Verify explicit params were preserved
    if enhanced_params3.get('date_restrict') == 'm1':
        print("✅ Explicit date_restrict preserved: m1")
    else:
        print(f"❌ Explicit date_restrict overridden: {enhanced_params3.get('date_restrict')}")
    
    if enhanced_params3.get('sort_by_date') == True:
        print("✅ Explicit sort_by_date preserved: True")
    
    try:
        result = await call_mcp_tool_unified(google_search_tool, 'google_search', params3)
        
        if 'error' in result:
            print(f"⚠️ Error: {result['error']}")
        elif 'content' in result and result['content']:
            content = result['content'][0].get('text', '')
            preview = content[:500] + '...' if len(content) > 500 else content
            print(f"✅ Success! Results preview:\n{preview}")
        else:
            print("⚠️ Unexpected response format")
            
    except Exception as e:
        print(f"❌ Error calling tool: {e}")
    
    print("\n" + "="*60)
    print("Integration test completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_google_search_with_temporal_params())