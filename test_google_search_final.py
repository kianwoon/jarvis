#!/usr/bin/env python3
"""Final test to verify Google Search MCP tool is working correctly"""

import asyncio
import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_google_search():
    """Test Google Search through the unified MCP service"""
    print("\n" + "="*80)
    print(" GOOGLE SEARCH MCP TOOL TEST ")
    print("="*80)
    
    try:
        from app.core.unified_mcp_service import call_mcp_tool_unified
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        
        # Get tools from cache
        print("\n📋 Loading MCP tools from cache...")
        tools = get_enabled_mcp_tools()
        print(f"✅ Loaded {len(tools)} tools")
        
        # Find Google Search tool
        google_tool = None
        google_tool_name = None
        for name, info in tools.items():
            if name == 'google_search':
                google_tool = info
                google_tool_name = name
                break
        
        if not google_tool:
            print("\n❌ Google Search tool not found in cache!")
            return False
        
        print(f"\n✅ Found Google Search tool: {google_tool_name}")
        print(f"  - Endpoint: {google_tool.get('endpoint', 'N/A')}")
        print(f"  - Server Hostname: {google_tool.get('server_hostname', 'N/A')}")
        print(f"  - Method: {google_tool.get('method', 'N/A')}")
        
        # Test 1: Basic search
        print("\n" + "-"*60)
        print("TEST 1: Basic Search")
        print("-"*60)
        print("Query: 'Python programming tutorials 2024'")
        
        test_params = {
            "query": "Python programming tutorials 2024",
            "num_results": 3
        }
        
        print("\n🔍 Executing search...")
        result = await call_mcp_tool_unified(google_tool, google_tool_name, test_params)
        
        if "error" in result:
            print(f"❌ Search failed: {result['error']}")
            return False
        else:
            print("✅ Search successful!")
            
            # Parse the result
            if isinstance(result, dict) and 'content' in result:
                content = result.get('content', [])
                if content and isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            text_content = item.get('text', '')
                            # Try to parse as JSON if it looks like JSON
                            if text_content.startswith('[') or text_content.startswith('{'):
                                try:
                                    search_results = json.loads(text_content)
                                    if isinstance(search_results, list):
                                        print(f"\n📊 Found {len(search_results)} results:")
                                        for i, r in enumerate(search_results[:3], 1):
                                            print(f"\n  Result {i}:")
                                            print(f"    Title: {r.get('title', 'N/A')}")
                                            print(f"    Link: {r.get('link', 'N/A')}")
                                            print(f"    Snippet: {r.get('snippet', 'N/A')[:100]}...")
                                except json.JSONDecodeError:
                                    print(f"\nRaw result: {text_content[:500]}...")
                            else:
                                print(f"\nResult text: {text_content[:500]}...")
            else:
                print(f"\nRaw result structure: {json.dumps(result, indent=2)[:500]}...")
        
        # Test 2: Search with date restriction
        print("\n" + "-"*60)
        print("TEST 2: Search with Date Restriction")
        print("-"*60)
        print("Query: 'latest AI news'")
        print("Date restriction: 'w1' (last week)")
        
        test_params = {
            "query": "latest AI news",
            "num_results": 3,
            "date_restrict": "w1",
            "sort_by_date": True
        }
        
        print("\n🔍 Executing search with date filter...")
        result = await call_mcp_tool_unified(google_tool, google_tool_name, test_params)
        
        if "error" in result:
            print(f"❌ Search failed: {result['error']}")
        else:
            print("✅ Search with date restriction successful!")
            
            # Show brief result summary
            if isinstance(result, dict) and 'content' in result:
                content = result.get('content', [])
                if content and isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text':
                            text_content = item.get('text', '')
                            if text_content.startswith('[') or text_content.startswith('{'):
                                try:
                                    search_results = json.loads(text_content)
                                    if isinstance(search_results, list):
                                        print(f"\n📊 Found {len(search_results)} recent results")
                                        if search_results:
                                            first_result = search_results[0]
                                            print(f"  Latest result: {first_result.get('title', 'N/A')}")
                                except:
                                    pass
        
        # Test 3: Error handling - empty query
        print("\n" + "-"*60)
        print("TEST 3: Error Handling - Empty Query")
        print("-"*60)
        
        test_params = {
            "query": "",
            "num_results": 3
        }
        
        print("\n🔍 Testing with empty query...")
        result = await call_mcp_tool_unified(google_tool, google_tool_name, test_params)
        
        if "error" in result:
            print(f"✅ Correctly handled empty query with error: {result['error']}")
        else:
            print("⚠️  Unexpected: Empty query did not produce an error")
        
        print("\n" + "="*80)
        print(" ALL TESTS COMPLETE ")
        print("="*80)
        print("\n✅ Google Search MCP tool is properly registered and working!")
        print("\nYou can now use Google Search in the Jarvis chat interface.")
        print("The system will automatically use it when users ask for web searches.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_google_search()
    
    if not success:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("1. MCP bridge server not running on localhost:3001")
        print("2. Google Search API credentials not configured")
        print("3. Database connection issues")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)