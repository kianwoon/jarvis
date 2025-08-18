#!/usr/bin/env python3
"""Test the fixed MCP parameter injector for google_search tool"""

import json
from app.core.mcp_parameter_injector import MCPParameterInjector, analyze_tool_capabilities


def test_google_search_with_sort_by_date():
    """Test that the injector properly adds sort_by_date for temporal queries"""
    print("\n" + "="*60)
    print("Testing Google Search Tool with sort_by_date parameter")
    print("="*60)
    
    # Simulate a Google Search tool with sort_by_date boolean parameter
    google_search_tool = {
        "name": "google_search",
        "description": "Search Google for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "default": 5,
                    "maximum": 10
                },
                "date_restrict": {
                    "type": "string",
                    "description": "Restrict results by date (d1, d3, w1, m1, m3, m6, y1, y2)"
                },
                "sort_by_date": {
                    "type": "boolean",
                    "description": "Sort results by date with newest first"
                }
            },
            "required": ["query"]
        }
    }
    
    injector = MCPParameterInjector()
    
    # Test 1: Query with "latest" - should add both date_restrict and sort_by_date
    print("\n--- Test 1: 'latest' keyword ---")
    params1 = {"query": "latest AI breakthroughs"}
    enhanced1 = injector.inject_parameters("google_search", params1, google_search_tool)
    print(f"Query: '{params1['query']}'")
    print(f"Original params: {params1}")
    print(f"Enhanced params: {enhanced1}")
    print(f"✓ Added date_restrict: {'date_restrict' in enhanced1} (value: {enhanced1.get('date_restrict', 'N/A')})")
    print(f"✓ Added sort_by_date: {'sort_by_date' in enhanced1} (value: {enhanced1.get('sort_by_date', 'N/A')})")
    assert enhanced1.get('date_restrict') == 'w1', f"Expected 'w1' for 'latest', got {enhanced1.get('date_restrict')}"
    assert enhanced1.get('sort_by_date') == True, f"Expected sort_by_date=True, got {enhanced1.get('sort_by_date')}"
    
    # Test 2: Query with "recent" - should add both parameters
    print("\n--- Test 2: 'recent' keyword ---")
    params2 = {"query": "recent developments in quantum computing"}
    enhanced2 = injector.inject_parameters("google_search", params2, google_search_tool)
    print(f"Query: '{params2['query']}'")
    print(f"Original params: {params2}")
    print(f"Enhanced params: {enhanced2}")
    print(f"✓ Added date_restrict: {'date_restrict' in enhanced2} (value: {enhanced2.get('date_restrict', 'N/A')})")
    print(f"✓ Added sort_by_date: {'sort_by_date' in enhanced2} (value: {enhanced2.get('sort_by_date', 'N/A')})")
    assert enhanced2.get('date_restrict') == 'w1', f"Expected 'w1' for 'recent', got {enhanced2.get('date_restrict')}"
    assert enhanced2.get('sort_by_date') == True, f"Expected sort_by_date=True, got {enhanced2.get('sort_by_date')}"
    
    # Test 3: Query with "current" - should use d3 (3 days) for more aggressive filtering
    print("\n--- Test 3: 'current' keyword ---")
    params3 = {"query": "current status of the economy"}
    enhanced3 = injector.inject_parameters("google_search", params3, google_search_tool)
    print(f"Query: '{params3['query']}'")
    print(f"Original params: {params3}")
    print(f"Enhanced params: {enhanced3}")
    print(f"✓ Added date_restrict: {'date_restrict' in enhanced3} (value: {enhanced3.get('date_restrict', 'N/A')})")
    print(f"✓ Added sort_by_date: {'sort_by_date' in enhanced3} (value: {enhanced3.get('sort_by_date', 'N/A')})")
    assert enhanced3.get('date_restrict') == 'd3', f"Expected 'd3' for 'current', got {enhanced3.get('date_restrict')}"
    assert enhanced3.get('sort_by_date') == True, f"Expected sort_by_date=True, got {enhanced3.get('sort_by_date')}"
    
    # Test 4: Query without temporal keywords - should not add parameters
    print("\n--- Test 4: No temporal keywords ---")
    params4 = {"query": "python programming tutorial"}
    enhanced4 = injector.inject_parameters("google_search", params4, google_search_tool)
    print(f"Query: '{params4['query']}'")
    print(f"Original params: {params4}")
    print(f"Enhanced params: {enhanced4}")
    print(f"✓ No temporal params added: {enhanced4 == params4}")
    assert 'date_restrict' not in enhanced4, f"Should not add date_restrict for non-temporal query"
    assert 'sort_by_date' not in enhanced4, f"Should not add sort_by_date for non-temporal query"
    
    # Test 5: Query with "today" - should use d1 (1 day)
    print("\n--- Test 5: 'today' keyword ---")
    params5 = {"query": "today's breaking news"}
    enhanced5 = injector.inject_parameters("google_search", params5, google_search_tool)
    print(f"Query: '{params5['query']}'")
    print(f"Original params: {params5}")
    print(f"Enhanced params: {enhanced5}")
    print(f"✓ Added date_restrict: {'date_restrict' in enhanced5} (value: {enhanced5.get('date_restrict', 'N/A')})")
    print(f"✓ Added sort_by_date: {'sort_by_date' in enhanced5} (value: {enhanced5.get('sort_by_date', 'N/A')})")
    assert enhanced5.get('date_restrict') == 'd1', f"Expected 'd1' for 'today', got {enhanced5.get('date_restrict')}"
    assert enhanced5.get('sort_by_date') == True, f"Expected sort_by_date=True, got {enhanced5.get('sort_by_date')}"
    
    # Analyze capabilities
    print("\n--- Tool Capabilities ---")
    capabilities = analyze_tool_capabilities(google_search_tool)
    print("Google Search capabilities:")
    for cap, supported in capabilities.items():
        print(f"  {cap}: {supported}")
    
    assert capabilities['supports_sort_by_date'] == True, "Should detect sort_by_date support"
    assert capabilities['supports_date_filtering'] == True, "Should detect date filtering support"
    
    print("\n✅ All tests passed! The MCP parameter injector correctly handles:")
    print("   1. date_restrict with aggressive filtering (w1 for 'latest', d3 for 'current', d1 for 'today')")
    print("   2. sort_by_date=True for all temporal queries")
    print("   3. No parameters added for non-temporal queries")


def test_backward_compatibility():
    """Test that the injector still works with tools that only have sort (string) parameter"""
    print("\n" + "="*60)
    print("Testing Backward Compatibility with sort (string) parameter")
    print("="*60)
    
    # Tool with only sort parameter (no sort_by_date)
    old_style_tool = {
        "name": "old_search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "dateRestrict": {"type": "string"},
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "date"]
                }
            }
        }
    }
    
    injector = MCPParameterInjector()
    
    params = {"query": "latest news"}
    enhanced = injector.inject_parameters("old_search", params, old_style_tool)
    print(f"Query: '{params['query']}'")
    print(f"Enhanced params: {enhanced}")
    print(f"✓ Added dateRestrict: {'dateRestrict' in enhanced}")
    print(f"✓ Added sort: {'sort' in enhanced} (value: {enhanced.get('sort', 'N/A')})")
    print(f"✓ No sort_by_date added: {'sort_by_date' not in enhanced}")
    
    assert enhanced.get('sort') == 'date', f"Expected sort='date', got {enhanced.get('sort')}"
    assert 'sort_by_date' not in enhanced, "Should not add sort_by_date when not in schema"
    
    print("\n✅ Backward compatibility test passed!")


if __name__ == "__main__":
    test_google_search_with_sort_by_date()
    test_backward_compatibility()
    print("\n" + "="*60)
    print("🎉 All tests passed successfully!")
    print("="*60)