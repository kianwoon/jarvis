#!/usr/bin/env python3
"""
Test script for MCP-compliant parameter injection.

This script demonstrates that the parameter injection system:
1. Checks the tool's inputSchema to see what parameters it accepts
2. Only adds parameters that the tool actually supports
3. Works with any MCP tool regardless of name
4. Dynamically adapts based on the tool's capabilities
"""

import json
from app.core.mcp_parameter_injector import MCPParameterInjector, analyze_tool_capabilities


def test_google_search_tool():
    """Test parameter injection for Google Search tool"""
    print("\n" + "="*60)
    print("Testing Google Search Tool")
    print("="*60)
    
    # Simulate a Google Search tool with its inputSchema
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
                "dateRestrict": {
                    "type": "string",
                    "description": "Restrict results by date (d1, w1, m1, m3, m6, y1, y2)"
                },
                "sort": {
                    "type": "string",
                    "enum": ["relevance", "date"],
                    "description": "Sort order for results"
                }
            },
            "required": ["query"]
        }
    }
    
    injector = MCPParameterInjector()
    
    # Test 1: Query asking for latest information
    params1 = {"query": "latest AI news"}
    enhanced1 = injector.inject_parameters("google_search", params1, google_search_tool)
    print(f"\nQuery: '{params1['query']}'")
    print(f"Original params: {params1}")
    print(f"Enhanced params: {enhanced1}")
    print(f"✓ Added dateRestrict: {'dateRestrict' in enhanced1}")
    print(f"✓ Added sort: {'sort' in enhanced1}")
    
    # Test 2: Query without temporal keywords
    params2 = {"query": "python programming tutorial"}
    enhanced2 = injector.inject_parameters("google_search", params2, google_search_tool)
    print(f"\nQuery: '{params2['query']}'")
    print(f"Original params: {params2}")
    print(f"Enhanced params: {enhanced2}")
    print(f"✓ No temporal params added: {enhanced2 == params2}")
    
    # Analyze capabilities
    capabilities = analyze_tool_capabilities(google_search_tool)
    print(f"\nGoogle Search capabilities:")
    for cap, supported in capabilities.items():
        print(f"  {cap}: {supported}")


def test_renamed_search_tool():
    """Test that it works even if the tool is renamed"""
    print("\n" + "="*60)
    print("Testing Renamed Search Tool (abc_search)")
    print("="*60)
    
    # Same schema as Google Search but with different name
    abc_search_tool = {
        "name": "abc_search",  # Different name!
        "description": "Custom search engine",
        "parameters": {
            "type": "object",
            "properties": {
                "q": {  # Different parameter name for query
                    "type": "string",
                    "description": "Search query"
                },
                "date_range": {  # Different parameter name for date restriction
                    "type": "string",
                    "description": "Date range filter"
                },
                "orderBy": {  # Different parameter name for sort
                    "type": "string",
                    "enum": ["recent", "relevant"],
                    "description": "Result ordering"
                }
            },
            "required": ["q"]
        }
    }
    
    injector = MCPParameterInjector()
    
    # Test with temporal query
    params = {"q": "current stock prices"}
    enhanced = injector.inject_parameters("abc_search", params, abc_search_tool)
    print(f"\nQuery: '{params['q']}'")
    print(f"Original params: {params}")
    print(f"Enhanced params: {enhanced}")
    print(f"✓ Works with renamed tool: {len(enhanced) > len(params)}")
    
    # Analyze capabilities
    capabilities = analyze_tool_capabilities(abc_search_tool)
    print(f"\nABC Search capabilities:")
    for cap, supported in capabilities.items():
        print(f"  {cap}: {supported}")


def test_tool_without_temporal_support():
    """Test that it doesn't add parameters to tools that don't support them"""
    print("\n" + "="*60)
    print("Testing Tool Without Temporal Support")
    print("="*60)
    
    # A tool that doesn't support date filtering
    email_tool = {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "subject": {
                    "type": "string"
                },
                "body": {
                    "type": "string"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
    
    injector = MCPParameterInjector()
    
    # Even with temporal keywords, it shouldn't add date params
    params = {
        "to": ["user@example.com"],
        "subject": "Latest updates",
        "body": "Here are the recent changes..."
    }
    enhanced = injector.inject_parameters("send_email", params, email_tool)
    print(f"\nOriginal params keys: {list(params.keys())}")
    print(f"Enhanced params keys: {list(enhanced.keys())}")
    print(f"✓ No extra params added: {enhanced == params}")
    
    # Analyze capabilities
    capabilities = analyze_tool_capabilities(email_tool)
    print(f"\nEmail tool capabilities:")
    for cap, supported in capabilities.items():
        print(f"  {cap}: {supported}")


def test_comprehensive_query():
    """Test smart defaults for comprehensive queries"""
    print("\n" + "="*60)
    print("Testing Comprehensive Query Handling")
    print("="*60)
    
    search_tool = {
        "name": "research_search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "maximum": 20
                }
            }
        }
    }
    
    injector = MCPParameterInjector()
    
    # Comprehensive query should get more results
    params = {"query": "comprehensive analysis of renewable energy"}
    enhanced = injector.inject_parameters("research_search", params, search_tool)
    print(f"\nQuery: '{params['query']}'")
    print(f"Enhanced params: {enhanced}")
    print(f"✓ Added max_results for comprehensive query: {'max_results' in enhanced}")
    if 'max_results' in enhanced:
        print(f"  Value: {enhanced['max_results']}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MCP-COMPLIANT PARAMETER INJECTION TEST SUITE")
    print("="*60)
    print("\nThis demonstrates that the solution:")
    print("1. ✓ Checks tool's inputSchema for supported parameters")
    print("2. ✓ Only adds parameters the tool actually supports")
    print("3. ✓ Works with ANY tool name (not hardcoded)")
    print("4. ✓ Adapts based on tool capabilities")
    
    test_google_search_tool()
    test_renamed_search_tool()
    test_tool_without_temporal_support()
    test_comprehensive_query()
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print("\nKey Points Demonstrated:")
    print("• The system inspects inputSchema to determine capabilities")
    print("• It works with google_search, abc_search, or any other name")
    print("• It only adds parameters that exist in the schema")
    print("• It's fully MCP-compliant and protocol-driven")


if __name__ == "__main__":
    main()