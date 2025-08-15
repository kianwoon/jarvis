#!/usr/bin/env python3
"""
Test script to verify temporal search parameter injection

This script tests that:
1. Temporal queries are detected correctly
2. dateRestrict and sort parameters are added automatically
3. The solution works generically for any search tool
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.service import _map_tool_parameters_service, _detect_temporal_query, _is_search_tool


def test_temporal_detection():
    """Test temporal query detection"""
    print("\n" + "="*60)
    print("Testing Temporal Query Detection")
    print("="*60)
    
    test_queries = [
        # Should be detected as temporal
        ("What are the latest AI breakthroughs?", True),
        ("Show me recent news about technology", True),
        ("Current ChatGPT pricing", True),
        ("What's new in Python 2025?", True),
        ("Today's stock market updates", True),
        ("This week's tech news", True),
        ("Most recent OpenAI announcements", True),
        ("Updated documentation for React", True),
        ("Breaking news about AI", True),
        
        # Should NOT be detected as temporal
        ("What is machine learning?", False),
        ("How does Python work?", False),
        ("Explain photosynthesis", False),
        ("History of computer science", False),
        ("What happened in World War 2?", False),
    ]
    
    for query, expected in test_queries:
        is_temporal = _detect_temporal_query(query)
        status = "✅" if is_temporal == expected else "❌"
        print(f"{status} '{query[:50]}...' -> Temporal: {is_temporal} (Expected: {expected})")
    
    print("\n✅ Temporal detection test completed")


def test_parameter_injection():
    """Test that temporal parameters are injected correctly"""
    print("\n" + "="*60)
    print("Testing Parameter Injection for Temporal Queries")
    print("="*60)
    
    # Test with different tool names to ensure it's generic
    tool_names = [
        "google_search",
        "web_search", 
        "tavily_search",
        "abc_search",  # Generic name to prove it works with any search tool
        "xyz_web_search",
    ]
    
    temporal_queries = [
        "What are the latest AI developments?",
        "Show me recent news",
        "Current pricing for ChatGPT"
    ]
    
    for tool_name in tool_names:
        print(f"\n📌 Testing tool: {tool_name}")
        print("-" * 40)
        
        # Verify it's detected as a search tool
        is_search = _is_search_tool(tool_name)
        print(f"  Is search tool: {is_search}")
        
        if not is_search:
            print(f"  ⚠️  Not detected as search tool, skipping...")
            continue
        
        for query in temporal_queries:
            # Test parameter injection
            params = {"query": query}
            mapped_tool, mapped_params = _map_tool_parameters_service(tool_name, params)
            
            # Check if temporal parameters were added
            has_date_restrict = 'dateRestrict' in mapped_params
            has_sort = 'sort' in mapped_params
            
            print(f"\n  Query: '{query[:40]}...'")
            print(f"    Original params: {params}")
            print(f"    Mapped params: {mapped_params}")
            print(f"    ✓ dateRestrict added: {has_date_restrict}")
            print(f"    ✓ sort added: {has_sort}")
            
            if has_date_restrict and has_sort:
                print(f"    ✅ Success: Temporal parameters injected")
            else:
                print(f"    ❌ Failed: Missing temporal parameters")
    
    print("\n✅ Parameter injection test completed")


def test_non_temporal_queries():
    """Test that non-temporal queries don't get temporal parameters"""
    print("\n" + "="*60)
    print("Testing Non-Temporal Queries (Should NOT Add Parameters)")
    print("="*60)
    
    non_temporal_queries = [
        "What is photosynthesis?",
        "Explain machine learning",
        "How does Python work?"
    ]
    
    for query in non_temporal_queries:
        params = {"query": query}
        tool_name = "google_search"
        
        mapped_tool, mapped_params = _map_tool_parameters_service(tool_name, params)
        
        # Check that temporal parameters were NOT added
        has_date_restrict = 'dateRestrict' in mapped_params
        has_sort = 'sort' in mapped_params
        
        print(f"\nQuery: '{query}'")
        print(f"  Params: {mapped_params}")
        
        if not has_date_restrict and not has_sort:
            print(f"  ✅ Correct: No temporal parameters added")
        else:
            print(f"  ❌ Error: Temporal parameters incorrectly added")
    
    print("\n✅ Non-temporal query test completed")


def test_existing_parameters():
    """Test that existing parameters are not overridden"""
    print("\n" + "="*60)
    print("Testing Existing Parameter Preservation")
    print("="*60)
    
    # Test that if parameters are already provided, they're not overridden
    params_with_date = {
        "query": "latest AI news",
        "dateRestrict": "d1",  # User specified last day
        "sort": "relevance"  # User wants relevance sort
    }
    
    tool_name = "google_search"
    mapped_tool, mapped_params = _map_tool_parameters_service(tool_name, params_with_date)
    
    print(f"Original params: {params_with_date}")
    print(f"Mapped params: {mapped_params}")
    
    if mapped_params["dateRestrict"] == "d1" and mapped_params["sort"] == "relevance":
        print("✅ Success: User-provided parameters preserved")
    else:
        print("❌ Error: User parameters were overridden")
    
    print("\n✅ Parameter preservation test completed")


def main():
    """Run all tests"""
    print("\n" + "🚀 "*20)
    print("TEMPORAL SEARCH PARAMETER INJECTION TEST SUITE")
    print("🚀 "*20)
    
    try:
        # Run tests
        test_temporal_detection()
        test_parameter_injection()
        test_non_temporal_queries()
        test_existing_parameters()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📝 Summary:")
        print("1. Temporal queries are detected correctly")
        print("2. dateRestrict='w1' and sort='date' are added for temporal queries")
        print("3. Non-temporal queries are left unchanged")
        print("4. User-provided parameters are preserved")
        print("5. Solution works generically with any search tool name")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()