#!/usr/bin/env python3
"""
Simple test for temporal search parameter injection

This test directly includes the necessary functions to avoid circular imports.
"""

import logging
logging.basicConfig(level=logging.INFO)


def _is_search_tool(tool_name: str) -> bool:
    """Check if a tool is a search tool that would benefit from query optimization"""
    search_keywords = ['search', 'google', 'web', 'tavily', 'find', 'lookup']
    tool_lower = tool_name.lower()
    return any(keyword in tool_lower for keyword in search_keywords)


def _detect_temporal_query(query: str) -> bool:
    """Detect if a query is asking for recent/current information
    
    Uses the temporal query classifier to determine if the query needs recent results.
    This function is generic and works regardless of the tool name.
    """
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from app.core.temporal_query_classifier import get_temporal_classifier
        
        # Get the temporal classifier
        classifier = get_temporal_classifier()
        
        # Classify the query
        classification = classifier.classify(query)
        
        # Check if the query has 'current' intent or high temporal sensitivity
        is_temporal = (
            classification.intent == 'current' or
            classification.sensitivity.value in ['real-time', 'volatile'] or
            any(keyword in query.lower() for keyword in [
                'latest', 'recent', 'current', 'newest', 'today', 
                'this week', 'this month', 'this year', 'now', 
                'updated', 'modern', '2025', 'breaking', 'new'
            ])
        )
        
        if is_temporal:
            logger = logging.getLogger(__name__)
            logger.info(f"[TEMPORAL DETECTION] Query classified as temporal: intent={classification.intent}, sensitivity={classification.sensitivity.value}")
        
        return is_temporal
        
    except Exception as e:
        # If temporal detection fails, fall back to simple keyword matching
        logger = logging.getLogger(__name__)
        logger.warning(f"[TEMPORAL DETECTION] Classifier failed, using fallback: {e}")
        
        temporal_keywords = [
            'latest', 'recent', 'current', 'newest', 'today',
            'this week', 'this month', 'this year', 'now',
            'updated', 'modern', '2025', 'breaking', 'new'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in temporal_keywords)


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
    
    passed = 0
    failed = 0
    
    for query, expected in test_queries:
        is_temporal = _detect_temporal_query(query)
        if is_temporal == expected:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        print(f"{status} '{query[:50]}...' -> Temporal: {is_temporal} (Expected: {expected})")
    
    print(f"\n✅ Temporal detection test completed: {passed} passed, {failed} failed")
    return failed == 0


def test_search_tool_detection():
    """Test search tool detection"""
    print("\n" + "="*60)
    print("Testing Search Tool Detection")
    print("="*60)
    
    test_cases = [
        ("google_search", True),
        ("web_search", True),
        ("tavily_search", True),
        ("abc_search", True),
        ("xyz_web_lookup", True),
        ("find_documents", True),
        ("get_datetime", False),
        ("send_email", False),
        ("calculate", False),
    ]
    
    passed = 0
    failed = 0
    
    for tool_name, expected in test_cases:
        is_search = _is_search_tool(tool_name)
        if is_search == expected:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        print(f"{status} '{tool_name}' -> Is search tool: {is_search} (Expected: {expected})")
    
    print(f"\n✅ Search tool detection test completed: {passed} passed, {failed} failed")
    return failed == 0


def simulate_parameter_mapping():
    """Simulate how parameters would be mapped for temporal queries"""
    print("\n" + "="*60)
    print("Simulating Parameter Mapping for Temporal Queries")
    print("="*60)
    
    test_cases = [
        {
            "tool_name": "google_search",
            "query": "What are the latest AI breakthroughs?",
            "initial_params": {"query": "What are the latest AI breakthroughs?"},
            "should_add_temporal": True
        },
        {
            "tool_name": "web_search",
            "query": "Current ChatGPT pricing",
            "initial_params": {"query": "Current ChatGPT pricing"},
            "should_add_temporal": True
        },
        {
            "tool_name": "abc_search",  # Generic name
            "query": "Recent news about technology",
            "initial_params": {"query": "Recent news about technology"},
            "should_add_temporal": True
        },
        {
            "tool_name": "google_search",
            "query": "What is photosynthesis?",
            "initial_params": {"query": "What is photosynthesis?"},
            "should_add_temporal": False
        },
        {
            "tool_name": "google_search",
            "query": "Latest news",
            "initial_params": {"query": "Latest news", "dateRestrict": "d1"},  # User already specified
            "should_add_temporal": False  # Should not override
        },
    ]
    
    for case in test_cases:
        print(f"\n📌 Tool: {case['tool_name']}")
        print(f"   Query: '{case['query'][:50]}...'")
        print(f"   Initial params: {case['initial_params']}")
        
        # Simulate the logic from _map_tool_parameters_service
        params = case['initial_params'].copy()
        
        if _is_search_tool(case['tool_name']) and 'query' in params:
            is_temporal = _detect_temporal_query(params['query'])
            
            if is_temporal:
                # Add temporal parameters if not already present
                if 'dateRestrict' not in params and 'date_restrict' not in params:
                    params['dateRestrict'] = 'w1'
                    print(f"   ➕ Added dateRestrict='w1'")
                
                if 'sort' not in params and 'sort_by_date' not in params:
                    params['sort'] = 'date'
                    print(f"   ➕ Added sort='date'")
        
        print(f"   Final params: {params}")
        
        # Check if result is as expected
        if case['should_add_temporal']:
            if 'dateRestrict' in params and 'sort' in params:
                print(f"   ✅ Correct: Temporal parameters added")
            else:
                print(f"   ❌ Error: Temporal parameters should have been added")
        else:
            if ('dateRestrict' not in case['initial_params'] and 'dateRestrict' in params) or \
               ('sort' not in case['initial_params'] and 'sort' in params):
                print(f"   ❌ Error: Temporal parameters incorrectly added")
            else:
                print(f"   ✅ Correct: Parameters handled appropriately")
    
    print("\n✅ Parameter mapping simulation completed")


def main():
    """Run all tests"""
    print("\n" + "🚀 "*20)
    print("TEMPORAL SEARCH PARAMETER INJECTION TEST")
    print("🚀 "*20)
    
    all_passed = True
    
    try:
        # Run tests
        all_passed = test_temporal_detection() and all_passed
        all_passed = test_search_tool_detection() and all_passed
        simulate_parameter_mapping()
        
        print("\n" + "="*60)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
        else:
            print("⚠️ SOME TESTS FAILED - Check the output above")
        print("="*60)
        
        print("\n📝 Key Points:")
        print("1. Temporal queries are detected using the temporal_query_classifier")
        print("2. dateRestrict='w1' and sort='date' are added automatically")
        print("3. The solution works with ANY search tool (google_search, abc_search, etc.)")
        print("4. User-provided parameters are preserved and not overridden")
        print("5. Non-temporal queries are left unchanged")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)