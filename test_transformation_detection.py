#!/usr/bin/env python3
"""
Test the transformation detection functionality for conversation state management.
Tests various user inputs to ensure proper detection of transformation requests.
"""

import sys
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.api.v1.endpoints.notebooks import _is_transformation_request


def test_transformation_detection():
    """Test various transformation request patterns."""
    print("🔍 Testing Transformation Request Detection\n")
    
    # Test cases organized by transformation type
    test_cases = {
        "Reordering/Sorting": [
            "order by date",
            "sort by name",
            "arrange by year",
            "organize by company",
            "rank by importance",
            "reorder the list",
            "rearrange by priority"
        ],
        "Filtering/Limiting": [
            "filter by 2023",
            "show only recent",
            "just show the first 5",
            "limit to 10 items",
            "only include projects from 2022",
            "exclude completed ones",
            "remove duplicates",
            "without the old data",
            "containing 'AI'",
            "from year 2023",
            "between 2020 and 2025",
            "after 2022",
            "before 2024",
            "since last year"
        ],
        "Formatting/Presentation": [
            "format as table",
            "make it a list",
            "convert to bullet points",
            "as a numbered list",
            "summarize this",
            "summary of the above",
            "brief version please",
            "short version"
        ],
        "Reference/Context": [
            "that list above",
            "those results",
            "the list from before",
            "previous list",
            "that data",
            "those items",
            "from above",
            "give me the list again",
            "show me the list",
            "show that list again",
            "list again",
            "show again",
            "display again"
        ],
        "Item-specific References": [
            "tell me more about item 3",
            "details about number 5",
            "explain item #2",
            "about the first one",
            "what is the second project",
            "who is the third person",
            "when was the fourth event",
            "where is item 1",
            "more about that company",
            "details on the startup"
        ]
    }
    
    # Non-transformation cases (should return False)
    non_transformation_cases = [
        "hello",
        "how are you",
        "what projects do you have",
        "find me some documents",
        "search for AI projects",
        "tell me about machine learning",
        "what is this document about",
        "new search query",
        "different topic entirely"
    ]
    
    # Test transformation cases
    total_tests = 0
    passed_tests = 0
    
    print("✅ TRANSFORMATION REQUEST TESTS:")
    for category, cases in test_cases.items():
        print(f"\n📂 {category}:")
        for case in cases:
            result = _is_transformation_request(case)
            total_tests += 1
            if result:
                print(f"  ✅ '{case}' → DETECTED")
                passed_tests += 1
            else:
                print(f"  ❌ '{case}' → NOT DETECTED (should be detected)")
    
    print(f"\n❌ NON-TRANSFORMATION REQUEST TESTS:")
    for case in non_transformation_cases:
        result = _is_transformation_request(case)
        total_tests += 1
        if not result:
            print(f"  ✅ '{case}' → CORRECTLY NOT DETECTED")
            passed_tests += 1
        else:
            print(f"  ❌ '{case}' → INCORRECTLY DETECTED (should not be detected)")
    
    print(f"\n📊 RESULTS:")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed - review detection patterns")
    
    return passed_tests == total_tests


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n🔧 Testing Edge Cases:")
    
    edge_cases = {
        "Empty/Short": ["", " ", "a", "ok"],
        "Mixed Case": ["ORDER BY date", "Sort By Name", "SHOW AGAIN"],
        "With Punctuation": ["order by date!", "sort by name?", "show, again"],
        "Multiple Keywords": ["sort and order by date", "filter and show only recent"],
        "Complex Sentences": [
            "can you please order the list by date descending",
            "I would like to see the results filtered by year 2023",
            "could you format this as bullet points for me"
        ]
    }
    
    for category, cases in edge_cases.items():
        print(f"\n📂 {category}:")
        for case in cases:
            result = _is_transformation_request(case)
            print(f"  {'✅' if result else '❌'} '{case}' → {'DETECTED' if result else 'NOT DETECTED'}")


if __name__ == "__main__":
    print("🚀 Starting Transformation Detection Tests\n")
    
    success = test_transformation_detection()
    test_edge_cases()
    
    print(f"\n{'🎉 All core tests passed!' if success else '⚠️ Some tests need attention'}")