#!/usr/bin/env python3
"""
Simple test for transformation detection function without importing the full app.
"""

def _is_transformation_request(message: str) -> bool:
    """
    Detect transformation requests that can be handled with existing conversation data.
    Covers reordering, filtering, formatting, and reference-based operations.
    
    Args:
        message: User's message
        
    Returns:
        bool: True if this is a transformation request
    """
    message_lower = message.lower().strip()
    
    # 1. Reordering/Sorting keywords
    reordering_keywords = [
        'order by', 'sort by', 'arrange by', 'organize by', 'rank by',
        'reorder', 'rearrange', 'sequence by'
    ]
    
    # 2. Filtering/Limiting keywords  
    filtering_keywords = [
        'filter', 'show only', 'just show', 'limit to', 'only include',
        'exclude', 'remove', 'without', 'containing', 'from year',
        'between', 'after', 'before', 'since'
    ]
    
    # 3. Formatting/Presentation keywords
    formatting_keywords = [
        'format as', 'make it a', 'convert to', 'as a table', 'as list',
        'bullet points', 'numbered list', 'summarize', 'summary of',
        'brief version', 'short version'
    ]
    
    # 4. Reference/Context keywords (referring to previous response)
    reference_keywords = [
        'that list', 'those results', 'the list above', 'previous list',
        'that data', 'those items', 'from above', 'the items',
        'give me the list again', 'show me the list', 'show that list',
        'list again', 'again', 'show again', 'display again'
    ]
    
    # 5. Item-specific references
    item_reference_patterns = [
        'item ', 'number ', '#', 'the first', 'the second', 'the third',
        'tell me more about', 'details about', 'explain the', 'about the'
    ]
    
    # Check all transformation categories
    all_keywords = (reordering_keywords + filtering_keywords + 
                   formatting_keywords + reference_keywords + 
                   item_reference_patterns)
    
    # Also check for specific patterns like "tell me more about item 3"
    item_reference_match = any(pattern in message_lower for pattern in [
        'about item', 'about number', 'about #', 'more about',
        'explain item', 'details on', 'tell me about the',
        'what is the first', 'what is the second', 'what is the third',
        'who is the first', 'who is the second', 'who is the third',
        'when was the first', 'when was the second', 'when was the third', 'when was the fourth',
        'where is the first', 'where is the second', 'where is the third', 'where is the fourth'
    ])
    
    return any(keyword in message_lower for keyword in all_keywords) or item_reference_match


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
            "containing AI",
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
        return True
    else:
        print("⚠️  Some tests failed - review detection patterns")
        return False


if __name__ == "__main__":
    print("🚀 Starting Transformation Detection Tests\n")
    success = test_transformation_detection()
    print(f"\n{'🎉 All tests passed!' if success else '⚠️ Some tests need attention'}")