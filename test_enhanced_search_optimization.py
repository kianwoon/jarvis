#!/usr/bin/env python3
"""Test suite for enhanced search optimization with entity preservation"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

test_cases = [
    {
        "name": "ChatGPT PRO subscription",
        "input": "what are the usage limit of chatgpt PRO subscription?",
        "expected_preserved": ["chatgpt PRO", "subscription"],
        "should_not_contain": ["Plus", "Premium"],
        "description": "Should preserve PRO, not change to Plus"
    },
    {
        "name": "GPT-4 API limits",
        "input": "GPT-4 API rate limits",
        "expected_preserved": ["GPT-4", "API"],
        "should_not_contain": ["GPT-3", "ChatGPT"],
        "description": "Should preserve exact model name"
    },
    {
        "name": "Comparison query",
        "input": "difference between ChatGPT Plus and ChatGPT PRO",
        "expected_preserved": ["ChatGPT Plus", "ChatGPT PRO"],
        "should_not_contain": [],
        "description": "Should preserve both product names in comparison"
    },
    {
        "name": "Version specific",
        "input": "how to use GPT-3.5 turbo API",
        "expected_preserved": ["GPT-3.5", "API"],
        "should_not_contain": ["GPT-4", "GPT-3"],
        "description": "Should preserve version numbers"
    },
    {
        "name": "Enterprise tier",
        "input": "what is included in ChatGPT Enterprise subscription",
        "expected_preserved": ["ChatGPT Enterprise"],
        "should_not_contain": ["Plus", "PRO", "Premium"],
        "description": "Should preserve Enterprise tier name"
    },
    {
        "name": "Technical query",
        "input": "can you tell me about the OpenAI SDK configuration",
        "expected_preserved": ["OpenAI SDK"],
        "should_not_contain": [],
        "description": "Should preserve technical terms"
    }
]

async def test_optimization():
    """Test the enhanced search query optimizer"""
    print("=" * 80)
    print("Testing Enhanced Search Query Optimization with Entity Preservation")
    print("=" * 80)
    
    try:
        from app.langchain.search_query_optimizer import get_search_query_optimizer
        
        optimizer = get_search_query_optimizer()
        
        passed = 0
        failed = 0
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test['name']}")
            print(f"Description: {test['description']}")
            print(f"Input: {test['input']}")
            
            try:
                # Get optimization result
                result = await optimizer.optimize_query(test['input'])
                
                print(f"  Original: {result['original']}")
                print(f"  Optimized: {result['optimized']}")
                print(f"  Confidence: {result['confidence']:.2f}")
                print(f"  Method: {result['method']}")
                print(f"  Entities preserved: {result['entities_preserved']}")
                
                # Check if optimization was appropriate based on confidence
                if result['confidence'] < 0.3:
                    print(f"  ℹ️  Low confidence ({result['confidence']:.2f}), optimization skipped")
                
                # Run validation checks
                test_passed = True
                
                # Check preserved terms
                for term in test['expected_preserved']:
                    # Check if term is preserved (case may vary but the term should exist)
                    found = False
                    for preserved in result['entities_preserved']:
                        if term.lower() in preserved.lower() or preserved.lower() in term.lower():
                            found = True
                            break
                    
                    # Also check in the actual optimized query
                    if term.lower() in result['optimized'].lower():
                        found = True
                    
                    if not found and result['method'] != 'skipped_low_confidence':
                        print(f"  ❌ Missing expected term: '{term}'")
                        test_passed = False
                    else:
                        print(f"  ✅ Preserved: '{term}'")
                
                # Check unwanted substitutions
                for term in test['should_not_contain']:
                    # Only check if the term wasn't in the original
                    if term.lower() not in test['input'].lower():
                        if term.lower() in result['optimized'].lower():
                            print(f"  ❌ Contains unwanted substitution: '{term}'")
                            test_passed = False
                        else:
                            print(f"  ✅ Avoided substitution: '{term}'")
                
                # Validate that optimization preserved intent
                if result['method'] not in ['skipped_low_confidence', 'none']:
                    is_valid = optimizer.validate_optimization(
                        result['original'], 
                        result['optimized']
                    )
                    if not is_valid:
                        print(f"  ⚠️  Validation failed but optimization was applied")
                        test_passed = False
                
                if test_passed:
                    print(f"  ✅ Test PASSED")
                    passed += 1
                else:
                    print(f"  ❌ Test FAILED")
                    failed += 1
                    
            except Exception as e:
                print(f"  ❌ Test FAILED with error: {e}")
                failed += 1
        
        # Summary
        print("\n" + "=" * 80)
        print(f"Test Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
        
        if failed == 0:
            print("✅ All tests passed! The optimization preserves entities correctly.")
        else:
            print(f"⚠️  {failed} test(s) failed. Review the implementation.")
        
        print("=" * 80)
        
    except ImportError as e:
        print(f"Error importing optimizer: {e}")
        print("Make sure you're running from the project directory")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

async def test_specific_query():
    """Test the specific problematic query"""
    print("\n" + "=" * 80)
    print("Testing Specific Query: ChatGPT PRO subscription limits")
    print("=" * 80)
    
    try:
        from app.langchain.search_query_optimizer import get_search_query_optimizer
        
        optimizer = get_search_query_optimizer()
        
        query = "what are the usage limit of chatgpt PRO subscription?"
        
        print(f"Original query: {query}")
        
        # Clear cache to ensure fresh optimization
        optimizer.clear_cache()
        
        result = await optimizer.optimize_query(query)
        
        print(f"\nOptimization Result:")
        print(f"  Original: {result['original']}")
        print(f"  Optimized: {result['optimized']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Method: {result['method']}")
        print(f"  Entities preserved: {result['entities_preserved']}")
        
        # Check if PRO was preserved
        if 'PRO' in result['optimized']:
            print(f"\n✅ SUCCESS: 'PRO' was preserved in the optimized query")
        else:
            print(f"\n❌ FAILURE: 'PRO' was not preserved in the optimized query")
        
        # Check if Plus was incorrectly substituted
        if 'Plus' in result['optimized'] and 'Plus' not in query:
            print(f"❌ FAILURE: 'Plus' was incorrectly substituted")
        else:
            print(f"✅ SUCCESS: No incorrect substitution to 'Plus'")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting enhanced search optimization tests...\n")
    
    # Run both test suites
    asyncio.run(test_optimization())
    asyncio.run(test_specific_query())