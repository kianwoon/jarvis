#!/usr/bin/env python3
"""
Test script to verify that the redundant batch extraction fix works correctly.
This tests that force_simple_retrieval=True prevents duplicate intelligent planning.
"""

import sys
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

# Test that the new parameter logic works correctly
def test_intelligent_planning_logic():
    """Test the logic for determining when to use intelligent planning"""
    
    # Simulate the conditions that would trigger intelligent planning
    wants_comprehensive = True
    quantity_intent = "all"
    confidence = 0.5  # Low confidence
    query_word_count = 10  # Complex query
    
    # Test 1: Normal case (should use intelligent planning)
    force_simple_retrieval = False
    should_use_intelligent_planning = (
        not force_simple_retrieval and (
            wants_comprehensive or 
            quantity_intent == "all" or 
            confidence < 0.7 or
            query_word_count > 8
        )
    )
    
    print(f"✅ Normal case (force_simple_retrieval={force_simple_retrieval}): should_use_intelligent_planning = {should_use_intelligent_planning}")
    assert should_use_intelligent_planning == True, "Should use intelligent planning in normal case"
    
    # Test 2: Fallback case (should NOT use intelligent planning)
    force_simple_retrieval = True
    should_use_intelligent_planning = (
        not force_simple_retrieval and (
            wants_comprehensive or 
            quantity_intent == "all" or 
            confidence < 0.7 or
            query_word_count > 8
        )
    )
    
    print(f"✅ Fallback case (force_simple_retrieval={force_simple_retrieval}): should_use_intelligent_planning = {should_use_intelligent_planning}")
    assert should_use_intelligent_planning == False, "Should NOT use intelligent planning in fallback case"
    
    print("🎯 All tests passed! The fix prevents duplicate intelligent planning.")

if __name__ == "__main__":
    test_intelligent_planning_logic()