#!/usr/bin/env python3
"""
Test script for cache bypass detection functionality.

This script tests the new programmatic cache bypass detection system to ensure
it correctly identifies when user queries require fresh data retrieval.
"""

import asyncio
import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.cache_bypass_detector import cache_bypass_detector

def test_bypass_detection():
    """Test various message patterns to validate cache bypass detection."""
    
    # Test cases that SHOULD trigger cache bypass
    bypass_cases = [
        "query all the projects from all sources again",
        "find all projects from all sources again", 
        "get all data from all sources again",
        "search all information from all sources again",
        "retrieve all documents from all sources again",
        "query all data again",
        "find all projects again",
        "get all information again",
        "search everything again", 
        "retrieve all content again",
        "from all sources again",
        "all projects again",
        "all data again",
        "fresh search for all data",
        "new query for all information",
        "refresh all results",
        "start over with all sources",
        "ignore previous and get all data again",
        "pull fresh data from all sources",
        "fetch all content again"
    ]
    
    # Test cases that should NOT trigger cache bypass
    no_bypass_cases = [
        "show me that list again",
        "display the results again", 
        "can you repeat that",
        "show me the same data",
        "order by end year, give me list again",
        "filter by 2023 projects",
        "sort those results differently",
        "show me only the first 5 items",
        "what was the last project you mentioned",
        "tell me more about project ABC",
        "explain that result in detail",
        "how many projects were there"
    ]
    
    # Edge cases to test
    edge_cases = [
        "get all data from sources", # missing "again"
        "query again", # missing scope
        "find all projects", # missing "again"
        "sources again", # missing scope/action
        "all again", # missing specificity
        "",
        "   ", 
        "a", # single char
        "show all projects from all sources again please" # polite version
    ]
    
    print("🧪 Testing Cache Bypass Detection System")
    print("=" * 60)
    
    # Test bypass cases
    print("\n1️⃣ Testing BYPASS cases (should return True):")
    print("-" * 50)
    bypass_correct = 0
    for msg in bypass_cases:
        result = cache_bypass_detector.should_bypass_cache(msg, "test-conv")
        status = "✅ PASS" if result['should_bypass'] else "❌ FAIL"
        confidence = result['confidence']
        reason = result['reason'][:50] + "..." if len(result['reason']) > 50 else result['reason']
        
        print(f"{status} | {confidence:.2f} | {msg}")
        print(f"        Reason: {reason}")
        
        if result['should_bypass']:
            bypass_correct += 1
    
    bypass_accuracy = bypass_correct / len(bypass_cases)
    print(f"\nBypass Detection Accuracy: {bypass_correct}/{len(bypass_cases)} = {bypass_accuracy:.1%}")
    
    # Test no-bypass cases  
    print(f"\n2️⃣ Testing NO-BYPASS cases (should return False):")
    print("-" * 50)
    no_bypass_correct = 0
    for msg in no_bypass_cases:
        result = cache_bypass_detector.should_bypass_cache(msg, "test-conv")
        status = "✅ PASS" if not result['should_bypass'] else "❌ FAIL"
        confidence = result['confidence']
        reason = result['reason'][:50] + "..." if len(result['reason']) > 50 else result['reason']
        
        print(f"{status} | {confidence:.2f} | {msg}")
        print(f"        Reason: {reason}")
        
        if not result['should_bypass']:
            no_bypass_correct += 1
    
    no_bypass_accuracy = no_bypass_correct / len(no_bypass_cases)
    print(f"\nNo-Bypass Detection Accuracy: {no_bypass_correct}/{len(no_bypass_cases)} = {no_bypass_accuracy:.1%}")
    
    # Test edge cases
    print(f"\n3️⃣ Testing EDGE cases:")
    print("-" * 50)
    for msg in edge_cases:
        result = cache_bypass_detector.should_bypass_cache(msg, "test-conv")
        bypass_status = "BYPASS" if result['should_bypass'] else "NO-BYPASS"
        confidence = result['confidence']
        reason = result['reason'][:50] + "..." if len(result['reason']) > 50 else result['reason']
        display_msg = repr(msg) if len(msg) <= 5 else msg
        
        print(f"{bypass_status} | {confidence:.2f} | {display_msg}")
        print(f"        Reason: {reason}")
    
    # Overall accuracy
    total_correct = bypass_correct + no_bypass_correct
    total_cases = len(bypass_cases) + len(no_bypass_cases)
    overall_accuracy = total_correct / total_cases
    
    print(f"\n📊 OVERALL RESULTS:")
    print("=" * 30)
    print(f"Bypass Cases Accuracy: {bypass_accuracy:.1%}")
    print(f"No-Bypass Cases Accuracy: {no_bypass_accuracy:.1%}")
    print(f"Overall Accuracy: {total_correct}/{total_cases} = {overall_accuracy:.1%}")
    
    # Success criteria
    success_threshold = 0.85
    if overall_accuracy >= success_threshold:
        print(f"✅ SUCCESS: Detection accuracy ({overall_accuracy:.1%}) meets threshold ({success_threshold:.1%})")
        return True
    else:
        print(f"❌ FAILURE: Detection accuracy ({overall_accuracy:.1%}) below threshold ({success_threshold:.1%})")
        return False

def test_pattern_analysis():
    """Test the pattern analysis feature."""
    print(f"\n🔍 Testing Pattern Analysis:")
    print("-" * 40)
    
    test_messages = [
        "query all projects from all sources again",
        "show me that list again",
        "get fresh data from all sources", 
        "find all information again",
        "display the same results"
    ]
    
    analysis = cache_bypass_detector.analyze_bypass_patterns(test_messages)
    
    print(f"Total messages analyzed: {analysis['total_messages']}")
    print(f"Bypass recommendations: {analysis['bypass_recommended']}")
    print(f"Bypass rate: {analysis['bypass_rate']:.1%}")
    print(f"Pattern matches: {analysis['pattern_matches']}")
    
    print(f"\nDetailed results:")
    for result in analysis['results']:
        status = "BYPASS" if result['should_bypass'] else "NO-BYPASS"
        print(f"  {status} | {result['confidence']:.2f} | {result['message']}")

async def test_integration():
    """Test integration with notebook endpoint logic (simulation)."""
    print(f"\n🔗 Testing Integration Simulation:")
    print("-" * 40)
    
    test_scenarios = [
        {
            'message': 'query all projects from all sources again',
            'conversation_id': 'test-conv-1',
            'expected_bypass': True
        },
        {
            'message': 'show me that list again',
            'conversation_id': 'test-conv-2', 
            'expected_bypass': False
        },
        {
            'message': 'get all data from all sources again',
            'conversation_id': 'test-conv-3',
            'expected_bypass': True
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing scenario: {scenario['message'][:50]}...")
        
        # Simulate the notebook endpoint logic
        bypass_decision = cache_bypass_detector.should_bypass_cache(
            scenario['message'], 
            scenario['conversation_id']
        )
        
        # Simulate cache decision
        would_use_cache = not bypass_decision['should_bypass']
        expected_cache_use = not scenario['expected_bypass']
        
        status = "✅ PASS" if would_use_cache == expected_cache_use else "❌ FAIL"
        
        print(f"   {status} Bypass: {bypass_decision['should_bypass']} (confidence: {bypass_decision['confidence']:.2f})")
        print(f"   Would use cache: {would_use_cache}")
        print(f"   Reason: {bypass_decision['reason']}")

def main():
    """Run all cache bypass detection tests."""
    print("🚀 Starting Cache Bypass Detection Tests")
    print("=" * 60)
    
    try:
        # Test basic detection
        detection_success = test_bypass_detection()
        
        # Test pattern analysis
        test_pattern_analysis()
        
        # Test integration simulation
        asyncio.run(test_integration())
        
        print(f"\n🎯 FINAL RESULT:")
        if detection_success:
            print("✅ Cache bypass detection system is working correctly!")
            print("   Ready for production use.")
        else:
            print("❌ Cache bypass detection needs improvement.")
            print("   Review pattern matching logic.")
            
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return detection_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)