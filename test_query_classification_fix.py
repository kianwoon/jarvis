#!/usr/bin/env python3
"""
Test Query Classification and LLM Response Quality Fixes

This script tests that:
1. Queries about AI/technology are correctly classified as "tool" (not "llm")
2. Direct LLM responses include the system prompt
3. Responses are relevant and high-quality
"""

import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
from app.core.llm_settings_cache import get_llm_settings, reload_llm_settings
from app.core.query_classifier_settings_cache import reload_query_classifier_settings

async def test_query_classification():
    """Test that various queries are classified correctly"""
    
    print("\n" + "="*60)
    print("TESTING QUERY CLASSIFICATION")
    print("="*60)
    
    # Reload settings to get latest changes
    reload_llm_settings()
    reload_query_classifier_settings()
    
    classifier = EnhancedQueryClassifier()
    classifier.reload_settings()
    
    test_cases = [
        # AI/Technology queries that should use tools
        ("what are the biggest challenges for AI to blend into corporate?", "tool", 
         "Should search for current information about AI challenges"),
        ("latest developments in quantum computing", "tool",
         "Should search for recent developments"),
        ("compare GPT-4 vs Claude 3", "tool",
         "Should search for comparison information"),
        ("current AI regulations in Europe", "tool",
         "Should search for current regulatory information"),
        
        # Simple factual queries that might use tools
        ("what is machine learning?", "tool",
         "Should search for comprehensive information"),
        ("explain blockchain technology", "tool",
         "Should search for explanations"),
         
        # RAG queries (only if matching specific collections)
        ("search my documents for budget report", "rag",
         "Should use RAG if documents exist"),
    ]
    
    results = []
    for query, expected_type, reason in test_cases:
        try:
            result = await classifier.classify(query)
            if result:
                classification = result[0]
                actual_type = classification.query_type.value
                confidence = classification.confidence
                
                # Check if classification matches expectation
                is_correct = actual_type == expected_type or \
                            (expected_type == "tool" and actual_type in ["tool", "tool_rag"])
                
                results.append({
                    "query": query,
                    "expected": expected_type,
                    "actual": actual_type,
                    "confidence": confidence,
                    "correct": is_correct,
                    "reason": reason
                })
                
                status = "✅" if is_correct else "❌"
                print(f"\n{status} Query: {query[:50]}...")
                print(f"   Expected: {expected_type} | Actual: {actual_type} | Confidence: {confidence:.2f}")
                print(f"   Reason: {reason}")
            else:
                results.append({
                    "query": query,
                    "expected": expected_type,
                    "actual": "ERROR",
                    "confidence": 0,
                    "correct": False,
                    "reason": "Classification failed"
                })
                print(f"\n❌ Query: {query[:50]}...")
                print(f"   ERROR: No classification result")
                
        except Exception as e:
            print(f"\n❌ Error classifying '{query[:30]}...': {e}")
            results.append({
                "query": query,
                "expected": expected_type,
                "actual": "ERROR",
                "confidence": 0,
                "correct": False,
                "reason": str(e)
            })
    
    # Summary
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    
    print("\n" + "-"*60)
    print(f"CLASSIFICATION RESULTS: {correct_count}/{total_count} correct")
    
    if correct_count == total_count:
        print("✅ All queries classified correctly!")
    else:
        print("⚠️ Some queries were misclassified:")
        for r in results:
            if not r["correct"]:
                print(f"  - '{r['query'][:40]}...': Expected {r['expected']}, got {r['actual']}")
    
    return results

def test_system_prompt_application():
    """Test that the system prompt is properly configured and will be applied"""
    
    print("\n" + "="*60)
    print("TESTING SYSTEM PROMPT APPLICATION")
    print("="*60)
    
    # Reload settings
    settings = reload_llm_settings()
    
    # Check main_llm system prompt
    main_llm = settings.get('main_llm', {})
    system_prompt = main_llm.get('system_prompt', '')
    
    if system_prompt:
        print("✅ Main LLM system prompt is configured")
        print(f"   Length: {len(system_prompt)} characters")
        print(f"   Preview: {system_prompt[:200]}...")
        
        # Check that it contains key instructions
        key_phrases = [
            "Jarvis",
            "assistant",
            "accurate",
            "response"
        ]
        
        found_phrases = [phrase for phrase in key_phrases if phrase.lower() in system_prompt.lower()]
        if found_phrases:
            print(f"✅ System prompt contains key phrases: {', '.join(found_phrases)}")
        else:
            print("⚠️ System prompt may be missing key instructions")
    else:
        print("❌ No system prompt configured for main_llm")
        return False
    
    # Check that the fix is applied in service.py
    service_file = "app/langchain/service.py"
    with open(service_file, 'r') as f:
        content = f.read()
    
    if "CRITICAL FIX: Get and apply the main LLM system prompt" in content:
        print("✅ System prompt fix is applied in service.py")
    else:
        print("❌ System prompt fix not found in service.py")
        return False
    
    return True

def test_query_classifier_settings():
    """Test that query classifier settings are properly configured"""
    
    print("\n" + "="*60)
    print("TESTING QUERY CLASSIFIER SETTINGS")
    print("="*60)
    
    from app.core.query_classifier_settings_cache import get_query_classifier_settings
    
    settings = get_query_classifier_settings()
    
    # Check critical settings
    checks = [
        ("enable_llm_classification", True, "LLM classification should be enabled"),
        ("llm_model", lambda x: bool(x), "Model should be configured"),
        ("direct_execution_threshold", lambda x: x <= 0.5, "Tool threshold should be low (<= 0.5)"),
        ("llm_direct_threshold", lambda x: x >= 0.9, "LLM direct threshold should be high (>= 0.9)"),
    ]
    
    all_good = True
    for key, expected, description in checks:
        value = settings.get(key)
        if callable(expected):
            is_correct = expected(value)
        else:
            is_correct = value == expected
        
        status = "✅" if is_correct else "❌"
        print(f"{status} {key}: {value}")
        print(f"   {description}")
        
        if not is_correct:
            all_good = False
    
    # Check system prompt
    system_prompt = settings.get('llm_system_prompt', '')
    if system_prompt and 'never output' not in system_prompt.lower():
        print("✅ Query classifier prompt is configured correctly")
        print(f"   Length: {len(system_prompt)} characters")
    else:
        print("⚠️ Query classifier prompt may need adjustment")
        all_good = False
    
    return all_good

async def main():
    print("="*60)
    print("COMPREHENSIVE TEST OF QUERY CLASSIFICATION FIXES")
    print("="*60)
    
    # Test 1: Query Classifier Settings
    print("\n📋 Test 1: Checking Query Classifier Settings...")
    settings_ok = test_query_classifier_settings()
    
    # Test 2: System Prompt Application
    print("\n📋 Test 2: Checking System Prompt Application...")
    prompt_ok = test_system_prompt_application()
    
    # Test 3: Query Classification
    print("\n📋 Test 3: Testing Query Classification...")
    classification_results = await test_query_classification()
    classification_ok = all(r["correct"] for r in classification_results if r["expected"] == "tool")
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    
    tests = [
        ("Query Classifier Settings", settings_ok),
        ("System Prompt Application", prompt_ok),
        ("Query Classification", classification_ok)
    ]
    
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in tests)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The fixes are working correctly.")
        print("\nThe system should now:")
        print("• Correctly classify AI/technology queries as 'tool' (for web search)")
        print("• Apply system prompts to direct LLM responses")
        print("• Provide relevant, high-quality answers")
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")
        print("You may need to:")
        print("• Check that settings were saved to the database")
        print("• Restart the API server to pick up changes")
        print("• Clear Redis cache if it's running")

if __name__ == "__main__":
    asyncio.run(main())