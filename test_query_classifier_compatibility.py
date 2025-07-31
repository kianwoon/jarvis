#!/usr/bin/env python3
"""
Test query classifier compatibility with both Qwen3 models.
This script validates that the enhanced query classifier works correctly
with both thinking and non-thinking model variants.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
from app.core.llm_settings_cache import get_llm_settings
from app.llm.response_analyzer import clear_behavior_cache

async def test_query_classifier_with_model(model_name: str, test_queries: list):
    """Test query classifier with a specific model"""
    
    print(f"\n{'='*80}")
    print(f"TESTING QUERY CLASSIFIER WITH: {model_name}")
    print("="*80)
    
    # Create classifier instance
    classifier = EnhancedQueryClassifier()
    
    # Temporarily update query classifier model in settings
    settings = get_llm_settings()
    original_model = settings.get('query_classifier', {}).get('model', '')
    
    # Update the query classifier model for testing
    if 'query_classifier' not in settings:
        settings['query_classifier'] = {}
    settings['query_classifier']['model'] = model_name
    
    test_results = []
    
    print(f"Running {len(test_queries)} test queries...")
    
    for i, (query, expected_type) in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/{len(test_queries)} ---")
        print(f"Query: {query}")
        print(f"Expected: {expected_type}")
        
        try:
            # Classify the query
            classification_results = await classifier.classify(query)
            
            # Extract the first (primary) classification result
            if classification_results:
                first_result = classification_results[0]
                query_type = first_result.query_type
                confidence = first_result.confidence  
                reason = first_result.metadata.get('reason', 'No reason provided')
            else:
                query_type = "NO_RESULT"
                confidence = 0.0
                reason = "No classification returned"
            
            # Store result
            result = {
                'query': query,
                'expected': expected_type,
                'actual': query_type,
                'confidence': confidence,
                'reason': reason,
                'success': query_type == expected_type or expected_type == 'ANY'
            }
            test_results.append(result)
            
            print(f"Result: {query_type} (confidence: {confidence:.2f})")
            print(f"Reason: {reason}")
            
            if result['success']:
                print("✅ PASS - Classification matches expectation")
            else:
                print("❌ FAIL - Classification mismatch")
                
        except Exception as e:
            print(f"❌ ERROR - Classification failed: {e}")
            test_results.append({
                'query': query,
                'expected': expected_type,
                'actual': 'ERROR',
                'confidence': 0.0,
                'reason': str(e),
                'success': False
            })
        
        # Brief pause between queries
        await asyncio.sleep(0.5)
    
    # Restore original model
    if original_model:
        settings['query_classifier']['model'] = original_model
    
    # Calculate success rate
    successful = sum(1 for r in test_results if r['success'])
    total = len(test_results)
    success_rate = (successful / total) * 100 if total > 0 else 0
    
    print(f"\n📊 RESULTS SUMMARY FOR {model_name}")
    print("-" * 60)
    print(f"Total queries: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Show failed cases
    failed_cases = [r for r in test_results if not r['success']]
    if failed_cases:
        print(f"\n❌ Failed classifications:")
        for case in failed_cases:
            print(f"  • Query: {case['query'][:50]}...")
            print(f"    Expected: {case['expected']}, Got: {case['actual']}")
    
    return test_results, success_rate

async def test_model_behavior_detection():
    """Test that the behavior detection is working in the classifier"""
    
    print(f"\n{'='*80}")
    print("TESTING MODEL BEHAVIOR DETECTION")
    print("="*80)
    
    from app.llm.response_analyzer import get_model_behavior_profile, response_analyzer
    
    # Check cached behavior profiles
    cache_stats = response_analyzer.get_cache_stats()
    
    print(f"Models in behavior cache: {cache_stats['total_models']}")
    print(f"Thinking models detected: {cache_stats['thinking_models']}")
    print(f"Non-thinking models detected: {cache_stats['non_thinking_models']}")
    
    if cache_stats['models']:
        print(f"\nBehavior profiles:")
        for model, profile in cache_stats['models'].items():
            behavior = "thinking" if profile['is_thinking'] else "non-thinking"
            print(f"  • {model}: {behavior} (confidence: {profile['confidence']:.2f}, samples: {profile['response_count']})")
    
    return cache_stats

async def main():
    """Run comprehensive query classifier compatibility tests"""
    
    print("Query Classifier Compatibility Test Suite")
    print("="*80)
    
    # Clear behavior cache for clean test
    clear_behavior_cache()
    
    # Test queries with expected classifications
    test_queries = [
        ("How do I search for files in a directory?", "TOOL"),
        ("What is the weather today?", "WEB_SEARCH"),
        ("Explain quantum computing", "LLM"),
        ("Find the latest news about AI", "WEB_SEARCH"),
        ("List all files in /home/user", "TOOL"),
        ("What is 2 + 2?", "LLM"),
        ("Search for Python tutorials", "WEB_SEARCH"),
        ("Create a new directory called test", "TOOL"),
        ("Who is the president of the United States?", "ANY"),  # Could be LLM or WEB_SEARCH
        ("Execute a complex data analysis workflow", "MULTI_AGENT")
    ]
    
    # Models to test
    models_to_test = [
        "qwen3:30b-a3b-q4_K_M",
        "qwen3:30b-a3b-instruct-2507-q4_K_M"
    ]
    
    all_results = {}
    
    # Test each model
    for model in models_to_test:
        try:
            results, success_rate = await test_query_classifier_with_model(model, test_queries)
            all_results[model] = {
                'results': results,
                'success_rate': success_rate
            }
        except Exception as e:
            print(f"❌ Failed to test model {model}: {e}")
            all_results[model] = {
                'results': [],
                'success_rate': 0.0,
                'error': str(e)
            }
    
    # Test behavior detection
    behavior_stats = await test_model_behavior_detection()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL COMPATIBILITY SUMMARY")
    print("="*80)
    
    for model, data in all_results.items():
        if 'error' in data:
            print(f"❌ {model}: ERROR - {data['error']}")
        else:
            success_rate = data['success_rate']
            if success_rate >= 90:
                status = "✅ EXCELLENT"
            elif success_rate >= 70:
                status = "🟡 GOOD"
            else:
                status = "❌ NEEDS IMPROVEMENT"
            
            print(f"{status} {model}: {success_rate:.1f}% success rate")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 60)
    
    if behavior_stats['total_models'] > 0:
        if behavior_stats['thinking_models'] > 0:
            print("✅ Thinking model detection working correctly")
        if behavior_stats['non_thinking_models'] > 0:
            print("✅ Non-thinking model detection working correctly")
        
        print("✅ Dynamic behavior detection is functional")
        print("✅ Query classifier should work with both model types")
    else:
        print("⚠️  No behavior profiles detected - ensure models are responding")
    
    # Check if both models work
    working_models = [model for model, data in all_results.items() 
                     if 'error' not in data and data['success_rate'] >= 70]
    
    if len(working_models) >= 2:
        print("🎉 SUCCESS: Both model variants are compatible with query classifier!")
    elif len(working_models) == 1:
        print(f"⚠️  PARTIAL: Only {working_models[0]} is working reliably")
    else:
        print("❌ FAILURE: Neither model is working reliably with query classifier")
    
    print(f"\n📋 NEXT STEPS")
    print("-" * 60)
    print("1. If tests pass: Query classifier is ready for production")
    print("2. If tests fail: Check model configuration and system prompts")
    print("3. Monitor classification accuracy in real usage")
    print("4. Adjust confidence thresholds if needed")

if __name__ == "__main__":
    asyncio.run(main())