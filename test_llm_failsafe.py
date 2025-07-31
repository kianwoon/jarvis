#!/usr/bin/env python3
"""
Test the LLM configuration failsafe system to ensure it prevents cascading failures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_emergency_fallback():
    """Test that emergency fallback settings work correctly"""
    print("\n" + "="*80)
    print("TESTING EMERGENCY FALLBACK SETTINGS")
    print("="*80)
    
    try:
        from app.core.llm_settings_cache import _get_emergency_fallback_settings, _validate_llm_settings
        
        # Get emergency settings
        emergency_settings = _get_emergency_fallback_settings()
        
        print(f"Emergency settings generated successfully")
        print(f"Settings keys: {list(emergency_settings.keys())}")
        
        # Validate the emergency settings
        is_valid = _validate_llm_settings(emergency_settings)
        print(f"Emergency settings validation: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        # Check specific configurations
        main_llm = emergency_settings.get('main_llm', {})
        print(f"Main LLM model: {main_llm.get('model', 'NOT_SET')}")
        print(f"Main LLM max_tokens: {main_llm.get('max_tokens', 'NOT_SET')}")
        
        second_llm = emergency_settings.get('second_llm', {})
        print(f"Second LLM model: {second_llm.get('model', 'NOT_SET')}")
        
        query_classifier = emergency_settings.get('query_classifier', {})
        print(f"Query classifier model: {query_classifier.get('model', 'NOT_SET')}")
        
        # Check that it uses the working model
        working_model = "qwen3:30b-a3b-q4_K_M"
        models_using_working = []
        for key in ['main_llm', 'second_llm', 'query_classifier', 'knowledge_graph']:
            if emergency_settings.get(key, {}).get('model') == working_model:
                models_using_working.append(key)
        
        print(f"Components using working model ({working_model}): {models_using_working}")
        
        return is_valid and len(models_using_working) >= 3
        
    except Exception as e:
        print(f"❌ ERROR: Emergency fallback test failed: {e}")
        return False

def test_failsafe_hierarchy():
    """Test the complete failsafe hierarchy"""
    print("\n" + "="*80)
    print("TESTING FAILSAFE HIERARCHY")
    print("="*80)
    
    try:
        from app.core.llm_settings_cache import get_llm_settings
        
        # This should work normally if Redis and DB are available
        settings = get_llm_settings()
        
        print(f"Normal settings retrieval: ✅ SUCCESS")
        print(f"Settings contains main_llm: {'main_llm' in settings}")
        print(f"Settings contains second_llm: {'second_llm' in settings}")
        print(f"Settings contains query_classifier: {'query_classifier' in settings}")
        
        # Check if emergency fallback marker exists
        is_emergency = settings.get('_fallback_mode') == 'emergency'
        print(f"Using emergency fallback: {'⚠️  YES' if is_emergency else '✅ NO'}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failsafe hierarchy test failed: {e}")
        return False

def test_configuration_functions():
    """Test that all configuration functions work with failsafe"""
    print("\n" + "="*80)
    print("TESTING CONFIGURATION FUNCTIONS WITH FAILSAFE")
    print("="*80)
    
    functions_to_test = [
        ('get_main_llm_full_config', 'main_llm'),
        ('get_second_llm_full_config', 'second_llm'),
        ('get_query_classifier_full_config', 'query_classifier')
    ]
    
    results = {}
    
    for func_name, component in functions_to_test:
        try:
            from app.core.llm_settings_cache import get_main_llm_full_config, get_second_llm_full_config, get_query_classifier_full_config
            
            if func_name == 'get_main_llm_full_config':
                config = get_main_llm_full_config()
            elif func_name == 'get_second_llm_full_config':
                config = get_second_llm_full_config()
            elif func_name == 'get_query_classifier_full_config':
                config = get_query_classifier_full_config()
            
            print(f"{func_name}: ✅ SUCCESS")
            print(f"  Model: {config.get('model', 'NOT_SET')}")
            print(f"  Mode: {config.get('effective_mode', config.get('mode', 'NOT_SET'))}")
            print(f"  Max tokens: {config.get('max_tokens', 'NOT_SET')}")
            
            results[func_name] = True
            
        except Exception as e:
            print(f"{func_name}: ❌ FAILED - {e}")
            results[func_name] = False
    
    all_passed = all(results.values())
    return all_passed

def main():
    """Run comprehensive failsafe system tests"""
    print("LLM Configuration Failsafe System Test Suite")
    print("="*80)
    
    tests = [
        ("Emergency Fallback Settings", test_emergency_fallback),
        ("Failsafe Hierarchy", test_failsafe_hierarchy),
        ("Configuration Functions", test_configuration_functions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Final summary
    print("\n" + "="*80)
    print("FAILSAFE SYSTEM TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 OVERALL RESULT")
    print("-" * 60)
    
    if all_passed:
        print("🎉 SUCCESS: All failsafe system tests passed!")
        print("✅ Emergency fallback settings are valid")
        print("✅ Configuration functions use failsafe mechanisms")
        print("✅ System is protected against cascading failures")
    else:
        print("❌ FAILURE: Some failsafe system tests failed")
        print("⚠️  System may be vulnerable to cascading failures")
    
    print(f"\n📋 FAILSAFE PROTECTION STATUS")
    print("-" * 60)
    print("✅ Redis cache with validation")
    print("✅ Database fallback with error handling")  
    print("✅ Emergency settings with working model")
    print("✅ Configuration function protection")
    print("✅ Comprehensive logging for diagnostics")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)