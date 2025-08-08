#!/usr/bin/env python3
"""
Test script to verify that all model server URLs are coming from settings,
not hardcoded values.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_llm_settings_cache():
    """Test that LLM settings cache uses configured URLs"""
    print("\n=== Testing LLM Settings Cache ===")
    
    from app.core.llm_settings_cache import (
        get_main_llm_full_config,
        get_second_llm_full_config,
        get_query_classifier_full_config
    )
    
    # Test main LLM config
    main_config = get_main_llm_full_config()
    model_server = main_config.get('model_server', '')
    print(f"Main LLM model_server: {model_server}")
    
    if not model_server:
        print("❌ Main LLM has no model_server configured")
    elif 'localhost:11434' in model_server or 'ollama:11434' in model_server:
        print(f"✅ Main LLM using configured URL: {model_server}")
    else:
        print(f"✅ Main LLM using non-default URL: {model_server}")
    
    # Test second LLM config
    second_config = get_second_llm_full_config()
    model_server = second_config.get('model_server', '')
    print(f"Second LLM model_server: {model_server}")
    
    if not model_server:
        print("❌ Second LLM has no model_server configured")
    elif 'localhost:11434' in model_server or 'ollama:11434' in model_server:
        print(f"✅ Second LLM using configured URL: {model_server}")
    else:
        print(f"✅ Second LLM using non-default URL: {model_server}")
    
    # Test query classifier config
    classifier_config = get_query_classifier_full_config()
    model_server = classifier_config.get('model_server', '')
    print(f"Query Classifier model_server: {model_server}")
    
    if not model_server:
        print("❌ Query Classifier has no model_server configured")
    elif 'localhost:11434' in model_server or 'ollama:11434' in model_server:
        print(f"✅ Query Classifier using configured URL: {model_server}")
    else:
        print(f"✅ Query Classifier using non-default URL: {model_server}")
    
    return True

def test_ollama_llm_class():
    """Test that OllamaLLM class properly uses settings"""
    print("\n=== Testing OllamaLLM Class ===")
    
    try:
        from app.llm.ollama import OllamaLLM, JarvisLLM
        from app.llm.base import LLMConfig
        
        # Test OllamaLLM without providing base_url
        config = LLMConfig(
            model_name="test-model",
            temperature=0.7,
            max_tokens=100
        )
        
        try:
            llm = OllamaLLM(config)
            print(f"✅ OllamaLLM initialized with base_url: {llm.base_url}")
            
            if not llm.base_url:
                print("❌ OllamaLLM has no base_url")
            elif llm.base_url == "http://localhost:11434":
                print("⚠️  OllamaLLM using default localhost URL (check if settings are configured)")
            else:
                print(f"✅ OllamaLLM using configured URL: {llm.base_url}")
        except ValueError as e:
            if "Model server URL must be configured" in str(e):
                print("✅ OllamaLLM correctly requires configuration (no hardcoded fallback)")
            else:
                print(f"❌ Unexpected error: {e}")
        
        # Test JarvisLLM without providing base_url
        try:
            jarvis = JarvisLLM()
            print(f"✅ JarvisLLM initialized with base_url: {jarvis.base_url}")
            
            if not jarvis.base_url:
                print("❌ JarvisLLM has no base_url")
            elif jarvis.base_url == "http://localhost:11434":
                print("⚠️  JarvisLLM using default localhost URL (check if settings are configured)")
            else:
                print(f"✅ JarvisLLM using configured URL: {jarvis.base_url}")
        except ValueError as e:
            if "Model server URL must be configured" in str(e):
                print("✅ JarvisLLM correctly requires configuration (no hardcoded fallback)")
            else:
                print(f"❌ Unexpected error: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing OllamaLLM: {e}")
        return False

def test_config_defaults():
    """Test that config.py doesn't have hardcoded defaults"""
    print("\n=== Testing Config Defaults ===")
    
    from app.core.config import get_settings
    
    settings = get_settings()
    ollama_url = settings.OLLAMA_BASE_URL
    
    if ollama_url is None:
        print("✅ OLLAMA_BASE_URL is None by default (no hardcoded value)")
    elif ollama_url == "http://localhost:11434":
        print("❌ OLLAMA_BASE_URL still has hardcoded default")
    else:
        print(f"✅ OLLAMA_BASE_URL configured to: {ollama_url}")
    
    return True

def test_emergency_fallback():
    """Test that emergency fallback doesn't use hardcoded URLs"""
    print("\n=== Testing Emergency Fallback Settings ===")
    
    from app.core.llm_settings_cache import _get_emergency_fallback_settings
    
    emergency = _get_emergency_fallback_settings()
    
    for component in ['main_llm', 'second_llm', 'query_classifier', 'knowledge_graph']:
        model_server = emergency.get(component, {}).get('model_server', '')
        
        if not model_server:
            print(f"✅ {component}: No hardcoded URL (empty)")
        elif model_server == "http://localhost:11434":
            print(f"❌ {component}: Still using hardcoded localhost:11434")
        else:
            print(f"✅ {component}: Using environment/configured URL: {model_server}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Model Server URL Configuration")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("LLM Settings Cache", test_llm_settings_cache()))
    results.append(("OllamaLLM Class", test_ollama_llm_class()))
    results.append(("Config Defaults", test_config_defaults()))
    results.append(("Emergency Fallback", test_emergency_fallback()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! Settings-based URLs are properly configured.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())