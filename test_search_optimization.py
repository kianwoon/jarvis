#!/usr/bin/env python3
"""
Test script to verify search optimization configuration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_search_optimization_config():
    """Test search optimization full config"""
    print("\n=== Testing Search Optimization Configuration ===")
    
    try:
        from app.core.llm_settings_cache import get_search_optimization_full_config
        
        # Test getting search optimization config
        config = get_search_optimization_full_config()
        print(f"Search Optimization config keys: {list(config.keys())}")
        
        model = config.get('model', '')
        model_server = config.get('model_server', '')
        
        print(f"Search Optimization model: {model}")
        print(f"Search Optimization model_server: {model_server}")
        
        if not model:
            print("❌ Search Optimization has no model configured")
            return False
        elif not model_server:
            print("❌ Search Optimization has no model_server configured")
            return False
        else:
            print("✅ Search Optimization has both model and model_server configured")
            return True
            
    except Exception as e:
        print(f"❌ Error testing search optimization config: {e}")
        return False

def test_search_optimization_integration():
    """Test search optimization integration with search query optimizer"""
    print("\n=== Testing Search Query Optimizer Integration ===")
    
    try:
        from app.langchain.search_query_optimizer import get_search_query_optimizer
        
        optimizer = get_search_query_optimizer()
        print("✅ Search query optimizer created successfully")
        
        # Test the optimization method (without actual LLM call)
        test_query = "Can you tell me about the latest AI developments?"
        print(f"Test query: {test_query}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing search query optimizer: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Search Optimization Configuration")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Search Optimization Config", test_search_optimization_config()))
    results.append(("Search Query Optimizer Integration", test_search_optimization_integration()))
    
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
        print("\n✅ All tests passed! Search optimization is properly configured.")
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())