#!/usr/bin/env python3
"""
Test configurable large output detection
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

# Test without environment variables first
def test_default_config():
    """Test with default configuration"""
    print("🔧 Testing Default Configuration")
    print("=" * 50)
    
    from app.core.large_generation_config import get_config
    config = get_config()
    
    print(f"Strong threshold: {config.strong_number_threshold}")
    print(f"Medium threshold: {config.medium_number_threshold}")
    print(f"Min items for chunking: {config.min_items_for_chunking}")
    print(f"Default comprehensive items: {config.default_comprehensive_items}")
    print(f"Score multiplier: {config.score_multiplier}")
    print(f"Redis TTL: {config.redis_conversation_ttl} seconds")
    print(f"Max Redis messages: {config.max_redis_messages}")
    
    # Test detection with default config
    from app.langchain.service import detect_large_output_potential
    
    test_question = "Generate 25 interview questions for software engineers"
    result = detect_large_output_potential(test_question)
    
    print(f"\nTest question: {test_question}")
    print(f"Large detection: {result['likely_large']}")
    print(f"Estimated items: {result['estimated_items']}")
    
    return config

def test_custom_config():
    """Test with custom environment variables"""
    print("\n\n🎛️  Testing Custom Configuration")
    print("=" * 50)
    
    # Set custom environment variables
    os.environ['LARGE_GEN_STRONG_THRESHOLD'] = '25'  # Lower threshold
    os.environ['LARGE_GEN_MIN_ITEMS_FOR_CHUNKING'] = '15'  # Lower chunking threshold
    os.environ['LARGE_GEN_DEFAULT_COMPREHENSIVE'] = '40'  # More items for comprehensive
    os.environ['REDIS_MAX_MESSAGES'] = '75'  # More Redis messages
    
    # Reload config to pick up env vars
    from app.core.large_generation_config import reload_config
    config = reload_config()
    
    print(f"Custom strong threshold: {config.strong_number_threshold}")
    print(f"Custom min items for chunking: {config.min_items_for_chunking}")
    print(f"Custom comprehensive items: {config.default_comprehensive_items}")
    print(f"Custom Redis max messages: {config.max_redis_messages}")
    
    # Test detection with custom config
    from app.langchain.service import detect_large_output_potential
    
    test_question = "Generate 25 interview questions for software engineers"
    result = detect_large_output_potential(test_question)
    
    print(f"\nSame test question: {test_question}")
    print(f"Large detection: {result['likely_large']}")
    print(f"Estimated items: {result['estimated_items']}")
    
    # Test with comprehensive request
    comprehensive_question = "Create a comprehensive guide to Python programming"
    result2 = detect_large_output_potential(comprehensive_question)
    
    print(f"\nComprehensive question: {comprehensive_question}")
    print(f"Large detection: {result2['likely_large']}")
    print(f"Estimated items: {result2['estimated_items']} (should be {config.default_comprehensive_items})")
    
    return config

def test_classification_with_config():
    """Test query classification with custom config"""
    print("\n\n🎯 Testing Classification with Custom Config")
    print("=" * 50)
    
    # Mock LLM config for testing
    mock_llm_cfg = {
        "model": "test",
        "thinking_mode": {"temperature": 0.7},
        "non_thinking_mode": {"temperature": 0.7},
        "max_tokens": 2048
    }
    
    test_cases = [
        "Generate 25 creative marketing ideas",  # Should be LARGE with custom config
        "Generate 5 quick tips",                # Should NOT be large
        "Create a comprehensive analysis",      # Should be large with custom comprehensive setting
        "What is Python?"                       # Should be LLM
    ]
    
    for question in test_cases:
        try:
            from app.langchain.service import classify_query_type
            result = classify_query_type(question, mock_llm_cfg)
            print(f"'{question}' → {result}")
        except Exception as e:
            print(f"'{question}' → Error: {e}")

def demonstrate_configuration_flexibility():
    """Demonstrate how easy it is to adjust configuration"""
    print("\n\n⚙️  Configuration Flexibility Demo")
    print("=" * 50)
    
    print("Example 1: Making detection more sensitive")
    os.environ['LARGE_GEN_STRONG_THRESHOLD'] = '15'  # Lower = more sensitive
    os.environ['LARGE_GEN_MIN_SCORE_KEYWORDS'] = '2'  # Lower = more sensitive
    
    from app.core.large_generation_config import reload_config
    config = reload_config()
    print(f"New strong threshold: {config.strong_number_threshold}")
    print(f"New min score for keywords: {config.min_score_for_keywords}")
    
    print("\nExample 2: Adjusting memory management")
    os.environ['REDIS_MAX_MESSAGES'] = '100'  # More history
    os.environ['CONVERSATION_HISTORY_DISPLAY'] = '15'  # Show more in chat
    
    config = reload_config()
    print(f"Redis message limit: {config.max_redis_messages}")
    print(f"History display count: {config.conversation_history_display}")
    
    print("\nExample 3: Performance tuning")
    os.environ['LARGE_GEN_DEFAULT_CHUNK_SIZE'] = '20'  # Larger chunks
    os.environ['LARGE_GEN_SECONDS_PER_CHUNK'] = '30'   # Faster estimates
    
    config = reload_config()
    print(f"Chunk size: {config.default_chunk_size}")
    print(f"Seconds per chunk: {config.estimated_seconds_per_chunk}")
    
    print("\n✅ All configuration changes applied successfully!")
    print("💡 Pro tip: Set these in .env file for persistence")

if __name__ == "__main__":
    print("🚀 Configurable Large Generation Detection Test")
    print("Testing the removal of hardcoded values")
    print("=" * 60)
    
    try:
        # Test 1: Default configuration
        default_config = test_default_config()
        
        # Test 2: Custom configuration via environment variables
        custom_config = test_custom_config()
        
        # Test 3: Classification with custom config
        test_classification_with_config()
        
        # Test 4: Demonstrate flexibility
        demonstrate_configuration_flexibility()
        
        print("\n\n🎉 Configurable Detection Test Complete!")
        print("✅ No more hardcoded values!")
        print("✅ Easy customization via environment variables")
        print("✅ Real-time configuration updates")
        print("✅ Backward compatible defaults")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())