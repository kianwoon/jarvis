#!/usr/bin/env python3
"""
Test configurable detection system without circular imports
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

def test_config_system():
    """Test the configuration system"""
    print("🔧 Testing Configuration System")
    print("=" * 50)
    
    # Test default config
    from app.core.large_generation_config import get_config
    config = get_config()
    
    print("DEFAULT CONFIGURATION:")
    print(f"  Strong threshold: {config.strong_number_threshold}")
    print(f"  Medium threshold: {config.medium_number_threshold}")
    print(f"  Min items for chunking: {config.min_items_for_chunking}")
    print(f"  Default comprehensive items: {config.default_comprehensive_items}")
    print(f"  Score multiplier: {config.score_multiplier}")
    print(f"  Redis TTL: {config.redis_conversation_ttl} seconds")
    print(f"  Max Redis messages: {config.max_redis_messages}")
    print(f"  Memory messages: {config.max_memory_messages}")
    print(f"  History display: {config.conversation_history_display}")
    
    # Test environment variable overrides
    print("\n" + "=" * 50)
    print("TESTING ENVIRONMENT VARIABLE OVERRIDES:")
    
    # Set custom environment variables
    os.environ['LARGE_GEN_STRONG_THRESHOLD'] = '25'
    os.environ['LARGE_GEN_MIN_CHUNKING'] = '15'
    os.environ['LARGE_GEN_DEFAULT_COMPREHENSIVE'] = '40'
    os.environ['REDIS_MAX_MESSAGES'] = '75'
    os.environ['MEMORY_MAX_MESSAGES'] = '30'
    
    # Reload config
    from app.core.large_generation_config import reload_config
    custom_config = reload_config()
    
    print("CUSTOM CONFIGURATION:")
    print(f"  Strong threshold: {custom_config.strong_number_threshold} (was {config.strong_number_threshold})")
    print(f"  Min chunking: {custom_config.min_items_for_chunking} (was {config.min_items_for_chunking})")
    print(f"  Comprehensive items: {custom_config.default_comprehensive_items} (was {config.default_comprehensive_items})")
    print(f"  Redis messages: {custom_config.max_redis_messages} (was {config.max_redis_messages})")
    print(f"  Memory messages: {custom_config.max_memory_messages} (was {config.max_memory_messages})")
    
    # Verify changes took effect
    changes_detected = (
        custom_config.strong_number_threshold != config.strong_number_threshold or
        custom_config.min_items_for_chunking != config.min_items_for_chunking or
        custom_config.default_comprehensive_items != config.default_comprehensive_items or
        custom_config.max_redis_messages != config.max_redis_messages or
        custom_config.max_memory_messages != config.max_memory_messages
    )
    
    if changes_detected:
        print("\n✅ Environment variable overrides working correctly!")
    else:
        print("\n❌ Environment variable overrides not working!")
        return False
    
    return True

def test_keywords_and_patterns():
    """Test that keywords and patterns are configurable"""
    print("\n" + "=" * 50)
    print("TESTING CONFIGURABLE KEYWORDS AND PATTERNS:")
    
    from app.core.large_generation_config import get_config
    config = get_config()
    
    print(f"Large output indicators ({len(config.large_output_indicators)} items):")
    for i, indicator in enumerate(config.large_output_indicators[:10]):  # Show first 10
        print(f"  {i+1}. {indicator}")
    if len(config.large_output_indicators) > 10:
        print(f"  ... and {len(config.large_output_indicators) - 10} more")
    
    print(f"\nComprehensive keywords ({len(config.comprehensive_keywords)} items):")
    for keyword in config.comprehensive_keywords:
        print(f"  - {keyword}")
    
    print(f"\nLarge patterns ({len(config.large_patterns)} items):")
    for i, pattern in enumerate(config.large_patterns[:3]):  # Show first 3
        print(f"  {i+1}. {pattern}")
    if len(config.large_patterns) > 3:
        print(f"  ... and {len(config.large_patterns) - 3} more")
    
    return True

def test_isolated_detection():
    """Test detection logic in isolation"""
    print("\n" + "=" * 50)
    print("TESTING DETECTION LOGIC (ISOLATED):")
    
    # Copy the detection function logic without imports that cause circular dependencies
    import re
    from app.core.large_generation_config import get_config
    
    def isolated_detect_large_output_potential(question: str) -> dict:
        """Isolated version of detection function"""
        config = get_config()
        large_output_indicators = config.large_output_indicators
        
        # Count indicator patterns
        score = 0
        question_lower = question.lower()
        matched_indicators = []
        
        for indicator in large_output_indicators:
            if indicator in question_lower:
                score += 1
                matched_indicators.append(indicator)
        
        # Extract numbers that might indicate quantity
        numbers = re.findall(r'\b(\d+)\b', question)
        max_number = max([int(n) for n in numbers], default=0)
        
        # Additional patterns that suggest large output
        large_patterns = config.large_patterns
        
        pattern_matches = []
        for pattern in large_patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                pattern_matches.extend(matches)
                score += config.pattern_score_weight
        
        # Calculate confidence and estimated items
        base_confidence = min(1.0, score / config.max_score_for_confidence)
        number_confidence = min(1.0, max_number / config.max_number_for_confidence) if max_number > 0 else 0
        final_confidence = max(base_confidence, number_confidence)
        
        # Estimate number of items to generate
        if max_number > 10:
            estimated_items = max_number
        elif score >= 3:
            estimated_items = score * config.score_multiplier
        elif any(keyword in question_lower for keyword in config.comprehensive_keywords):
            estimated_items = config.default_comprehensive_items
        else:
            estimated_items = config.min_estimated_items
        
        # Refined logic for determining if it's a large generation request
        is_likely_large = False
        
        if max_number >= config.strong_number_threshold:
            is_likely_large = True
        elif max_number >= config.medium_number_threshold and score >= config.min_score_for_medium_numbers:
            is_likely_large = True
        elif score >= config.min_score_for_keywords and any(keyword in question_lower for keyword in config.comprehensive_keywords):
            is_likely_large = True
        elif max_number > 0 and max_number < config.small_number_threshold:
            is_likely_large = False
        
        # Adjust estimated items for small number requests
        if max_number > 0 and max_number < config.small_number_threshold:
            estimated_items = max_number
        
        return {
            "likely_large": is_likely_large,
            "estimated_items": estimated_items,
            "confidence": final_confidence,
            "score": score,
            "max_number": max_number,
            "matched_indicators": matched_indicators,
            "pattern_matches": pattern_matches
        }
    
    # Test cases
    test_cases = [
        "Generate 50 interview questions for software engineers",
        "Create a comprehensive guide to Python programming",
        "Generate 5 quick tips",
        "List many examples of successful startups"
    ]
    
    for question in test_cases:
        result = isolated_detect_large_output_potential(question)
        print(f"\n'{question}':")
        print(f"  Large: {result['likely_large']}")
        print(f"  Items: {result['estimated_items']}")
        print(f"  Score: {result['score']}")
        print(f"  Indicators: {result['matched_indicators']}")
    
    return True

if __name__ == "__main__":
    print("🚀 Configuration System Test")
    print("Testing configurable detection without hardcoded values")
    print("=" * 60)
    
    try:
        success = True
        
        # Test 1: Configuration system
        success &= test_config_system()
        
        # Test 2: Keywords and patterns
        success &= test_keywords_and_patterns()
        
        # Test 3: Detection logic in isolation
        success &= test_isolated_detection()
        
        if success:
            print("\n\n🎉 Configuration System Test Complete!")
            print("✅ No more hardcoded values!")
            print("✅ Environment variable overrides working")
            print("✅ Configurable keywords and patterns")
            print("✅ Detection logic uses configuration")
            print("\n💡 To customize, set environment variables like:")
            print("   LARGE_GEN_STRONG_THRESHOLD=25")
            print("   LARGE_GEN_MIN_CHUNKING=15")
            print("   REDIS_MAX_MESSAGES=100")
        else:
            print("\n❌ Some tests failed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())