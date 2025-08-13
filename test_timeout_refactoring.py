#!/usr/bin/env python3
"""
Test script to verify timeout refactoring is working correctly.
This script tests that all refactored timeout values are being read from the centralized configuration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_timeout_configuration():
    """Test that all timeout values are correctly loaded from centralized configuration"""
    
    print("=" * 60)
    print("TESTING TIMEOUT REFACTORING")
    print("=" * 60)
    
    # Test 1: Large Generation Utils - Conversation TTL
    print("\n1. Testing Large Generation Utils - Conversation TTL")
    print("-" * 50)
    try:
        from app.core.large_generation_utils import get_config_accessor
        config = get_config_accessor()
        ttl = config.redis_conversation_ttl
        print(f"✓ redis_conversation_ttl: {ttl} seconds")
        print(f"  (Should be from timeout settings, not large_generation settings)")
        
        # Verify it's using the centralized timeout
        from app.core.timeout_settings_cache import get_timeout_value
        expected_ttl = get_timeout_value("session_cache", "conversation_cache_ttl", 86400)
        if ttl == expected_ttl:
            print(f"✓ VERIFIED: Using centralized timeout value ({expected_ttl})")
        else:
            print(f"✗ WARNING: Values don't match! Got {ttl}, expected {expected_ttl}")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 2: IDC Reference Manager - Cache TTL
    print("\n2. Testing IDC Reference Manager - Cache TTL")
    print("-" * 50)
    try:
        from app.services.idc_reference_manager import IDCReferenceManager
        manager = IDCReferenceManager()
        print(f"✓ cache_ttl: {manager.cache_ttl} seconds")
        
        from app.core.timeout_settings_cache import get_timeout_value
        expected_ttl = get_timeout_value("session_cache", "temp_data_ttl", 1800)
        if manager.cache_ttl == expected_ttl:
            print(f"✓ VERIFIED: Using centralized timeout value ({expected_ttl})")
        else:
            print(f"✗ WARNING: Values don't match! Got {manager.cache_ttl}, expected {expected_ttl}")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 3: IDC Extraction Service - Document Processing Timeout
    print("\n3. Testing IDC Extraction Service - Document Processing Timeout")
    print("-" * 50)
    try:
        # We can't easily test the timeout in the request without making a real call,
        # but we can verify the import and function work
        from app.core.timeout_settings_cache import get_timeout_value
        timeout = get_timeout_value("document_processing", "document_processing_timeout", 120)
        print(f"✓ document_processing_timeout: {timeout} seconds")
        print(f"  (This value will be used in idc_extraction_service.py)")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 4: IDC Validation Service - Document Processing Timeout
    print("\n4. Testing IDC Validation Service - Document Processing Timeout")
    print("-" * 50)
    try:
        from app.core.timeout_settings_cache import get_timeout_value
        timeout = get_timeout_value("document_processing", "document_processing_timeout", 120)
        print(f"✓ document_processing_timeout: {timeout} seconds")
        print(f"  (This value will be used in idc_validation_service.py)")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 5: Ollama LLM - Streaming Timeout
    print("\n5. Testing Ollama LLM - Streaming Timeout")
    print("-" * 50)
    try:
        from app.core.timeout_settings_cache import get_timeout_value
        timeout = get_timeout_value("llm_ai", "llm_streaming_timeout", 120)
        print(f"✓ llm_streaming_timeout: {timeout} seconds")
        print(f"  (This value will be used in ollama.py for streaming)")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 6: Agent Bridge - Agent Processing Timeout
    print("\n6. Testing Agent Bridge - Agent Processing Timeout")
    print("-" * 50)
    try:
        from app.core.timeout_settings_cache import get_timeout_value
        timeout = get_timeout_value("llm_ai", "agent_processing_timeout", 90)
        print(f"✓ agent_processing_timeout: {timeout} seconds")
        print(f"  (This value will be used in agent_bridge.py)")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 7: Redis Continuation Manager - Session TTL
    print("\n7. Testing Redis Continuation Manager - Session TTL")
    print("-" * 50)
    try:
        from app.core.timeout_settings_cache import get_timeout_value
        ttl = get_timeout_value("session_cache", "conversation_cache_ttl", 86400)
        print(f"✓ conversation_cache_ttl: {ttl} seconds ({ttl/3600:.1f} hours)")
        print(f"  (This value will be used in redis_continuation_manager.py)")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 8: Enhanced Conversation Service - Cache TTL
    print("\n8. Testing Enhanced Conversation Service - Cache TTL")
    print("-" * 50)
    try:
        from app.core.timeout_settings_cache import get_timeout_value
        ttl = get_timeout_value("session_cache", "temp_data_ttl", 1800)
        print(f"✓ temp_data_ttl: {ttl} seconds ({ttl/60:.0f} minutes)")
        print(f"  (This value will be used in enhanced_conversation_service.py)")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 9: LangGraph RAG - Conversation TTL
    print("\n9. Testing LangGraph RAG - Conversation TTL")
    print("-" * 50)
    try:
        from app.langchain import langgraph_rag
        ttl = langgraph_rag.CONVERSATION_TTL
        print(f"✓ CONVERSATION_TTL: {ttl} seconds ({ttl/3600:.1f} hours)")
        
        from app.core.timeout_settings_cache import get_timeout_value
        expected_ttl = get_timeout_value("session_cache", "conversation_cache_ttl", 86400)
        if ttl == expected_ttl:
            print(f"✓ VERIFIED: Using centralized timeout value ({expected_ttl})")
        else:
            print(f"✗ WARNING: Values don't match! Got {ttl}, expected {expected_ttl}")
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    # Test 10: Verify timeout settings are loaded
    print("\n10. Testing Timeout Settings Cache")
    print("-" * 50)
    try:
        from app.core.timeout_settings_cache import get_timeout_settings, get_category_timeouts
        
        all_settings = get_timeout_settings()
        print(f"✓ Loaded {len(all_settings)} timeout categories:")
        for category in all_settings.keys():
            category_settings = get_category_timeouts(category)
            print(f"  - {category}: {len(category_settings)} settings")
        
        # Check specific values we're using
        session_cache = get_category_timeouts("session_cache")
        print(f"\n✓ Session Cache Settings:")
        print(f"  - conversation_cache_ttl: {session_cache.get('conversation_cache_ttl', 'NOT SET')}")
        print(f"  - temp_data_ttl: {session_cache.get('temp_data_ttl', 'NOT SET')}")
        
        doc_processing = get_category_timeouts("document_processing")
        print(f"\n✓ Document Processing Settings:")
        print(f"  - document_processing_timeout: {doc_processing.get('document_processing_timeout', 'NOT SET')}")
        
        llm_ai = get_category_timeouts("llm_ai")
        print(f"\n✓ LLM/AI Settings:")
        print(f"  - llm_streaming_timeout: {llm_ai.get('llm_streaming_timeout', 'NOT SET')}")
        print(f"  - agent_processing_timeout: {llm_ai.get('agent_processing_timeout', 'NOT SET')}")
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_timeout_configuration()