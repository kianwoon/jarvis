#!/usr/bin/env python3
"""
Test script to verify that multi-agent system uses settings correctly
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_settings_integration():
    """Test that model_server URLs are read from settings"""
    
    print("=" * 80)
    print("TESTING SETTINGS INTEGRATION FOR MULTI-AGENT SYSTEM")
    print("=" * 80)
    
    # Test 1: Check LLM settings cache
    print("\n1. Testing LLM Settings Cache...")
    try:
        from app.core.llm_settings_cache import (
            get_main_llm_full_config, 
            get_second_llm_full_config,
            get_query_classifier_full_config
        )
        
        main_config = get_main_llm_full_config()
        second_config = get_second_llm_full_config()
        query_config = get_query_classifier_full_config()
        
        print(f"✓ Main LLM model_server: {main_config.get('model_server', 'NOT SET')}")
        print(f"✓ Second LLM model_server: {second_config.get('model_server', 'NOT SET')}")
        print(f"✓ Query Classifier model_server: {query_config.get('model_server', 'NOT SET')}")
        
        # Check if Docker detection is working
        is_docker = os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER")
        print(f"✓ Docker environment detected: {is_docker}")
        
    except Exception as e:
        print(f"✗ Failed to get LLM settings: {e}")
        return False
    
    # Test 2: Check Multi-Agent System initialization
    print("\n2. Testing Multi-Agent System Initialization...")
    try:
        from app.langchain.langgraph_multi_agent_system import LangGraphMultiAgentSystem
        
        system = LangGraphMultiAgentSystem()
        
        print(f"✓ Multi-agent system initialized")
        print(f"✓ Ollama base URL: {system.ollama_base_url}")
        
        # Check if it's using settings-based URL
        if "localhost" in second_config.get('model_server', '') and is_docker:
            expected_url = second_config.get('model_server', '').replace('localhost', 'host.docker.internal')
        else:
            expected_url = second_config.get('model_server', 'http://localhost:11434')
        
        if system.ollama_base_url == expected_url:
            print(f"✓ URL matches expected settings-based URL: {expected_url}")
        else:
            print(f"⚠ URL mismatch - Expected: {expected_url}, Got: {system.ollama_base_url}")
        
    except ImportError as e:
        print(f"⚠ LangGraph not installed (expected): {e}")
    except Exception as e:
        print(f"✗ Failed to initialize multi-agent system: {e}")
        return False
    
    # Test 3: Check Enhanced Query Classifier
    print("\n3. Testing Enhanced Query Classifier...")
    try:
        from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
        
        classifier = EnhancedQueryClassifier()
        print(f"✓ Enhanced query classifier initialized")
        
        # Test classification (this will use the settings-based URL)
        result = await classifier.classify("What is the weather today?")
        print(f"✓ Classification test completed: {result.get('category', 'UNKNOWN')}")
        
    except Exception as e:
        print(f"✗ Failed to test query classifier: {e}")
    
    # Test 4: Check Search Query Optimizer
    print("\n4. Testing Search Query Optimizer...")
    try:
        from app.langchain.search_query_optimizer import SearchQueryOptimizer
        
        optimizer = SearchQueryOptimizer()
        print(f"✓ Search query optimizer initialized")
        
        # The optimizer should use settings internally
        test_query = "latest AI developments"
        optimized = await optimizer.optimize_query(test_query)
        print(f"✓ Optimization test completed: '{test_query}' -> '{optimized}'")
        
    except Exception as e:
        print(f"✗ Failed to test search optimizer: {e}")
    
    print("\n" + "=" * 80)
    print("SETTINGS INTEGRATION TEST COMPLETED")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    # Run the async test
    result = asyncio.run(test_settings_integration())
    sys.exit(0 if result else 1)