#!/usr/bin/env python3
"""Quick test of single query classification to debug the issue"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.langchain.enhanced_query_classifier import EnhancedQueryClassifier
from app.core.llm_settings_cache import get_llm_settings
from app.llm.response_analyzer import clear_behavior_cache

async def test_single_query():
    print("Single Query Classification Test")
    print("="*50)
    
    # Clear cache
    clear_behavior_cache()
    
    # Create classifier
    classifier = EnhancedQueryClassifier()
    
    # Test with instruct-2507 model
    settings = get_llm_settings()
    if 'query_classifier' not in settings:
        settings['query_classifier'] = {}
    settings['query_classifier']['model'] = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    
    query = "How do I search for files in a directory?"
    print(f"Query: {query}")
    print(f"Model: {settings['query_classifier']['model']}")
    
    try:
        results = await classifier.classify(query)
        
        if results:
            result = results[0]
            print(f"Classification: {result.query_type}")
            print(f"Confidence: {result.confidence}")
            print(f"Metadata: {result.metadata}")
            
            # Check behavior cache
            from app.llm.response_analyzer import response_analyzer
            stats = response_analyzer.get_cache_stats()
            print(f"\nBehavior cache stats: {stats}")
            
        else:
            print("No results returned")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_query())