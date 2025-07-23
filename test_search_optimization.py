#!/usr/bin/env python3
"""
Test script for search query optimization functionality
"""
import asyncio
import sys
import os

# Add the app directory to path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

async def test_search_optimization():
    """Test the search query optimizer directly"""
    
    print("Testing Search Query Optimization...")
    
    try:
        # Test the search query optimizer
        from app.langchain.search_query_optimizer import optimize_search_query
        
        test_queries = [
            "Can you tell me about the latest AI developments?",
            "What's the weather like today?", 
            "I want to know about Python vs JavaScript",
            "Please search for information about renewable energy"
        ]
        
        for query in test_queries:
            print(f"\nOriginal: {query}")
            try:
                optimized = await optimize_search_query(query)
                print(f"Optimized: {optimized}")
                
                if optimized != query:
                    print("✅ Query was optimized")
                else:
                    print("⚠️  Query unchanged (optimization disabled or failed)")
                    
            except Exception as e:
                print(f"❌ Optimization failed: {e}")
        
        print("\n" + "="*50)
        
        # Test configuration loading
        print("\nTesting Configuration...")
        try:
            from app.core.llm_settings_cache import get_search_optimization_config
            config = get_search_optimization_config()
            print(f"Search optimization enabled: {config.get('enable_search_optimization')}")
            print(f"Optimization timeout: {config.get('optimization_timeout')} seconds")
            print(f"Prompt length: {len(config.get('optimization_prompt', ''))} characters")
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            
    except Exception as e:
        print(f"❌ Import or test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search_optimization())