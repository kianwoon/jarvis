#!/usr/bin/env python3
"""
Direct test of web search functionality
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.radiating.extraction.web_search_integration import WebSearchIntegration

async def test_direct_web_search():
    """Test web search directly"""
    web_search = WebSearchIntegration()
    
    # Test queries from different categories
    test_queries = [
        "What are the latest developments in the Ukraine conflict?",
        "What is Elon Musk working on recently?",
        "Apple's newest product announcements",
        "Who won the latest NBA championship?",
        "Recent discoveries in quantum computing",
        "Current state of the stock market",
        "Weather forecast for New York",
        "Latest policy changes in healthcare",
        "Best smartphones released this year",
        "analyze my code"  # Should skip web search
    ]
    
    print("=" * 80)
    print("TESTING WEB SEARCH TRIGGER LOGIC")
    print("=" * 80)
    
    for query in test_queries:
        should_search = web_search.should_use_web_search(query)
        status = "✓ TRIGGERS" if should_search else "✗ SKIPS"
        print(f"{status}: {query[:60]}...")
    
    print("\n" + "=" * 80)
    print("TESTING WEB SEARCH EXECUTION")
    print("=" * 80)
    
    # Test actual web search execution
    test_query = "Latest AI developments 2024"
    print(f"\nExecuting web search for: {test_query}")
    
    try:
        # Test the private method directly
        results = await web_search._execute_web_search(test_query)
        print(f"Search returned {len(results)} results")
        
        if results:
            print("\nFirst result:")
            print(f"  Title: {results[0].get('title', 'N/A')}")
            print(f"  Snippet: {results[0].get('snippet', 'N/A')[:100]}...")
            print(f"  URL: {results[0].get('url', 'N/A')}")
    except Exception as e:
        print(f"Error executing web search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test entity extraction from web search
    print("\n" + "=" * 80)
    print("TESTING ENTITY EXTRACTION FROM WEB")
    print("=" * 80)
    
    try:
        entities = await web_search.search_for_entities("Latest OpenAI GPT models")
        print(f"Found {len(entities)} entities from web search")
        
        for entity in entities[:5]:
            print(f"  - {entity.get('text')} ({entity.get('type')}) - confidence: {entity.get('confidence', 0):.2f}")
    except Exception as e:
        print(f"Error extracting entities: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct_web_search())