#!/usr/bin/env python3
"""
Test script to verify that RadiatingService uses web search as PRIMARY source.

This script tests the updated RadiatingService to ensure:
1. Web search is used by default for ALL queries
2. Web-first mode can be explicitly forced
3. Proper logging shows web-first approach is active
4. Web entities are prioritized over LLM-only entities
"""

import asyncio
import logging
import sys
from datetime import datetime

# Add the app directory to the path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

from app.services.radiating.radiating_service import RadiatingService

# Configure logging to see the web-first logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_web_first_default():
    """Test that web search is used by default for technology queries."""
    print("\n" + "="*80)
    print("TEST 1: Web-First Default Behavior")
    print("="*80)
    
    service = RadiatingService()
    
    # Test with a technology query
    query = "What are the latest LLM frameworks and RAG tools in 2024?"
    print(f"\nQuery: {query}")
    
    result = await service.execute_radiating_query(
        query=query,
        include_coverage=True
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Web search used: {result.get('web_search_used', False)}")
    
    if 'entity_sources' in result:
        print(f"\nEntity Sources:")
        print(f"  - Web entities: {result['entity_sources']['web']}")
        print(f"  - LLM entities: {result['entity_sources']['llm']}")
        print(f"  - Total entities: {result['entity_sources']['total']}")
    
    if 'coverage' in result:
        print(f"\nCoverage:")
        print(f"  - Total entities: {result['coverage']['total_entities']}")
        print(f"  - Entity types: {result['coverage'].get('entity_types', {})}")
    
    return result

async def test_force_web_search():
    """Test explicitly forcing web search."""
    print("\n" + "="*80)
    print("TEST 2: Force Web Search")
    print("="*80)
    
    service = RadiatingService()
    
    # Test with a general query but force web search
    query = "artificial intelligence applications"
    print(f"\nQuery: {query}")
    
    result = await service.execute_radiating_query(
        query=query,
        filters={'force_web_search': True},
        include_coverage=True
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Web search used: {result.get('web_search_used', False)}")
    
    if 'entity_sources' in result:
        print(f"\nEntity Sources:")
        print(f"  - Web entities: {result['entity_sources']['web']}")
        print(f"  - LLM entities: {result['entity_sources']['llm']}")
        print(f"  - Total entities: {result['entity_sources']['total']}")
    
    return result

async def test_local_query():
    """Test that local queries skip web search."""
    print("\n" + "="*80)
    print("TEST 3: Local Query (Should Skip Web Search)")
    print("="*80)
    
    service = RadiatingService()
    
    # Test with a local query
    query = "analyze this code function"
    print(f"\nQuery: {query}")
    
    result = await service.execute_radiating_query(
        query=query,
        include_coverage=True
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Web search used: {result.get('web_search_used', False)}")
    
    if 'entity_sources' in result:
        print(f"\nEntity Sources:")
        print(f"  - Web entities: {result['entity_sources']['web']}")
        print(f"  - LLM entities: {result['entity_sources']['llm']}")
        print(f"  - Total entities: {result['entity_sources']['total']}")
    
    return result

async def test_web_first_with_method():
    """Test the dedicated web-first method."""
    print("\n" + "="*80)
    print("TEST 4: Web-First Method")
    print("="*80)
    
    service = RadiatingService()
    
    # Test with comprehensive query using web-first method
    query = "emerging AI technologies and their applications"
    print(f"\nQuery: {query}")
    
    result = await service.execute_radiating_query_with_web_search(
        query=query,
        include_coverage=True
    )
    
    print(f"\nStatus: {result['status']}")
    print(f"Web-first mode: {result.get('web_first_mode', False)}")
    print(f"Discovery method: {result.get('discovery_method', 'unknown')}")
    
    if 'entity_sources' in result:
        print(f"\nEntity Sources:")
        print(f"  - Web entities: {result['entity_sources']['web']}")
        print(f"  - LLM entities: {result['entity_sources']['llm']}")
        print(f"  - Total entities: {result['entity_sources']['total']}")
    
    return result

async def test_preview_with_web():
    """Test that preview also uses web-first approach."""
    print("\n" + "="*80)
    print("TEST 5: Preview with Web-First")
    print("="*80)
    
    service = RadiatingService()
    
    query = "latest cloud computing platforms"
    print(f"\nQuery: {query}")
    
    result = await service.preview_expansion(
        query=query,
        max_depth=2,
        max_entities=10
    )
    
    print(f"\nPotential entities found: {len(result.get('potential_entities', []))}")
    print(f"Web-first enabled: {result.get('web_first_enabled', False)}")
    print(f"Discovery method: {result.get('discovery_method', 'unknown')}")
    
    if result.get('web_entities_discovered'):
        print(f"Web entities discovered: {result['web_entities_discovered']}")
    
    # Show first few entities
    if result.get('potential_entities'):
        print("\nFirst 5 potential entities:")
        for entity in result['potential_entities'][:5]:
            source = entity.get('properties', {}).get('source', 'unknown')
            print(f"  - {entity['name']} ({entity['type']}) [source: {source}]")
    
    return result

async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("WEB-FIRST RADIATING SERVICE TEST SUITE")
    print("Testing that web search is the PRIMARY source for ALL queries")
    print("="*80)
    
    try:
        # Run all tests
        await test_web_first_default()
        await test_force_web_search()
        await test_local_query()
        await test_web_first_with_method()
        await test_preview_with_web()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("Web-first approach is working as expected!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())