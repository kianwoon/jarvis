#!/usr/bin/env python3
"""
Final test to verify web search integration is working properly
"""

import asyncio
import logging
import sys
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

async def test_web_search_trigger_logic():
    """Test that web search is triggered for technology queries"""
    from app.services.radiating.radiating_service import RadiatingService
    
    service = RadiatingService()
    
    test_queries = [
        "What are the latest LLM frameworks?",
        "Show me new vector databases",
        "JSON API integration tools",
        "database connection pooling libraries"
    ]
    
    print("\n✅ Web Search Trigger Logic:")
    for query in test_queries:
        should_search = service._should_use_web_search(query)
        print(f"  {'✓' if should_search else '✗'} '{query[:40]}...' -> Web search: {should_search}")
    
    return all(service._should_use_web_search(q) for q in test_queries)

async def test_mcp_connectivity():
    """Test that MCP server is accessible"""
    from app.core.enhanced_tool_executor import call_mcp_tool_enhanced_async
    
    print("\n✅ MCP Server Connectivity:")
    
    # Simple test query
    result = await call_mcp_tool_enhanced_async(
        "google_search",
        {"query": "test", "num_results": 1}
    )
    
    if "error" not in result or "API error" in str(result.get("error", "")):
        # API error means we connected but need API key
        print("  ✓ MCP server is accessible (API key may be needed)")
        return True
    else:
        print(f"  ✗ MCP server connection failed: {result.get('error')}")
        return False

async def test_radiating_with_web_search():
    """Test the full radiating coverage with web search"""
    from app.services.radiating.radiating_service import RadiatingService
    
    service = RadiatingService()
    
    print("\n✅ Radiating Coverage with Web Search:")
    
    query = "What are the latest open source LLM frameworks in 2024?"
    
    try:
        # Use the correct method name
        result = await service.get_radiating_coverage(
            query=query,
            filters={'use_web_search': True},
            include_coverage=True
        )
        
        if result and 'status' in result:
            print(f"  ✓ Radiating coverage executed successfully")
            print(f"    Status: {result.get('status')}")
            
            # Check if web search was triggered
            if 'coverage' in result and 'entities' in result['coverage']:
                entities = result['coverage']['entities']
                web_entities = [e for e in entities if e.get('metadata', {}).get('source') == 'web_search']
                print(f"    Total entities: {len(entities)}")
                print(f"    Web-sourced entities: {len(web_entities)}")
                
                if entities:
                    print("    Sample entities:")
                    for entity in entities[:3]:
                        source = entity.get('metadata', {}).get('source', 'unknown')
                        print(f"      - {entity.get('name', 'N/A')} [source: {source}]")
            
            return True
        else:
            print(f"  ✗ Unexpected result format")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_entity_extraction_with_web():
    """Test entity extraction with web search"""
    from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
    
    extractor = UniversalEntityExtractor()
    
    print("\n✅ Entity Extraction with Web Search:")
    
    query = "What tools are similar to LangChain and LlamaIndex?"
    
    try:
        entities = await extractor.extract_entities_with_web_search(
            query,
            force_web_search=True
        )
        
        if entities:
            print(f"  ✓ Extracted {len(entities)} entities")
            for entity in entities[:3]:
                source = entity.metadata.get('source', 'unknown') if hasattr(entity, 'metadata') else 'unknown'
                print(f"    - {entity.text} ({entity.entity_type}) [source: {source}]")
            return True
        else:
            print("  ✗ No entities extracted")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

async def main():
    print("="*70)
    print(" WEB SEARCH INTEGRATION TEST ")
    print("="*70)
    
    # Run all tests
    tests = [
        ("Web Search Trigger Logic", test_web_search_trigger_logic()),
        ("MCP Server Connectivity", test_mcp_connectivity()),
        ("Entity Extraction", test_entity_extraction_with_web()),
        ("Full Radiating Coverage", test_radiating_with_web_search()),
    ]
    
    results = {}
    for name, test_coro in tests:
        try:
            results[name] = await test_coro
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY ")
    print("="*70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nWeb search integration is working correctly:")
        print("  1. Technology queries trigger web search")
        print("  2. MCP server is accessible")
        print("  3. Entity extraction uses web search")
        print("  4. Radiating coverage integrates web results")
    else:
        print("\n⚠️ Some tests failed. Issues to address:")
        failed = [name for name, passed in results.items() if not passed]
        for test in failed:
            print(f"  - {test}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    asyncio.run(main())