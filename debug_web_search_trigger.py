#!/usr/bin/env python3
"""
Debug script to diagnose why web search isn't being triggered
and test the entire flow from query to web search execution.
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

async def test_mcp_tools():
    """Test if MCP tools are accessible and web search is available."""
    print("\n" + "="*60)
    print("TESTING MCP TOOLS AVAILABILITY")
    print("="*60)
    
    try:
        from app.core.mcp_tools_cache import get_enabled_mcp_tools
        
        tools = get_enabled_mcp_tools()
        print(f"\n✅ Found {len(tools)} enabled MCP tools")
        
        # Check for search tools
        search_tools = [t for t in tools if 'search' in t.lower() or 'web' in t.lower()]
        print(f"\n🔍 Search-related tools: {search_tools}")
        
        if 'google_search' in tools:
            print("\n✅ google_search tool is AVAILABLE")
            tool_info = tools['google_search']
            print(f"   Endpoint: {tool_info.get('endpoint', 'N/A')}")
            print(f"   Method: {tool_info.get('method', 'N/A')}")
        else:
            print("\n❌ google_search tool is NOT available")
            
        return 'google_search' in tools
        
    except Exception as e:
        print(f"\n❌ Error checking MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_web_search_integration():
    """Test the WebSearchIntegration class directly."""
    print("\n" + "="*60)
    print("TESTING WEB SEARCH INTEGRATION")
    print("="*60)
    
    try:
        from app.services.radiating.extraction.web_search_integration import WebSearchIntegration
        
        web_search = WebSearchIntegration()
        
        # Test queries that should trigger web search
        test_queries = [
            "What are the latest LLM frameworks in 2024?",
            "new AI tools for RAG",
            "current vector databases",
            "best open source LLMs",
            "emerging technologies in AI",
            "JSON API integration tools",  # Generic terms that were problematic
            "system architecture patterns",
            "database connection pooling"
        ]
        
        print("\n📝 Testing should_use_web_search() for various queries:")
        for query in test_queries:
            should_search = web_search.should_use_web_search(query)
            icon = "✅" if should_search else "❌"
            print(f"   {icon} '{query[:50]}...' -> {should_search}")
        
        # Test actual web search execution
        print("\n🌐 Testing actual web search execution:")
        test_query = "latest open source LLM frameworks 2024"
        print(f"   Query: '{test_query}'")
        
        results = await web_search._execute_web_search(test_query)
        if results:
            print(f"   ✅ Got {len(results)} search results")
            if results:
                print(f"   First result: {json.dumps(results[0], indent=2)[:200]}...")
        else:
            print("   ❌ No results returned")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing WebSearchIntegration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_radiating_service():
    """Test the RadiatingService's web search decision logic."""
    print("\n" + "="*60)
    print("TESTING RADIATING SERVICE")
    print("="*60)
    
    try:
        from app.services.radiating.radiating_service import RadiatingService
        
        service = RadiatingService()
        
        # Test queries
        test_queries = [
            "What are the latest LLM frameworks?",
            "Show me new AI tools",
            "List current vector databases",
            "JSON API integration",
            "database connection patterns"
        ]
        
        print("\n🔍 Testing _should_use_web_search() in RadiatingService:")
        for query in test_queries:
            should_search = service._should_use_web_search(query)
            icon = "✅" if should_search else "❌"
            print(f"   {icon} '{query[:50]}...' -> {should_search}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing RadiatingService: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_entity_extractor():
    """Test the UniversalEntityExtractor with web search."""
    print("\n" + "="*60)
    print("TESTING UNIVERSAL ENTITY EXTRACTOR")
    print("="*60)
    
    try:
        from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
        
        extractor = UniversalEntityExtractor()
        
        # Test entity extraction with web search
        test_queries = [
            "What are the latest LLM frameworks like LangChain?",
            "Show me tools similar to ChromaDB and Pinecone"
        ]
        
        for query in test_queries:
            print(f"\n📝 Extracting entities from: '{query[:50]}...'")
            
            # Test with force_web_search
            entities = await extractor.extract_entities_with_web_search(
                query,
                force_web_search=True
            )
            
            if entities:
                print(f"   ✅ Extracted {len(entities)} entities:")
                for entity in entities[:5]:
                    source = entity.metadata.get('source', 'unknown') if hasattr(entity, 'metadata') else 'unknown'
                    print(f"      - {entity.text} ({entity.entity_type}) [source: {source}]")
            else:
                print("   ❌ No entities extracted")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing UniversalEntityExtractor: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_radiating_coverage():
    """Test the full radiating_coverage flow."""
    print("\n" + "="*60)
    print("TESTING FULL RADIATING COVERAGE FLOW")
    print("="*60)
    
    try:
        from app.services.radiating.radiating_service import RadiatingService
        
        service = RadiatingService()
        
        # Test query that should trigger web search
        query = "What are the latest open source LLM frameworks and vector databases in 2024?"
        
        print(f"\n🚀 Testing full radiating_coverage for:")
        print(f"   Query: '{query}'")
        
        # Call radiating_coverage
        result = await service.radiating_coverage(
            query=query,
            filters={'use_web_search': True},  # Force web search
            include_coverage=True
        )
        
        print(f"\n📊 Results:")
        print(f"   Query ID: {result.get('query_id', 'N/A')}")
        print(f"   Status: {result.get('status', 'N/A')}")
        
        if 'coverage' in result:
            coverage = result['coverage']
            if 'entities' in coverage:
                entities = coverage['entities']
                print(f"   Entities found: {len(entities)}")
                
                # Count web-sourced entities
                web_entities = [e for e in entities if e.get('metadata', {}).get('source') == 'web_search']
                print(f"   Web-sourced entities: {len(web_entities)}")
                
                if entities:
                    print("\n   Sample entities:")
                    for entity in entities[:5]:
                        source = entity.get('metadata', {}).get('source', 'unknown')
                        print(f"      - {entity.get('name', 'N/A')} [source: {source}]")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing full radiating coverage: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_tool_execution():
    """Test direct MCP tool execution."""
    print("\n" + "="*60)
    print("TESTING DIRECT MCP TOOL EXECUTION")
    print("="*60)
    
    try:
        from app.core.enhanced_tool_executor import call_mcp_tool_enhanced_async
        
        # Test a simple search
        query = "latest LLM frameworks 2024"
        print(f"\n🔧 Testing direct MCP tool call with: '{query}'")
        
        result = await call_mcp_tool_enhanced_async(
            "google_search",
            {"query": query, "num_results": 5}
        )
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Success! Got response:")
            print(f"   {json.dumps(result, indent=2)[:500]}...")
        
        return "error" not in result
        
    except Exception as e:
        print(f"\n❌ Error testing MCP tool execution: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all debug tests."""
    print("\n" + "="*80)
    print(" WEB SEARCH TRIGGER DEBUG SCRIPT ")
    print("="*80)
    
    # Run all tests
    results = {
        "MCP Tools Available": await test_mcp_tools(),
        "Web Search Integration": await test_web_search_integration(),
        "Radiating Service": await test_radiating_service(),
        "Entity Extractor": await test_entity_extractor(),
        "Direct MCP Execution": await test_mcp_tool_execution(),
        "Full Radiating Coverage": await test_full_radiating_coverage(),
    }
    
    # Summary
    print("\n" + "="*80)
    print(" SUMMARY ")
    print("="*80)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {test_name}: {'PASSED' if passed else 'FAILED'}")
    
    if all_passed:
        print("\n✅ All tests passed! Web search should be working.")
    else:
        print("\n❌ Some tests failed. Issues identified:")
        failed_tests = [name for name, passed in results.items() if not passed]
        for test in failed_tests:
            print(f"   - {test}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())