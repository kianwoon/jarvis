#!/usr/bin/env python3
"""
Test script for the enhanced web-first entity extraction in UniversalEntityExtractor.

This script tests:
1. The new extract_entities_web_first method
2. The prefer_web_search parameter in extract_entities
3. The enhanced extract_entities_with_web_search method
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_web_first_extraction():
    """Test the web-first extraction approach."""
    logger.info("=" * 80)
    logger.info("Testing Web-First Entity Extraction")
    logger.info("=" * 80)
    
    # Initialize the extractor
    extractor = UniversalEntityExtractor()
    
    # Test queries that should trigger web search
    test_queries = [
        "What are the latest LLM frameworks and tools for building AI applications?",
        "List the essential technologies for RAG systems in 2024",
        "What are the newest open source vector databases?",
        "Show me modern AI agent frameworks and orchestration tools",
        "What technologies are used for prompt engineering and LLM optimization?"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: {query}")
        logger.info("-" * 40)
        
        try:
            # Test 1: extract_entities with prefer_web_search=True (default)
            logger.info("\n1. Testing extract_entities with prefer_web_search=True")
            entities_web_first = await extractor.extract_entities(
                query, 
                prefer_web_search=True
            )
            logger.info(f"   Found {len(entities_web_first)} entities with web-first approach")
            if entities_web_first:
                logger.info("   Top 5 entities:")
                for entity in entities_web_first[:5]:
                    source = entity.metadata.get('source', 'unknown')
                    logger.info(f"   - {entity.text} ({entity.entity_type}) [confidence: {entity.confidence:.2f}, source: {source}]")
            
            # Test 2: extract_entities with prefer_web_search=False (LLM only)
            logger.info("\n2. Testing extract_entities with prefer_web_search=False")
            entities_llm_only = await extractor.extract_entities(
                query,
                prefer_web_search=False
            )
            logger.info(f"   Found {len(entities_llm_only)} entities with LLM-only approach")
            if entities_llm_only:
                logger.info("   Top 5 entities:")
                for entity in entities_llm_only[:5]:
                    source = entity.metadata.get('source', 'llm')
                    logger.info(f"   - {entity.text} ({entity.entity_type}) [confidence: {entity.confidence:.2f}, source: {source}]")
            
            # Test 3: Direct call to extract_entities_web_first
            logger.info("\n3. Testing extract_entities_web_first directly")
            entities_web_direct = await extractor.extract_entities_web_first(query)
            logger.info(f"   Found {len(entities_web_direct)} entities with direct web-first method")
            
            # Count sources
            web_count = sum(1 for e in entities_web_direct if e.metadata.get('source') == 'web_search')
            llm_count = sum(1 for e in entities_web_direct if e.metadata.get('extraction_method') == 'llm_fallback')
            logger.info(f"   Source breakdown: Web: {web_count}, LLM: {llm_count}")
            
            # Test 4: Enhanced extract_entities_with_web_search
            logger.info("\n4. Testing enhanced extract_entities_with_web_search")
            entities_enhanced = await extractor.extract_entities_with_web_search(
                query,
                force_web_search=True
            )
            logger.info(f"   Found {len(entities_enhanced)} entities with enhanced method")
            
            # Show unique entities found only via web search
            if entities_web_first and entities_llm_only:
                web_only_texts = set(e.text.lower() for e in entities_web_first)
                llm_only_texts = set(e.text.lower() for e in entities_llm_only)
                unique_to_web = web_only_texts - llm_only_texts
                
                if unique_to_web:
                    logger.info(f"\n   Unique entities found only via web search ({len(unique_to_web)}):")
                    for text in list(unique_to_web)[:10]:
                        logger.info(f"   - {text}")
            
            logger.info("\n" + "=" * 40)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("Web-First Extraction Test Complete")
    logger.info("=" * 80)


async def test_comparison():
    """Compare extraction methods side by side."""
    logger.info("\n" + "=" * 80)
    logger.info("Comparing Extraction Methods")
    logger.info("=" * 80)
    
    extractor = UniversalEntityExtractor()
    
    # Test with a comprehensive technology query
    query = "What are all the essential technologies, frameworks, and tools for building modern LLM-powered applications with RAG?"
    
    logger.info(f"\nQuery: {query}")
    logger.info("-" * 80)
    
    # Run all extraction methods
    logger.info("\nRunning all extraction methods...")
    
    # Method 1: LLM-only
    entities_llm = await extractor.extract_entities(query, prefer_web_search=False)
    
    # Method 2: Web-first
    entities_web_first = await extractor.extract_entities_web_first(query)
    
    # Method 3: Enhanced web search
    entities_enhanced = await extractor.extract_entities_with_web_search(query, force_web_search=True)
    
    # Compare results
    logger.info("\n" + "=" * 40)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 40)
    
    logger.info(f"\n1. LLM-Only Extraction: {len(entities_llm)} entities")
    logger.info("   Top 10 entities:")
    for entity in entities_llm[:10]:
        logger.info(f"   - {entity.text} ({entity.entity_type}) [conf: {entity.confidence:.2f}]")
    
    logger.info(f"\n2. Web-First Extraction: {len(entities_web_first)} entities")
    web_sourced = sum(1 for e in entities_web_first if e.metadata.get('source') == 'web_search')
    logger.info(f"   Web-sourced: {web_sourced}, LLM-fallback: {len(entities_web_first) - web_sourced}")
    logger.info("   Top 10 entities:")
    for entity in entities_web_first[:10]:
        source = entity.metadata.get('source', 'unknown')
        logger.info(f"   - {entity.text} ({entity.entity_type}) [conf: {entity.confidence:.2f}, src: {source}]")
    
    logger.info(f"\n3. Enhanced Web Search: {len(entities_enhanced)} entities")
    logger.info("   Top 10 entities:")
    for entity in entities_enhanced[:10]:
        source = entity.metadata.get('source', 'unknown')
        logger.info(f"   - {entity.text} ({entity.entity_type}) [conf: {entity.confidence:.2f}]")
    
    # Find unique discoveries
    llm_texts = set(e.text.lower() for e in entities_llm)
    web_texts = set(e.text.lower() for e in entities_web_first)
    enhanced_texts = set(e.text.lower() for e in entities_enhanced)
    
    web_unique = web_texts - llm_texts
    enhanced_unique = enhanced_texts - llm_texts
    
    logger.info("\n" + "=" * 40)
    logger.info("UNIQUE DISCOVERIES")
    logger.info("=" * 40)
    
    if web_unique:
        logger.info(f"\nEntities found by Web-First but not LLM ({len(web_unique)}):")
        for text in list(web_unique)[:15]:
            logger.info(f"   - {text}")
    
    if enhanced_unique:
        logger.info(f"\nEntities found by Enhanced but not LLM ({len(enhanced_unique)}):")
        for text in list(enhanced_unique)[:15]:
            logger.info(f"   - {text}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Comparison Test Complete")
    logger.info("=" * 80)


async def main():
    """Run all tests."""
    try:
        # Test 1: Web-first extraction
        await test_web_first_extraction()
        
        # Add a delay between tests
        await asyncio.sleep(2)
        
        # Test 2: Method comparison
        await test_comparison()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())