#!/usr/bin/env python3
"""
Test script to verify entity extraction limits are working correctly.
Tests both web search and LLM extraction paths.
"""

import asyncio
import logging
from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_entity_extraction():
    """Test entity extraction with different methods"""
    
    # Initialize extractor
    extractor = UniversalEntityExtractor()
    
    # Test query
    query = "What are the latest LLM frameworks and tools for building AI applications in 2024?"
    
    logger.info(f"Testing query: {query}")
    logger.info("=" * 60)
    
    # Test 1: Standard extraction (should use LLM since web search is failing)
    logger.info("\n1. Testing standard extraction with prefer_web_search=False")
    entities = await extractor.extract_entities(query, prefer_web_search=False)
    logger.info(f"   Found {len(entities)} entities")
    if entities:
        logger.info("   Sample entities:")
        for entity in entities[:10]:
            source = entity.metadata.get('source', 'unknown') if hasattr(entity, 'metadata') else 'llm'
            logger.info(f"   - {entity.text} ({entity.entity_type}) [confidence: {entity.confidence:.2f}, source: {source}]")
    
    # Test 2: Web-first extraction with fallback
    logger.info("\n2. Testing web-first extraction (should fallback to LLM if web fails)")
    entities_web = await extractor.extract_entities_web_first(query)
    logger.info(f"   Found {len(entities_web)} entities")
    if entities_web:
        logger.info("   Sample entities:")
        for entity in entities_web[:10]:
            source = entity.metadata.get('source', 'unknown') if hasattr(entity, 'metadata') else 'llm'
            logger.info(f"   - {entity.text} ({entity.entity_type}) [confidence: {entity.confidence:.2f}, source: {source}]")
    
    # Test 3: Enhanced extraction with web search
    logger.info("\n3. Testing enhanced extraction with web search")
    entities_enhanced = await extractor.extract_entities_with_web_search(query, force_web_search=True)
    logger.info(f"   Found {len(entities_enhanced)} entities")
    if entities_enhanced:
        logger.info("   Sample entities:")
        for entity in entities_enhanced[:10]:
            source = entity.metadata.get('source', 'unknown') if hasattr(entity, 'metadata') else 'llm'
            logger.info(f"   - {entity.text} ({entity.entity_type}) [confidence: {entity.confidence:.2f}, source: {source}]")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY:")
    logger.info(f"Standard extraction: {len(entities)} entities")
    logger.info(f"Web-first extraction: {len(entities_web)} entities")
    logger.info(f"Enhanced extraction: {len(entities_enhanced)} entities")
    
    # Check if limits are working
    if len(entities) >= 20:
        logger.info("✅ Entity limits increased successfully (>= 20 entities)")
    else:
        logger.warning(f"⚠️ Entity count still low: {len(entities)} entities")

if __name__ == "__main__":
    asyncio.run(test_entity_extraction())