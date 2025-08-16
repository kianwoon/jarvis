#!/usr/bin/env python3
"""
Test script for the enhanced radiating entity extraction with web search integration.

This script tests the UniversalEntityExtractor's ability to:
1. Extract entities using LLM
2. Trigger web search for temporal queries
3. Discover latest AI/LLM technologies
4. Merge and deduplicate entities from both sources
"""

import asyncio
import logging
from typing import List
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_extraction():
    """Test basic entity extraction without web search."""
    from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
    
    extractor = UniversalEntityExtractor()
    
    query = "What are the main components of a RAG system?"
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST 1: Basic extraction (no web search expected)")
    logger.info(f"Query: {query}")
    logger.info(f"{'='*60}")
    
    entities = await extractor.extract_entities(query)
    
    logger.info(f"Found {len(entities)} entities:")
    for entity in entities[:10]:
        logger.info(f"  - {entity.text} ({entity.entity_type}): {entity.confidence:.2f}")
    
    return entities


async def test_web_search_triggered():
    """Test entity extraction with web search for latest technologies."""
    from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
    
    extractor = UniversalEntityExtractor()
    
    query = "What are the latest open source LLM frameworks and tools in 2024?"
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST 2: Web search triggered extraction")
    logger.info(f"Query: {query}")
    logger.info(f"{'='*60}")
    
    entities = await extractor.extract_entities_with_web_search(query)
    
    logger.info(f"Found {len(entities)} entities (with web search):")
    
    # Separate entities by source
    llm_entities = [e for e in entities if not hasattr(e, 'metadata') or e.metadata.get('source') != 'web_search']
    web_entities = [e for e in entities if hasattr(e, 'metadata') and e.metadata.get('source') == 'web_search']
    
    logger.info(f"\nLLM-extracted entities ({len(llm_entities)}):")
    for entity in llm_entities[:10]:
        logger.info(f"  - {entity.text} ({entity.entity_type}): {entity.confidence:.2f}")
    
    logger.info(f"\nWeb-discovered entities ({len(web_entities)}):")
    for entity in web_entities[:10]:
        logger.info(f"  - {entity.text} ({entity.entity_type}): {entity.confidence:.2f}")
    
    return entities


async def test_forced_web_search():
    """Test forcing web search even when not automatically triggered."""
    from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
    
    extractor = UniversalEntityExtractor()
    
    query = "Popular AI frameworks for building applications"
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST 3: Forced web search")
    logger.info(f"Query: {query}")
    logger.info(f"{'='*60}")
    
    # First without forcing
    entities_no_force = await extractor.extract_entities_with_web_search(query, force_web_search=False)
    
    # Then with forcing
    entities_forced = await extractor.extract_entities_with_web_search(query, force_web_search=True)
    
    logger.info(f"Without force: {len(entities_no_force)} entities")
    logger.info(f"With force: {len(entities_forced)} entities")
    
    # Show the difference
    if len(entities_forced) > len(entities_no_force):
        logger.info("\nAdditional entities from forced web search:")
        forced_texts = {e.text.lower() for e in entities_forced}
        no_force_texts = {e.text.lower() for e in entities_no_force}
        new_entities = forced_texts - no_force_texts
        for text in list(new_entities)[:10]:
            logger.info(f"  - {text}")
    
    return entities_forced


async def test_comprehensive_technology_query():
    """Test comprehensive technology query that should return many entities."""
    from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
    
    extractor = UniversalEntityExtractor()
    
    query = "What are the essential technologies and tools for building modern LLM applications in 2024?"
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST 4: Comprehensive technology query")
    logger.info(f"Query: {query}")
    logger.info(f"{'='*60}")
    
    entities = await extractor.extract_entities_with_web_search(query)
    
    logger.info(f"Found {len(entities)} entities total")
    
    # Group by entity type
    entity_types = {}
    for entity in entities:
        entity_type = entity.entity_type
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity)
    
    logger.info("\nEntities by type:")
    for entity_type, type_entities in entity_types.items():
        logger.info(f"\n{entity_type} ({len(type_entities)}):")
        for entity in type_entities[:5]:
            source = entity.metadata.get('source', 'llm') if hasattr(entity, 'metadata') else 'llm'
            logger.info(f"  - {entity.text} [{source}]: {entity.confidence:.2f}")
    
    return entities


async def test_specific_domain_search():
    """Test web search for specific domain (e.g., vector databases)."""
    from app.services.radiating.extraction.universal_entity_extractor import UniversalEntityExtractor
    
    extractor = UniversalEntityExtractor()
    
    query = "What are the latest vector databases for RAG systems in 2024?"
    domain_hints = ["vector_database", "rag"]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TEST 5: Domain-specific search")
    logger.info(f"Query: {query}")
    logger.info(f"Domain hints: {domain_hints}")
    logger.info(f"{'='*60}")
    
    entities = await extractor.extract_entities_with_web_search(
        query,
        domain_hints=domain_hints
    )
    
    # Filter for vector database entities
    vector_db_entities = [
        e for e in entities 
        if 'vector' in e.entity_type.lower() or 'database' in e.entity_type.lower()
    ]
    
    logger.info(f"Found {len(entities)} total entities")
    logger.info(f"Found {len(vector_db_entities)} vector database entities:")
    
    for entity in vector_db_entities[:15]:
        source = entity.metadata.get('source', 'llm') if hasattr(entity, 'metadata') else 'llm'
        logger.info(f"  - {entity.text} ({entity.entity_type}) [{source}]: {entity.confidence:.2f}")
    
    return entities


async def main():
    """Run all tests."""
    logger.info("Starting Radiating Web Search Integration Tests")
    logger.info("=" * 80)
    
    try:
        # Test 1: Basic extraction
        await test_basic_extraction()
        
        # Test 2: Web search triggered
        await test_web_search_triggered()
        
        # Test 3: Forced web search
        await test_forced_web_search()
        
        # Test 4: Comprehensive query
        await test_comprehensive_technology_query()
        
        # Test 5: Domain-specific search
        await test_specific_domain_search()
        
        logger.info("\n" + "=" * 80)
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())