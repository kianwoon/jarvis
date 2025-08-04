#!/usr/bin/env python3
"""
Simple Knowledge Graph Ingestion Test

Tests only the most critical parts of knowledge graph ingestion.
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

import asyncio
import logging
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.services.knowledge_graph_service import get_knowledge_graph_service
from app.services.neo4j_service import get_neo4j_service
from app.document_handlers.base import ExtractedChunk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_ingestion():
    """Test basic knowledge graph ingestion"""
    
    logger.info("🚀 Simple Knowledge Graph Ingestion Test")
    logger.info("=" * 50)
    
    # 1. Test configuration loading
    logger.info("1️⃣ Testing configuration...")
    try:
        kg_settings = get_knowledge_graph_settings()
        logger.info(f"✅ Config loaded: mode={kg_settings.get('mode', 'unknown')}")
        logger.info(f"   Max entities: {kg_settings.get('max_entities_per_chunk', 'unknown')}")
        logger.info(f"   Max relationships: {kg_settings.get('max_relationships_per_chunk', 'unknown')}")
    except Exception as e:
        logger.error(f"❌ Config failed: {e}")
        return False
    
    # 2. Test Neo4j connectivity
    logger.info("\n2️⃣ Testing Neo4j...")
    try:
        neo4j_service = get_neo4j_service()
        if not neo4j_service.is_enabled():
            logger.error("❌ Neo4j not enabled")
            return False
        
        entity_count = neo4j_service.get_total_entity_count()
        relationship_count = neo4j_service.get_total_relationship_count()
        logger.info(f"✅ Neo4j connected: {entity_count} entities, {relationship_count} relationships")
    except Exception as e:
        logger.error(f"❌ Neo4j failed: {e}")
        return False
    
    # 3. Test entity extraction
    logger.info("\n3️⃣ Testing extraction...")
    test_text = """
    DBS Bank is evaluating OceanBase database technology for core banking systems.
    The bank is considering migration from PostgreSQL to improve performance.
    Singapore is the primary market for this digital transformation initiative.
    """
    
    try:
        test_chunk = ExtractedChunk(
            content=test_text,
            metadata={'chunk_id': 'test_001', 'source': 'test.txt'}
        )
        
        kg_service = get_knowledge_graph_service()
        result = await kg_service.extract_from_chunk(test_chunk, "test_doc")
        
        logger.info(f"✅ Extraction done: {len(result.entities)} entities, {len(result.relationships)} relationships")
        
        # Show extracted entities
        if result.entities:
            logger.info("   Entities:")
            for entity in result.entities[:5]:  # Show first 5
                logger.info(f"     - {entity.canonical_form} ({entity.label})")
        
        # Show relationships
        if result.relationships:
            logger.info("   Relationships:")
            for rel in result.relationships[:3]:  # Show first 3
                logger.info(f"     - {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
        
        return len(result.entities) > 0
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return False

async def test_storage():
    """Test Neo4j storage"""
    logger.info("\n4️⃣ Testing storage...")
    
    try:
        # First extract
        test_chunk = ExtractedChunk(
            content="DBS Bank uses advanced technology for digital banking services.",
            metadata={'chunk_id': 'test_storage', 'source': 'storage_test.txt'}
        )
        
        kg_service = get_knowledge_graph_service()
        extraction_result = await kg_service.extract_from_chunk(test_chunk, "storage_test")
        
        if len(extraction_result.entities) == 0:
            logger.warning("⚠️  No entities to store")
            return True
        
        # Get pre-storage counts
        neo4j_service = get_neo4j_service()
        pre_entities = neo4j_service.get_total_entity_count()
        pre_relationships = neo4j_service.get_total_relationship_count()
        
        # Store in Neo4j
        storage_result = await kg_service.store_in_neo4j(extraction_result, "storage_test")
        
        # Get post-storage counts
        post_entities = neo4j_service.get_total_entity_count()
        post_relationships = neo4j_service.get_total_relationship_count()
        
        entities_added = post_entities - pre_entities
        relationships_added = post_relationships - pre_relationships
        
        logger.info(f"✅ Storage result: {storage_result.get('success', False)}")
        logger.info(f"   Added: {entities_added} entities, {relationships_added} relationships")
        
        return storage_result.get('success', False)
        
    except Exception as e:
        logger.error(f"❌ Storage failed: {e}")
        return False

async def main():
    """Run simple tests"""
    
    # Test basic ingestion
    extraction_works = await test_basic_ingestion()
    
    # Test storage if extraction works
    storage_works = False
    if extraction_works:
        storage_works = await test_storage()
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("🎯 TEST SUMMARY")
    logger.info(f"   Extraction: {'✅ WORKING' if extraction_works else '❌ BROKEN'}")
    logger.info(f"   Storage: {'✅ WORKING' if storage_works else '❌ BROKEN'}")
    
    if extraction_works and storage_works:
        logger.info("🎉 KNOWLEDGE GRAPH INGESTION IS WORKING!")
    elif extraction_works:
        logger.info("⚠️  PARTIAL SUCCESS: Extraction works, storage has issues")
    else:
        logger.info("💥 CRITICAL FAILURE: Basic ingestion is broken")
    
    return extraction_works and storage_works

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)