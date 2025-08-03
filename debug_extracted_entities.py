#!/usr/bin/env python3
"""
Debug extracted entities to see what's being passed to entity linking
"""

import asyncio
import logging
from app.services.knowledge_graph_service import KnowledgeGraphExtractionService
from app.services.entity_linking_service import get_entity_linking_service
from app.document_handlers.base import ExtractedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_extracted_entities():
    """Debug what entities are being extracted vs what's in the database"""
    
    kg_service = KnowledgeGraphExtractionService()
    entity_linking_service = get_entity_linking_service()
    
    logger.info("🔍 DEBUGGING EXTRACTED ENTITIES")
    
    test_content = """
    DBS Bank uses PostgreSQL databases for core banking operations.
    Apache Kafka streaming is deployed across Singapore and Indonesia.
    Temenos platform integrates with Hadoop clusters in India.
    """
    
    logger.info(f"📝 Test content: {test_content.strip()}")
    
    # Extract entities
    chunk = ExtractedChunk(
        content=test_content,
        metadata={'title': 'Test', 'document_id': 'debug_test'}
    )
    
    extraction_result = await kg_service.extract_from_chunk(chunk)
    entities = extraction_result.entities
    
    logger.info(f"\n📊 EXTRACTED ENTITIES:")
    for i, entity in enumerate(entities):
        logger.info(f"   {i+1}. Name: '{entity.canonical_form}', Label: {entity.label}, Text: '{entity.text}'")
    
    # Test entity linking for each entity
    logger.info(f"\n🔗 TESTING ENTITY LINKING:")
    for entity in entities:
        logger.info(f"\n   Testing: '{entity.canonical_form}' (type: {entity.label})")
        
        # Get compatible types
        compatible_types = entity_linking_service._get_compatible_types(entity.label)
        logger.info(f"      Compatible types: {compatible_types}")
        
        # Test finding candidates
        candidates = await entity_linking_service._find_entity_candidates(entity)
        logger.info(f"      Candidates found: {len(candidates)}")
        
        for candidate in candidates:
            logger.info(f"         - ID: {candidate.entity_id}, Name: '{candidate.entity_name}', Score: {candidate.similarity_score:.3f}")

if __name__ == "__main__":
    asyncio.run(debug_extracted_entities())