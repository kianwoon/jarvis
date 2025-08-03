#!/usr/bin/env python3
"""
Script to fix existing entity types in Neo4j using the new classification logic
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.services.neo4j_service import get_neo4j_service
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fix_existing_entity_types():
    """Fix entity types for existing entities in Neo4j"""
    
    logger.info("🔧 Fixing existing entity types in Neo4j")
    
    # Initialize services
    extractor = LLMKnowledgeExtractor()
    neo4j_service = get_neo4j_service()
    
    if not neo4j_service.is_enabled():
        logger.error("❌ Neo4j service not enabled")
        return
    
    # Get all entities from Neo4j
    get_entities_query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
    RETURN n.id as id, n.name as name, n.type as current_type
    ORDER BY n.name
    """
    
    entities = neo4j_service.execute_cypher(get_entities_query)
    logger.info(f"📊 Found {len(entities)} entities to reclassify")
    
    if not entities:
        logger.info("No entities found in Neo4j")
        return
    
    # Track changes
    type_changes = {}
    updated_count = 0
    
    for entity in entities:
        entity_id = entity['id']
        entity_name = entity['name']
        current_type = entity.get('current_type', 'CONCEPT')
        
        # Get new type using enhanced classification
        new_type = extractor._classify_entity_type(entity_name)
        
        if new_type != current_type:
            logger.info(f"🔄 '{entity_name}': {current_type} -> {new_type}")
            
            # Update the entity type in Neo4j
            update_query = """
            MATCH (n)
            WHERE n.id = $entity_id
            SET n.type = $new_type
            RETURN n.name as name, n.type as updated_type
            """
            
            try:
                result = neo4j_service.execute_cypher(update_query, {
                    'entity_id': entity_id,
                    'new_type': new_type
                })
                
                if result:
                    updated_count += 1
                    
                    # Track changes
                    change_key = f"{current_type} -> {new_type}"
                    if change_key not in type_changes:
                        type_changes[change_key] = []
                    type_changes[change_key].append(entity_name)
                
            except Exception as e:
                logger.error(f"❌ Failed to update {entity_name}: {e}")
        
        else:
            logger.debug(f"✅ '{entity_name}': {current_type} (no change needed)")
    
    logger.info(f"\n🎯 Update Summary:")
    logger.info(f"   Total entities processed: {len(entities)}")
    logger.info(f"   Entities updated: {updated_count}")
    logger.info(f"   Entities unchanged: {len(entities) - updated_count}")
    
    if type_changes:
        logger.info(f"\n📊 Type Changes Made:")
        for change, entity_names in type_changes.items():
            logger.info(f"   {change}: {len(entity_names)} entities")
            for name in entity_names[:3]:  # Show first 3 examples
                logger.info(f"      - {name}")
            if len(entity_names) > 3:
                logger.info(f"      ... and {len(entity_names) - 3} more")
    
    # Verify the changes
    logger.info("\n🔍 Verifying updated type distribution...")
    
    type_distribution_query = """
    MATCH (n)
    WHERE n.name IS NOT NULL
    WITH n.type as entity_type, count(n) as count
    RETURN entity_type, count
    ORDER BY count DESC
    """
    
    type_results = neo4j_service.execute_cypher(type_distribution_query)
    
    logger.info("📊 New Type Distribution:")
    for result in type_results:
        logger.info(f"   {result['entity_type']}: {result['count']} entities")
    
    unique_types = len(type_results)
    logger.info(f"\n🎉 Result: {unique_types} unique entity types in Neo4j")
    
    if unique_types >= 4:
        logger.info("✅ SUCCESS: Entity type diversity achieved - nodes should now show different colors!")
    else:
        logger.warning("⚠️  Still limited type diversity - may need more classification rules")

if __name__ == "__main__":
    asyncio.run(fix_existing_entity_types())