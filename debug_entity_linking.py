#!/usr/bin/env python3
"""
Debug entity linking issues
"""

import asyncio
import logging
from app.services.neo4j_service import get_neo4j_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_entity_linking():
    """Debug what entities exist and why linking isn't working"""
    
    neo4j_service = get_neo4j_service()
    
    if not neo4j_service.is_enabled():
        logger.error("Neo4j not enabled")
        return
    
    logger.info("🔍 DEBUGGING ENTITY LINKING")
    
    # Check what entities exist
    query = """
    MATCH (n) 
    RETURN n.id as id, n.name as name, n.type as type, labels(n) as labels
    ORDER BY n.name
    """
    
    results = neo4j_service.execute_cypher(query, {})
    
    logger.info(f"📋 EXISTING ENTITIES IN DATABASE:")
    for result in results:
        logger.info(f"   ID: {result['id']}, Name: '{result['name']}', Type: {result['type']}, Labels: {result['labels']}")
    
    # Test specific entity lookups
    test_entities = ['PostgreSQL', 'Apache Kafka', 'Hadoop', 'DBS Bank', 'Temenos']
    
    for entity_name in test_entities:
        logger.info(f"\n🔍 Testing lookup for: '{entity_name}'")
        
        # Test exact match
        exact_query = """
        MATCH (n)
        WHERE n.name = $name
        RETURN n.id as id, n.name as name, n.type as type, labels(n) as labels
        """
        exact_results = neo4j_service.execute_cypher(exact_query, {'name': entity_name})
        logger.info(f"   Exact match results: {len(exact_results)}")
        for result in exact_results:
            logger.info(f"      {result}")
        
        # Test type-based lookup
        type_query = """
        MATCH (n)
        WHERE (labels(n)[0] IN ['TECHNOLOGY', 'ORG', 'ORGANIZATION'] OR n.type IN ['TECHNOLOGY', 'ORG', 'ORGANIZATION'])
        AND n.name = $name
        RETURN n.id as id, n.name as name, n.type as type, labels(n) as labels
        """
        type_results = neo4j_service.execute_cypher(type_query, {'name': entity_name})
        logger.info(f"   Type-based match results: {len(type_results)}")
        for result in type_results:
            logger.info(f"      {result}")

if __name__ == "__main__":
    asyncio.run(debug_entity_linking())