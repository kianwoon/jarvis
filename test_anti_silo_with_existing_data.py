#!/usr/bin/env python3
"""
Test anti-silo improvements with existing data
Creates initial entities and then tests linking functionality
"""

import asyncio
import logging
from typing import List, Dict, Any

from app.services.knowledge_graph_service import KnowledgeGraphExtractionService
from app.services.entity_linking_service import get_entity_linking_service
from app.services.neo4j_service import get_neo4j_service
from app.document_handlers.base import ExtractedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_anti_silo_with_existing_data():
    """Test anti-silo improvements with existing entities in the graph"""
    
    kg_service = KnowledgeGraphExtractionService()
    entity_linking_service = get_entity_linking_service()
    neo4j_service = get_neo4j_service()
    
    logger.info("🔧 SETTING UP TEST DATA")
    
    # Clear existing data
    if neo4j_service.is_enabled():
        neo4j_service.execute_cypher("MATCH (n) DETACH DELETE n", {})
        logger.info("   ✅ Cleared existing Neo4j data")
        
        # Create some initial entities that should be linked to
        initial_entities = [
            "CREATE (n:TECHNOLOGY {id: 'tech_1', name: 'PostgreSQL', type: 'TECHNOLOGY', confidence: 0.9})",
            "CREATE (n:TECHNOLOGY {id: 'tech_2', name: 'Apache Kafka', type: 'TECHNOLOGY', confidence: 0.9})",
            "CREATE (n:TECHNOLOGY {id: 'tech_3', name: 'Hadoop', type: 'TECHNOLOGY', confidence: 0.9})", 
            "CREATE (n:LOCATION {id: 'loc_1', name: 'Singapore', type: 'LOCATION', confidence: 0.9})",
            "CREATE (n:LOCATION {id: 'loc_2', name: 'Indonesia', type: 'LOCATION', confidence: 0.9})",
            "CREATE (n:LOCATION {id: 'loc_3', name: 'India', type: 'LOCATION', confidence: 0.9})",
            "CREATE (n:ORGANIZATION {id: 'org_1', name: 'DBS Bank', type: 'ORGANIZATION', confidence: 0.9})",
            "CREATE (n:ORGANIZATION {id: 'org_2', name: 'Temenos', type: 'ORGANIZATION', confidence: 0.9})"
        ]
        
        for query in initial_entities:
            neo4j_service.execute_cypher(query, {})
        
        logger.info(f"   ✅ Created {len(initial_entities)} initial entities for linking tests")
    
    logger.info("\n🧪 TESTING ANTI-SILO WITH EXISTING DATA")
    logger.info("=" * 50)
    
    # Test document with entities that should link to existing ones
    test_doc = {
        "id": "test_linking",
        "content": """
        DBS Bank uses PostgreSQL databases for core banking operations.
        Apache Kafka streaming is deployed across Singapore and Indonesia.
        Temenos platform integrates with Hadoop clusters in India.
        """,
        "title": "Banking Technology Integration"
    }
    
    logger.info(f"📄 Processing: {test_doc['title']}")
    
    # Create chunk and extract
    chunk = ExtractedChunk(
        content=test_doc['content'],
        metadata={'title': test_doc['title'], 'document_id': test_doc['id']}
    )
    
    extraction_result = await kg_service.extract_from_chunk(chunk)
    entities = extraction_result.entities
    relationships = extraction_result.relationships
    
    logger.info(f"   ✅ Extracted {len(entities)} entities, {len(relationships)} relationships")
    
    # Test entity linking
    linking_results = await entity_linking_service.link_entities_in_document(
        entities, test_doc['id']
    )
    
    # Analyze linking results
    linked_count = sum(1 for r in linking_results if not r.is_new_entity)
    logger.info(f"   🔗 Linked {linked_count}/{len(entities)} entities to existing entities")
    
    logger.info("\n📊 DETAILED LINKING ANALYSIS:")
    for result in linking_results:
        entity_name = result.original_entity.canonical_form
        if not result.is_new_entity:
            logger.info(f"   ✅ '{entity_name}' → LINKED (confidence: {result.linking_confidence:.2f})")
        else:
            logger.info(f"   ❌ '{entity_name}' → NEW ENTITY ({result.reasoning})")
    
    # Test storing in Neo4j to create cross-document relationships
    if neo4j_service.is_enabled():
        logger.info("\n💾 STORING EXTRACTION RESULTS...")
        store_result = await kg_service.store_in_neo4j(extraction_result, test_doc['id'])
        logger.info(f"   ✅ Stored: {store_result.get('entities_created', 0)} entities, {store_result.get('relationships_created', 0)} relationships")
        
        # Check for anti-silo relationships
        anti_silo_result = await kg_service._discover_anti_silo_relationships(
            [e.canonical_form for e in entities], test_doc['id']
        )
        logger.info(f"   🔗 Created {len(anti_silo_result)} anti-silo relationships")
        
        # Final verification - count total entities and relationships
        entity_count_query = "MATCH (n) RETURN count(n) as total"
        relationship_count_query = "MATCH ()-[r]->() RETURN count(r) as total"
        
        entity_count = neo4j_service.execute_cypher(entity_count_query, {})[0]['total']
        relationship_count = neo4j_service.execute_cypher(relationship_count_query, {})[0]['total']
        
        logger.info(f"\n🎯 FINAL GRAPH STATE:")
        logger.info(f"   Total entities: {entity_count}")
        logger.info(f"   Total relationships: {relationship_count}")
        
        # Check connectivity of problematic entities
        problematic_entities = ['PostgreSQL', 'Indonesia', 'Temenos', 'India', 'DBS Bank', 'Apache Kafka', 'Hadoop', 'Singapore']
        connected_entities = 0
        
        for entity in problematic_entities:
            connectivity_query = """
            MATCH (n {name: $name})-[r]-()
            RETURN count(r) as connections
            """
            result = neo4j_service.execute_cypher(connectivity_query, {'name': entity})
            connections = result[0]['connections'] if result else 0
            
            if connections > 0:
                connected_entities += 1
                logger.info(f"   ✅ '{entity}': {connections} connections")
            else:
                logger.info(f"   ❌ '{entity}': No connections (silo)")
        
        connectivity_rate = (connected_entities / len(problematic_entities) * 100)
        logger.info(f"\n🏆 ANTI-SILO SUCCESS RATE: {connectivity_rate:.1f}%")
        
        if connectivity_rate >= 80:
            logger.info("   🎉 EXCELLENT: Anti-silo improvements working very well!")
        elif connectivity_rate >= 60:
            logger.info("   ✅ GOOD: Anti-silo improvements working well!")
        elif connectivity_rate >= 40:
            logger.info("   ⚠️ MODERATE: Some anti-silo improvements working")
        else:
            logger.info("   ❌ POOR: Anti-silo improvements need more work")

if __name__ == "__main__":
    asyncio.run(test_anti_silo_with_existing_data())