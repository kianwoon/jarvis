#!/usr/bin/env python3
"""
Test comprehensive anti-silo improvements
Tests the enhanced entity recognition, validation, and linking to ensure
problematic silo entities are properly connected across documents.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any

from app.services.knowledge_graph_service import KnowledgeGraphExtractionService
from app.services.entity_linking_service import get_entity_linking_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.document_handlers.base import ExtractedChunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AntiSiloTester:
    def __init__(self):
        self.kg_service = KnowledgeGraphExtractionService()
        self.entity_linking_service = get_entity_linking_service()
        
    async def test_anti_silo_improvements(self):
        """Test that previously problematic silo entities are now properly linked"""
        
        # Test documents containing the problematic entities
        test_documents = [
            {
                "id": "test_doc_1",
                "content": """
                Apache Kafka is widely adopted in India for real-time data processing.
                PostgreSQL databases are being deployed across Indonesia by major banks.
                Temenos core banking platform serves financial institutions throughout Asia.
                Hadoop clusters process big data in Singapore and Hong Kong.
                Legacy mainframe systems are being modernized across East Asia.
                """,
                "title": "Technology Adoption in Asia"
            },
            {
                "id": "test_doc_2", 
                "content": """
                DBS Bank in Singapore uses PostgreSQL for transaction processing.
                India's fintech sector relies heavily on Apache Kafka for streaming.
                Temenos partnership with banks in Indonesia drives digital transformation.
                Hadoop deployments in Asia Pacific region continue to grow.
                Mainframe modernization projects span from Hong Kong to Thailand.
                """,
                "title": "Financial Technology Trends"
            },
            {
                "id": "test_doc_3",
                "content": """
                Tomorrow's banking will integrate Kafka, PostgreSQL, and Temenos.
                Treasury operations in Asia leverage big data platforms like Hadoop.
                Indonesia and India lead Southeast Asian digital transformation.
                Mainframe systems in Hong Kong banks are being upgraded.
                """,
                "title": "Future Banking Technology"
            }
        ]
        
        logger.info("🧪 TESTING COMPREHENSIVE ANTI-SILO IMPROVEMENTS")
        logger.info("=" * 60)
        
        all_entities = []
        all_relationships = []
        linking_results = []
        
        # Process each document
        for doc in test_documents:
            logger.info(f"\n📄 Processing Document: {doc['title']}")
            logger.info(f"Content: {doc['content'][:100]}...")
            
            try:
                # Create chunk and extract entities/relationships 
                chunk = ExtractedChunk(
                    content=doc['content'],
                    metadata={'title': doc['title'], 'document_id': doc['id']}
                )
                
                extraction_result = await self.kg_service.extract_from_chunk(chunk)
                entities = extraction_result.entities
                relationships = extraction_result.relationships
                
                logger.info(f"   ✅ Extracted {len(entities)} entities, {len(relationships)} relationships")
                
                # Test entity linking
                doc_linking_results = await self.entity_linking_service.link_entities_in_document(
                    entities, doc['id']
                )
                
                linked_count = sum(1 for r in doc_linking_results if not r.is_new_entity)
                logger.info(f"   🔗 Linked {linked_count}/{len(entities)} entities to existing entities")
                
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                linking_results.extend(doc_linking_results)
                
            except Exception as e:
                logger.error(f"   ❌ Failed to process document {doc['id']}: {e}")
        
        # Analyze results
        await self._analyze_anti_silo_results(all_entities, all_relationships, linking_results)
        
    async def _analyze_anti_silo_results(self, entities: List, relationships: List, linking_results: List):
        """Analyze the results to verify anti-silo improvements"""
        
        logger.info("\n📊 ANTI-SILO ANALYSIS RESULTS")
        logger.info("=" * 40)
        
        # Previously problematic entities to check
        problematic_entities = [
            'postgresql', 'indonesia', 'temenos', 'india', 'asia', 
            'hadoop', 'mainframe', 'tomorrow', 'treasury', 'singapore',
            'hong kong', 'dbs bank', 'kafka'
        ]
        
        # Track entity extraction
        extracted_entities = {}
        for entity in entities:
            name_lower = entity.canonical_form.lower()
            if name_lower not in extracted_entities:
                extracted_entities[name_lower] = []
            extracted_entities[name_lower].append(entity)
        
        logger.info(f"📋 Total unique entities extracted: {len(extracted_entities)}")
        
        # Check each problematic entity
        for prob_entity in problematic_entities:
            if prob_entity in extracted_entities:
                entity_instances = extracted_entities[prob_entity]
                logger.info(f"   ✅ '{prob_entity}': {len(entity_instances)} instances extracted")
                
                # Check if properly linked
                linked_instances = 0
                for instance in entity_instances:
                    for link_result in linking_results:
                        if (link_result.original_entity.canonical_form.lower() == prob_entity and 
                            not link_result.is_new_entity):
                            linked_instances += 1
                            break
                
                if linked_instances > 0:
                    logger.info(f"      🔗 {linked_instances} instances successfully linked (anti-silo working)")
                else:
                    logger.info(f"      ⚠️  No instances linked - potential silo")
            else:
                logger.info(f"   ❌ '{prob_entity}': Not extracted (recognition issue)")
        
        # Analyze relationship connectivity
        logger.info(f"\n🔄 Total relationships extracted: {len(relationships)}")
        
        # Check cross-entity relationships
        cross_entity_relationships = []
        for rel in relationships:
            source_clean = rel.source_entity.lower()
            target_clean = rel.target_entity.lower()
            
            # Check if relationship connects problematic entities
            if (source_clean in problematic_entities or target_clean in problematic_entities):
                cross_entity_relationships.append(rel)
        
        logger.info(f"🌐 Cross-entity relationships (connecting problematic entities): {len(cross_entity_relationships)}")
        
        # Sample some relationships
        if cross_entity_relationships:
            logger.info("   Sample relationships:")
            for i, rel in enumerate(cross_entity_relationships[:5]):
                logger.info(f"      {i+1}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
        
        # Check linking effectiveness
        total_entities = len(entities)
        total_linked = sum(1 for r in linking_results if not r.is_new_entity)
        linking_rate = (total_linked / total_entities * 100) if total_entities > 0 else 0
        
        logger.info(f"\n📈 LINKING EFFECTIVENESS:")
        logger.info(f"   Total entities: {total_entities}")
        logger.info(f"   Successfully linked: {total_linked}")
        logger.info(f"   Linking rate: {linking_rate:.1f}%")
        
        if linking_rate > 40:
            logger.info("   ✅ Good linking rate - anti-silo improvements working")
        elif linking_rate > 20:
            logger.info("   ⚠️  Moderate linking rate - some improvement but can be better")
        else:
            logger.info("   ❌ Low linking rate - anti-silo improvements need more work")
        
        # Check specific improvements
        logger.info(f"\n🎯 SPECIFIC IMPROVEMENTS CHECK:")
        
        # Geographic entity linking
        geographic_entities = ['india', 'indonesia', 'singapore', 'hong kong', 'asia']
        geographic_linked = 0
        for geo_entity in geographic_entities:
            for link_result in linking_results:
                if (link_result.original_entity.canonical_form.lower() == geo_entity and 
                    not link_result.is_new_entity):
                    geographic_linked += 1
                    break
        
        logger.info(f"   🌍 Geographic entities linked: {geographic_linked}/{len(geographic_entities)}")
        
        # Technology entity linking
        tech_entities = ['postgresql', 'kafka', 'hadoop', 'temenos', 'mainframe']
        tech_linked = 0
        for tech_entity in tech_entities:
            for link_result in linking_results:
                if (link_result.original_entity.canonical_form.lower() == tech_entity and 
                    not link_result.is_new_entity):
                    tech_linked += 1
                    break
        
        logger.info(f"   💻 Technology entities linked: {tech_linked}/{len(tech_entities)}")
        
        # Company entity linking
        company_entities = ['dbs bank', 'temenos']
        company_linked = 0
        for company_entity in company_entities:
            for link_result in linking_results:
                if (link_result.original_entity.canonical_form.lower() == company_entity and 
                    not link_result.is_new_entity):
                    company_linked += 1
                    break
        
        logger.info(f"   🏢 Company entities linked: {company_linked}/{len(company_entities)}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ANTI-SILO TESTING COMPLETE")

async def main():
    """Run the comprehensive anti-silo test"""
    try:
        tester = AntiSiloTester()
        await tester.test_anti_silo_improvements()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())