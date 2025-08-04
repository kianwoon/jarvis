#!/usr/bin/env python3
"""
Test script to verify knowledge graph extraction balance improvements
Tests entity and relationship extraction ratios
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the app directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.knowledge_graph_service import KnowledgeGraphExtractionService, get_knowledge_graph_service
from services.neo4j_service import get_neo4j_service
from document_handlers.base import ExtractedChunk
from core.knowledge_graph_settings_cache import get_knowledge_graph_settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test document - a rich business strategy document
TEST_DOCUMENT = """
DBS Bank Digital Transformation Strategy 2024-2027

Executive Summary:
DBS Bank, headquartered in Singapore, is embarking on an ambitious digital transformation journey. 
The bank's CEO, Piyush Gupta, has outlined a comprehensive strategy involving multiple technology partners 
including Microsoft Azure, Amazon Web Services (AWS), Google Cloud Platform, and IBM Watson.

Key Strategic Initiatives:

1. Cloud Migration Program
   - Partnership with Microsoft Azure for core banking infrastructure
   - Migration timeline: Q1 2024 to Q4 2025
   - Budget allocation: $450 million USD
   - Project lead: Sarah Chen, CTO of DBS Bank
   - Supporting vendors: Accenture, Deloitte, PwC

2. AI and Machine Learning Implementation
   - Collaboration with Google Cloud AI and IBM Watson
   - Focus areas: Credit risk assessment, fraud detection, customer service
   - AI Lab established in Singapore, Hong Kong, and Shanghai
   - Research partnership with National University of Singapore (NUS)
   - Team size: 200+ data scientists and ML engineers

3. Blockchain and Digital Assets
   - Strategic alliance with Ant Group and Tencent
   - Development of DBS Digital Exchange (DDEx)
   - Support for Bitcoin, Ethereum, and enterprise blockchain solutions
   - Partnership with R3 Corda and Hyperledger Fabric
   - Regulatory collaboration with Monetary Authority of Singapore (MAS)

4. API Banking Ecosystem
   - Open banking platform development with Visa and Mastercard
   - Integration with fintech partners: Grab, Gojek, Sea Group
   - API gateway powered by Kong and MuleSoft
   - Developer portal launched for third-party integrations

5. Cybersecurity Enhancement
   - Partnership with Palo Alto Networks and CrowdStrike
   - Implementation of zero-trust architecture
   - Security operations center (SOC) in Singapore
   - Collaboration with Interpol and regional cybersecurity agencies

Regional Expansion:
- Indonesia: Partnership with Bank Mandiri and Bank BCA
- Thailand: Collaboration with Kasikorn Bank and Siam Commercial Bank
- Vietnam: Joint venture with Vietcombank
- India: Technology partnership with HDFC Bank and ICICI Bank
- China: Fintech collaboration through DBS China subsidiary

Technology Stack:
- Core Banking: Temenos T24 on Oracle Database
- Data Platform: Snowflake, Databricks, Apache Kafka
- Analytics: Tableau, Power BI, Qlik Sense
- DevOps: Kubernetes, Docker, Jenkins, GitLab
- Monitoring: Datadog, New Relic, Splunk

Financial Projections:
- Total investment: $2.5 billion over 4 years
- Expected ROI: 35% by 2027
- Cost savings: $800 million annually through automation
- Revenue growth: 25% CAGR in digital channels

Leadership Team:
- Piyush Gupta - CEO
- Sarah Chen - CTO
- Michael Tan - Head of Digital Banking
- Jennifer Lee - Chief Data Officer
- Robert Kumar - CISO
- Amanda Wong - Head of Innovation

This transformation positions DBS Bank as a leading digital bank in Asia-Pacific, competing with 
traditional banks like HSBC, Standard Chartered, and Citibank, as well as digital challengers 
like Revolut, N26, and local neobanks.
"""

async def test_extraction_balance():
    """Test the balance of entity and relationship extraction"""
    logger.info("🚀 Testing Knowledge Graph Extraction Balance")
    logger.info("=" * 70)
    
    try:
        # Get services
        kg_service = get_knowledge_graph_service()
        neo4j_service = get_neo4j_service()
        
        # Verify Neo4j connection
        if not neo4j_service.is_enabled():
            logger.error("❌ Neo4j is not enabled")
            return False
            
        # Get current settings
        settings = get_knowledge_graph_settings()
        extraction_mode = settings.get('extraction', {}).get('mode', 'standard')
        logger.info(f"📊 Current extraction mode: {extraction_mode}")
        
        # Clear any existing test data
        logger.info("🧹 Clearing previous test data...")
        with neo4j_service.driver.session() as session:
            session.run("MATCH (n) WHERE n.document_id = 'test_balance_doc' DETACH DELETE n")
        
        # Create test chunk
        test_chunk = ExtractedChunk(
            chunk_id="test_chunk_001",
            text=TEST_DOCUMENT,
            metadata={
                "document_id": "test_balance_doc",
                "chunk_index": 0,
                "total_chunks": 1,
                "document_type": "business_strategy"
            }
        )
        
        logger.info(f"📄 Test document size: {len(TEST_DOCUMENT):,} characters")
        logger.info("🔍 Starting extraction...")
        
        # Extract knowledge graph
        start_time = datetime.now()
        extraction_result = await kg_service.extract_from_chunk(test_chunk, "test_balance_doc")
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        entity_count = len(extraction_result.entities)
        relationship_count = len(extraction_result.relationships)
        ratio = relationship_count / max(entity_count, 1)
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 EXTRACTION RESULTS")
        logger.info("=" * 70)
        logger.info(f"⏱️  Extraction time: {extraction_time:.1f} seconds")
        logger.info(f"📍 Entities extracted: {entity_count}")
        logger.info(f"🔗 Relationships extracted: {relationship_count}")
        logger.info(f"📈 Relationship/Entity ratio: {ratio:.1f}")
        logger.info(f"📏 Entity density: {entity_count / (len(TEST_DOCUMENT) / 1000):.1f} per 1K chars")
        
        # Display entity breakdown by type
        entity_types = {}
        for entity in extraction_result.entities:
            entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
        
        logger.info("\n📍 Entity Breakdown by Type:")
        for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   - {entity_type}: {count}")
        
        # Display relationship breakdown by type
        relationship_types = {}
        for rel in extraction_result.relationships:
            relationship_types[rel.relationship_type] = relationship_types.get(rel.relationship_type, 0) + 1
        
        logger.info("\n🔗 Relationship Breakdown by Type:")
        for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   - {rel_type}: {count}")
        
        # Check for co-occurrence relationships
        cooccurrence_count = sum(1 for rel in extraction_result.relationships 
                                if rel.properties.get('inference_method') == 'cooccurrence')
        logger.info(f"\n💡 Co-occurrence relationships: {cooccurrence_count} ({cooccurrence_count/max(relationship_count,1)*100:.1f}%)")
        
        # Store in Neo4j to verify
        logger.info("\n💾 Storing in Neo4j...")
        storage_result = await kg_service.store_in_neo4j(extraction_result, "test_balance_doc")
        
        if storage_result['success']:
            logger.info(f"✅ Stored {storage_result['entities_stored']} entities")
            logger.info(f"✅ Stored {storage_result['relationships_stored']} relationships")
            if storage_result.get('relationship_failures', 0) > 0:
                logger.warning(f"⚠️  Failed relationships: {storage_result['relationship_failures']}")
        else:
            logger.error(f"❌ Storage failed: {storage_result.get('error')}")
        
        # Evaluate balance
        logger.info("\n" + "=" * 70)
        logger.info("🎯 BALANCE EVALUATION")
        logger.info("=" * 70)
        
        if entity_count < 40:
            logger.warning(f"⚠️  UNDER-EXTRACTION: Only {entity_count} entities (expected 60-100 for this document)")
        elif entity_count > 150:
            logger.warning(f"⚠️  OVER-EXTRACTION: {entity_count} entities (expected 60-100)")
        else:
            logger.info(f"✅ GOOD ENTITY COUNT: {entity_count} entities")
        
        if ratio > 5:
            logger.warning(f"⚠️  TOO MANY RELATIONSHIPS: {ratio:.1f} per entity (healthy: 2-4)")
        elif ratio < 1:
            logger.warning(f"⚠️  TOO FEW RELATIONSHIPS: {ratio:.1f} per entity (healthy: 2-4)")
        else:
            logger.info(f"✅ HEALTHY RATIO: {ratio:.1f} relationships per entity")
        
        # Overall assessment
        is_balanced = (40 <= entity_count <= 150) and (1 <= ratio <= 5)
        
        if is_balanced:
            logger.info("\n🎉 EXTRACTION IS WELL BALANCED!")
        else:
            logger.info("\n❌ EXTRACTION NEEDS FURTHER TUNING")
            logger.info("\nRecommendations:")
            if entity_count < 40:
                logger.info("- Lower entity confidence thresholds")
                logger.info("- Reduce entity deduplication aggressiveness")
                logger.info("- Enhance entity extraction prompts")
            if ratio > 5:
                logger.info("- Increase relationship confidence thresholds")
                logger.info("- Reduce co-occurrence generation")
                logger.info("- Apply stricter relationship limits")
        
        # Cleanup test data
        logger.info("\n🧹 Cleaning up test data...")
        with neo4j_service.driver.session() as session:
            result = session.run("MATCH (n) WHERE n.document_id = 'test_balance_doc' DETACH DELETE n")
            summary = result.consume()
            logger.info(f"✅ Deleted {summary.counters.nodes_deleted} test nodes")
        
        return is_balanced
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.exception("Full error details:")
        return False

async def main():
    """Main test execution"""
    success = await test_extraction_balance()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)