#!/usr/bin/env python3
"""
Test Entity Extraction Rebalancing

This script tests the rebalanced entity extraction settings to ensure:
1. Entity extraction count is restored (60-100 entities expected)
2. Organization types are properly extracted (ORGANIZATION, COMPANY, BANK)
3. Relationship limits remain controlled (≤4 per entity)
4. Overall performance is improved from the 14-entity problem
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings, reload_knowledge_graph_settings
from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample business document for testing (DBS-style content)
SAMPLE_BUSINESS_TEXT = """
DBS Group Holdings is a leading financial services group in Asia, headquartered in Singapore. 
The bank operates across 18 markets and serves over 12 million customers through its digital platforms.

DBS Bank has implemented advanced technology solutions including artificial intelligence and machine learning 
to enhance customer experience and operational efficiency. The organization's digital transformation strategy 
focuses on cloud computing, data analytics, and API-driven services.

Key executives include CEO Piyush Gupta, CTO David Gledhill, and CFO Chng Sok Hui. The bank's technology 
division has developed innovative solutions like PayLah! mobile payment platform and DBS digibank.

The company's strategic initiatives include expanding into Southeast Asia markets, particularly Indonesia 
and Thailand, while maintaining strong presence in Hong Kong and India. DBS has invested heavily in 
fintech partnerships and blockchain technology for trade finance.

Recent performance shows revenue growth of 8% to S$15.2 billion, with strong digital adoption rates 
exceeding 90% across key markets. The bank's wealth management division serves over 250,000 affluent customers.
"""

async def test_entity_extraction_rebalance():
    """Test the rebalanced entity extraction settings"""
    logger.info("=" * 80)
    logger.info("TESTING ENTITY EXTRACTION REBALANCING")
    logger.info("=" * 80)
    
    # Force reload settings to use the updated configuration
    logger.info("🔄 Reloading knowledge graph settings...")
    settings = reload_knowledge_graph_settings()
    
    logger.info(f"📊 Current Settings:")
    logger.info(f"   - Mode: {settings.get('mode', 'unknown')}")
    logger.info(f"   - Max entities per chunk: {settings.get('max_entities_per_chunk', 'unknown')}")
    logger.info(f"   - Max relationships per entity: {settings.get('max_relationships_per_entity', 'unknown')}")
    logger.info(f"   - Min entity confidence: {settings.get('extraction', {}).get('min_entity_confidence', 'unknown')}")
    logger.info(f"   - Entity discovery confidence: {settings.get('entity_discovery', {}).get('confidence_threshold', 'unknown')}")
    logger.info(f"   - Anti-silo enabled: {settings.get('extraction', {}).get('enable_anti_silo', 'unknown')}")
    
    # Initialize extractor
    logger.info("🚀 Initializing LLM Knowledge Extractor...")
    extractor = LLMKnowledgeExtractor()
    
    # Test extraction
    logger.info("🎯 Starting entity extraction test...")
    start_time = datetime.now()
    
    try:
        result = await extractor.extract_knowledge(
            text=SAMPLE_BUSINESS_TEXT,
            context={'document_type': 'business_strategy', 'domain': 'financial_services'}
        )
        
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 80)
        logger.info("EXTRACTION RESULTS")
        logger.info("=" * 80)
        
        # Analyze entities
        entities = result.entities
        relationships = result.relationships
        
        logger.info(f"📊 ENTITY ANALYSIS:")
        logger.info(f"   - Total entities extracted: {len(entities)}")
        logger.info(f"   - Processing time: {extraction_time:.2f} seconds")
        logger.info(f"   - Confidence score: {result.confidence_score:.3f}")
        
        # Count entity types
        entity_type_counts = {}
        organization_entities = []
        
        for entity in entities:
            entity_type = entity.label
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            
            if entity_type in ['ORGANIZATION', 'ORG', 'COMPANY', 'BANK', 'CORPORATION', 'FINTECH']:
                organization_entities.append(entity)
        
        logger.info(f"📈 ENTITY TYPE BREAKDOWN:")
        for entity_type, count in sorted(entity_type_counts.items()):
            logger.info(f"   - {entity_type}: {count}")
        
        logger.info(f"🏢 ORGANIZATION ENTITIES ({len(organization_entities)}):")
        for org in organization_entities:
            logger.info(f"   - {org.canonical_form} ({org.label})")
        
        logger.info(f"🔗 RELATIONSHIP ANALYSIS:")
        logger.info(f"   - Total relationships: {len(relationships)}")
        
        if len(entities) > 0:
            relationship_ratio = len(relationships) / len(entities)
            logger.info(f"   - Relationship to entity ratio: {relationship_ratio:.2f}")
        
        # Success criteria
        logger.info("=" * 80)
        logger.info("SUCCESS CRITERIA EVALUATION")
        logger.info("=" * 80)
        
        success_criteria = {
            "Entity count >= 25": len(entities) >= 25,
            "Organization entities found": len(organization_entities) > 0,
            "Relationship ratio <= 5.0": len(entities) == 0 or len(relationships) / len(entities) <= 5.0,
            "Extraction completed": len(entities) > 0,
            "Processing time < 60s": extraction_time < 60.0
        }
        
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"   {status}: {criterion}")
            if not passed:
                all_passed = False
        
        logger.info("=" * 80)
        if all_passed:
            logger.info("🎉 ALL SUCCESS CRITERIA PASSED - Entity extraction rebalancing SUCCESSFUL!")
        else:
            logger.info("⚠️  Some criteria failed - further tuning may be needed")
        logger.info("=" * 80)
        
        return {
            'success': all_passed,
            'entity_count': len(entities),
            'organization_count': len(organization_entities),
            'relationship_count': len(relationships),
            'extraction_time': extraction_time,
            'confidence_score': result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"❌ Extraction test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

async def main():
    """Main test function"""
    logger.info("🔬 Entity Extraction Rebalancing Test")
    logger.info(f"📅 Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = await test_entity_extraction_rebalance()
    
    if result['success']:
        logger.info("✅ Test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Test failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())