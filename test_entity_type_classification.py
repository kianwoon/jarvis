#!/usr/bin/env python3
"""
Test script to verify entity type classification improvements
"""

import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_entity_classification():
    """Test the enhanced entity type classification"""
    
    logger.info("🧪 Testing Entity Type Classification")
    
    # Initialize the extractor
    extractor = LLMKnowledgeExtractor()
    
    # Test entities from the DBS document
    test_entities = [
        "DBS Bank",
        "OceanBase", 
        "SOFAStack",
        "TDSQL",
        "Singapore",
        "China",
        "Alipay",
        "Digital Transformation",
        "MariaDB",
        "PostgreSQL",
        "Ant Group",
        "Alibaba",
        "Hong Kong",
        "Performance",
        "Scalability"
    ]
    
    logger.info(f"🔍 Testing classification for {len(test_entities)} entities:")
    
    type_distribution = {}
    
    for entity in test_entities:
        entity_type = extractor._classify_entity_type(entity)
        logger.info(f"   '{entity}' -> {entity_type}")
        
        # Count distribution
        if entity_type not in type_distribution:
            type_distribution[entity_type] = 0
        type_distribution[entity_type] += 1
    
    logger.info("\n📊 Type Distribution:")
    for entity_type, count in sorted(type_distribution.items()):
        logger.info(f"   {entity_type}: {count} entities")
    
    # Check if we have diversity
    unique_types = len(type_distribution)
    logger.info(f"\n🎯 Result: {unique_types} unique entity types found")
    
    if unique_types >= 4:
        logger.info("✅ SUCCESS: Good entity type diversity - this should fix the color issue!")
    else:
        logger.warning("⚠️  LIMITED: Only a few entity types - may still have color issues")
    
    return type_distribution

if __name__ == "__main__":
    asyncio.run(test_entity_classification())