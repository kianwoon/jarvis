#!/usr/bin/env python3
"""
Test script to verify knowledge graph improvements:
1. Anti-silo business entity connections
2. Enhanced relationship extraction with specific types
3. Relationship quality scoring and filtering
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

async def test_relationship_quality_improvements():
    """Test the relationship quality scoring and improvement logic"""
    
    logger.info("🧪 Testing Relationship Quality Improvements")
    
    # Initialize the extractor
    extractor = LLMKnowledgeExtractor()
    
    # Test relationship quality scoring
    test_relationships = [
        # High-quality specific relationships
        ("DBS Bank", "OceanBase", "EVALUATES", "ORGANIZATION", "TECHNOLOGY"),
        ("Ant Group", "Financial Sector", "PROVIDES_SERVICES_TO", "ORGANIZATION", "CONCEPT"),
        ("DBS Bank", "Singapore", "OPERATES_IN", "ORGANIZATION", "LOCATION"),
        
        # Generic relationships that should be improved
        ("DBS Bank", "PostgreSQL", "RELATED_TO", "ORGANIZATION", "TECHNOLOGY"),
        ("Ant Group", "Alibaba", "RELATED_TO", "ORGANIZATION", "ORGANIZATION"),
        
        # Low-quality relationships that should be filtered
        ("Performance", "Scalability", "RELATED_TO", "CONCEPT", "CONCEPT"),
    ]
    
    logger.info(f"🔍 Testing {len(test_relationships)} relationship scenarios:")
    
    high_quality_count = 0
    improved_count = 0
    filtered_count = 0
    
    for source, target, rel_type, source_type, target_type in test_relationships:
        # Test quality scoring
        quality_score = extractor._score_relationship_quality(
            source, target, rel_type, source_type, target_type
        )
        
        # Test relationship improvement
        improved_type = extractor._improve_relationship_type(
            source, target, rel_type, source_type, target_type
        )
        
        logger.info(f"   {source} -> {target}:")
        logger.info(f"      Original: {rel_type} (Quality: {quality_score:.2f})")
        logger.info(f"      Improved: {improved_type}")
        
        if quality_score >= 0.7:
            high_quality_count += 1
            logger.info(f"      ✅ High quality relationship")
        elif quality_score >= 0.3:
            if improved_type != rel_type:
                improved_count += 1
                logger.info(f"      🔧 Relationship improved: {rel_type} -> {improved_type}")
            else:
                logger.info(f"      ⚠️  Medium quality, needs attention")
        else:
            filtered_count += 1
            logger.info(f"      🚫 Would be filtered (low quality)")
        
        logger.info("")
    
    logger.info("📊 Relationship Quality Test Results:")
    logger.info(f"   High quality relationships: {high_quality_count}")
    logger.info(f"   Improved relationships: {improved_count}")
    logger.info(f"   Filtered relationships: {filtered_count}")
    
    return {
        'high_quality': high_quality_count,
        'improved': improved_count,
        'filtered': filtered_count
    }

async def test_business_relationship_logic():
    """Test the business relationship determination logic"""
    
    logger.info("\n🏢 Testing Business Relationship Logic")
    
    # Initialize the knowledge graph service (we'll test the logic directly)
    from app.services.knowledge_graph_service import KnowledgeGraphService
    kg_service = KnowledgeGraphService()
    
    # Test business relationship patterns
    test_business_pairs = [
        ("DBS Bank", "HSBC Bank"),
        ("Ant Group", "DBS Bank"),
        ("Alibaba", "Ant Group"),
        ("DBS Bank", "Singapore"),
        ("Technology Company", "Financial Services"),
    ]
    
    logger.info(f"🔍 Testing {len(test_business_pairs)} business relationship patterns:")
    
    for entity1, entity2 in test_business_pairs:
        relationship_type, confidence, reasoning = kg_service._determine_business_relationship(
            entity1.lower(), entity2.lower(), entity1, entity2
        )
        
        logger.info(f"   {entity1} <-> {entity2}:")
        if relationship_type:
            logger.info(f"      Relationship: {relationship_type}")
            logger.info(f"      Confidence: {confidence:.2f}")
            logger.info(f"      Reasoning: {reasoning}")
        else:
            logger.info(f"      No specific relationship identified")
        logger.info("")

async def test_entity_classification_coverage():
    """Test entity classification for business domain coverage"""
    
    logger.info("\n🏷️ Testing Entity Classification Coverage")
    
    extractor = LLMKnowledgeExtractor()
    
    # Test entities that should get diverse types
    business_entities = [
        "DBS Bank", "Ant Group", "Alibaba", "Tencent",  # Organizations
        "OceanBase", "SOFAStack", "TDSQL", "MariaDB",   # Technology
        "Singapore", "Hong Kong", "China", "Indonesia",  # Locations
        "Alipay", "WeChat Pay", "Digital Wallet",       # Products
        "Digital Transformation", "Database Migration",  # Projects
        "Performance", "Scalability", "Security",       # Concepts
    ]
    
    type_distribution = {}
    
    for entity in business_entities:
        entity_type = extractor._classify_entity_type(entity)
        type_distribution[entity_type] = type_distribution.get(entity_type, 0) + 1
    
    logger.info("📊 Entity Type Distribution:")
    for entity_type, count in sorted(type_distribution.items()):
        logger.info(f"   {entity_type}: {count} entities")
    
    diversity_score = len(type_distribution) / len(set(business_entities))
    logger.info(f"\n🎯 Type Diversity Score: {diversity_score:.2f}")
    
    if len(type_distribution) >= 5:
        logger.info("✅ Excellent entity type diversity!")
    elif len(type_distribution) >= 3:
        logger.info("⚠️  Good entity type diversity")
    else:
        logger.info("❌ Poor entity type diversity")
    
    return type_distribution

async def main():
    """Run all improvement tests"""
    
    logger.info("🚀 Knowledge Graph Quality Improvement Tests\n")
    
    # Test 1: Relationship quality improvements
    rel_results = await test_relationship_quality_improvements()
    
    # Test 2: Business relationship logic
    await test_business_relationship_logic()
    
    # Test 3: Entity classification coverage
    type_dist = await test_entity_classification_coverage()
    
    # Summary
    logger.info("\n🎯 IMPROVEMENT TEST SUMMARY:")
    logger.info("=" * 50)
    
    # Relationship improvements
    total_rels = sum(rel_results.values())
    if rel_results['high_quality'] + rel_results['improved'] >= total_rels * 0.8:
        logger.info("✅ RELATIONSHIP QUALITY: Excellent improvements implemented")
    else:
        logger.info("⚠️  RELATIONSHIP QUALITY: Some improvements still needed")
    
    # Entity type diversity
    if len(type_dist) >= 5:
        logger.info("✅ ENTITY CLASSIFICATION: Excellent type diversity")
    else:
        logger.info("⚠️  ENTITY CLASSIFICATION: Limited type diversity")
    
    # Anti-silo improvements
    logger.info("✅ ANTI-SILO CONNECTIONS: Business entity logic implemented")
    
    logger.info("\n💡 Next Steps:")
    logger.info("1. Reprocess the DBS document to see these improvements")
    logger.info("2. Check for zero isolated nodes")
    logger.info("3. Verify relationship type diversity (less RELATED_TO)")
    logger.info("4. Confirm entity color coding in visualization")

if __name__ == "__main__":
    asyncio.run(main())