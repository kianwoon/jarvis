#!/usr/bin/env python3
"""
Test script for LLM Relationship Discovery in Radiating System

Tests the ability to discover 50+ relationships between 30 AI technology entities
using LLM knowledge when Neo4j returns empty results.
"""

import asyncio
import logging
from typing import List
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the necessary modules
from app.services.radiating.extraction.llm_relationship_discoverer import LLMRelationshipDiscoverer
from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_context import RadiatingContext, DomainContext, TraversalStrategy
from app.services.radiating.engine.radiating_traverser import RadiatingTraverser


async def create_test_entities() -> List[RadiatingEntity]:
    """Create a list of 30 AI technology entities for testing"""
    
    ai_entities = [
        # AI Frameworks & Libraries
        ("LangChain", "FRAMEWORK", "Python framework for LLM applications"),
        ("LlamaIndex", "FRAMEWORK", "Data framework for LLM applications"),
        ("Hugging Face", "PLATFORM", "ML model hub and tools"),
        ("TensorFlow", "FRAMEWORK", "Deep learning framework by Google"),
        ("PyTorch", "FRAMEWORK", "Deep learning framework by Meta"),
        
        # LLM Providers
        ("OpenAI", "COMPANY", "AI research company, creator of GPT"),
        ("Anthropic", "COMPANY", "AI safety company, creator of Claude"),
        ("Google", "COMPANY", "Tech giant with Gemini/Bard"),
        ("Meta", "COMPANY", "Tech company with Llama models"),
        ("Mistral AI", "COMPANY", "French AI company"),
        
        # Vector Databases
        ("Milvus", "DATABASE", "Open-source vector database"),
        ("Pinecone", "DATABASE", "Managed vector database service"),
        ("Qdrant", "DATABASE", "Vector similarity search engine"),
        ("Weaviate", "DATABASE", "Vector search engine"),
        ("ChromaDB", "DATABASE", "Embedding database"),
        
        # Cloud Platforms
        ("AWS", "PLATFORM", "Amazon cloud services"),
        ("Azure", "PLATFORM", "Microsoft cloud services"),
        ("Google Cloud", "PLATFORM", "Google cloud services"),
        
        # Development Tools
        ("Python", "LANGUAGE", "Programming language"),
        ("Docker", "TOOL", "Containerization platform"),
        ("Kubernetes", "TOOL", "Container orchestration"),
        ("Git", "TOOL", "Version control system"),
        
        # AI Models
        ("GPT-4", "MODEL", "OpenAI's latest model"),
        ("Claude", "MODEL", "Anthropic's AI assistant"),
        ("Llama 2", "MODEL", "Meta's open-source LLM"),
        ("Gemini", "MODEL", "Google's multimodal AI"),
        
        # Databases & Storage
        ("PostgreSQL", "DATABASE", "Relational database"),
        ("Redis", "DATABASE", "In-memory data store"),
        ("MongoDB", "DATABASE", "NoSQL database"),
        ("Elasticsearch", "DATABASE", "Search and analytics engine")
    ]
    
    entities = []
    for name, entity_type, description in ai_entities:
        entity = RadiatingEntity(
            canonical_form=name,
            label=entity_type,
            confidence=1.0,
            properties={
                "description": description,
                "domain": "AI/Technology"
            },
            relevance_score=0.8,
            domain_metadata={
                "domain": "technology",
                "description": description
            }
        )
        entities.append(entity)
    
    return entities


async def test_llm_discovery():
    """Test LLM relationship discovery"""
    
    logger.info("="*80)
    logger.info("Starting LLM Relationship Discovery Test")
    logger.info("="*80)
    
    # Create test entities
    entities = await create_test_entities()
    logger.info(f"Created {len(entities)} test entities")
    
    # Initialize the LLM discoverer
    discoverer = LLMRelationshipDiscoverer()
    
    # Discover relationships
    logger.info("\nDiscovering relationships using LLM...")
    start_time = datetime.now()
    
    relationships = await discoverer.discover_relationships(
        entities=entities,
        max_relationships_per_pair=5,  # Allow up to 5 relationships between any pair
        confidence_threshold=0.5  # Lower threshold to get more relationships
    )
    
    discovery_time = (datetime.now() - start_time).total_seconds()
    
    # Analyze results
    logger.info("\n" + "="*80)
    logger.info("DISCOVERY RESULTS")
    logger.info("="*80)
    logger.info(f"Total relationships discovered: {len(relationships)}")
    logger.info(f"Discovery time: {discovery_time:.2f} seconds")
    
    # Group relationships by type
    relationship_types = {}
    for rel in relationships:
        rel_type = rel.relationship_type
        if rel_type not in relationship_types:
            relationship_types[rel_type] = []
        relationship_types[rel_type].append(
            f"{rel.source_entity} -> {rel.target_entity}"
        )
    
    logger.info(f"\nRelationship type distribution ({len(relationship_types)} types):")
    for rel_type, examples in sorted(relationship_types.items(), 
                                    key=lambda x: len(x[1]), reverse=True):
        logger.info(f"  {rel_type}: {len(examples)} relationships")
        # Show first 3 examples
        for example in examples[:3]:
            logger.info(f"    - {example}")
    
    # Show some interesting relationships with context
    logger.info("\n" + "="*80)
    logger.info("SAMPLE RELATIONSHIPS WITH CONTEXT")
    logger.info("="*80)
    
    sample_size = min(10, len(relationships))
    for i, rel in enumerate(relationships[:sample_size], 1):
        logger.info(f"\n{i}. {rel.source_entity} {rel.relationship_type} {rel.target_entity}")
        logger.info(f"   Confidence: {rel.confidence:.2f}")
        logger.info(f"   Context: {rel.context}")
    
    # Test with RadiatingTraverser integration
    logger.info("\n" + "="*80)
    logger.info("TESTING RADIATING TRAVERSER INTEGRATION")
    logger.info("="*80)
    
    # Create a context with LLM discovery enabled
    context = RadiatingContext(
        original_query="AI technology relationships",
        query_domain=DomainContext.TECHNOLOGY,
        depth_limit=2,
        relevance_threshold=0.3,
        max_entities_per_level=20,
        traversal_strategy=TraversalStrategy.BEST_FIRST,
        enable_llm_discovery=True,  # Enable LLM discovery
        llm_discovery_entities=entities[:15],  # Provide subset for discovery
        llm_max_entities_for_discovery=30
    )
    
    # Initialize traverser
    traverser = RadiatingTraverser()
    
    # Test relationship discovery for a specific entity
    test_entity = entities[0]  # LangChain
    logger.info(f"\nTesting relationship discovery for: {test_entity.canonical_form}")
    
    # Discover relationships (this will use LLM when Neo4j returns empty)
    entity_relationships = await traverser._get_entity_relationships(
        test_entity.get_entity_id(),
        context
    )
    
    logger.info(f"Found {len(entity_relationships)} relationships for {test_entity.canonical_form}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    success = len(relationships) >= 50
    logger.info(f"Target: 50+ relationships")
    logger.info(f"Discovered: {len(relationships)} relationships")
    logger.info(f"Success: {'✓ YES' if success else '✗ NO'}")
    
    if success:
        logger.info("\n🎉 Successfully discovered 50+ relationships using LLM!")
    else:
        logger.info(f"\n⚠️ Only discovered {len(relationships)} relationships. Need optimization.")
    
    # Save results to file
    output_file = "/tmp/llm_relationship_discovery_results.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "entities_count": len(entities),
        "relationships_count": len(relationships),
        "discovery_time_seconds": discovery_time,
        "relationship_types": {
            rel_type: len(examples) 
            for rel_type, examples in relationship_types.items()
        },
        "sample_relationships": [
            {
                "source": rel.source_entity,
                "target": rel.target_entity,
                "type": rel.relationship_type,
                "confidence": rel.confidence,
                "context": rel.context
            }
            for rel in relationships[:20]
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return relationships


async def main():
    """Main test function"""
    try:
        relationships = await test_llm_discovery()
        logger.info("\n✅ Test completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())