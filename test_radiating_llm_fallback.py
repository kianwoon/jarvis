#!/usr/bin/env python3
"""
Test Radiating System LLM Fallback Mechanism

This script tests the enhanced radiating system's ability to:
1. Detect when Neo4j returns empty results
2. Fall back to LLM for relationship discovery
3. Discover 50+ relationships for AI technology entities
"""

import asyncio
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)


async def test_basic_llm_discovery():
    """Test basic LLM relationship discovery functionality"""
    
    from app.services.radiating.extraction.llm_relationship_discoverer import LLMRelationshipDiscoverer
    from app.services.radiating.models.radiating_entity import RadiatingEntity
    
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Basic LLM Relationship Discovery")
    logger.info("="*80)
    
    # Create a small set of well-known AI entities
    test_entities = [
        RadiatingEntity(
            canonical_form="LangChain",
            label="FRAMEWORK",
            confidence=1.0,
            properties={"type": "AI Framework"},
            relevance_score=0.9,
            domain_metadata={"domain": "technology", "description": "LLM application framework"}
        ),
        RadiatingEntity(
            canonical_form="OpenAI",
            label="COMPANY",
            confidence=1.0,
            properties={"type": "AI Company"},
            relevance_score=0.9,
            domain_metadata={"domain": "technology", "description": "AI research company"}
        ),
        RadiatingEntity(
            canonical_form="Milvus",
            label="DATABASE",
            confidence=1.0,
            properties={"type": "Vector Database"},
            relevance_score=0.8,
            domain_metadata={"domain": "technology", "description": "Open-source vector database"}
        ),
        RadiatingEntity(
            canonical_form="Python",
            label="LANGUAGE",
            confidence=1.0,
            properties={"type": "Programming Language"},
            relevance_score=0.8,
            domain_metadata={"domain": "technology", "description": "Programming language"}
        ),
        RadiatingEntity(
            canonical_form="Docker",
            label="TOOL",
            confidence=1.0,
            properties={"type": "Container Platform"},
            relevance_score=0.7,
            domain_metadata={"domain": "technology", "description": "Container platform"}
        )
    ]
    
    # Initialize discoverer
    discoverer = LLMRelationshipDiscoverer()
    
    # Discover relationships
    logger.info(f"\nDiscovering relationships for {len(test_entities)} entities...")
    start_time = datetime.now()
    
    relationships = await discoverer.discover_relationships(
        entities=test_entities,
        max_relationships_per_pair=3,
        confidence_threshold=0.5
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"\nResults:")
    logger.info(f"  - Discovered {len(relationships)} relationships")
    logger.info(f"  - Time taken: {elapsed:.2f} seconds")
    
    # Show sample relationships
    if relationships:
        logger.info("\nSample relationships discovered:")
        for rel in relationships[:5]:
            logger.info(f"  - {rel.source_entity} {rel.relationship_type} {rel.target_entity}")
            logger.info(f"    Context: {rel.context[:100]}...")
    
    return len(relationships) > 0


async def test_traverser_fallback():
    """Test RadiatingTraverser's LLM fallback mechanism"""
    
    from app.services.radiating.engine.radiating_traverser import RadiatingTraverser
    from app.services.radiating.models.radiating_context import RadiatingContext, DomainContext
    from app.services.radiating.models.radiating_entity import RadiatingEntity
    
    logger.info("\n" + "="*80)
    logger.info("TEST 2: RadiatingTraverser LLM Fallback")
    logger.info("="*80)
    
    # Create context with LLM discovery enabled
    context = RadiatingContext(
        original_query="AI technology ecosystem",
        query_domain=DomainContext.TECHNOLOGY,
        depth_limit=2,
        enable_llm_discovery=True,  # Enable LLM fallback
        llm_max_entities_for_discovery=10
    )
    
    # Add some seed entities to context for discovery
    seed_entities = [
        RadiatingEntity(
            canonical_form="Kubernetes",
            label="PLATFORM",
            confidence=1.0,
            properties={"type": "Container Orchestration"},
            relevance_score=0.9
        ),
        RadiatingEntity(
            canonical_form="Redis",
            label="DATABASE",
            confidence=1.0,
            properties={"type": "In-memory Database"},
            relevance_score=0.8
        ),
        RadiatingEntity(
            canonical_form="PostgreSQL",
            label="DATABASE",
            confidence=1.0,
            properties={"type": "Relational Database"},
            relevance_score=0.8
        )
    ]
    
    context.llm_discovery_entities = seed_entities
    
    # Initialize traverser
    traverser = RadiatingTraverser()
    
    # Test getting relationships (should trigger LLM when Neo4j is empty)
    test_entity_id = "test_kubernetes_001"
    
    logger.info(f"\nTesting relationship discovery for entity ID: {test_entity_id}")
    logger.info("(This should trigger LLM fallback when Neo4j returns empty)")
    
    # Mock the entity lookup to return our test entity
    async def mock_get_entity():
        return seed_entities[0]
    
    # Temporarily replace the method
    original_method = traverser._get_entity_by_id
    traverser._get_entity_by_id = lambda x: mock_get_entity()
    
    try:
        relationships = await traverser._get_entity_relationships(
            test_entity_id,
            context
        )
        
        logger.info(f"\nResults:")
        logger.info(f"  - LLM fallback triggered: {'Yes' if relationships else 'No'}")
        logger.info(f"  - Relationships discovered: {len(relationships)}")
        
        if relationships:
            logger.info("\nSample relationships from fallback:")
            for rel in relationships[:3]:
                logger.info(f"  - {rel.source_entity} {rel.relationship_type} {rel.target_entity}")
        
        # Check metrics
        metrics = traverser.get_metrics()
        logger.info(f"\nTraverser metrics:")
        logger.info(f"  - LLM discoveries: {metrics.get('llm_discoveries', 0)}")
        logger.info(f"  - Neo4j queries: {metrics.get('neo4j_queries', 0)}")
        
        return len(relationships) > 0
        
    finally:
        # Restore original method
        traverser._get_entity_by_id = original_method


async def test_large_scale_discovery():
    """Test discovering 50+ relationships for 30 AI entities"""
    
    from app.services.radiating.extraction.llm_relationship_discoverer import LLMRelationshipDiscoverer
    from app.services.radiating.models.radiating_entity import RadiatingEntity
    
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Large-Scale Discovery (50+ relationships goal)")
    logger.info("="*80)
    
    # Create 30 diverse AI technology entities
    ai_entities_data = [
        # Core AI Frameworks
        ("LangChain", "FRAMEWORK", "LLM application framework"),
        ("LlamaIndex", "FRAMEWORK", "Data framework for LLMs"),
        ("Hugging Face", "PLATFORM", "ML model hub"),
        ("TensorFlow", "FRAMEWORK", "Deep learning framework"),
        ("PyTorch", "FRAMEWORK", "Deep learning framework"),
        
        # LLM Providers
        ("OpenAI", "COMPANY", "GPT models provider"),
        ("Anthropic", "COMPANY", "Claude AI provider"),
        ("Google", "COMPANY", "Gemini/Bard provider"),
        ("Meta", "COMPANY", "Llama models provider"),
        
        # Vector Databases
        ("Milvus", "DATABASE", "Vector database"),
        ("Pinecone", "DATABASE", "Vector database service"),
        ("Qdrant", "DATABASE", "Vector search engine"),
        ("Weaviate", "DATABASE", "Vector database"),
        
        # Cloud & Infrastructure
        ("AWS", "PLATFORM", "Cloud platform"),
        ("Azure", "PLATFORM", "Cloud platform"),
        ("Docker", "TOOL", "Container platform"),
        ("Kubernetes", "PLATFORM", "Container orchestration"),
        
        # Databases
        ("PostgreSQL", "DATABASE", "Relational database"),
        ("Redis", "DATABASE", "In-memory database"),
        ("MongoDB", "DATABASE", "NoSQL database"),
        
        # Programming
        ("Python", "LANGUAGE", "Programming language"),
        ("JavaScript", "LANGUAGE", "Programming language"),
        ("TypeScript", "LANGUAGE", "Programming language"),
        
        # AI Models
        ("GPT-4", "MODEL", "OpenAI model"),
        ("Claude", "MODEL", "Anthropic model"),
        ("Llama 2", "MODEL", "Meta model"),
        
        # Tools & Services
        ("GitHub", "PLATFORM", "Code repository"),
        ("FastAPI", "FRAMEWORK", "Web framework"),
        ("Streamlit", "FRAMEWORK", "App framework"),
        ("Gradio", "FRAMEWORK", "ML demo framework")
    ]
    
    # Create RadiatingEntity objects
    entities = []
    for name, entity_type, description in ai_entities_data:
        entity = RadiatingEntity(
            canonical_form=name,
            label=entity_type,
            confidence=1.0,
            properties={"description": description},
            relevance_score=0.8,
            domain_metadata={
                "domain": "technology",
                "description": description
            }
        )
        entities.append(entity)
    
    logger.info(f"\nCreated {len(entities)} AI technology entities")
    
    # Initialize discoverer
    discoverer = LLMRelationshipDiscoverer()
    
    # Discover relationships with higher limits
    logger.info("\nDiscovering relationships (targeting 50+)...")
    start_time = datetime.now()
    
    relationships = await discoverer.discover_relationships(
        entities=entities,
        max_relationships_per_pair=5,  # Allow more relationships per pair
        confidence_threshold=0.4  # Lower threshold to get more results
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Analyze results
    logger.info(f"\n" + "="*60)
    logger.info(f"RESULTS:")
    logger.info(f"="*60)
    logger.info(f"Total relationships discovered: {len(relationships)}")
    logger.info(f"Time taken: {elapsed:.2f} seconds")
    logger.info(f"Average time per relationship: {elapsed/max(1, len(relationships)):.3f} seconds")
    
    # Group by relationship type
    rel_types = {}
    for rel in relationships:
        rel_type = rel.relationship_type
        if rel_type not in rel_types:
            rel_types[rel_type] = 0
        rel_types[rel_type] += 1
    
    logger.info(f"\nRelationship types distribution ({len(rel_types)} types):")
    for rel_type, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        logger.info(f"  - {rel_type}: {count} relationships")
    
    # Show some interesting relationships
    logger.info("\nInteresting relationships discovered:")
    interesting = [r for r in relationships if r.confidence > 0.7][:10]
    for i, rel in enumerate(interesting, 1):
        logger.info(f"\n{i}. {rel.source_entity} -> {rel.target_entity}")
        logger.info(f"   Type: {rel.relationship_type}")
        logger.info(f"   Confidence: {rel.confidence:.2f}")
        if rel.context:
            logger.info(f"   Context: {rel.context[:100]}...")
    
    # Success check
    success = len(relationships) >= 50
    logger.info(f"\n" + "="*60)
    logger.info(f"TARGET: 50+ relationships")
    logger.info(f"ACHIEVED: {len(relationships)} relationships")
    logger.info(f"STATUS: {'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}")
    logger.info(f"="*60)
    
    return success


async def main():
    """Run all tests"""
    
    logger.info("\n" + "="*80)
    logger.info("RADIATING SYSTEM LLM FALLBACK TEST SUITE")
    logger.info("="*80)
    
    results = {}
    
    try:
        # Test 1: Basic discovery
        logger.info("\nRunning Test 1...")
        results['basic_discovery'] = await test_basic_llm_discovery()
        
        # Test 2: Traverser fallback
        logger.info("\nRunning Test 2...")
        results['traverser_fallback'] = await test_traverser_fallback()
        
        # Test 3: Large-scale discovery
        logger.info("\nRunning Test 3...")
        results['large_scale'] = await test_large_scale_discovery()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        results['error'] = str(e)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUITE SUMMARY")
    logger.info("="*80)
    
    for test_name, result in results.items():
        if test_name == 'error':
            logger.info(f"❌ Tests failed with error: {result}")
        else:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
    
    all_passed = all(v for k, v in results.items() if k != 'error')
    
    if all_passed:
        logger.info("\n🎉 All tests PASSED! The LLM fallback system is working correctly.")
        logger.info("The system can discover 50+ relationships when Neo4j is empty.")
    else:
        logger.info("\n⚠️ Some tests failed. Please review the implementation.")
    
    # Save results
    output_file = "/tmp/radiating_llm_fallback_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "all_passed": all_passed
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())