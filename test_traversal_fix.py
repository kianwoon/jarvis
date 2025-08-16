#!/usr/bin/env python3
"""
Test script to verify the radiating traverser fix for entity processing
"""

import asyncio
import logging
from typing import List

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_context import RadiatingContext, TraversalStrategy, DomainContext
from app.services.radiating.engine.radiating_traverser import RadiatingTraverser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_traversal_with_entities():
    """Test that starting entities are properly processed during traversal"""
    
    logger.info("=" * 80)
    logger.info("TESTING RADIATING TRAVERSER ENTITY PROCESSING FIX")
    logger.info("=" * 80)
    
    # Create test entities with proper constructor arguments
    test_entities = [
        RadiatingEntity(
            text="TensorFlow",
            label="ML_FRAMEWORK",
            start_char=0,
            end_char=10,
            canonical_form="TensorFlow",
            confidence=0.95,
            relevance_score=0.9,
            properties={"type": "machine_learning", "open_source": True}
        ),
        RadiatingEntity(
            text="PyTorch",
            label="ML_FRAMEWORK",
            start_char=20,
            end_char=27,
            canonical_form="PyTorch",
            confidence=0.95,
            relevance_score=0.9,
            properties={"type": "machine_learning", "open_source": True}
        ),
        RadiatingEntity(
            text="Scikit-learn",
            label="ML_LIBRARY",
            start_char=40,
            end_char=52,
            canonical_form="scikit-learn",
            confidence=0.9,
            relevance_score=0.85,
            properties={"type": "machine_learning", "open_source": True}
        )
    ]
    
    logger.info(f"Created {len(test_entities)} test entities")
    for entity in test_entities:
        logger.info(f"  - {entity.canonical_form} ({entity.label})")
    
    # Create context with LLM discovery enabled
    context = RadiatingContext(
        original_query="machine learning frameworks",
        depth_limit=2,
        max_total_entities=100,
        max_entities_per_level=20,
        traversal_strategy=TraversalStrategy.BREADTH_FIRST,
        relevance_threshold=0.3,
        enable_llm_discovery=True,
        llm_max_entities_for_discovery=30
    )
    
    # Set domain preferences
    context.set_domain_preferences(
        DomainContext.TECHNOLOGY,
        entity_types=["ML_FRAMEWORK", "ML_LIBRARY", "TOOL"],
        relationship_types=["COMPETES_WITH", "INTEGRATES_WITH", "DEPENDS_ON"]
    )
    
    # Create traverser
    traverser = RadiatingTraverser()
    
    logger.info("\nStarting traversal...")
    logger.info("-" * 40)
    
    # Run traversal
    graph = await traverser.traverse(context, test_entities)
    
    # Get metrics
    metrics = traverser.get_metrics()
    
    logger.info("\nTraversal Results:")
    logger.info("-" * 40)
    logger.info(f"✓ Nodes in graph: {graph.total_nodes}")
    logger.info(f"✓ Edges in graph: {graph.total_edges}")
    logger.info(f"✓ Entities processed: {metrics['entities_processed']}")
    logger.info(f"✓ Relationships discovered: {metrics['relationships_discovered']}")
    logger.info(f"✓ LLM discoveries: {metrics.get('llm_discoveries', 0)}")
    logger.info(f"✓ Neo4j queries: {metrics['neo4j_queries']}")
    logger.info(f"✓ Cache hits: {metrics['cache_hits']}")
    logger.info(f"✓ Cache misses: {metrics['cache_misses']}")
    logger.info(f"✓ Traversal time: {metrics['traversal_time_ms']:.2f}ms")
    
    # Check if entities were added to traversal queue
    logger.info("\nContext Statistics:")
    logger.info("-" * 40)
    logger.info(f"✓ Total entities discovered: {context.total_entities_discovered}")
    logger.info(f"✓ Total relationships discovered: {context.total_relationships_discovered}")
    logger.info(f"✓ Visited entity IDs: {len(context.visited_entity_ids)}")
    logger.info(f"✓ Max depth reached: {context.current_depth}")
    
    # Verify fix worked
    success = metrics['entities_processed'] > 0
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ SUCCESS: Entities were processed! The fix is working.")
        logger.info("=" * 80)
        
        # Show some discovered entities
        if graph.total_nodes > 3:
            logger.info("\nSample entities in graph:")
            for entity_id, entity in list(graph.nodes.items())[:5]:
                logger.info(f"  - {entity.canonical_form} ({entity.label})")
                
        # Show some relationships if any
        if graph.total_edges > 0:
            logger.info(f"\nSample relationships discovered:")
            for rel_id, rel in list(graph.edges.items())[:5]:
                logger.info(f"  - {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
    else:
        logger.info("\n" + "=" * 80)
        logger.info("❌ FAILURE: No entities were processed. The fix didn't work.")
        logger.info("=" * 80)
        
        # Debug info
        logger.info("\nDebug Information:")
        logger.info(f"  - Traversal queue empty? {len(context.traversal_queue) == 0}")
        logger.info(f"  - Visited entities: {context.visited_entity_ids}")
        logger.info(f"  - Expanded entities count: {len(context.expanded_entities)}")
    
    return success


if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_traversal_with_entities())
    
    # Exit with appropriate code
    exit(0 if success else 1)