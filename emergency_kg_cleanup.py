#!/usr/bin/env python3
"""
Emergency Knowledge Graph Cleanup Script
Reduces relationships from 486 to ≤188 (≤4 per entity) for browser performance
"""

import asyncio
import logging
from typing import Dict, List, Any
from app.services.neo4j_service import get_neo4j_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

logger = logging.getLogger(__name__)

async def get_current_stats() -> Dict[str, Any]:
    """Get current knowledge graph statistics"""
    try:
        neo4j_service = get_neo4j_service()
        
        # Get entity count
        entity_result = await neo4j_service.run_query(
            "MATCH (n) RETURN count(n) as entity_count"
        )
        entity_count = entity_result[0]['entity_count'] if entity_result else 0
        
        # Get relationship count
        rel_result = await neo4j_service.run_query(
            "MATCH ()-[r]->() RETURN count(r) as rel_count"
        )
        rel_count = rel_result[0]['rel_count'] if rel_result else 0
        
        # Calculate ratio
        ratio = rel_count / max(entity_count, 1)
        
        return {
            'entities': entity_count,
            'relationships': rel_count,
            'ratio': ratio
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {'entities': 0, 'relationships': 0, 'ratio': 0}

async def cleanup_low_confidence_competes_with():
    """Remove low-confidence COMPETES_WITH relationships (major cause of 486 relationships)"""
    try:
        neo4j_service = get_neo4j_service()
        
        query = """
        MATCH ()-[r:COMPETES_WITH]->()
        WHERE r.confidence < 0.8 OR r.confidence IS NULL
        WITH r
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await neo4j_service.run_query(query)
        deleted = result[0]['deleted_count'] if result else 0
        
        logger.info(f"✅ Removed {deleted} low-confidence COMPETES_WITH relationships")
        return deleted
    except Exception as e:
        logger.error(f"Failed to cleanup COMPETES_WITH: {e}")
        return 0

async def cap_related_technology_relationships():
    """Cap RELATED_TECHNOLOGY connections to max 3 per entity"""
    try:
        neo4j_service = get_neo4j_service()
        
        query = """
        MATCH (t:TECHNOLOGY)-[r:RELATED_TECHNOLOGY]-()
        WITH t, collect(r) as rels
        WHERE size(rels) > 3
        WITH t, rels[3..] as excess_rels
        UNWIND excess_rels as r
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await neo4j_service.run_query(query)
        deleted = result[0]['deleted_count'] if result else 0
        
        logger.info(f"✅ Capped RELATED_TECHNOLOGY: removed {deleted} excess relationships")
        return deleted
    except Exception as e:
        logger.error(f"Failed to cap RELATED_TECHNOLOGY: {e}")
        return 0

async def remove_anti_silo_relationships():
    """Remove anti-silo created relationships"""
    try:
        neo4j_service = get_neo4j_service()
        
        query = """
        MATCH ()-[r]->()
        WHERE r.created_by IS NOT NULL AND r.created_by CONTAINS 'anti_silo'
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await neo4j_service.run_query(query)
        deleted = result[0]['deleted_count'] if result else 0
        
        logger.info(f"✅ Removed {deleted} anti-silo created relationships")
        return deleted
    except Exception as e:
        logger.error(f"Failed to remove anti-silo relationships: {e}")
        return 0

async def enforce_four_relationship_limit():
    """Final enforcement: 4-relationship limit per entity"""
    try:
        neo4j_service = get_neo4j_service()
        
        # Get entities with more than 4 relationships and remove excess
        query = """
        MATCH (n)-[r]-()
        WITH n, collect(r) as rels
        WHERE size(rels) > 4
        WITH n, rels[0..4] as keep, rels[4..] as remove
        UNWIND remove as r
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await neo4j_service.run_query(query)
        deleted = result[0]['deleted_count'] if result else 0
        
        logger.info(f"✅ Enforced 4-relationship limit: removed {deleted} excess relationships")
        return deleted
    except Exception as e:
        logger.error(f"Failed to enforce 4-relationship limit: {e}")
        return 0

async def verify_cleanup_results():
    """Verify that cleanup achieved target ratio ≤4.0"""
    stats = await get_current_stats()
    
    logger.info(f"📊 Post-cleanup statistics:")
    logger.info(f"   Entities: {stats['entities']}")
    logger.info(f"   Relationships: {stats['relationships']}")
    logger.info(f"   Ratio: {stats['ratio']:.2f}")
    
    target_achieved = stats['ratio'] <= 4.0
    performance_target = stats['relationships'] <= 188
    
    logger.info(f"   ✅ Target ratio ≤4.0: {'ACHIEVED' if target_achieved else 'NOT ACHIEVED'}")
    logger.info(f"   ✅ Performance target ≤188: {'ACHIEVED' if performance_target else 'NOT ACHIEVED'}")
    
    return {
        'target_achieved': target_achieved,
        'performance_target': performance_target,
        'stats': stats
    }

async def main():
    """Execute emergency knowledge graph cleanup"""
    logger.info("🚨 Starting Emergency Knowledge Graph Cleanup")
    logger.info("Target: Reduce relationships from 486 to ≤188 (≤4 per entity)")
    
    # Get initial stats
    initial_stats = await get_current_stats()
    logger.info(f"📊 Initial statistics:")
    logger.info(f"   Entities: {initial_stats['entities']}")
    logger.info(f"   Relationships: {initial_stats['relationships']}")
    logger.info(f"   Ratio: {initial_stats['ratio']:.2f}")
    
    if initial_stats['ratio'] <= 4.0:
        logger.info("✅ Already within target ratio. No cleanup needed.")
        return
    
    total_deleted = 0
    
    # Phase 1: Remove low-confidence COMPETES_WITH relationships
    logger.info("\n🔧 Phase 1: Removing low-confidence COMPETES_WITH relationships...")
    deleted = await cleanup_low_confidence_competes_with()
    total_deleted += deleted
    
    # Phase 2: Cap RELATED_TECHNOLOGY connections
    logger.info("\n🔧 Phase 2: Capping RELATED_TECHNOLOGY connections...")
    deleted = await cap_related_technology_relationships()
    total_deleted += deleted
    
    # Phase 3: Remove anti-silo relationships
    logger.info("\n🔧 Phase 3: Removing anti-silo created relationships...")
    deleted = await remove_anti_silo_relationships()
    total_deleted += deleted
    
    # Phase 4: Final 4-relationship limit enforcement
    logger.info("\n🔧 Phase 4: Enforcing 4-relationship limit per entity...")
    deleted = await enforce_four_relationship_limit()
    total_deleted += deleted
    
    # Verify results
    logger.info(f"\n📊 Total relationships deleted: {total_deleted}")
    results = await verify_cleanup_results()
    
    if results['target_achieved'] and results['performance_target']:
        logger.info("🎉 SUCCESS: Emergency cleanup completed successfully!")
        logger.info("   ✅ Ratio ≤4.0 achieved")
        logger.info("   ✅ Performance target ≤188 achieved")
        logger.info("   ✅ Browser performance should be significantly improved")
    else:
        logger.warning("⚠️  Cleanup completed but targets not fully achieved")
        if not results['target_achieved']:
            logger.warning(f"   ❌ Ratio still {results['stats']['ratio']:.2f} (target: ≤4.0)")
        if not results['performance_target']:
            logger.warning(f"   ❌ Relationships still {results['stats']['relationships']} (target: ≤188)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())