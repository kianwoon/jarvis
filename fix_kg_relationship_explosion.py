"""
Emergency Fix for Knowledge Graph Relationship Explosion

This script implements aggressive relationship reduction strategies to bring
the knowledge graph down from 1001 relationships to ~200-300.
"""

import asyncio
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime
from app.services.neo4j_service import get_neo4j_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings, invalidate_knowledge_graph_cache
from app.core.db import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphRelationshipReducer:
    """Emergency relationship reduction service"""
    
    def __init__(self):
        self.neo4j_service = get_neo4j_service()
        self.target_relationships = 250  # Target ~250 relationships total
        self.target_ratio = 3.0  # Target 3:1 entity:relationship ratio
        
    async def emergency_reduction(self) -> Dict[str, Any]:
        """Execute emergency relationship reduction"""
        logger.info("🚨 EMERGENCY RELATIONSHIP REDUCTION STARTING")
        
        # Step 1: Get current state
        current_stats = self._get_graph_statistics()
        logger.info(f"📊 Current state: {current_stats['entities']} entities, {current_stats['relationships']} relationships")
        logger.info(f"📊 Current ratio: {current_stats['ratio']:.1f}:1")
        
        if current_stats['relationships'] <= self.target_relationships:
            logger.info("✅ Already within target range")
            return current_stats
        
        # Step 2: Apply aggressive reduction strategies
        results = {}
        
        # Strategy 1: Remove low-confidence relationships
        logger.info("🔥 Strategy 1: Removing low-confidence relationships")
        removed_low_conf = self._remove_low_confidence_relationships(threshold=0.8)
        results['removed_low_confidence'] = removed_low_conf
        
        # Strategy 2: Limit relationships per entity to 1
        logger.info("🔥 Strategy 2: Limiting to 1 relationship per entity")
        removed_excess = self._limit_relationships_per_entity(max_per_entity=1)
        results['removed_excess_per_entity'] = removed_excess
        
        # Strategy 3: Remove cross-chunk relationships
        logger.info("🔥 Strategy 3: Removing cross-chunk relationships")
        removed_cross_chunk = self._remove_cross_chunk_relationships()
        results['removed_cross_chunk'] = removed_cross_chunk
        
        # Strategy 4: Remove anti-silo generated relationships
        logger.info("🔥 Strategy 4: Removing anti-silo relationships")
        removed_anti_silo = self._remove_anti_silo_relationships()
        results['removed_anti_silo'] = removed_anti_silo
        
        # Strategy 5: Keep only highest confidence relationships globally
        current_count = self.neo4j_service.get_total_relationship_count()
        if current_count > self.target_relationships:
            logger.info(f"🔥 Strategy 5: Keeping only top {self.target_relationships} relationships")
            removed_final = self._keep_top_n_relationships(self.target_relationships)
            results['removed_final_trim'] = removed_final
        
        # Get final state
        final_stats = self._get_graph_statistics()
        results['initial_stats'] = current_stats
        results['final_stats'] = final_stats
        results['total_removed'] = current_stats['relationships'] - final_stats['relationships']
        
        logger.info(f"✅ REDUCTION COMPLETE: {final_stats['relationships']} relationships (removed {results['total_removed']})")
        logger.info(f"✅ Final ratio: {final_stats['ratio']:.1f}:1")
        
        return results
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics"""
        query = """
        MATCH (n)
        WITH count(n) as entity_count
        MATCH ()-[r]->()
        WITH entity_count, count(r) as relationship_count
        RETURN entity_count, relationship_count, 
               CASE WHEN relationship_count > 0 
                    THEN toFloat(entity_count) / toFloat(relationship_count)
                    ELSE 0.0 END as ratio
        """
        
        result = self.neo4j_service.execute_cypher(query)
        if result:
            return {
                'entities': result[0]['entity_count'],
                'relationships': result[0]['relationship_count'],
                'ratio': result[0]['ratio']
            }
        return {'entities': 0, 'relationships': 0, 'ratio': 0.0}
    
    def _remove_low_confidence_relationships(self, threshold: float = 0.8) -> int:
        """Remove relationships below confidence threshold"""
        query = """
        MATCH ()-[r]->()
        WHERE r.confidence < $threshold
        WITH r
        DELETE r
        RETURN count(r) as removed
        """
        
        result = self.neo4j_service.execute_cypher(query, {'threshold': threshold})
        removed = result[0]['removed'] if result else 0
        logger.info(f"   Removed {removed} low-confidence relationships (< {threshold})")
        return removed
    
    def _limit_relationships_per_entity(self, max_per_entity: int = 1) -> int:
        """Keep only the highest confidence relationships per entity"""
        query = """
        MATCH (n)-[r]->()
        WITH n, r
        ORDER BY n.id, r.confidence DESC
        WITH n, collect(r) as relationships
        UNWIND range(0, size(relationships)-1) as idx
        WITH n, relationships[idx] as r, idx
        WHERE idx >= $max_per_entity
        DELETE r
        RETURN count(r) as removed
        """
        
        result = self.neo4j_service.execute_cypher(query, {'max_per_entity': max_per_entity})
        removed = result[0]['removed'] if result else 0
        logger.info(f"   Removed {removed} excess relationships (>{max_per_entity} per entity)")
        return removed
    
    def _remove_cross_chunk_relationships(self) -> int:
        """Remove relationships created by cross-chunk analysis"""
        query = """
        MATCH ()-[r]->()
        WHERE r.created_by IN ['anti_silo_analysis', 'aggressive_anti_silo', 'nuclear_anti_silo']
           OR r.chunk_id CONTAINS 'cross_chunk'
           OR r.aggressive_mode = true
        DELETE r
        RETURN count(r) as removed
        """
        
        result = self.neo4j_service.execute_cypher(query)
        removed = result[0]['removed'] if result else 0
        logger.info(f"   Removed {removed} cross-chunk/anti-silo relationships")
        return removed
    
    def _remove_anti_silo_relationships(self) -> int:
        """Remove all anti-silo generated relationships"""
        query = """
        MATCH ()-[r]->()
        WHERE r.created_by CONTAINS 'anti_silo'
           OR r.nuclear_connection = true
           OR r.connection_strategy IS NOT NULL
        DELETE r
        RETURN count(r) as removed
        """
        
        result = self.neo4j_service.execute_cypher(query)
        removed = result[0]['removed'] if result else 0
        logger.info(f"   Removed {removed} anti-silo relationships")
        return removed
    
    def _keep_top_n_relationships(self, n: int) -> int:
        """Keep only the top N highest confidence relationships"""
        # First, get all relationships sorted by confidence
        query = """
        MATCH ()-[r]->()
        RETURN id(r) as rel_id, r.confidence as confidence
        ORDER BY r.confidence DESC, id(r)
        """
        
        all_relationships = self.neo4j_service.execute_cypher(query)
        if not all_relationships:
            return 0
        
        # Identify relationships to delete (those beyond top N)
        if len(all_relationships) <= n:
            return 0
        
        relationships_to_delete = all_relationships[n:]
        rel_ids_to_delete = [r['rel_id'] for r in relationships_to_delete]
        
        # Delete in batches
        batch_size = 100
        total_deleted = 0
        
        for i in range(0, len(rel_ids_to_delete), batch_size):
            batch_ids = rel_ids_to_delete[i:i + batch_size]
            delete_query = """
            UNWIND $rel_ids as rel_id
            MATCH ()-[r]->()
            WHERE id(r) = rel_id
            DELETE r
            RETURN count(r) as deleted
            """
            
            result = self.neo4j_service.execute_cypher(delete_query, {'rel_ids': batch_ids})
            if result:
                total_deleted += result[0]['deleted']
        
        logger.info(f"   Kept top {n} relationships, removed {total_deleted}")
        return total_deleted

async def update_extraction_settings():
    """Update extraction settings for aggressive reduction"""
    logger.info("📝 Updating extraction settings for aggressive reduction")
    
    db = SessionLocal()
    try:
        # Create aggressive extraction settings
        extraction_settings = {
            'mode': 'simple',  # Use simple mode for minimal relationships
            'min_entity_confidence': 0.8,
            'min_relationship_confidence': 0.85,  # Very high threshold
            'max_relationships_per_entity': 1,    # Strict limit
            'enable_multi_chunk_relationships': False,  # Disable cross-chunk
            'enable_nuclear_option': False,       # Disable nuclear anti-silo
            'max_entities_per_chunk': 10,        # Limit entities per chunk
            'max_relationships_per_chunk': 5,    # Limit relationships per chunk
            'global_relationship_cap': 250,      # Strict global cap
            'deduplicate_entities': True,
            'deduplicate_relationships': True,
            'progressive_storage_batch_size': 1  # Process one chunk at a time
        }
        
        # Update in database
        query = """
        UPDATE settings 
        SET value = %s::jsonb 
        WHERE category = 'knowledge_graph' 
        AND key = 'extraction'
        """
        
        import json
        db.execute(query, (json.dumps(extraction_settings),))
        
        # If no existing setting, insert it
        if db.rowcount == 0:
            insert_query = """
            INSERT INTO settings (category, key, value)
            VALUES ('knowledge_graph', 'extraction', %s::jsonb)
            """
            db.execute(insert_query, (json.dumps(extraction_settings),))
        
        db.commit()
        
        # Invalidate cache to force reload
        invalidate_knowledge_graph_cache()
        
        logger.info("✅ Extraction settings updated successfully")
        
        # Verify the update
        updated_settings = get_knowledge_graph_settings()
        logger.info(f"📊 New extraction settings: {updated_settings.get('extraction', {})}")
        
    except Exception as e:
        logger.error(f"❌ Failed to update settings: {e}")
        db.rollback()
    finally:
        db.close()

async def main():
    """Main execution"""
    logger.info("🚀 Knowledge Graph Relationship Explosion Fix")
    logger.info("=" * 60)
    
    # Step 1: Update extraction settings
    await update_extraction_settings()
    
    # Step 2: Execute emergency reduction
    reducer = KnowledgeGraphRelationshipReducer()
    results = await reducer.emergency_reduction()
    
    # Step 3: Display results
    logger.info("\n📊 FINAL RESULTS:")
    logger.info(f"Initial: {results['initial_stats']['entities']} entities, {results['initial_stats']['relationships']} relationships")
    logger.info(f"Final: {results['final_stats']['entities']} entities, {results['final_stats']['relationships']} relationships")
    logger.info(f"Removed: {results['total_removed']} relationships")
    logger.info(f"New ratio: {results['final_stats']['ratio']:.1f}:1")
    
    # Step 4: Provide recommendations
    logger.info("\n🎯 RECOMMENDATIONS:")
    logger.info("1. The extraction settings have been updated to 'simple' mode")
    logger.info("2. Multi-chunk relationships have been disabled")
    logger.info("3. Max 1 relationship per entity is now enforced")
    logger.info("4. Global cap set to 250 relationships")
    logger.info("5. Future documents will generate far fewer relationships")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())