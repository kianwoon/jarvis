"""
Monitor Knowledge Graph Relationships in Real-Time

This script provides detailed monitoring of the knowledge graph state
and relationship distribution.
"""

import asyncio
import logging
from typing import Dict, List, Any
from app.services.neo4j_service import get_neo4j_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphMonitor:
    """Monitor knowledge graph statistics and relationship distribution"""
    
    def __init__(self):
        self.neo4j_service = get_neo4j_service()
    
    async def analyze_graph_state(self) -> Dict[str, Any]:
        """Comprehensive analysis of current graph state"""
        logger.info("🔍 KNOWLEDGE GRAPH ANALYSIS")
        logger.info("=" * 60)
        
        analysis = {}
        
        # 1. Overall statistics
        overall_stats = self._get_overall_statistics()
        analysis['overall'] = overall_stats
        logger.info(f"\n📊 OVERALL STATISTICS:")
        logger.info(f"   Total Entities: {overall_stats['total_entities']:,}")
        logger.info(f"   Total Relationships: {overall_stats['total_relationships']:,}")
        logger.info(f"   Ratio: {overall_stats['ratio']:.1f}:1 (entities:relationships)")
        
        # 2. Relationship distribution by confidence
        confidence_dist = self._get_confidence_distribution()
        analysis['confidence_distribution'] = confidence_dist
        logger.info(f"\n📈 RELATIONSHIP CONFIDENCE DISTRIBUTION:")
        for range_label, count in confidence_dist.items():
            logger.info(f"   {range_label}: {count:,} relationships")
        
        # 3. Relationships per entity distribution
        entity_dist = self._get_relationships_per_entity_distribution()
        analysis['entity_distribution'] = entity_dist
        logger.info(f"\n📊 RELATIONSHIPS PER ENTITY:")
        for rel_count, entity_count in entity_dist.items():
            logger.info(f"   {rel_count} relationships: {entity_count:,} entities")
        
        # 4. Relationship types breakdown
        type_breakdown = self._get_relationship_type_breakdown()
        analysis['relationship_types'] = type_breakdown
        logger.info(f"\n🏷️  TOP RELATIONSHIP TYPES:")
        for rel_type, count in list(type_breakdown.items())[:10]:
            logger.info(f"   {rel_type}: {count:,}")
        
        # 5. Creation source analysis
        source_analysis = self._get_creation_source_analysis()
        analysis['creation_sources'] = source_analysis
        logger.info(f"\n🔧 RELATIONSHIP CREATION SOURCES:")
        for source, count in source_analysis.items():
            logger.info(f"   {source}: {count:,}")
        
        # 6. Document contribution analysis
        doc_analysis = self._get_document_contribution()
        analysis['document_contribution'] = doc_analysis
        logger.info(f"\n📄 DOCUMENT CONTRIBUTION:")
        logger.info(f"   Total documents: {doc_analysis['total_documents']}")
        logger.info(f"   Avg relationships per document: {doc_analysis['avg_relationships_per_doc']:.1f}")
        logger.info(f"   Max relationships from single doc: {doc_analysis['max_relationships_from_doc']}")
        
        # 7. Problem entities (with too many relationships)
        problem_entities = self._get_problem_entities()
        analysis['problem_entities'] = problem_entities
        if problem_entities:
            logger.info(f"\n⚠️  ENTITIES WITH EXCESSIVE RELATIONSHIPS:")
            for entity in problem_entities[:5]:  # Show top 5
                logger.info(f"   {entity['name']} ({entity['type']}): {entity['relationship_count']} relationships")
        
        # 8. Configuration check
        kg_settings = get_knowledge_graph_settings()
        extraction_config = kg_settings.get('extraction', {})
        analysis['current_config'] = extraction_config
        logger.info(f"\n⚙️  CURRENT EXTRACTION CONFIGURATION:")
        logger.info(f"   Mode: {extraction_config.get('mode', 'standard')}")
        logger.info(f"   Multi-chunk enabled: {extraction_config.get('enable_multi_chunk_relationships', True)}")
        logger.info(f"   Max per entity: {extraction_config.get('max_relationships_per_entity', 'not set')}")
        logger.info(f"   Global cap: {extraction_config.get('global_relationship_cap', 'not set')}")
        
        return analysis
    
    def _get_overall_statistics(self) -> Dict[str, Any]:
        """Get basic graph statistics"""
        query = """
        MATCH (n)
        WITH count(DISTINCT n) as entity_count
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
                'total_entities': result[0]['entity_count'],
                'total_relationships': result[0]['relationship_count'],
                'ratio': result[0]['ratio']
            }
        return {'total_entities': 0, 'total_relationships': 0, 'ratio': 0.0}
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of relationships by confidence score"""
        query = """
        MATCH ()-[r]->()
        WITH r.confidence as confidence
        RETURN 
            CASE 
                WHEN confidence >= 0.9 THEN '0.9-1.0 (Very High)'
                WHEN confidence >= 0.8 THEN '0.8-0.9 (High)'
                WHEN confidence >= 0.7 THEN '0.7-0.8 (Medium)'
                WHEN confidence >= 0.6 THEN '0.6-0.7 (Low)'
                ELSE '< 0.6 (Very Low)'
            END as confidence_range,
            count(*) as count
        ORDER BY confidence_range DESC
        """
        
        result = self.neo4j_service.execute_cypher(query)
        return {r['confidence_range']: r['count'] for r in result} if result else {}
    
    def _get_relationships_per_entity_distribution(self) -> Dict[str, int]:
        """Get distribution of how many relationships each entity has"""
        query = """
        MATCH (n)
        WITH n, size([(n)-[]-() | 1]) as rel_count
        WITH rel_count, count(n) as entity_count
        RETURN rel_count, entity_count
        ORDER BY rel_count
        LIMIT 20
        """
        
        result = self.neo4j_service.execute_cypher(query)
        return {f"{r['rel_count']}": r['entity_count'] for r in result} if result else {}
    
    def _get_relationship_type_breakdown(self) -> Dict[str, int]:
        """Get breakdown of relationship types"""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        
        result = self.neo4j_service.execute_cypher(query)
        return {r['rel_type']: r['count'] for r in result} if result else {}
    
    def _get_creation_source_analysis(self) -> Dict[str, int]:
        """Analyze where relationships came from"""
        query = """
        MATCH ()-[r]->()
        WITH 
            CASE 
                WHEN r.created_by CONTAINS 'anti_silo' THEN 'Anti-Silo Analysis'
                WHEN r.created_by CONTAINS 'nuclear' THEN 'Nuclear Anti-Silo'
                WHEN r.created_by = 'llm' THEN 'Direct LLM Extraction'
                WHEN r.chunk_id CONTAINS 'cross_chunk' THEN 'Cross-Chunk Analysis'
                WHEN r.aggressive_mode = true THEN 'Aggressive Mode'
                ELSE COALESCE(r.created_by, 'Unknown')
            END as source,
            count(*) as count
        RETURN source, count
        ORDER BY count DESC
        """
        
        result = self.neo4j_service.execute_cypher(query)
        return {r['source']: r['count'] for r in result} if result else {}
    
    def _get_document_contribution(self) -> Dict[str, Any]:
        """Analyze relationship contribution by document"""
        query = """
        MATCH ()-[r]->()
        WHERE r.document_id IS NOT NULL
        WITH r.document_id as doc_id, count(r) as rel_count
        WITH collect({doc_id: doc_id, count: rel_count}) as docs, 
             count(DISTINCT doc_id) as total_docs,
             sum(rel_count) as total_rels
        RETURN total_docs, 
               toFloat(total_rels) / toFloat(total_docs) as avg_per_doc,
               [d IN docs | d.count][0..1][0] as max_from_single_doc
        """
        
        result = self.neo4j_service.execute_cypher(query)
        if result:
            return {
                'total_documents': result[0]['total_docs'],
                'avg_relationships_per_doc': result[0]['avg_per_doc'],
                'max_relationships_from_doc': result[0]['max_from_single_doc']
            }
        return {'total_documents': 0, 'avg_relationships_per_doc': 0, 'max_relationships_from_doc': 0}
    
    def _get_problem_entities(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """Find entities with excessive relationships"""
        query = """
        MATCH (n)
        WITH n, size([(n)-[]-() | 1]) as rel_count
        WHERE rel_count > $threshold
        RETURN n.name as name, n.type as type, rel_count as relationship_count
        ORDER BY rel_count DESC
        LIMIT 10
        """
        
        result = self.neo4j_service.execute_cypher(query, {'threshold': threshold})
        return result if result else []

async def main():
    """Main monitoring execution"""
    monitor = KnowledgeGraphMonitor()
    
    # Run analysis
    analysis = await monitor.analyze_graph_state()
    
    # Provide recommendations
    logger.info("\n\n🎯 RECOMMENDATIONS BASED ON ANALYSIS:")
    
    total_rels = analysis['overall']['total_relationships']
    if total_rels > 300:
        logger.info(f"⚠️  CRITICAL: {total_rels} relationships exceeds target of 250-300")
        logger.info("   ACTION: Run fix_kg_relationship_explosion.py immediately")
    
    # Check for anti-silo explosion
    anti_silo_count = analysis['creation_sources'].get('Anti-Silo Analysis', 0)
    nuclear_count = analysis['creation_sources'].get('Nuclear Anti-Silo', 0)
    if anti_silo_count + nuclear_count > 100:
        logger.info(f"⚠️  Anti-silo generated {anti_silo_count + nuclear_count} relationships")
        logger.info("   ACTION: Disable anti-silo analysis in settings")
    
    # Check problem entities
    if analysis['problem_entities']:
        logger.info(f"⚠️  Found {len(analysis['problem_entities'])} entities with >10 relationships")
        logger.info("   ACTION: Apply stricter per-entity limits")
    
    # Check extraction mode
    current_mode = analysis['current_config'].get('mode', 'standard')
    if current_mode != 'simple' and total_rels > 300:
        logger.info(f"⚠️  Extraction mode '{current_mode}' is too permissive")
        logger.info("   ACTION: Switch to 'simple' mode")
    
    return analysis

if __name__ == "__main__":
    asyncio.run(main())