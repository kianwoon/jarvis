#!/usr/bin/env python3
"""
Knowledge Graph Performance Optimization Analysis & Plan

This script analyzes the current performance issues and provides a comprehensive
optimization strategy to achieve ≤4 relationships per entity and smooth browser performance.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Import services
from app.services.neo4j_service import get_neo4j_service
from app.services.knowledge_graph_service import get_knowledge_graph_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

class KnowledgeGraphPerformanceAnalyzer:
    """Analyze and optimize knowledge graph performance"""
    
    def __init__(self):
        self.neo4j_service = get_neo4j_service()
        self.kg_service = get_knowledge_graph_service()
        
    async def analyze_current_state(self) -> Dict[str, Any]:
        """Analyze current knowledge graph state and performance metrics"""
        print("\n🔍 ANALYZING CURRENT KNOWLEDGE GRAPH STATE...")
        
        # Get entity and relationship counts
        entity_count = self.neo4j_service.get_total_entity_count()
        relationship_count = self.neo4j_service.get_total_relationship_count()
        
        # Calculate ratio
        ratio = relationship_count / entity_count if entity_count > 0 else 0
        
        # Get relationship distribution
        distribution_query = """
        MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as rel_count
        RETURN 
            min(rel_count) as min_rels,
            max(rel_count) as max_rels,
            avg(rel_count) as avg_rels,
            stdev(rel_count) as stdev_rels,
            percentileCont(rel_count, 0.5) as median_rels,
            percentileCont(rel_count, 0.75) as p75_rels,
            percentileCont(rel_count, 0.95) as p95_rels
        """
        
        dist_result = self.neo4j_service.execute_cypher(distribution_query)
        distribution = dist_result[0] if dist_result else {}
        
        # Find high-degree nodes
        high_degree_query = """
        MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as rel_count
        WHERE rel_count > 10
        RETURN n.name as name, n.type as type, rel_count
        ORDER BY rel_count DESC
        LIMIT 10
        """
        
        high_degree_nodes = self.neo4j_service.execute_cypher(high_degree_query)
        
        # Analyze relationship types
        rel_type_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        
        rel_types = self.neo4j_service.execute_cypher(rel_type_query)
        
        # Check for duplicate relationships
        duplicate_query = """
        MATCH (a)-[r1]->(b)
        MATCH (a)-[r2]->(b)
        WHERE id(r1) < id(r2) AND type(r1) = type(r2)
        RETURN count(*) as duplicate_count
        """
        
        dup_result = self.neo4j_service.execute_cypher(duplicate_query)
        duplicate_count = dup_result[0]['duplicate_count'] if dup_result else 0
        
        analysis = {
            'entity_count': entity_count,
            'relationship_count': relationship_count,
            'ratio': ratio,
            'distribution': distribution,
            'high_degree_nodes': high_degree_nodes,
            'relationship_types': rel_types,
            'duplicate_relationships': duplicate_count,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print analysis
        print(f"\n📊 CURRENT STATE:")
        print(f"   Entities: {entity_count}")
        print(f"   Relationships: {relationship_count}")
        print(f"   Ratio: {ratio:.2f} relationships per entity")
        print(f"   Target: ≤4 relationships per entity")
        print(f"   Excess: {relationship_count - (entity_count * 4)} relationships to remove")
        
        print(f"\n📈 DISTRIBUTION:")
        print(f"   Min: {distribution.get('min_rels', 0):.0f}")
        print(f"   Median: {distribution.get('median_rels', 0):.1f}")
        print(f"   Average: {distribution.get('avg_rels', 0):.1f}")
        print(f"   75th percentile: {distribution.get('p75_rels', 0):.1f}")
        print(f"   95th percentile: {distribution.get('p95_rels', 0):.1f}")
        print(f"   Max: {distribution.get('max_rels', 0):.0f}")
        
        print(f"\n🔥 HIGH-DEGREE NODES (>10 relationships):")
        for node in high_degree_nodes[:5]:
            print(f"   - {node['name']} ({node['type']}): {node['rel_count']} relationships")
        
        print(f"\n🔗 RELATIONSHIP TYPES:")
        total_rels = sum(rt['count'] for rt in rel_types)
        for rt in rel_types[:5]:
            percentage = (rt['count'] / total_rels * 100) if total_rels > 0 else 0
            print(f"   - {rt['rel_type']}: {rt['count']} ({percentage:.1f}%)")
        
        if duplicate_count > 0:
            print(f"\n⚠️  DUPLICATES: {duplicate_count} duplicate relationships found!")
        
        return analysis

    async def identify_enforcement_failures(self) -> Dict[str, Any]:
        """Identify why enforcement layers are failing"""
        print("\n🔍 ANALYZING ENFORCEMENT FAILURES...")
        
        # Check relationships created by different systems
        creation_source_query = """
        MATCH ()-[r]->()
        RETURN 
            COALESCE(r.created_by, 'unknown') as source,
            count(r) as count
        ORDER BY count DESC
        """
        
        creation_sources = self.neo4j_service.execute_cypher(creation_source_query)
        
        # Check confidence distribution
        confidence_query = """
        MATCH ()-[r]->()
        WHERE r.confidence IS NOT NULL
        RETURN 
            CASE 
                WHEN r.confidence >= 0.8 THEN 'high (≥0.8)'
                WHEN r.confidence >= 0.6 THEN 'medium (0.6-0.8)'
                WHEN r.confidence >= 0.4 THEN 'low (0.4-0.6)'
                ELSE 'very low (<0.4)'
            END as confidence_level,
            count(r) as count
        ORDER BY count DESC
        """
        
        confidence_dist = self.neo4j_service.execute_cypher(confidence_query)
        
        # Check anti-silo relationships
        anti_silo_query = """
        MATCH ()-[r]->()
        WHERE r.created_by CONTAINS 'anti_silo' OR r.nuclear_connection = true
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        """
        
        anti_silo_rels = self.neo4j_service.execute_cypher(anti_silo_query)
        
        failures = {
            'creation_sources': creation_sources,
            'confidence_distribution': confidence_dist,
            'anti_silo_relationships': anti_silo_rels
        }
        
        print(f"\n🔧 RELATIONSHIP CREATION SOURCES:")
        for source in creation_sources:
            print(f"   - {source['source']}: {source['count']} relationships")
        
        print(f"\n📊 CONFIDENCE DISTRIBUTION:")
        for conf in confidence_dist:
            print(f"   - {conf['confidence_level']}: {conf['count']} relationships")
        
        print(f"\n🔗 ANTI-SILO RELATIONSHIPS:")
        anti_silo_total = sum(rel['count'] for rel in anti_silo_rels)
        print(f"   Total anti-silo: {anti_silo_total} relationships")
        for rel in anti_silo_rels[:3]:
            print(f"   - {rel['rel_type']}: {rel['count']}")
        
        return failures

    async def generate_optimization_plan(self) -> Dict[str, Any]:
        """Generate comprehensive optimization plan"""
        print("\n📋 GENERATING OPTIMIZATION PLAN...")
        
        # Current state
        entity_count = self.neo4j_service.get_total_entity_count()
        relationship_count = self.neo4j_service.get_total_relationship_count()
        current_ratio = relationship_count / entity_count if entity_count > 0 else 0
        
        # Target state
        target_ratio = 4.0
        target_relationships = int(entity_count * target_ratio)
        relationships_to_remove = relationship_count - target_relationships
        
        optimization_plan = {
            'current_state': {
                'entities': entity_count,
                'relationships': relationship_count,
                'ratio': current_ratio
            },
            'target_state': {
                'entities': entity_count,
                'relationships': target_relationships,
                'ratio': target_ratio
            },
            'reduction_required': relationships_to_remove,
            'strategies': []
        }
        
        print(f"\n🎯 OPTIMIZATION TARGET:")
        print(f"   Current: {relationship_count} relationships ({current_ratio:.1f} per entity)")
        print(f"   Target: {target_relationships} relationships ({target_ratio:.1f} per entity)")
        print(f"   Must remove: {relationships_to_remove} relationships")
        
        # Strategy 1: Remove low-confidence relationships
        print(f"\n📌 STRATEGY 1: Remove Low-Confidence Relationships")
        low_conf_query = """
        MATCH ()-[r]->()
        WHERE r.confidence < 0.6
        RETURN count(r) as count
        """
        low_conf_result = self.neo4j_service.execute_cypher(low_conf_query)
        low_conf_count = low_conf_result[0]['count'] if low_conf_result else 0
        print(f"   Can remove: {low_conf_count} relationships (confidence < 0.6)")
        
        optimization_plan['strategies'].append({
            'name': 'Remove Low-Confidence Relationships',
            'potential_reduction': low_conf_count,
            'query': """
                MATCH ()-[r]->()
                WHERE r.confidence < 0.6
                DELETE r
            """
        })
        
        # Strategy 2: Remove duplicate relationships
        print(f"\n📌 STRATEGY 2: Remove Duplicate Relationships")
        dup_query = """
        MATCH (a)-[r1]->(b)
        MATCH (a)-[r2]->(b)
        WHERE id(r1) < id(r2) AND type(r1) = type(r2)
        RETURN count(r2) as count
        """
        dup_result = self.neo4j_service.execute_cypher(dup_query)
        dup_count = dup_result[0]['count'] if dup_result else 0
        print(f"   Can remove: {dup_count} duplicate relationships")
        
        optimization_plan['strategies'].append({
            'name': 'Remove Duplicate Relationships',
            'potential_reduction': dup_count,
            'query': """
                MATCH (a)-[r1]->(b)
                MATCH (a)-[r2]->(b)
                WHERE id(r1) < id(r2) AND type(r1) = type(r2)
                DELETE r2
            """
        })
        
        # Strategy 3: Remove anti-silo nuclear relationships
        print(f"\n📌 STRATEGY 3: Remove Nuclear Anti-Silo Relationships")
        nuclear_query = """
        MATCH ()-[r]->()
        WHERE r.nuclear_connection = true OR r.created_by CONTAINS 'nuclear'
        RETURN count(r) as count
        """
        nuclear_result = self.neo4j_service.execute_cypher(nuclear_query)
        nuclear_count = nuclear_result[0]['count'] if nuclear_result else 0
        print(f"   Can remove: {nuclear_count} nuclear anti-silo relationships")
        
        optimization_plan['strategies'].append({
            'name': 'Remove Nuclear Anti-Silo Relationships',
            'potential_reduction': nuclear_count,
            'query': """
                MATCH ()-[r]->()
                WHERE r.nuclear_connection = true OR r.created_by CONTAINS 'nuclear'
                DELETE r
            """
        })
        
        # Strategy 4: Cap high-degree nodes
        print(f"\n📌 STRATEGY 4: Cap High-Degree Nodes")
        high_degree_query = """
        MATCH (n)-[r]-()
        WITH n, collect(r) as rels, count(r) as rel_count
        WHERE rel_count > 8
        RETURN sum(rel_count - 8) as excess_count
        """
        high_degree_result = self.neo4j_service.execute_cypher(high_degree_query)
        high_degree_excess = high_degree_result[0]['excess_count'] if high_degree_result else 0
        print(f"   Can remove: {high_degree_excess} relationships from nodes with >8 connections")
        
        optimization_plan['strategies'].append({
            'name': 'Cap High-Degree Nodes',
            'potential_reduction': high_degree_excess,
            'query': """
                MATCH (n)-[r]-()
                WITH n, collect(r) as rels
                WHERE size(rels) > 8
                WITH n, rels, [i in range(0, size(rels)-8) | rels[i]] as to_delete
                UNWIND to_delete as r
                DELETE r
            """
        })
        
        # Calculate total potential reduction
        total_potential = sum(s['potential_reduction'] for s in optimization_plan['strategies'])
        optimization_plan['total_potential_reduction'] = total_potential
        optimization_plan['sufficient'] = total_potential >= relationships_to_remove
        
        print(f"\n✅ TOTAL POTENTIAL REDUCTION: {total_potential} relationships")
        print(f"   Required: {relationships_to_remove}")
        print(f"   Sufficient: {'YES' if optimization_plan['sufficient'] else 'NO'}")
        
        return optimization_plan

    async def recommend_settings_changes(self) -> Dict[str, Any]:
        """Recommend settings changes to prevent future issues"""
        print("\n⚙️  RECOMMENDED SETTINGS CHANGES:")
        
        recommendations = {
            'extraction_mode': {
                'current': get_knowledge_graph_settings().get('extraction', {}).get('mode', 'standard'),
                'recommended': 'simple',
                'reason': 'Simple mode enforces strictest limits'
            },
            'relationship_limits': {
                'max_relationships_per_entity': 2,
                'max_relationships_per_chunk': 3,
                'global_relationship_cap': 150,
                'reason': 'Aggressive limits to maintain performance'
            },
            'confidence_thresholds': {
                'min_relationship_confidence': 0.7,
                'reason': 'Higher threshold reduces low-quality relationships'
            },
            'anti_silo_settings': {
                'enable_anti_silo': False,
                'enable_nuclear_option': False,
                'reason': 'Anti-silo analysis creates excessive relationships'
            },
            'enforcement_improvements': {
                'enforce_chunk_limits_first': True,
                'apply_global_budget_strictly': True,
                'post_process_ratio_check': True,
                'reason': 'Multiple enforcement layers to prevent ratio explosion'
            }
        }
        
        print(json.dumps(recommendations, indent=2))
        
        return recommendations

    async def generate_browser_optimizations(self) -> Dict[str, Any]:
        """Generate browser/visualization optimizations"""
        print("\n🖥️  BROWSER OPTIMIZATION RECOMMENDATIONS:")
        
        optimizations = {
            'force_simulation': {
                'reduce_iterations': {
                    'current': 300,
                    'recommended': 150,
                    'reason': 'Fewer iterations = faster initial render'
                },
                'optimize_forces': {
                    'charge_strength': -300,
                    'link_distance': 50,
                    'collision_radius': 30,
                    'reason': 'Balanced forces for dense graphs'
                }
            },
            'rendering': {
                'use_canvas_instead_of_svg': True,
                'implement_node_culling': True,
                'viewport_based_rendering': True,
                'reason': 'Canvas performs better for >100 nodes'
            },
            'interaction': {
                'debounce_drag_events': 16,  # ms
                'throttle_zoom_events': 50,  # ms
                'disable_animations_on_drag': True,
                'reason': 'Reduce computation during interaction'
            },
            'data_management': {
                'implement_relationship_filtering': True,
                'default_filter': 'confidence >= 0.7',
                'progressive_loading': True,
                'reason': 'Show high-quality relationships first'
            },
            'ui_features': {
                'relationship_visibility_toggle': True,
                'node_clustering_option': True,
                'performance_mode_toggle': True,
                'reason': 'User control over visualization complexity'
            }
        }
        
        print(json.dumps(optimizations, indent=2))
        
        return optimizations

    async def execute_emergency_cleanup(self) -> Dict[str, Any]:
        """Execute emergency cleanup to achieve target ratio"""
        print("\n🚨 EMERGENCY CLEANUP EXECUTION:")
        
        # Get optimization plan
        plan = await self.generate_optimization_plan()
        
        if not plan['sufficient']:
            print("⚠️  WARNING: Planned strategies may not achieve target ratio!")
        
        results = {
            'strategies_executed': [],
            'total_removed': 0,
            'errors': []
        }
        
        # Execute each strategy
        for strategy in plan['strategies']:
            try:
                print(f"\n🔧 Executing: {strategy['name']}")
                print(f"   Expected reduction: {strategy['potential_reduction']}")
                
                # For safety, let's just print the queries instead of executing
                print(f"   Query: {strategy['query'][:100]}...")
                
                # Uncomment below to actually execute
                # result = self.neo4j_service.execute_cypher(strategy['query'])
                # removed = strategy['potential_reduction']
                # results['strategies_executed'].append(strategy['name'])
                # results['total_removed'] += removed
                # print(f"   ✅ Removed: {removed} relationships")
                
            except Exception as e:
                error_msg = f"Failed to execute {strategy['name']}: {str(e)}"
                results['errors'].append(error_msg)
                print(f"   ❌ {error_msg}")
        
        # Verify final state
        if results['total_removed'] > 0:
            final_state = await self.analyze_current_state()
            results['final_state'] = final_state
            print(f"\n📊 FINAL STATE:")
            print(f"   Ratio: {final_state['ratio']:.2f} relationships per entity")
            print(f"   Success: {'YES' if final_state['ratio'] <= 4.0 else 'NO'}")
        
        return results

async def main():
    """Main execution function"""
    print("=" * 80)
    print("KNOWLEDGE GRAPH PERFORMANCE OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    analyzer = KnowledgeGraphPerformanceAnalyzer()
    
    # 1. Analyze current state
    current_state = await analyzer.analyze_current_state()
    
    # 2. Identify enforcement failures
    failures = await analyzer.identify_enforcement_failures()
    
    # 3. Generate optimization plan
    optimization_plan = await analyzer.generate_optimization_plan()
    
    # 4. Recommend settings changes
    settings_recommendations = await analyzer.recommend_settings_changes()
    
    # 5. Generate browser optimizations
    browser_optimizations = await analyzer.generate_browser_optimizations()
    
    # 6. Option to execute emergency cleanup
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n🎯 OPTIMIZATION SUMMARY:")
    print(f"   Current ratio: {current_state['ratio']:.2f}")
    print(f"   Target ratio: 4.0")
    print(f"   Relationships to remove: {optimization_plan['reduction_required']}")
    print(f"   Can be achieved: {'YES' if optimization_plan['sufficient'] else 'NO'}")
    
    print(f"\n💡 KEY RECOMMENDATIONS:")
    print(f"   1. Switch to 'simple' extraction mode")
    print(f"   2. Disable anti-silo analysis")
    print(f"   3. Increase confidence threshold to 0.7")
    print(f"   4. Implement browser optimizations")
    print(f"   5. Add relationship filtering UI")
    
    # Save full analysis
    analysis_report = {
        'timestamp': datetime.now().isoformat(),
        'current_state': current_state,
        'enforcement_failures': failures,
        'optimization_plan': optimization_plan,
        'settings_recommendations': settings_recommendations,
        'browser_optimizations': browser_optimizations
    }
    
    with open('kg_performance_analysis.json', 'w') as f:
        json.dump(analysis_report, f, indent=2)
    
    print(f"\n📄 Full analysis saved to: kg_performance_analysis.json")
    
    # print("\n🚨 Execute emergency cleanup? (uncomment code in execute_emergency_cleanup to enable)")
    # cleanup_results = await analyzer.execute_emergency_cleanup()

if __name__ == "__main__":
    asyncio.run(main())