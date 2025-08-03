#!/usr/bin/env python3
"""
Analyze the reduction in knowledge graph data to determine if it's correct or indicates a problem.
"""

import logging
from app.services.neo4j_service import get_neo4j_service
from app.core.config import get_settings
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_knowledge_graph_state():
    """Comprehensive analysis of the current knowledge graph state"""
    neo4j_service = get_neo4j_service()
    
    if not neo4j_service.is_enabled():
        logger.error("Neo4j is not enabled")
        return
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH REDUCTION ANALYSIS")
    print("="*80)
    
    # 1. Basic statistics
    print("\n1. CURRENT GRAPH STATISTICS:")
    db_info = neo4j_service.get_database_info()
    print(f"   - Total nodes: {db_info.get('node_count', 0)}")
    print(f"   - Total relationships: {db_info.get('relationship_count', 0)}")
    
    # 2. Node type distribution
    print("\n2. NODE TYPE DISTRIBUTION:")
    node_types_query = """
    MATCH (n)
    UNWIND labels(n) as label
    WITH label, count(*) as count
    ORDER BY count DESC
    RETURN label, count
    """
    node_types = neo4j_service.execute_cypher(node_types_query)
    for result in node_types:
        print(f"   - {result['label']}: {result['count']} nodes")
    
    # 3. Relationship type distribution
    print("\n3. RELATIONSHIP TYPE DISTRIBUTION:")
    rel_types_query = """
    MATCH ()-[r]->()
    WITH type(r) as rel_type, count(*) as count
    ORDER BY count DESC
    RETURN rel_type, count
    """
    rel_types = neo4j_service.execute_cypher(rel_types_query)
    for result in rel_types:
        print(f"   - {result['rel_type']}: {result['count']} relationships")
    
    # 4. Check for duplicate entities
    print("\n4. CHECKING FOR DUPLICATE ENTITIES:")
    duplicates_query = """
    MATCH (n)
    WITH n.name as name, collect(n) as nodes, count(n) as count
    WHERE count > 1 AND name IS NOT NULL
    RETURN name, count, [node IN nodes | labels(node)[0]] as types
    ORDER BY count DESC
    LIMIT 10
    """
    duplicates = neo4j_service.execute_cypher(duplicates_query)
    if duplicates:
        print("   Found duplicate entities:")
        for dup in duplicates:
            print(f"   - '{dup['name']}': {dup['count']} instances (types: {dup['types']})")
    else:
        print("   ✅ No duplicate entities found")
    
    # 5. Check entity properties to understand deduplication
    print("\n5. SAMPLE ENTITY PROPERTIES (to understand deduplication):")
    sample_entities_query = """
    MATCH (n)
    WHERE n.id IS NOT NULL
    WITH n, labels(n)[0] as type
    RETURN n.id as id, n.name as name, type, n.confidence as confidence, 
           n.document_count as doc_count, n.created_at as created_at
    ORDER BY n.created_at DESC
    LIMIT 10
    """
    sample_entities = neo4j_service.execute_cypher(sample_entities_query)
    for entity in sample_entities[:5]:
        print(f"   - {entity['name']} ({entity['type']})")
        print(f"     ID: {entity['id']}")
        print(f"     Confidence: {entity.get('confidence', 'N/A')}")
        print(f"     Doc count: {entity.get('doc_count', 'N/A')}")
    
    # 6. Check for isolated nodes
    print("\n6. ISOLATED NODES ANALYSIS:")
    isolated_query = """
    MATCH (n)
    WHERE NOT EXISTS((n)-[]-())
    WITH labels(n)[0] as type, count(*) as count
    RETURN type, count
    ORDER BY count DESC
    """
    isolated_types = neo4j_service.execute_cypher(isolated_query)
    total_isolated = sum(result['count'] for result in isolated_types)
    print(f"   Total isolated nodes: {total_isolated}")
    for result in isolated_types:
        print(f"   - {result['type']}: {result['count']} isolated nodes")
    
    # 7. Check for hub nodes
    print("\n7. HUB NODES ANALYSIS:")
    hub_query = """
    MATCH (n:HUB)
    OPTIONAL MATCH (n)-[r]-()
    WITH n, count(r) as connections
    RETURN n.name as name, n.type as type, connections
    ORDER BY connections DESC
    """
    hubs = neo4j_service.execute_cypher(hub_query)
    if hubs:
        print(f"   Found {len(hubs)} hub nodes:")
        for hub in hubs:
            print(f"   - {hub['name']} ({hub['type']}): {hub['connections']} connections")
    else:
        print("   ⚠️  No hub nodes found")
    
    # 8. Check relationship patterns
    print("\n8. RELATIONSHIP PATTERNS:")
    pattern_query = """
    MATCH (a)-[r]->(b)
    WITH labels(a)[0] as from_type, type(r) as rel_type, labels(b)[0] as to_type, count(*) as count
    WHERE count > 1
    RETURN from_type + ' --[' + rel_type + ']--> ' + to_type as pattern, count
    ORDER BY count DESC
    LIMIT 10
    """
    patterns = neo4j_service.execute_cypher(pattern_query)
    for pattern in patterns:
        print(f"   - {pattern['pattern']}: {pattern['count']} instances")
    
    # 9. Check for recently deleted nodes (if we have audit/history)
    print("\n9. ANTI-SILO PROCESSING IMPACT:")
    # Check for entities that were recently linked
    recent_links_query = """
    MATCH (a)-[r]->(b)
    WHERE r.created_at IS NOT NULL
    WITH r, a, b
    ORDER BY r.created_at DESC
    LIMIT 10
    RETURN a.name as source, type(r) as rel_type, b.name as target, 
           r.created_at as created_at, r.discovered_by as discovered_by
    """
    recent_links = neo4j_service.execute_cypher(recent_links_query)
    if recent_links:
        print("   Recent relationships created:")
        for link in recent_links[:5]:
            print(f"   - {link['source']} --[{link['rel_type']}]--> {link['target']}")
            print(f"     Created: {link.get('created_at', 'Unknown')}, By: {link.get('discovered_by', 'Unknown')}")
    
    # 10. Document coverage analysis
    print("\n10. DOCUMENT COVERAGE:")
    doc_coverage_query = """
    MATCH (n)
    WHERE n.document_id IS NOT NULL
    WITH DISTINCT n.document_id as doc_id, count(n) as entities_per_doc
    RETURN count(doc_id) as total_documents, avg(entities_per_doc) as avg_entities_per_doc,
           min(entities_per_doc) as min_entities, max(entities_per_doc) as max_entities
    """
    coverage = neo4j_service.execute_cypher(doc_coverage_query)
    if coverage:
        stats = coverage[0]
        print(f"   - Total documents: {stats.get('total_documents', 0)}")
        print(f"   - Avg entities per doc: {stats.get('avg_entities_per_doc', 0):.1f}")
        print(f"   - Min/Max entities: {stats.get('min_entities', 0)}/{stats.get('max_entities', 0)}")
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY:")
    print("="*80)
    
    # Calculate ratios
    if db_info.get('node_count', 0) > 0:
        rel_to_node_ratio = db_info.get('relationship_count', 0) / db_info.get('node_count', 1)
        isolated_ratio = total_isolated / db_info.get('node_count', 1) * 100
        
        print(f"\n📊 Key Metrics:")
        print(f"   - Relationship to Node Ratio: {rel_to_node_ratio:.2f}")
        print(f"   - Isolated Nodes: {isolated_ratio:.1f}%")
        
        if rel_to_node_ratio < 1.5:
            print("\n⚠️  LOW CONNECTIVITY: The graph has fewer relationships than typical.")
            print("   This could indicate:")
            print("   1. Aggressive deduplication removed valid relationships")
            print("   2. Anti-silo processing is not creating enough connections")
            print("   3. Entity linking threshold is too strict")
        
        if isolated_ratio > 20:
            print("\n⚠️  HIGH ISOLATION: Many nodes have no connections.")
            print("   Consider running anti-silo analysis to connect isolated entities.")
    
    print("\n🔍 PROBABLE CAUSES OF REDUCTION:")
    print("   1. Entity Deduplication: Multiple instances of same entity merged")
    print("   2. Hub Node Cleanup: Empty hub nodes removed")
    print("   3. Relationship Normalization: Duplicate relationships consolidated")
    print("   4. Data Quality Improvements: Invalid or low-confidence data removed")
    
    print("\n✅ RECOMMENDED ACTIONS:")
    print("   1. Review entity linking threshold (currently may be too aggressive)")
    print("   2. Check if anti-silo processing is creating appropriate connections")
    print("   3. Verify that relationship extraction prompts are working correctly")
    print("   4. Consider re-processing a test document to validate extraction")

if __name__ == "__main__":
    analyze_knowledge_graph_state()