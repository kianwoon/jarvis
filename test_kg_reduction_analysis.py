#!/usr/bin/env python3
"""
Test script to analyze knowledge graph reduction via API endpoints
"""

import requests
import json
from datetime import datetime

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def pretty_print_json(data, title=""):
    """Pretty print JSON data with a title"""
    if title:
        print(f"\n{title}:")
        print("-" * len(title))
    print(json.dumps(data, indent=2))

def analyze_kg_reduction():
    """Analyze knowledge graph reduction through API endpoints"""
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH REDUCTION ANALYSIS")
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Get graph statistics
    print("\n1. GETTING GRAPH STATISTICS...")
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge-graph/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   - Total entities: {stats.get('total_entities', 0)}")
            print(f"   - Total relationships: {stats.get('total_relationships', 0)}")
            print(f"   - Documents processed: {stats.get('documents_processed', 0)}")
            
            # Show entity type distribution
            if 'entity_types' in stats:
                print("\n   Entity Type Distribution:")
                for entity_type, count in stats['entity_types'].items():
                    print(f"     - {entity_type}: {count}")
            
            # Show relationship type distribution
            if 'relationship_types' in stats:
                print("\n   Relationship Type Distribution:")
                for rel_type, count in stats['relationship_types'].items():
                    print(f"     - {rel_type}: {count}")
        else:
            print(f"   ❌ Failed to get stats: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting stats: {e}")
    
    # 2. Check isolated nodes
    print("\n2. CHECKING ISOLATED NODES...")
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge-graph/isolated-nodes")
        if response.status_code == 200:
            isolated_data = response.json()
            isolated_nodes = isolated_data.get('isolated_nodes', [])
            print(f"   - Total isolated nodes: {isolated_data.get('total_isolated', len(isolated_nodes))}")
            
            # Group by type
            type_counts = {}
            for node in isolated_nodes:
                node_type = node.get('type', 'Unknown')
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
            
            if type_counts:
                print("\n   Isolated Nodes by Type:")
                for node_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"     - {node_type}: {count}")
            
            # Show sample isolated nodes
            if isolated_nodes:
                print("\n   Sample Isolated Nodes:")
                for node in isolated_nodes[:5]:
                    print(f"     - {node.get('name', 'Unknown')} ({node.get('type', 'Unknown')})")
        else:
            print(f"   ❌ Failed to get isolated nodes: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting isolated nodes: {e}")
    
    # 3. Get silo analysis
    print("\n3. SILO ANALYSIS...")
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge-graph/silo-analysis")
        if response.status_code == 200:
            silo_data = response.json()
            print(f"   - Total silos: {silo_data.get('total_silos', 0)}")
            print(f"   - Nodes in silos: {silo_data.get('nodes_in_silos', 0)}")
            
            if 'silo_types' in silo_data:
                print("\n   Silo Distribution by Type:")
                for silo_type, info in silo_data['silo_types'].items():
                    print(f"     - {silo_type}: {info.get('count', 0)} nodes")
        else:
            print(f"   ❌ Failed to get silo analysis: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting silo analysis: {e}")
    
    # 4. Get connectivity analysis
    print("\n4. CONNECTIVITY ANALYSIS...")
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge-graph/connectivity-analysis")
        if response.status_code == 200:
            connectivity = response.json()
            
            # Hub analysis
            if 'hub_analysis' in connectivity:
                hubs = connectivity['hub_analysis']
                print(f"\n   Top Hub Nodes (most connected):")
                for hub in hubs[:5]:
                    print(f"     - {hub['name']} ({hub['type']}): {hub['connections']} connections")
            
            # Component analysis
            if 'components' in connectivity:
                components = connectivity['components']
                print(f"\n   Connected Components:")
                print(f"     - Total components: {len(components)}")
                if components:
                    largest = max(components, key=lambda x: x['size'])
                    print(f"     - Largest component: {largest['size']} nodes")
                    print(f"     - Smallest component: {min(components, key=lambda x: x['size'])['size']} nodes")
        else:
            print(f"   ❌ Failed to get connectivity analysis: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting connectivity analysis: {e}")
    
    # 5. Debug and diagnostics
    print("\n5. RUNNING DIAGNOSTICS...")
    try:
        response = requests.get(f"{API_BASE_URL}/knowledge-graph/debug")
        if response.status_code == 200:
            debug_info = response.json()
            
            # Recent operations
            if 'recent_operations' in debug_info:
                print("\n   Recent Operations:")
                for op in debug_info['recent_operations'][:5]:
                    print(f"     - {op.get('operation', 'Unknown')}: {op.get('timestamp', 'Unknown')}")
                    if 'details' in op:
                        print(f"       {op['details']}")
            
            # System state
            if 'system_state' in debug_info:
                state = debug_info['system_state']
                print(f"\n   System State:")
                print(f"     - Neo4j connected: {state.get('neo4j_connected', False)}")
                print(f"     - Last deduplication: {state.get('last_deduplication', 'Never')}")
                print(f"     - Last anti-silo run: {state.get('last_anti_silo', 'Never')}")
        else:
            print(f"   ❌ Failed to get debug info: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting debug info: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n🔍 FINDINGS:")
    print("   The significant reduction from 67 nodes/530 relationships to 32 nodes/37 relationships")
    print("   appears to be caused by:")
    print("\n   1. AGGRESSIVE DEDUPLICATION:")
    print("      - Entity consolidation based on deterministic IDs")
    print("      - Merging of entities with similar names across documents")
    print("      - Hub node cleanup removing empty organizational nodes")
    
    print("\n   2. RELATIONSHIP CONSOLIDATION:")
    print("      - The original 530 relationships likely counted bidirectional edges")
    print("      - Neo4j counts only directed edges (A->B, not A<->B)")
    print("      - Duplicate relationships between same entities were merged")
    
    print("\n   3. ANTI-SILO PROCESSING:")
    print("      - Isolated nodes may have been removed if they had no semantic value")
    print("      - Empty hub nodes were cleaned up")
    print("      - Low-confidence entities might have been filtered")
    
    print("\n⚠️  POTENTIAL ISSUES:")
    print("   1. Entity linking threshold might be too aggressive")
    print("   2. Relationship extraction might need tuning")
    print("   3. Some valid relationships might have been lost during deduplication")
    
    print("\n✅ RECOMMENDATIONS:")
    print("   1. Review and adjust entity linking similarity threshold (currently 0.5)")
    print("   2. Check relationship extraction prompts for proper JSON formatting")
    print("   3. Consider re-processing a test document to validate extraction")
    print("   4. Monitor future ingestions to ensure proper entity/relationship creation")
    print("   5. Run anti-silo analysis with less aggressive settings")

if __name__ == "__main__":
    analyze_kg_reduction()