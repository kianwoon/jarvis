#!/usr/bin/env python3
"""
Critical Anti-Silo System Debugging Script
Diagnoses why system reports 0 isolated nodes but user sees silo nodes
"""

import asyncio
import json
from neo4j import GraphDatabase
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AntiSiloDebugger:
    def __init__(self):
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "jarvis_neo4j_password")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
        
    def close(self):
        self.driver.close()
        
    def run_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    
    def debug_isolated_nodes(self):
        """Debug isolated node detection logic"""
        print("\n" + "="*80)
        print("🔍 ANTI-SILO SYSTEM DEBUGGING")
        print("="*80)
        
        # 1. Count ALL nodes
        all_nodes_query = "MATCH (n) RETURN count(n) as total_nodes"
        total_nodes = self.run_query(all_nodes_query)[0]['total_nodes']
        print(f"\n📊 Total nodes in database: {total_nodes}")
        
        # 2. Count ALL relationships
        all_rels_query = "MATCH ()-[r]->() RETURN count(r) as total_relationships"
        total_rels = self.run_query(all_rels_query)[0]['total_relationships']
        print(f"📊 Total relationships in database: {total_rels}")
        
        # 3. Simple isolated node check (what the viewer might be using)
        simple_isolated_query = """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN count(n) as isolated_count
        """
        simple_isolated = self.run_query(simple_isolated_query)[0]['isolated_count']
        print(f"\n🚨 Simple isolated node count: {simple_isolated}")
        
        # 4. List actual isolated nodes
        if simple_isolated > 0:
            list_isolated_query = """
            MATCH (n)
            WHERE NOT (n)--()
            RETURN n.id as id, n.name as name, n.type as type, labels(n) as labels
            LIMIT 10
            """
            isolated_nodes = self.run_query(list_isolated_query)
            print(f"\n🔴 Sample isolated nodes:")
            for node in isolated_nodes:
                print(f"   - {node['name']} (id: {node['id']}, type: {node['type']}, labels: {node['labels']})")
        
        # 5. Check the complex anti-silo query (what the system uses)
        complex_isolated_query = """
        MATCH (n)
        WITH n.id as entity_id, collect(n) as nodes
        WITH entity_id, nodes[0] as representative_node
        OPTIONAL MATCH (any_instance)
        WHERE any_instance.id = entity_id
        OPTIONAL MATCH (any_instance)-[r]-()
        WITH entity_id, representative_node, count(r) as total_relationships
        WHERE total_relationships = 0
        RETURN count(entity_id) as isolated_count
        """
        complex_isolated = self.run_query(complex_isolated_query)[0]['isolated_count']
        print(f"\n🤔 Complex anti-silo query count: {complex_isolated}")
        
        # 6. Check for nodes with only self-relationships
        self_rel_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-(other)
        WITH n, collect(DISTINCT other) as connected_nodes
        WHERE size(connected_nodes) = 0 OR (size(connected_nodes) = 1 AND connected_nodes[0] = n)
        RETURN count(n) as self_connected_only
        """
        self_connected = self.run_query(self_rel_query)[0]['self_connected_only']
        print(f"\n🔄 Nodes with only self-relationships: {self_connected}")
        
        # 7. Check relationship types
        rel_types_query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(r) as count
        ORDER BY count DESC
        LIMIT 10
        """
        rel_types = self.run_query(rel_types_query)
        print(f"\n📋 Relationship types distribution:")
        for rel in rel_types:
            print(f"   - {rel['rel_type']}: {rel['count']}")
        
        # 8. Check for duplicate nodes
        duplicate_query = """
        MATCH (n)
        WITH n.id as entity_id, count(n) as duplicates
        WHERE duplicates > 1
        RETURN entity_id, duplicates
        ORDER BY duplicates DESC
        LIMIT 10
        """
        duplicates = self.run_query(duplicate_query)
        if duplicates:
            print(f"\n⚠️  Duplicate nodes found:")
            for dup in duplicates:
                print(f"   - Entity ID {dup['entity_id']}: {dup['duplicates']} instances")
        
        # 9. Verify anti-silo created relationships
        anti_silo_rels_query = """
        MATCH ()-[r]->()
        WHERE r.created_by IN ['anti_silo_analysis', 'nuclear_anti_silo']
        RETURN type(r) as rel_type, count(r) as count, r.created_by as created_by
        ORDER BY count DESC
        """
        anti_silo_rels = self.run_query(anti_silo_rels_query)
        print(f"\n🔗 Anti-silo created relationships:")
        for rel in anti_silo_rels:
            print(f"   - {rel['rel_type']} ({rel['created_by']}): {rel['count']}")
        
        # 10. Check nodes without proper IDs
        no_id_query = """
        MATCH (n)
        WHERE n.id IS NULL OR n.id = ''
        RETURN count(n) as nodes_without_id
        """
        no_id_count = self.run_query(no_id_query)[0]['nodes_without_id']
        print(f"\n❌ Nodes without proper ID: {no_id_count}")
        
        # 11. Analyze node connectivity distribution
        connectivity_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as connections
        RETURN connections, count(n) as node_count
        ORDER BY connections
        LIMIT 20
        """
        connectivity = self.run_query(connectivity_query)
        print(f"\n📊 Node connectivity distribution:")
        for conn in connectivity:
            print(f"   - {conn['node_count']} nodes have {conn['connections']} connections")
        
        # 12. Check for broken relationships
        broken_rels_query = """
        MATCH ()-[r]->()
        WHERE r.source_entity IS NULL OR r.target_entity IS NULL
        RETURN count(r) as broken_relationships
        """
        broken_rels = self.run_query(broken_rels_query)[0]['broken_relationships']
        print(f"\n💔 Broken relationships: {broken_rels}")
        
        print("\n" + "="*80)
        print("🎯 DIAGNOSIS SUMMARY")
        print("="*80)
        
        if simple_isolated > 0 and complex_isolated == 0:
            print("\n❗ CRITICAL BUG DETECTED:")
            print("   The anti-silo detection query is FLAWED!")
            print("   Simple query finds isolated nodes but complex query doesn't.")
            print("   This explains why the system reports success but user sees isolated nodes.")
            
        if self_connected > 0:
            print("\n❗ SELF-RELATIONSHIP ISSUE:")
            print("   Some nodes only connect to themselves.")
            print("   These appear connected but are effectively isolated.")
            
        if duplicates:
            print("\n❗ DUPLICATE NODE ISSUE:")
            print("   Multiple instances of same entity exist.")
            print("   This can confuse the anti-silo logic.")
            
        if no_id_count > 0:
            print("\n❗ MISSING ID ISSUE:")
            print("   Some nodes lack proper IDs.")
            print("   This breaks relationship creation.")

def main():
    debugger = AntiSiloDebugger()
    try:
        debugger.debug_isolated_nodes()
    finally:
        debugger.close()

if __name__ == "__main__":
    main()