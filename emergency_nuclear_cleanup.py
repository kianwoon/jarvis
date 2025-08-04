#!/usr/bin/env python3
"""
NUCLEAR CLEANUP: Complete knowledge graph reset
Removes ALL relationships and entities to stop the explosion
"""
import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.neo4j_service import get_neo4j_service
import logging

def nuclear_cleanup():
    """Complete database cleanup - removes EVERYTHING"""
    neo4j_service = get_neo4j_service()
    
    if not neo4j_service.is_enabled():
        print("❌ Neo4j not enabled")
        return False
    
    try:
        with neo4j_service.driver.session() as session:
            # Get current counts
            count_result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                RETURN count(DISTINCT n) as nodes, count(r) as relationships
            """)
            record = count_result.single()
            nodes = record['nodes']
            relationships = record['relationships']
            
            print(f"🔥 BEFORE CLEANUP: {nodes} nodes, {relationships} relationships")
            
            # NUCLEAR OPTION: Delete everything
            print("💣 NUCLEAR CLEANUP: Deleting ALL nodes and relationships...")
            
            # Delete all relationships first
            session.run("MATCH ()-[r]-() DELETE r")
            
            # Delete all nodes
            session.run("MATCH (n) DELETE n")
            
            # Verify cleanup
            verify_result = session.run("""
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                RETURN count(DISTINCT n) as nodes, count(r) as relationships
            """)
            verify_record = verify_result.single()
            final_nodes = verify_record['nodes']
            final_relationships = verify_record['relationships']
            
            print(f"✅ AFTER CLEANUP: {final_nodes} nodes, {final_relationships} relationships")
            print(f"📉 REMOVED: {nodes} nodes, {relationships} relationships")
            
            return True
            
    except Exception as e:
        print(f"❌ Nuclear cleanup failed: {e}")
        return False

if __name__ == "__main__":
    print("🚨 EMERGENCY NUCLEAR CLEANUP")
    print("This will DELETE ALL knowledge graph data!")
    confirm = input("Type 'NUCLEAR' to confirm: ")
    
    if confirm == "NUCLEAR":
        success = nuclear_cleanup()
        if success:
            print("💥 Nuclear cleanup completed successfully")
        else:
            print("❌ Nuclear cleanup failed")
    else:
        print("❌ Cleanup cancelled")