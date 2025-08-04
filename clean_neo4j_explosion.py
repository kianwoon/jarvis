#!/usr/bin/env python3
"""
Clean up the Neo4j database after relationship explosion
"""

from app.services.neo4j_service import get_neo4j_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_neo4j_database():
    """Clean up Neo4j database to remove explosion"""
    
    neo4j = get_neo4j_service()
    
    if not neo4j.is_enabled():
        logger.error("Neo4j is not enabled")
        return
    
    try:
        with neo4j.driver.session() as session:
            # First, get counts
            count_query = """
            MATCH (n)
            WITH count(n) as node_count
            MATCH ()-[r]->()
            RETURN node_count, count(r) as relationship_count
            """
            
            result = session.run(count_query).single()
            if result:
                logger.info(f"Current state: {result['node_count']} nodes, {result['relationship_count']} relationships")
            
            # Option 1: Delete ALL relationships (keep nodes)
            logger.info("🗑️  Deleting ALL relationships...")
            delete_rels_query = """
            MATCH ()-[r]->()
            DELETE r
            RETURN count(r) as deleted_count
            """
            
            result = session.run(delete_rels_query).single()
            logger.info(f"✅ Deleted {result['deleted_count']} relationships")
            
            # Option 2: Delete orphaned nodes (nodes with no relationships after cleanup)
            logger.info("🗑️  Deleting orphaned nodes...")
            delete_orphans_query = """
            MATCH (n)
            WHERE NOT EXISTS((n)-[]-())
            DELETE n
            RETURN count(n) as deleted_count
            """
            
            result = session.run(delete_orphans_query).single()
            logger.info(f"✅ Deleted {result['deleted_count']} orphaned nodes")
            
            # Final count
            final_count_query = """
            MATCH (n)
            RETURN count(n) as node_count
            """
            
            result = session.run(final_count_query).single()
            logger.info(f"Final state: {result['node_count']} nodes remaining")
            
    except Exception as e:
        logger.error(f"Failed to clean database: {e}")

def nuclear_option_full_reset():
    """Nuclear option: Delete EVERYTHING"""
    
    neo4j = get_neo4j_service()
    
    if not neo4j.is_enabled():
        logger.error("Neo4j is not enabled")
        return
    
    try:
        with neo4j.driver.session() as session:
            logger.warning("☢️  NUCLEAR OPTION: Deleting ALL nodes and relationships...")
            
            nuclear_query = """
            MATCH (n)
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            
            result = session.run(nuclear_query).single()
            logger.info(f"☢️  Nuclear cleanup complete: Deleted {result['deleted_count']} nodes and all relationships")
            
    except Exception as e:
        logger.error(f"Nuclear cleanup failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--nuclear":
        logger.warning("☢️  NUCLEAR MODE ACTIVATED!")
        response = input("This will DELETE EVERYTHING in Neo4j. Are you sure? (yes/no): ")
        if response.lower() == "yes":
            nuclear_option_full_reset()
        else:
            logger.info("Nuclear option cancelled")
    else:
        clean_neo4j_database()
        logger.info("\nTo completely reset Neo4j, run: python clean_neo4j_explosion.py --nuclear")