#!/usr/bin/env python3
"""
EMERGENCY RELATIONSHIP CLEANUP SCRIPT
Reduces knowledge graph relationships from 1001 to ~250 by removing low-confidence connections.

This script prioritizes relationships by:
1. Removing anti-silo generated relationships (lowest priority)
2. Removing low confidence relationships (< 0.7)
3. Keeping only top 250 relationships by confidence

USAGE: python emergency_cleanup_relationships.py
"""
import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.neo4j_service import get_neo4j_service
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_relationships():
    """Remove low-confidence and excessive relationships to reach ~250 target"""
    neo4j_service = get_neo4j_service()
    
    if not neo4j_service.is_enabled():
        print("❌ Neo4j not enabled - cannot perform cleanup")
        return False
    
    try:
        with neo4j_service.driver.session() as session:
            print("🧹 EMERGENCY KNOWLEDGE GRAPH CLEANUP STARTED")
            print("=" * 60)
            
            # Get current counts
            count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            current_count = count_result.single()['count']
            print(f"📊 Current relationships: {current_count}")
            
            if current_count <= 250:
                print(f"✅ Already within target range ({current_count} <= 250)")
                return True
            
            # Step 1: Remove anti-silo generated relationships (lowest priority)
            print("\n🗑️  STEP 1: Removing anti-silo generated relationships...")
            anti_silo_query = """
            MATCH ()-[r]->() 
            WHERE r.created_by IN ['anti_silo_analysis', 'aggressive_anti_silo', 'nuclear_anti_silo', 
                                  'anti_silo_business_analysis', 'nuclear_document_cooccurrence',
                                  'nuclear_isolated_cluster']
            DELETE r
            RETURN count(r) as deleted
            """
            result = session.run(anti_silo_query)
            deleted = result.single()['deleted']
            print(f"   🗑️  Deleted {deleted} anti-silo relationships")
            
            # Check count after anti-silo cleanup
            count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            after_antisilo_count = count_result.single()['count']
            print(f"   📊 After anti-silo cleanup: {after_antisilo_count} relationships")
            
            if after_antisilo_count <= 250:
                print(f"✅ Target reached after anti-silo cleanup!")
                return True
            
            # Step 2: Remove low confidence relationships (< 0.7)
            print("\n🗑️  STEP 2: Removing low-confidence relationships...")
            low_conf_query = """
            MATCH ()-[r]->() 
            WHERE r.confidence < 0.7
            DELETE r
            RETURN count(r) as deleted
            """
            result = session.run(low_conf_query)
            deleted = result.single()['deleted']
            print(f"   🗑️  Deleted {deleted} low-confidence relationships")
            
            # Check count after low confidence cleanup
            count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            after_lowconf_count = count_result.single()['count']
            print(f"   📊 After low-confidence cleanup: {after_lowconf_count} relationships")
            
            if after_lowconf_count <= 250:
                print(f"✅ Target reached after low-confidence cleanup!")
                return True
            
            # Step 3: Remove nuclear/aggressive connections (very low priority)
            print("\n🗑️  STEP 3: Removing nuclear/aggressive connections...")
            nuclear_query = """
            MATCH ()-[r]->() 
            WHERE r.nuclear_connection = true OR r.aggressive_mode = true
            DELETE r
            RETURN count(r) as deleted
            """
            result = session.run(nuclear_query)
            deleted = result.single()['deleted']
            print(f"   🗑️  Deleted {deleted} nuclear/aggressive connections")
            
            # Check count after nuclear cleanup
            count_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            after_nuclear_count = count_result.single()['count']
            print(f"   📊 After nuclear cleanup: {after_nuclear_count} relationships")
            
            if after_nuclear_count <= 250:
                print(f"✅ Target reached after nuclear cleanup!")
                return True
            
            # Step 4: Keep only top 250 relationships by confidence (final cut)
            print("\n🗑️  STEP 4: Keeping only top 250 relationships by confidence...")
            
            # First, get the 250th highest confidence value
            confidence_threshold_query = """
            MATCH ()-[r]->()
            WHERE r.confidence IS NOT NULL
            RETURN r.confidence as conf
            ORDER BY conf DESC
            SKIP 249
            LIMIT 1
            """
            threshold_result = session.run(confidence_threshold_query)
            threshold_record = threshold_result.single()
            
            if threshold_record:
                confidence_threshold = threshold_record['conf']
                print(f"   📊 Using confidence threshold: {confidence_threshold}")
                
                # Delete relationships below the 250th highest confidence
                final_cleanup_query = """
                MATCH ()-[r]->()
                WHERE r.confidence IS NOT NULL AND r.confidence < $threshold
                DELETE r
                RETURN count(r) as deleted
                """
                result = session.run(final_cleanup_query, {'threshold': confidence_threshold})
                deleted = result.single()['deleted']
                print(f"   🗑️  Deleted {deleted} relationships below confidence threshold")
            else:
                # Fallback: delete relationships with lowest confidence until we reach 250
                print("   ⚠️  No confidence threshold found, using fallback method...")
                fallback_query = """
                MATCH (a)-[r]->(b)
                WITH r ORDER BY COALESCE(r.confidence, 0) ASC
                SKIP 250
                DELETE r
                RETURN count(r) as deleted
                """
                result = session.run(fallback_query)
                deleted = result.single()['deleted']
                print(f"   🗑️  Deleted {deleted} excess relationships (keeping top 250)")
            
            # Final count verification
            final_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            final_count = final_result.single()['count']
            
            print("\n" + "=" * 60)
            print("🎯 CLEANUP RESULTS:")
            print(f"   📉 Initial count: {current_count}")
            print(f"   📊 Final count: {final_count}")
            print(f"   🎯 Reduction: {current_count - final_count} relationships removed")
            print(f"   📈 Efficiency: {100*(current_count-final_count)/current_count:.1f}% reduction")
            
            if final_count <= 250:
                print(f"   ✅ SUCCESS: Target reached ({final_count} <= 250)")
                return True
            else:
                print(f"   ⚠️  WARNING: Still above target ({final_count} > 250)")
                return False
            
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        logger.exception("Full exception details:")
        return False

def get_relationship_stats():
    """Get detailed statistics about current relationships"""
    neo4j_service = get_neo4j_service()
    
    if not neo4j_service.is_enabled():
        print("❌ Neo4j not enabled")
        return
    
    try:
        with neo4j_service.driver.session() as session:
            print("\n📊 RELATIONSHIP STATISTICS:")
            print("-" * 40)
            
            # Total count
            total_query = "MATCH ()-[r]->() RETURN count(r) as count"
            total = session.run(total_query).single()['count']
            print(f"Total relationships: {total}")
            
            # By created_by
            by_creator_query = """
            MATCH ()-[r]->()
            RETURN r.created_by as creator, count(r) as count
            ORDER BY count DESC
            """
            creators = session.run(by_creator_query)
            print("\nBy creator:")
            for record in creators:
                creator = record['creator'] or 'unknown'
                count = record['count']
                print(f"  {creator}: {count}")
            
            # By confidence ranges
            confidence_query = """
            MATCH ()-[r]->()
            WHERE r.confidence IS NOT NULL
            RETURN 
                sum(CASE WHEN r.confidence >= 0.9 THEN 1 ELSE 0 END) as very_high,
                sum(CASE WHEN r.confidence >= 0.8 AND r.confidence < 0.9 THEN 1 ELSE 0 END) as high,
                sum(CASE WHEN r.confidence >= 0.7 AND r.confidence < 0.8 THEN 1 ELSE 0 END) as medium,
                sum(CASE WHEN r.confidence < 0.7 THEN 1 ELSE 0 END) as low
            """
            conf_result = session.run(confidence_query).single()
            print(f"\nBy confidence:")
            print(f"  Very High (>=0.9): {conf_result['very_high']}")
            print(f"  High (0.8-0.9):    {conf_result['high']}")
            print(f"  Medium (0.7-0.8):  {conf_result['medium']}")
            print(f"  Low (<0.7):        {conf_result['low']}")
            
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")

if __name__ == "__main__":
    print("🚨 EMERGENCY KNOWLEDGE GRAPH RELATIONSHIP CLEANUP")
    print("This script will reduce relationships from ~1001 to ~250")
    print()
    
    # Show current stats
    get_relationship_stats()
    
    # Confirm before proceeding
    response = input("\n⚠️  Continue with cleanup? This will permanently delete relationships. (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Cleanup cancelled")
        sys.exit(0)
    
    # Perform cleanup
    success = cleanup_relationships()
    
    # Show final stats
    print("\n" + "=" * 60)
    get_relationship_stats()
    
    if success:
        print("\n✅ Emergency cleanup completed successfully!")
        print("💡 You can now process documents with the new relationship limits.")
    else:
        print("\n⚠️  Cleanup completed but target not fully reached.")
        print("💡 Consider running the script again or adjusting confidence thresholds.")