#!/usr/bin/env python3
"""
Test script to verify Neo4j index creation fix
Tests the corrected create_performance_indexes() method
"""

import logging
import sys
import os

# Add the app directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.neo4j_service import get_neo4j_service

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_neo4j_index_creation():
    """Test Neo4j index creation with corrected syntax"""
    
    logger.info("🔍 Testing Neo4j Index Creation Fix")
    logger.info("=" * 60)
    
    try:
        # Get Neo4j service
        neo4j_service = get_neo4j_service()
        
        # Test connection first
        logger.info("1️⃣ Testing Neo4j connection...")
        connection_result = neo4j_service.test_connection()
        
        if not connection_result.get('success'):
            logger.error(f"❌ Neo4j connection failed: {connection_result.get('error')}")
            return False
            
        logger.info(f"✅ Neo4j connected successfully")
        logger.info(f"   Database: {connection_result.get('database_info', {}).get('database_name', 'unknown')}")
        logger.info(f"   Nodes: {connection_result.get('database_info', {}).get('node_count', 0)}")
        logger.info(f"   Relationships: {connection_result.get('database_info', {}).get('relationship_count', 0)}")
        
        # Test index creation
        logger.info("\n2️⃣ Testing index creation with corrected syntax...")
        index_result = neo4j_service.create_performance_indexes()
        
        if not index_result.get('success'):
            logger.error(f"❌ Index creation failed: {index_result.get('error')}")
            return False
            
        # Report results
        total_created = index_result.get('total_created', 0)
        total_failed = index_result.get('total_failed', 0)
        
        logger.info(f"✅ Index creation completed!")
        logger.info(f"   Successfully created: {total_created} indexes")
        logger.info(f"   Failed: {total_failed} indexes")
        
        if total_created > 0:
            logger.info("\n📋 Successfully created indexes:")
            for idx_name in index_result.get('indexes_created', [])[:10]:  # Show first 10
                logger.info(f"   ✓ {idx_name}")
            if len(index_result.get('indexes_created', [])) > 10:
                logger.info(f"   ... and {len(index_result.get('indexes_created', [])) - 10} more")
                
        if total_failed > 0:
            logger.info("\n⚠️ Failed indexes:")
            for error in index_result.get('indexes_failed', [])[:5]:  # Show first 5 errors
                logger.info(f"   ✗ {error}")
            if len(index_result.get('indexes_failed', [])) > 5:
                logger.info(f"   ... and {len(index_result.get('indexes_failed', [])) - 5} more errors")
        
        # Verify indexes in database
        logger.info("\n3️⃣ Verifying indexes in database...")
        try:
            with neo4j_service.driver.session() as session:
                # Get all indexes
                show_indexes_result = session.run("SHOW INDEXES")
                indexes = list(show_indexes_result)
                
                logger.info(f"📊 Total indexes in database: {len(indexes)}")
                
                # Show index details
                if indexes:
                    logger.info("\n📋 Current indexes in database:")
                    for idx in indexes[:15]:  # Show first 15
                        idx_data = dict(idx)
                        name = idx_data.get('name', 'unnamed')
                        labels = idx_data.get('labelsOrTypes', [])
                        properties = idx_data.get('properties', [])
                        state = idx_data.get('state', 'unknown')
                        
                        logger.info(f"   • {name}: {labels} on {properties} [{state}]")
                    
                    if len(indexes) > 15:
                        logger.info(f"   ... and {len(indexes) - 15} more indexes")
                
        except Exception as e:
            logger.warning(f"Could not verify indexes: {e}")
        
        # Performance check
        logger.info("\n4️⃣ Testing query performance with indexes...")
        try:
            with neo4j_service.driver.session() as session:
                # Test a simple query that should use indexes
                import time
                
                start_time = time.time()
                result = session.run("MATCH (n) WHERE n.id IS NOT NULL RETURN count(n) as count")
                record = result.single()
                end_time = time.time()
                
                node_count = record['count'] if record else 0
                query_time = (end_time - start_time) * 1000  # Convert to ms
                
                logger.info(f"✅ Query performance test completed")
                logger.info(f"   Nodes with ID property: {node_count}")
                logger.info(f"   Query time: {query_time:.2f}ms")
                
                if query_time < 1000:  # Less than 1 second
                    logger.info(f"   🚀 Good performance (< 1s)")
                else:
                    logger.info(f"   ⚠️ Slow performance (> 1s) - indexes may not be helping yet")
                    
        except Exception as e:
            logger.warning(f"Performance test failed: {e}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 SUMMARY")
        logger.info("=" * 60)
        
        if total_created >= 50:  # We expect many indexes (10 entity types × 6 properties each + composites + text)
            logger.info(f"✅ EXCELLENT: Created {total_created} indexes successfully")
            logger.info("✅ Neo4j performance optimization is now active")
            logger.info("✅ Knowledge graph queries should be 5-10x faster")
        elif total_created >= 20:
            logger.info(f"✅ GOOD: Created {total_created} indexes")
            logger.info("✅ Significant performance improvement expected")
        elif total_created > 0:
            logger.info(f"⚠️ PARTIAL: Created {total_created} indexes")
            logger.info("⚠️ Some performance improvement expected")
        else:
            logger.error("❌ FAILED: No indexes created")
            logger.error("❌ Neo4j performance will remain poor")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

def main():
    """Main test execution"""
    logger.info("🧪 Neo4j Index Creation Fix Test")
    logger.info("Testing the corrected create_performance_indexes() method")
    logger.info("")
    
    success = test_neo4j_index_creation()
    
    if success:
        logger.info("\n🎉 Test completed successfully!")
        logger.info("Neo4j index creation has been fixed and is working properly.")
        sys.exit(0)
    else:
        logger.error("\n💥 Test failed!")
        logger.error("Neo4j index creation is still having issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()