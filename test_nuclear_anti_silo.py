#!/usr/bin/env python3
"""Test nuclear anti-silo option to eliminate remaining isolated nodes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from app.services.knowledge_graph_service import get_knowledge_graph_service
from app.services.neo4j_service import get_neo4j_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_silo_nodes():
    """Check for current silo nodes in the graph"""
    try:
        neo4j_service = get_neo4j_service()
        
        # Query for isolated nodes
        isolated_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        WITH n, [(n)-[r]-(other) | r] as relationships
        WHERE size(relationships) = 0
        RETURN n.id as id, n.name as name, n.type as type, labels(n)[0] as label
        ORDER BY n.name
        """
        
        isolated_nodes = neo4j_service.execute_cypher(isolated_query)
        
        print(f"🔍 Found {len(isolated_nodes)} isolated/silo nodes:")
        for node in isolated_nodes:
            print(f"   • {node['name']} ({node.get('type', node.get('label', 'UNKNOWN'))})")
        
        return isolated_nodes
        
    except Exception as e:
        logger.error(f"Failed to check silo nodes: {e}")
        return []

async def trigger_nuclear_anti_silo():
    """Manually trigger nuclear anti-silo processing"""
    try:
        logger.info("🚀 Starting nuclear anti-silo test...")
        
        # Get services
        kg_service = get_knowledge_graph_service()
        neo4j_service = get_neo4j_service()
        
        # Check current state
        print("\n📊 BEFORE Nuclear Processing:")
        print("=" * 50)
        isolated_before = await check_silo_nodes()
        
        if not isolated_before:
            print("✅ No silo nodes found - graph is already fully connected!")
            return True
        
        # Manually trigger nuclear elimination
        print(f"\n☢️  Triggering NUCLEAR ANTI-SILO processing...")
        print("=" * 50)
        
        nuclear_connections = await kg_service._nuclear_anti_silo_elimination(neo4j_service)
        print(f"☢️  Nuclear processing made {nuclear_connections} connections")
        
        # Check after processing
        print("\n📊 AFTER Nuclear Processing:")
        print("=" * 50)
        isolated_after = await check_silo_nodes()
        
        # Results summary
        print(f"\n📈 NUCLEAR ANTI-SILO RESULTS:")
        print("=" * 50)
        print(f"   Before: {len(isolated_before)} silo nodes")
        print(f"   After:  {len(isolated_after)} silo nodes")
        print(f"   Eliminated: {len(isolated_before) - len(isolated_after)} silo nodes")
        print(f"   Connections made: {nuclear_connections}")
        
        if len(isolated_after) == 0:
            print("🎉 SUCCESS: All silo nodes eliminated!")
            return True
        else:
            print(f"⚠️  WARNING: {len(isolated_after)} silo nodes still remain:")
            for node in isolated_after:
                print(f"      • {node['name']} ({node.get('type', 'UNKNOWN')})")
            return False
            
    except Exception as e:
        logger.error(f"Nuclear anti-silo test failed: {e}")
        logger.exception("Test exception details:")
        return False

async def get_graph_stats():
    """Get overall graph connectivity statistics"""
    try:
        neo4j_service = get_neo4j_service()
        
        # Total nodes and relationships
        stats_query = """
        MATCH (n) 
        OPTIONAL MATCH (n)-[r]-()
        RETURN count(DISTINCT n) as total_nodes, 
               count(DISTINCT r) as total_relationships,
               count(DISTINCT n) - count(DISTINCT CASE WHEN size([(n)--()]) = 0 THEN n ELSE null END) as connected_nodes
        """
        
        result = neo4j_service.execute_cypher(stats_query)
        if result:
            stats = result[0]
            total_nodes = stats['total_nodes']
            total_relationships = stats['total_relationships']
            connected_nodes = stats['connected_nodes']
            
            connectivity_percentage = (connected_nodes / total_nodes * 100) if total_nodes > 0 else 0
            
            print(f"\n📊 GRAPH CONNECTIVITY STATISTICS:")
            print("=" * 50)
            print(f"   Total nodes: {total_nodes}")
            print(f"   Connected nodes: {connected_nodes}")
            print(f"   Isolated nodes: {total_nodes - connected_nodes}")
            print(f"   Total relationships: {total_relationships}")
            print(f"   Connectivity: {connectivity_percentage:.1f}%")
            
            return {
                'total_nodes': total_nodes,
                'connected_nodes': connected_nodes,
                'isolated_nodes': total_nodes - connected_nodes,
                'total_relationships': total_relationships,
                'connectivity_percentage': connectivity_percentage
            }
        
    except Exception as e:
        logger.error(f"Failed to get graph stats: {e}")
        return None

async def main():
    """Main test function"""
    print("🧪 NUCLEAR ANTI-SILO TEST")
    print("=" * 60)
    
    # Get initial stats
    await get_graph_stats()
    
    # Run nuclear test
    success = await trigger_nuclear_anti_silo()
    
    # Get final stats
    await get_graph_stats()
    
    # Final result
    if success:
        print("\n🎉 NUCLEAR ANTI-SILO TEST PASSED!")
        print("   All silo nodes have been eliminated.")
        print("   Knowledge graph is now fully connected.")
    else:
        print("\n❌ NUCLEAR ANTI-SILO TEST FAILED!")
        print("   Some silo nodes still remain after nuclear processing.")
        print("   Manual investigation may be required.")
    
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        sys.exit(1)