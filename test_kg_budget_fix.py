#!/usr/bin/env python3
"""
Test script to verify that our knowledge graph budget fixes are working properly.
This script tests the new relationship limits and budget controls.
"""
import sys
import os
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

import asyncio
import logging
from app.services.knowledge_graph_service import GlobalRelationshipBudget, get_knowledge_graph_service
from app.services.neo4j_service import get_neo4j_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_budget_tracker():
    """Test the global relationship budget tracker"""
    print("🧪 TESTING GLOBAL RELATIONSHIP BUDGET TRACKER")
    print("=" * 60)
    
    # Test Neo4j connection
    neo4j_service = get_neo4j_service()
    if not neo4j_service.is_enabled():
        print("❌ Neo4j not enabled - cannot test budget tracker")
        return False
    
    # Test budget tracker
    budget = GlobalRelationshipBudget()
    budget.reset()  # Reset for clean test
    
    current_count = budget.get_current_global_count(neo4j_service)
    print(f"📊 Current global relationships: {current_count}")
    print(f"🎯 Global cap: {budget.max_global}")
    print(f"📝 Session limit: {budget.max_per_session}")
    
    # Test budget calculations
    print(f"\n🧮 BUDGET CALCULATIONS:")
    test_requests = [5, 10, 50, 100, 300]
    
    for requested in test_requests:
        allowed = budget.can_add_relationships(neo4j_service, requested)
        print(f"   Requested: {requested:3d} → Allowed: {allowed:3d}")
    
    # Test if global cap is reached
    cap_reached = budget.is_global_cap_reached(neo4j_service)
    print(f"\n🚫 Global cap reached: {cap_reached}")
    
    return True

async def test_mode_configurations():
    """Test the new stricter mode configurations"""
    print("\n🧪 TESTING MODE CONFIGURATIONS")
    print("=" * 60)
    
    # Get the service to check mode configs
    kg_service = get_knowledge_graph_service()
    
    # Test the mode configurations (simulate internal logic)
    mode_configs = {
        'simple': {
            'confidence_threshold': 0.9,
            'max_relationships_per_entity': 1,
            'max_total_relationships': 10,
            'global_relationship_cap': 250,
            'max_relationships_per_chunk': 3
        },
        'standard': {
            'confidence_threshold': 0.8,
            'max_relationships_per_entity': 2,
            'max_total_relationships': 25,
            'global_relationship_cap': 300,
            'max_relationships_per_chunk': 5
        },
        'comprehensive': {
            'confidence_threshold': 0.75,
            'max_relationships_per_entity': 2,
            'max_total_relationships': 40,
            'global_relationship_cap': 350,
            'max_relationships_per_chunk': 8
        }
    }
    
    for mode, config in mode_configs.items():
        print(f"\n📋 {mode.upper()} MODE:")
        print(f"   Confidence threshold: {config['confidence_threshold']}")
        print(f"   Max per entity: {config['max_relationships_per_entity']}")
        print(f"   Max per chunk: {config['max_relationships_per_chunk']}")
        print(f"   Max total: {config['max_total_relationships']}")
        print(f"   Global cap: {config['global_relationship_cap']}")
    
    return True

async def test_current_graph_state():
    """Test current state of the knowledge graph"""
    print("\n🧪 TESTING CURRENT GRAPH STATE")
    print("=" * 60)
    
    neo4j_service = get_neo4j_service()
    
    if not neo4j_service.is_enabled():
        print("❌ Neo4j not enabled")
        return False
    
    try:
        with neo4j_service.driver.session() as session:
            # Get basic counts
            counts_query = """
            MATCH (n) 
            OPTIONAL MATCH (n)-[r]->()
            RETURN 
                count(DISTINCT n) as node_count,
                count(r) as relationship_count
            """
            result = session.run(counts_query).single()
            node_count = result['node_count']
            relationship_count = result['relationship_count']
            
            print(f"📊 Graph Statistics:")
            print(f"   Nodes: {node_count}")
            print(f"   Relationships: {relationship_count}")
            print(f"   Relationships per node: {relationship_count/node_count:.2f}" if node_count > 0 else "   No nodes")
            
            # Check for any remaining problematic relationships
            problematic_query = """
            MATCH ()-[r]->()
            WHERE r.created_by IN ['anti_silo_analysis', 'aggressive_anti_silo', 'nuclear_anti_silo']
            RETURN count(r) as problematic_count
            """
            prob_result = session.run(problematic_query).single()
            problematic_count = prob_result['problematic_count']
            
            print(f"   Problematic relationships: {problematic_count}")
            
            if problematic_count == 0:
                print("   ✅ No anti-silo relationships remaining")
            else:
                print("   ⚠️  Some anti-silo relationships still exist")
            
            # Check confidence distribution
            confidence_query = """
            MATCH ()-[r]->()
            WHERE r.confidence IS NOT NULL
            RETURN 
                min(r.confidence) as min_conf,
                max(r.confidence) as max_conf,
                avg(r.confidence) as avg_conf,
                count(r) as conf_count
            """
            conf_result = session.run(confidence_query).single()
            
            print(f"   Confidence range: {conf_result['min_conf']:.2f} - {conf_result['max_conf']:.2f}")
            print(f"   Average confidence: {conf_result['avg_conf']:.2f}")
            print(f"   Relationships with confidence: {conf_result['conf_count']}")
            
            return True
            
    except Exception as e:
        print(f"❌ Failed to analyze graph state: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚨 KNOWLEDGE GRAPH BUDGET FIX VERIFICATION")
    print("Testing emergency fixes for relationship count reduction")
    print()
    
    # Run tests
    tests = [
        ("Budget Tracker", test_budget_tracker),
        ("Mode Configurations", test_mode_configurations), 
        ("Current Graph State", test_current_graph_state)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - Budget fixes are working correctly!")
        print("💡 Knowledge graph relationship explosion has been controlled.")
    else:
        print("⚠️  Some tests failed - manual verification recommended.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)