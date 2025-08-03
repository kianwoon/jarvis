#!/usr/bin/env python3
"""
Fix existing entity types and test knowledge graph improvements
"""

import asyncio
import logging
import json
from app.services.neo4j_service import get_neo4j_service
from app.services.knowledge_graph_service import get_knowledge_graph_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_entity_type(entity_name: str) -> str:
    """Classify entity type based on name patterns"""
    if not entity_name:
        return 'CONCEPT'
    
    name_lower = entity_name.lower()
    
    # Organization patterns
    if any(org_term in name_lower for org_term in [
        'bank', 'corp', 'corporation', 'company', 'inc', 'ltd', 'group', 
        'authority', 'ministry', 'government', 'agency', 'foundation'
    ]):
        return 'ORGANIZATION'
    
    # Technology patterns
    if any(tech_term in name_lower for tech_term in [
        'database', 'db', 'sql', 'stack', 'base', 'cloud', 'platform',
        'system', 'framework', 'api', 'service', 'software', 'app'
    ]):
        return 'TECHNOLOGY'
    
    # Location patterns
    if any(loc_term in name_lower for loc_term in [
        'singapore', 'hong kong', 'china', 'india', 'indonesia', 'thailand',
        'vietnam', 'malaysia', 'philippines', 'korea', 'japan', 'country',
        'city', 'region', 'state', 'province'
    ]):
        return 'LOCATION'
    
    # Product/Service patterns
    if any(prod_term in name_lower for prod_term in [
        'pay', 'payment', 'wallet', 'trial', 'poc', 'proof-of-concept',
        'solution', 'offering', 'product'
    ]):
        return 'PRODUCT'
    
    # Project patterns
    if any(proj_term in name_lower for proj_term in [
        'project', 'initiative', 'program', 'transformation', 'migration',
        'implementation', 'deployment', 'rollout'
    ]):
        return 'PROJECT'
    
    # Event patterns
    if any(event_term in name_lower for event_term in [
        'conference', 'meeting', 'summit', 'workshop', 'training',
        'launch', 'release', 'announcement'
    ]):
        return 'EVENT'
    
    # Person patterns
    if any(person_term in name_lower for person_term in [
        'ceo', 'cto', 'manager', 'director', 'analyst', 'engineer',
        'developer', 'architect', 'consultant'
    ]):
        return 'PERSON'
    
    # Default to CONCEPT for abstract ideas
    return 'CONCEPT'

async def fix_entity_types():
    """Fix existing entity types and run anti-silo analysis"""
    neo4j_service = get_neo4j_service()
    kg_service = get_knowledge_graph_service()
    
    if not neo4j_service.is_enabled():
        print("❌ Neo4j service is not enabled")
        return
    
    print("🔧 Starting entity type fixes and anti-silo analysis...")
    
    try:
        # Step 1: Get all entities and fix their types
        query = """
        MATCH (n) 
        RETURN n.id as id, n.name as name, labels(n)[0] as current_label
        """
        
        entities = neo4j_service.execute_cypher(query)
        print(f"📊 Found {len(entities)} entities to process")
        
        type_changes = {}
        entities_updated = 0
        
        for entity in entities:
            entity_id = entity['id']
            entity_name = entity['name']
            current_label = entity['current_label']
            
            # Classify the entity type
            new_type = classify_entity_type(entity_name)
            
            if new_type != current_label:
                type_changes[entity_id] = {
                    'name': entity_name,
                    'old_type': current_label,
                    'new_type': new_type
                }
            
            # Update the entity's type property and label if needed
            update_query = f"""
            MATCH (n {{id: $entity_id}})
            SET n.type = $new_type
            RETURN n.id as id, n.name as name, n.type as type
            """
            
            result = neo4j_service.execute_cypher(update_query, {
                'entity_id': entity_id,
                'new_type': new_type
            })
            
            if result:
                entities_updated += 1
                print(f"✅ Updated {entity_name}: {current_label} -> {new_type}")
        
        print(f"\n📈 Summary:")
        print(f"  - Total entities processed: {len(entities)}")
        print(f"  - Entities updated: {entities_updated}")
        print(f"  - Type changes: {len(type_changes)}")
        
        if type_changes:
            print("\n🔄 Type Changes Made:")
            for entity_id, changes in type_changes.items():
                print(f"  - {changes['name']}: {changes['old_type']} → {changes['new_type']}")
        
        # Step 2: Run anti-silo analysis
        print("\n🔗 Running anti-silo analysis...")
        anti_silo_result = await kg_service.run_global_anti_silo_analysis()
        
        if anti_silo_result.get('success'):
            print(f"✅ Anti-silo analysis completed:")
            print(f"  - Initial silo count: {anti_silo_result.get('initial_silo_count', 0)}")
            print(f"  - Final silo count: {anti_silo_result.get('final_silo_count', 0)}")
            print(f"  - Connections made: {anti_silo_result.get('connections_made', 0)}")
            print(f"  - Nodes removed: {anti_silo_result.get('nodes_removed', 0)}")
            print(f"  - Reduction: {anti_silo_result.get('reduction', 0)}")
        else:
            print(f"❌ Anti-silo analysis failed: {anti_silo_result.get('error', 'Unknown error')}")
        
        # Step 3: Show final stats
        print("\n📊 Getting updated knowledge graph stats...")
        import requests
        try:
            response = requests.get("http://localhost:8000/api/v1/knowledge-graph/stats")
            if response.status_code == 200:
                stats = response.json()
                print("📈 Updated Knowledge Graph Stats:")
                print(f"  - Total entities: {stats['total_entities']}")
                print(f"  - Total relationships: {stats['total_relationships']}")
                print(f"  - Entity types: {json.dumps(stats['entity_types'], indent=4)}")
                print(f"  - Relationship types: {json.dumps(stats['relationship_types'], indent=4)}")
            else:
                print(f"❌ Failed to get stats: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
        
        # Step 4: Show remaining silo analysis
        print("\n🎯 Checking remaining silo nodes...")
        try:
            response = requests.get("http://localhost:8000/api/v1/knowledge-graph/silo-analysis")
            if response.status_code == 200:
                silo_data = response.json()
                analysis = silo_data['analysis']
                print(f"🏝️ Remaining Silo Analysis:")
                print(f"  - Isolated nodes: {len(analysis['isolated_nodes'])}")
                print(f"  - Weakly connected: {len(analysis['weakly_connected'])}")
                print(f"  - Total silos: {analysis['total_silos']}")
                
                if analysis['isolated_nodes']:
                    print("🏝️ Remaining isolated nodes:")
                    for node in analysis['isolated_nodes'][:5]:  # Show first 5
                        print(f"    - {node['name']} ({node.get('type', 'N/A')})")
            else:
                print(f"❌ Failed to get silo analysis: {response.status_code}")
        except Exception as e:
            print(f"❌ Error getting silo analysis: {e}")
            
        print("\n🎉 Entity type fixes and anti-silo analysis completed!")
        
    except Exception as e:
        logger.error(f"❌ Error during fixes: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(fix_entity_types())