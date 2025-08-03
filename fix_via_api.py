#!/usr/bin/env python3
"""
Fix entity types via Neo4j API calls
"""

import requests
import json

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
        'system', 'framework', 'api', 'service', 'software', 'app', 'mainframe'
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
        'implementation', 'deployment', 'rollout', '2.0'
    ]):
        return 'PROJECT'
    
    # Event patterns
    if any(event_term in name_lower for event_term in [
        'conference', 'meeting', 'summit', 'workshop', 'training',
        'launch', 'release', 'announcement'
    ]):
        return 'EVENT'
    
    # Default to CONCEPT for abstract ideas
    return 'CONCEPT'

def main():
    base_url = "http://localhost:8000/api/v1/knowledge-graph"
    
    print("🔧 Starting entity type fixes via API...")
    
    # Step 1: Get all entities
    query_payload = {
        "query": "MATCH (n) RETURN n.id as id, n.name as name, labels(n)[0] as current_label"
    }
    
    response = requests.post(f"{base_url}/query", json=query_payload)
    
    if response.status_code != 200:
        print(f"❌ Failed to get entities: {response.status_code}")
        return
    
    entities_data = response.json()
    entities = entities_data.get('results', [])
    print(f"📊 Found {len(entities)} entities to process")
    
    # Step 2: Update entity types
    entities_updated = 0
    type_changes = {}
    
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
        
        # Update the entity's type property
        update_query = f"""
        MATCH (n {{id: '{entity_id}'}})
        SET n.type = '{new_type}'
        RETURN n.id as id, n.name as name, n.type as type
        """
        
        update_payload = {"query": update_query}
        update_response = requests.post(f"{base_url}/query", json=update_payload)
        
        if update_response.status_code == 200:
            entities_updated += 1
            print(f"✅ Updated {entity_name}: {current_label} -> {new_type}")
        else:
            print(f"❌ Failed to update {entity_name}: {update_response.status_code}")
    
    print(f"\n📈 Update Summary:")
    print(f"  - Total entities processed: {len(entities)}")
    print(f"  - Entities updated: {entities_updated}")
    print(f"  - Type changes: {len(type_changes)}")
    
    if type_changes:
        print("\n🔄 Type Changes Made:")
        for entity_id, changes in type_changes.items():
            print(f"  - {changes['name']}: {changes['old_type']} → {changes['new_type']}")
    
    # Step 3: Run anti-silo analysis
    print("\n🔗 Running anti-silo analysis...")
    anti_silo_response = requests.post(f"{base_url}/global-anti-silo-cleanup")
    
    if anti_silo_response.status_code == 200:
        anti_silo_result = anti_silo_response.json()
        if anti_silo_result.get('success'):
            print(f"✅ Anti-silo analysis completed:")
            results = anti_silo_result.get('results', {})
            print(f"  - Initial silo count: {results.get('initial_silo_count', 0)}")
            print(f"  - Final silo count: {results.get('final_silo_count', 0)}")
            print(f"  - Connections made: {results.get('connections_made', 0)}")
            print(f"  - Nodes removed: {results.get('nodes_removed', 0)}")
            print(f"  - Reduction: {results.get('reduction', 0)}")
        else:
            print(f"❌ Anti-silo analysis failed: {anti_silo_result.get('message', 'Unknown error')}")
    else:
        print(f"❌ Failed to run anti-silo analysis: {anti_silo_response.status_code}")
    
    # Step 4: Show final stats
    print("\n📊 Getting updated knowledge graph stats...")
    stats_response = requests.get(f"{base_url}/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print("📈 Updated Knowledge Graph Stats:")
        print(f"  - Total entities: {stats['total_entities']}")
        print(f"  - Total relationships: {stats['total_relationships']}")
        print(f"  - Entity types: {json.dumps(stats['entity_types'], indent=4)}")
        print(f"  - Relationship types: {json.dumps(stats['relationship_types'], indent=4)}")
    else:
        print(f"❌ Failed to get stats: {stats_response.status_code}")
    
    # Step 5: Show remaining silo analysis
    print("\n🎯 Checking remaining silo nodes...")
    silo_response = requests.get(f"{base_url}/silo-analysis")
    if silo_response.status_code == 200:
        silo_data = silo_response.json()
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
        print(f"❌ Failed to get silo analysis: {silo_response.status_code}")
        
    print("\n🎉 Entity type fixes and anti-silo analysis completed!")

if __name__ == "__main__":
    main()