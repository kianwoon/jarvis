#!/usr/bin/env python3
"""
Test to understand the discrepancy between backend and viewer
The backend might report 0 isolated nodes if relationships exist,
but the viewer shows nodes as isolated if they're not visually connected
"""

import requests
import json

def test_graph_data():
    """Test what data the viewer actually receives"""
    
    print("\n" + "="*80)
    print("🔍 VIEWER VS BACKEND DISCREPANCY ANALYSIS")
    print("="*80)
    
    # 1. Get entities
    entities_url = "http://localhost:8000/api/v1/knowledge-graph/query"
    entities_query = {
        "query": "MATCH (n) RETURN n.id as id, n.name as name, n.type as type, n.confidence as confidence LIMIT 100"
    }
    
    try:
        response = requests.post(entities_url, json=entities_query)
        if response.status_code == 200:
            data = response.json()
            print(f"   DEBUG: Response type: {type(data)}")
            if isinstance(data, dict):
                print(f"   DEBUG: Response keys: {list(data.keys())}")
            
            # Handle the response which might be wrapped
            if isinstance(data, dict) and 'results' in data:
                entities = data['results']
            elif isinstance(data, dict) and 'result' in data:
                entities = data['result']
            elif isinstance(data, list):
                entities = data
            else:
                entities = []
                print(f"   DEBUG: Unexpected response format: {data}")
            
            print(f"\n📊 Total entities returned to viewer: {len(entities)}")
            
            # Create a set of entity IDs
            entity_ids = {e['id'] for e in entities if isinstance(e, dict) and 'id' in e}
            print(f"   Unique entity IDs: {len(entity_ids)}")
            
            # Show sample entities
            if entities:
                print("\n   Sample entities:")
                for e in entities[:5]:
                    print(f"     - {e.get('name', 'Unknown')} (ID: {e.get('id', 'no-id')})")
        else:
            print(f"❌ Error fetching entities: {response.status_code}")
            return
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Get relationships
    relationships_query = {
        "query": "MATCH (a)-[r]->(b) RETURN a.id as source_entity, b.id as target_entity, type(r) as relationship_type, r.confidence as confidence LIMIT 100"
    }
    
    try:
        response = requests.post(entities_url, json=relationships_query)
        if response.status_code == 200:
            data = response.json()
            # Handle the response which might be wrapped
            if isinstance(data, dict) and 'results' in data:
                relationships = data['results']
            elif isinstance(data, dict) and 'result' in data:
                relationships = data['result']
            elif isinstance(data, list):
                relationships = data
            else:
                relationships = []
            
            print(f"\n📊 Total relationships returned to viewer: {len(relationships)}")
            
            # Track which entities have relationships
            connected_entities = set()
            for rel in relationships:
                connected_entities.add(rel.get('source_entity'))
                connected_entities.add(rel.get('target_entity'))
            
            print(f"   Entities with connections: {len(connected_entities)}")
            
            # Find entities that appear isolated in the viewer
            viewer_isolated = entity_ids - connected_entities
            print(f"\n🚨 Entities that appear isolated in viewer: {len(viewer_isolated)}")
            
            if viewer_isolated:
                # Get details about these "viewer-isolated" entities
                print("\n   Entities that APPEAR isolated in viewer:")
                for entity_id in list(viewer_isolated)[:10]:
                    entity = next((e for e in entities if e['id'] == entity_id), None)
                    if entity:
                        print(f"     - {entity.get('name', 'Unknown')} (ID: {entity_id})")
                
                # Now check if these entities ACTUALLY have relationships in Neo4j
                print("\n   Checking if these entities ACTUALLY have relationships...")
                check_actual_relationships(list(viewer_isolated)[:5])
            
        else:
            print(f"❌ Error fetching relationships: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("🎯 DIAGNOSIS")
    print("="*80)
    
    if viewer_isolated:
        print("\n❗ KEY FINDING:")
        print("   The viewer shows nodes as 'isolated' if they're not in the")
        print("   LIMITED set of relationships returned by the query!")
        print("\n   The viewer query has 'LIMIT 100' on relationships.")
        print("   If a node's relationships are beyond this limit,")
        print("   it appears isolated in the viewer but isn't actually isolated.")
        print("\n   This is an ARCHITECTURAL ISSUE with the viewer's data fetching.")

def check_actual_relationships(entity_ids):
    """Check if entities actually have relationships in Neo4j"""
    for entity_id in entity_ids:
        query = {
            "query": f"""
            MATCH (n {{id: '{entity_id}'}})
            OPTIONAL MATCH (n)-[r]-()
            RETURN n.name as name, count(r) as relationship_count
            """
        }
        
        try:
            response = requests.post("http://localhost:8000/api/v1/knowledge-graph/query", json=query)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'results' in data:
                    result = data['results']
                elif isinstance(data, list):
                    result = data
                else:
                    result = []
                if result:
                    name = result[0].get('name', 'Unknown')
                    rel_count = result[0].get('relationship_count', 0)
                    if rel_count > 0:
                        print(f"       ⚠️  {name}: Has {rel_count} relationships but NOT shown in viewer!")
                    else:
                        print(f"       ✓ {name}: Actually isolated (0 relationships)")
        except:
            pass

def test_query_limits():
    """Test the impact of query limits"""
    print("\n📊 TESTING QUERY LIMITS:")
    
    # Count total relationships
    count_query = {
        "query": "MATCH ()-[r]->() RETURN count(r) as total"
    }
    
    try:
        response = requests.post("http://localhost:8000/api/v1/knowledge-graph/query", json=count_query)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'results' in data:
                result = data['results']
            elif isinstance(data, dict) and 'result' in data:
                result = data['result']
            elif isinstance(data, list):
                result = data
            else:
                result = []
            total_rels = result[0]['total'] if result and len(result) > 0 else 0
            print(f"   Total relationships in database: {total_rels}")
            print(f"   Viewer query limit: 100")
            print(f"   Relationships not shown: {max(0, total_rels - 100)}")
            
            if total_rels > 100:
                print("\n   ⚠️  WARNING: Viewer is only showing 100 out of {total_rels} relationships!")
                print("      This causes nodes to appear isolated when they're not!")
    except:
        pass

if __name__ == "__main__":
    test_graph_data()
    test_query_limits()