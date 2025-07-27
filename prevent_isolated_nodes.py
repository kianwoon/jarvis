#!/usr/bin/env python3
"""
Isolation Prevention Script

Implements rules to prevent isolated nodes during knowledge graph operations:
1. When splitting multi-concept entities, preserve or create relationships
2. Remove entities that have no meaningful connections
3. Set minimum relationship requirements for new entities
"""

import requests
import json
import re
from typing import Dict, List, Tuple, Optional

class IsolationPrevention:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
        # Minimum relationships required for an entity to be kept
        self.min_relationships = 1
        
        # Patterns for entities that are likely noise if isolated
        self.noise_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[A-Z]{1,3}$',  # Short abbreviations
            r'[,;]',  # Multiple concepts in name
            r'^(and|or|the|a|an|of|in|on|at|by|for|with|to)$',  # Stop words
        ]
    
    def execute_cypher(self, query: str, description: str = "") -> List[Dict]:
        """Execute a Cypher query"""
        response = requests.post(
            f"{self.base_url}/api/v1/knowledge-graph/query",
            headers={"Content-Type": "application/json"},
            json={"query": query, "query_type": "cypher"}
        )
        
        if response.status_code == 200:
            data = response.json()
            if description:
                print(f"✅ {description}: {len(data['results'])} results")
            return data['results']
        else:
            if description:
                print(f"❌ {description} failed: {response.status_code}")
            return []
    
    def is_likely_noise(self, entity_name: str) -> bool:
        """Check if an entity name matches noise patterns"""
        for pattern in self.noise_patterns:
            if re.match(pattern, entity_name.strip(), re.IGNORECASE):
                return True
        return False
    
    def find_potential_isolated_entities(self) -> List[Dict]:
        """Find entities that are at risk of becoming isolated"""
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as relationship_count
        WHERE relationship_count < 2
        RETURN n.id as id, n.name as name, labels(n)[0] as type, 
               relationship_count, n.confidence as confidence
        ORDER BY relationship_count ASC, n.confidence ASC
        """
        
        results = self.execute_cypher(query, "Found entities with few relationships")
        return results
    
    def clean_noise_entities(self) -> int:
        """Remove entities that are clearly noise"""
        print("🧹 Cleaning noise entities...")
        
        # Find entities that match noise patterns
        query = """
        MATCH (n)
        WHERE NOT (n)-[]-()
        RETURN n.id as id, n.name as name, labels(n)[0] as type
        """
        
        isolated_entities = self.execute_cypher(query, "Found isolated entities")
        
        removed_count = 0
        for entity in isolated_entities:
            name = entity['name'].strip()
            entity_id = entity['id']
            
            if self.is_likely_noise(name):
                delete_query = f'MATCH (n {{id: "{entity_id}"}}) DELETE n'
                self.execute_cypher(delete_query, f"Removed noise entity: {name}")
                removed_count += 1
        
        return removed_count
    
    def improve_entity_connections(self) -> int:
        """Try to connect weakly connected entities by finding implicit relationships"""
        print("🔗 Improving entity connections...")
        
        # Find entities with only 1 relationship that could be connected to others
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH n, count(r) as rel_count
        WHERE rel_count = 1
        RETURN n.id as id, n.name as name, labels(n)[0] as type, n.document_id as doc_id
        LIMIT 10
        """
        
        weakly_connected = self.execute_cypher(query, "Found weakly connected entities")
        
        improved_count = 0
        for entity in weakly_connected:
            name = entity['name']
            entity_type = entity['type']
            doc_id = entity.get('doc_id', '')
            
            # Try to find entities from the same document to connect
            if doc_id:
                same_doc_query = f"""
                MATCH (source {{id: "{entity['id']}"}}), (target)
                WHERE target.document_id = "{doc_id}" 
                  AND target.id <> "{entity['id']}"
                  AND NOT (source)-[]-(target)
                WITH source, target
                LIMIT 1
                CREATE (source)-[r:CONTEXTUALLY_RELATED]->(target)
                SET r.confidence = 0.4,
                    r.context = "Same document context",
                    r.document_id = "{doc_id}",
                    r.created_at = datetime(),
                    r.source = "isolation_prevention"
                RETURN count(r) as created
                """
                
                result = self.execute_cypher(same_doc_query, f"Connected {name} to same-document entity")
                if result and result[0].get('created', 0) > 0:
                    improved_count += 1
        
        return improved_count
    
    def implement_quality_rules(self) -> Dict[str, int]:
        """Implement comprehensive isolation prevention rules"""
        print("🚀 Implementing Isolation Prevention Rules")
        print("=" * 60)
        
        # Get baseline metrics
        baseline_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH count(DISTINCT n) as total_entities, 
             count(DISTINCT CASE WHEN r IS NULL THEN n END) as isolated_entities
        RETURN total_entities, isolated_entities
        """
        
        baseline = self.execute_cypher(baseline_query, "Baseline metrics")
        baseline_isolated = baseline[0]['isolated_entities'] if baseline else 0
        baseline_total = baseline[0]['total_entities'] if baseline else 0
        
        # Apply rules
        removed_noise = self.clean_noise_entities()
        improved_connections = self.improve_entity_connections()
        
        # Get final metrics
        final = self.execute_cypher(baseline_query, "Final metrics")
        final_isolated = final[0]['isolated_entities'] if final else 0
        final_total = final[0]['total_entities'] if final else 0
        
        print("\n" + "=" * 60)
        print(f"📋 Isolation Prevention Summary:")
        print(f"   Removed noise entities: {removed_noise}")
        print(f"   Improved connections: {improved_connections}")
        print(f"   Isolated entities: {baseline_isolated} -> {final_isolated}")
        print(f"   Total entities: {baseline_total} -> {final_total}")
        print(f"   Isolation rate: {final_isolated/final_total*100:.1f}%" if final_total > 0 else "   Isolation rate: 0%")
        
        return {
            'removed_noise': removed_noise,
            'improved_connections': improved_connections,
            'final_isolated': final_isolated,
            'final_total': final_total
        }
    
    def add_isolation_check_to_quality_assessment(self):
        """Add isolation penalty to quality scoring"""
        print("📊 Adding isolation penalty to quality metrics...")
        
        # This would ideally modify the quality assessment service
        # For now, we'll create a standalone isolation quality metric
        
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH count(DISTINCT n) as total_entities,
             count(DISTINCT CASE WHEN r IS NULL THEN n END) as isolated_entities
        RETURN total_entities, isolated_entities,
               (total_entities - isolated_entities) * 1.0 / total_entities as connectivity_score
        """
        
        result = self.execute_cypher(query, "Calculated connectivity score")
        if result:
            score = result[0]['connectivity_score']
            print(f"🎯 Current connectivity quality score: {score:.3f}")
            
            if score >= 0.99:
                print("✅ Excellent connectivity (>99% connected)")
            elif score >= 0.95:
                print("✅ Good connectivity (>95% connected)")
            elif score >= 0.90:
                print("⚠️  Acceptable connectivity (>90% connected)")
            else:
                print("❌ Poor connectivity (<90% connected)")
        
        return result[0] if result else None

if __name__ == "__main__":
    prevention = IsolationPrevention()
    
    # Run isolation prevention
    results = prevention.implement_quality_rules()
    
    # Add quality assessment
    connectivity = prevention.add_isolation_check_to_quality_assessment()
    
    if results['final_isolated'] == 0:
        print(f"\n🎉 Perfect! Zero isolated nodes achieved.")
    else:
        print(f"\n⚠️  Still have {results['final_isolated']} isolated nodes to address.")