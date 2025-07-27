#!/usr/bin/env python3
"""
Knowledge Graph Quality Fix Script

Fixes specific quality issues identified in the real data:
1. Nonsensical LOCATED_IN relationships between organizations
2. Entity naming issues (multiple concepts in one entity)
3. Classification errors
"""

import requests
import json
import re
from typing import Dict, List, Tuple

class KnowledgeGraphQualityFixer:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
        # Mapping of problematic relationships to better ones based on context keywords
        self.relationship_fixes = {
            "appointed": "APPOINTED",
            "hired": "HIRED",
            "recruit": "RECRUITED_FROM", 
            "former": "RECRUITED_FROM",
            "worked with": "PARTNERS_WITH",
            "collaboration": "PARTNERS_WITH",
            "compliance": "REGULATED_BY",
            "guideline": "REGULATED_BY",
            "explore": "STUDIES",
            "similar": "STUDIES",
            "guidance": "GUIDED_BY"
        }
    
    def execute_cypher(self, query: str, description: str = "") -> List[Dict]:
        """Execute a Cypher query"""
        response = requests.post(
            f"{self.base_url}/api/v1/knowledge-graph/query",
            headers={"Content-Type": "application/json"},
            json={"query": query, "query_type": "cypher"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {description}: {len(data['results'])} results")
            return data['results']
        else:
            print(f"❌ {description} failed: {response.status_code}")
            return []
    
    def fix_org_located_in_relationships(self):
        """Fix ORG->ORG LOCATED_IN relationships"""
        print("🔧 Fixing nonsensical LOCATED_IN relationships...")
        
        # Get all problematic relationships with context
        query = """
        MATCH (a:ORGANIZATION)-[r:LOCATED_IN]->(b:ORGANIZATION) 
        RETURN a.name as source, b.name as target, r.context as context, 
               r.confidence as confidence, r.chunk_id as chunk_id,
               r.document_id as document_id
        """
        
        problematic_rels = self.execute_cypher(query, "Found ORG->ORG LOCATED_IN relationships")
        
        fixed_count = 0
        for rel in problematic_rels:
            source = rel['source']
            target = rel['target'] 
            context = rel.get('context', '').lower()
            confidence = rel.get('confidence', 0.7)
            chunk_id = rel.get('chunk_id', '')
            document_id = rel.get('document_id', '')
            
            # Determine better relationship type based on context
            new_relationship = self._determine_better_relationship(context)
            
            if new_relationship != "LOCATED_IN":
                # Delete the old relationship and create a new one
                delete_query = f"""
                MATCH (a:ORGANIZATION {{name: "{source}"}})-[r:LOCATED_IN]->(b:ORGANIZATION {{name: "{target}"}})
                WHERE r.context = "{rel.get('context', '')}"
                DELETE r
                """
                
                create_query = f"""
                MATCH (a:ORGANIZATION {{name: "{source}"}}), (b:ORGANIZATION {{name: "{target}"}})
                CREATE (a)-[r:{new_relationship}]->(b)
                SET r.confidence = {confidence},
                    r.context = "{rel.get('context', '')}",
                    r.chunk_id = "{chunk_id}",
                    r.document_id = "{document_id}",
                    r.created_at = datetime()
                """
                
                # Execute the fix
                self.execute_cypher(delete_query, f"Deleted LOCATED_IN: {source} -> {target}")
                self.execute_cypher(create_query, f"Created {new_relationship}: {source} -> {target}")
                fixed_count += 1
        
        print(f"✅ Fixed {fixed_count} nonsensical LOCATED_IN relationships")
        return fixed_count
    
    def _determine_better_relationship(self, context: str) -> str:
        """Determine better relationship type based on context"""
        context_lower = context.lower()
        
        # Check for specific patterns
        for keyword, relationship in self.relationship_fixes.items():
            if keyword in context_lower:
                return relationship
        
        # Default fallback
        if any(word in context_lower for word in ["work", "partner", "collab"]):
            return "PARTNERS_WITH"
        elif any(word in context_lower for word in ["regulat", "complian", "guideline"]):
            return "REGULATED_BY"
        elif any(word in context_lower for word in ["study", "explor", "similar", "like"]):
            return "STUDIES"
        else:
            return "CONTEXTUALLY_RELATED"
    
    def fix_entity_naming_issues(self):
        """Fix entities with multiple concepts in one name"""
        print("🔧 Fixing entity naming issues...")
        
        # Find entities with commas or semicolons (multiple concepts)
        query = """
        MATCH (n) 
        WHERE n.name CONTAINS "," OR n.name CONTAINS ";"
        RETURN n.id as id, n.name as name, labels(n)[0] as type,
               n.document_id as document_id, n.chunk_id as chunk_id
        """
        
        problematic_entities = self.execute_cypher(query, "Found entities with naming issues")
        
        fixed_count = 0
        for entity in problematic_entities:
            name = entity['name']
            entity_type = entity['type']
            entity_id = entity['id']
            
            # Split on commas and semicolons
            parts = re.split(r'[,;]\s*', name)
            
            if len(parts) > 1:
                # Remove the original entity
                delete_query = f"""
                MATCH (n {{id: "{entity_id}"}})
                DETACH DELETE n
                """
                
                self.execute_cypher(delete_query, f"Deleted multi-concept entity: {name}")
                
                # Create separate entities for each part
                for i, part in enumerate(parts):
                    part = part.strip()
                    if len(part) > 2:  # Only create if meaningful
                        create_query = f"""
                        CREATE (n:{entity_type})
                        SET n.id = "{entity_id}_{i}",
                            n.name = "{part}",
                            n.type = "{entity_type}",
                            n.confidence = 0.8,
                            n.document_id = "{entity.get('document_id', '')}",
                            n.chunk_id = "{entity.get('chunk_id', '')}",
                            n.created_at = datetime(),
                            n.source = "split_from_multi_concept"
                        """
                        
                        self.execute_cypher(create_query, f"Created entity: {part}")
                
                fixed_count += 1
        
        print(f"✅ Fixed {fixed_count} entity naming issues")
        return fixed_count
    
    def fix_classification_errors(self):
        """Fix specific classification errors identified in quality assessment"""
        print("🔧 Fixing classification errors...")
        
        classification_fixes = [
            ("Ping An Technology", "ORGANIZATION", "CONCEPT"),
            ("Mas Technology Risk Management (Trm", "ORGANIZATION", "CONCEPT"), 
            ("Modernizing Core Systems", "ORGANIZATION", "CONCEPT")
        ]
        
        fixed_count = 0
        for name, current_type, correct_type in classification_fixes:
            # Find and update the entity
            query = f"""
            MATCH (n:{current_type} {{name: "{name}"}})
            REMOVE n:{current_type}
            SET n:{correct_type}, n.type = "{correct_type}"
            RETURN n.name as name
            """
            
            results = self.execute_cypher(query, f"Fixed classification: {name} -> {correct_type}")
            if results:
                fixed_count += 1
        
        print(f"✅ Fixed {fixed_count} classification errors")
        return fixed_count
    
    def run_quality_fixes(self):
        """Run all quality fixes"""
        print("🚀 Starting Knowledge Graph Quality Fixes")
        print("=" * 60)
        
        # Get baseline metrics
        baseline_query = """
        MATCH ()-[r:LOCATED_IN]->() 
        RETURN count(r) as total_located_in
        """
        baseline = self.execute_cypher(baseline_query, "Baseline LOCATED_IN count")
        baseline_count = baseline[0]['total_located_in'] if baseline else 0
        
        # Run fixes
        fix1 = self.fix_org_located_in_relationships()
        fix2 = self.fix_entity_naming_issues() 
        fix3 = self.fix_classification_errors()
        
        # Get final metrics
        final = self.execute_cypher(baseline_query, "Final LOCATED_IN count")
        final_count = final[0]['total_located_in'] if final else 0
        
        print("\n" + "=" * 60)
        print(f"📋 Quality Fix Summary:")
        print(f"   Fixed LOCATED_IN relationships: {fix1}")
        print(f"   Fixed entity naming issues: {fix2}")
        print(f"   Fixed classification errors: {fix3}")
        print(f"   LOCATED_IN relationships: {baseline_count} -> {final_count}")
        print(f"   Total fixes applied: {fix1 + fix2 + fix3}")
        
        return fix1 + fix2 + fix3

if __name__ == "__main__":
    fixer = KnowledgeGraphQualityFixer()
    total_fixes = fixer.run_quality_fixes()
    
    if total_fixes > 0:
        print(f"\n🎉 Successfully applied {total_fixes} quality fixes!")
        print("💡 Run the quality assessment again to see improvements")
    else:
        print(f"\n⚠️  No fixes were applied. Check the data or try again.")