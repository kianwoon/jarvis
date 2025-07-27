#!/usr/bin/env python3
"""
Comprehensive Knowledge Graph Cleanup

Addresses fundamental quality issues:
1. Removes meaningless entities (ordinals, numbers, common words)
2. Fixes all nonsensical LOCATED_IN relationships
3. Removes low-confidence noise relationships
4. Improves overall semantic quality
"""

import requests
import json
import re
from typing import Dict, List, Set

class ComprehensiveKGCleanup:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
        # Entities that should NOT be in a knowledge graph
        self.meaningless_entities = {
            # Ordinals and positional words
            "First", "Second", "Third", "Fourth", "Fifth", "Last", "Next", "Previous",
            # Common numbers that are not significant
            "30", "90%", "30–70%", "22,000", "707 Million", "200K",
            # Temporal words that should be attributes
            "Days", "Months", "Years", "Hours", "Minutes",
            # Generic concepts
            "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            # Measurement units
            "Percent", "Percentage", "%", "Million", "Billion", "Thousand",
            # Common adjectives/adverbs treated as entities
            "Better", "Faster", "Slower", "Higher", "Lower", "Best", "Worst",
            # Articles and prepositions that got extracted
            "The", "And", "Or", "But", "In", "On", "At", "By", "For", "With", "To"
        }
        
        # Relationship types that are too generic to be useful
        self.generic_relationships = {
            "CONTEXTUALLY_RELATED", "RELATED_TO", "ASSOCIATED_WITH", 
            "MENTIONED_WITH", "DOCUMENT_RELATED"
        }
        
        # Confidence threshold for removing noise relationships
        self.min_confidence = 0.5
        
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
    
    def get_baseline_metrics(self) -> Dict:
        """Get baseline metrics before cleanup"""
        print("📊 Getting baseline metrics...")
        
        # Total entities and relationships
        stats_query = """
        MATCH (n) 
        OPTIONAL MATCH ()-[r]->()
        RETURN count(DISTINCT n) as total_entities, count(r) as total_relationships
        """
        stats = self.execute_cypher(stats_query, "Basic stats")
        
        # Relationship type distribution
        rel_types_query = """
        MATCH ()-[r]->() 
        RETURN type(r) as rel_type, count(r) as count 
        ORDER BY count DESC
        """
        rel_types = self.execute_cypher(rel_types_query, "Relationship types")
        
        # Generic relationship count
        generic_count = sum(r['count'] for r in rel_types if r['rel_type'] in self.generic_relationships)
        total_rels = stats[0]['total_relationships'] if stats else 0
        
        return {
            'total_entities': stats[0]['total_entities'] if stats else 0,
            'total_relationships': total_rels,
            'generic_relationships': generic_count,
            'generic_ratio': generic_count / total_rels if total_rels > 0 else 0,
            'relationship_types': {r['rel_type']: r['count'] for r in rel_types}
        }
    
    def remove_meaningless_entities(self) -> int:
        """Remove entities that are just noise"""
        print("🗑️  Removing meaningless entities...")
        
        removed_count = 0
        
        # Remove specific meaningless entities
        for entity_name in self.meaningless_entities:
            query = f"""
            MATCH (n) 
            WHERE n.name = "{entity_name}" OR n.name = "{entity_name.lower()}" OR n.name = "{entity_name.upper()}"
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            result = self.execute_cypher(query, f"Removed entity: {entity_name}")
            if result and result[0]['deleted'] > 0:
                removed_count += result[0]['deleted']
        
        # Remove entities that are just numbers
        number_query = """
        MATCH (n)
        WHERE n.name =~ '^\\\\d+$' OR n.name =~ '^\\\\d+%$' OR n.name =~ '^\\\\d+[KMB]$'
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        result = self.execute_cypher(number_query, "Removed pure number entities")
        if result:
            removed_count += result[0]['deleted']
        
        # Remove very short entities that are likely noise
        short_query = """
        MATCH (n)
        WHERE length(n.name) <= 2 AND n.name =~ '^[a-zA-Z]+$'
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        result = self.execute_cypher(short_query, "Removed short word entities")
        if result:
            removed_count += result[0]['deleted']
        
        print(f"🗑️  Removed {removed_count} meaningless entities")
        return removed_count
    
    def fix_all_located_in_relationships(self) -> int:
        """Fix ALL nonsensical LOCATED_IN relationships"""
        print("🔧 Fixing all LOCATED_IN relationships...")
        
        # Get all LOCATED_IN relationships
        query = """
        MATCH (a)-[r:LOCATED_IN]->(b)
        RETURN a.name as source, b.name as target, 
               labels(a)[0] as source_type, labels(b)[0] as target_type,
               r.context as context, r.confidence as confidence,
               r.chunk_id as chunk_id, r.document_id as document_id
        """
        
        located_in_rels = self.execute_cypher(query, "Found all LOCATED_IN relationships")
        
        fixed_count = 0
        for rel in located_in_rels:
            source_type = rel['source_type']
            target_type = rel['target_type']
            source = rel['source']
            target = rel['target']
            context = rel.get('context', '').lower()
            
            # Check if this is a valid LOCATED_IN relationship
            should_fix = False
            new_relationship = "LOCATED_IN"
            
            # ORG -> ORG should not be LOCATED_IN
            if source_type == "ORGANIZATION" and target_type == "ORGANIZATION":
                should_fix = True
                # Determine better relationship based on context
                if any(word in context for word in ["appoint", "hire", "former", "ceo", "cio"]):
                    new_relationship = "RECRUITED_FROM"
                elif any(word in context for word in ["partner", "work", "collaboration"]):
                    new_relationship = "PARTNERS_WITH"
                elif any(word in context for word in ["study", "explore", "similar", "like"]):
                    new_relationship = "STUDIES"
                elif any(word in context for word in ["regulat", "complian", "guideline"]):
                    new_relationship = "REGULATED_BY"
                else:
                    new_relationship = "CONTEXTUALLY_RELATED"
            
            # PERSON -> PERSON should not be LOCATED_IN  
            elif source_type == "PERSON" and target_type == "PERSON":
                should_fix = True
                new_relationship = "KNOWS"
            
            # CONCEPT -> CONCEPT should not be LOCATED_IN
            elif source_type == "CONCEPT" and target_type == "CONCEPT":
                should_fix = True
                new_relationship = "RELATED_TO"
            
            # Self-references should be removed
            elif source == target:
                should_fix = True
                new_relationship = "DELETE"
            
            if should_fix:
                # Delete the old relationship
                delete_query = f"""
                MATCH (a {{name: "{source}"}})-[r:LOCATED_IN]->(b {{name: "{target}"}})
                WHERE r.context = "{rel.get('context', '')}"
                DELETE r
                """
                self.execute_cypher(delete_query, f"Deleted bad LOCATED_IN: {source} -> {target}")
                
                # Create new relationship if not deleting
                if new_relationship != "DELETE":
                    create_query = f"""
                    MATCH (a {{name: "{source}"}}), (b {{name: "{target}"}})
                    CREATE (a)-[r:{new_relationship}]->(b)
                    SET r.confidence = {rel.get('confidence', 0.7)},
                        r.context = "{rel.get('context', '')}",
                        r.chunk_id = "{rel.get('chunk_id', '')}",
                        r.document_id = "{rel.get('document_id', '')}",
                        r.created_at = datetime(),
                        r.source = "located_in_fix"
                    """
                    self.execute_cypher(create_query, f"Created {new_relationship}: {source} -> {target}")
                
                fixed_count += 1
        
        print(f"🔧 Fixed {fixed_count} LOCATED_IN relationships")
        return fixed_count
    
    def remove_low_confidence_noise(self) -> int:
        """Remove relationships with low confidence that are clearly noise"""
        print("🧹 Removing low-confidence noise relationships...")
        
        # Remove very low confidence generic relationships
        query = f"""
        MATCH ()-[r]->()
        WHERE r.confidence < {self.min_confidence} 
        AND type(r) IN {list(self.generic_relationships)}
        DELETE r
        RETURN count(r) as deleted
        """
        
        result = self.execute_cypher(query, "Removed low-confidence generic relationships")
        deleted_count = result[0]['deleted'] if result else 0
        
        print(f"🧹 Removed {deleted_count} low-confidence noise relationships")
        return deleted_count
    
    def run_comprehensive_cleanup(self) -> Dict:
        """Run all cleanup operations"""
        print("🚀 Starting Comprehensive Knowledge Graph Cleanup")
        print("=" * 70)
        
        # Get baseline
        baseline = self.get_baseline_metrics()
        print(f"📊 Baseline: {baseline['total_entities']} entities, {baseline['total_relationships']} relationships")
        print(f"📊 Generic ratio: {baseline['generic_ratio']:.1%}")
        
        # Run cleanup operations
        removed_entities = self.remove_meaningless_entities()
        fixed_located_in = self.fix_all_located_in_relationships() 
        removed_noise = self.remove_low_confidence_noise()
        
        # Get final metrics
        final = self.get_baseline_metrics()
        
        print("\n" + "=" * 70)
        print(f"📋 Comprehensive Cleanup Summary:")
        print(f"   Removed meaningless entities: {removed_entities}")
        print(f"   Fixed LOCATED_IN relationships: {fixed_located_in}")
        print(f"   Removed noise relationships: {removed_noise}")
        print(f"   Entities: {baseline['total_entities']} -> {final['total_entities']} ({final['total_entities'] - baseline['total_entities']:+d})")
        print(f"   Relationships: {baseline['total_relationships']} -> {final['total_relationships']} ({final['total_relationships'] - baseline['total_relationships']:+d})")
        print(f"   Generic ratio: {baseline['generic_ratio']:.1%} -> {final['generic_ratio']:.1%}")
        
        improvement_score = (baseline['generic_ratio'] - final['generic_ratio']) * 100
        print(f"   Quality improvement: {improvement_score:+.1f} percentage points")
        
        return {
            'baseline': baseline,
            'final': final,
            'removed_entities': removed_entities,
            'fixed_relationships': fixed_located_in,
            'removed_noise': removed_noise,
            'improvement_score': improvement_score
        }

if __name__ == "__main__":
    cleanup = ComprehensiveKGCleanup()
    results = cleanup.run_comprehensive_cleanup()
    
    if results['improvement_score'] > 5:
        print(f"\n🎉 Significant improvement achieved! (+{results['improvement_score']:.1f}pp)")
    elif results['improvement_score'] > 0:
        print(f"\n✅ Moderate improvement achieved (+{results['improvement_score']:.1f}pp)")
    else:
        print(f"\n⚠️  Limited improvement. May need deeper changes to extraction logic.")