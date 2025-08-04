#!/usr/bin/env python3

"""
Simulation test for aggressive knowledge graph relationship filtering.
Tests the filtering logic without requiring LLM server connection.
"""

import sys
import os
from typing import List, Dict

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.llm_knowledge_extractor import LLMKnowledgeExtractor
from services.knowledge_graph_types import ExtractedEntity, ExtractedRelationship

def create_test_entities() -> List[ExtractedEntity]:
    """Create test entities simulating a business document extraction"""
    entities = [
        ExtractedEntity(text="DBS Bank", label="ORGANIZATION", start_char=0, end_char=8, confidence=0.9, canonical_form="DBS Bank"),
        ExtractedEntity(text="Piyush Gupta", label="EXECUTIVE", start_char=10, end_char=22, confidence=0.8, canonical_form="Piyush Gupta"),
        ExtractedEntity(text="AWS", label="TECHNOLOGY", start_char=30, end_char=33, confidence=0.9, canonical_form="AWS"),
        ExtractedEntity(text="Singapore", label="LOCATION", start_char=40, end_char=49, confidence=0.8, canonical_form="Singapore"),
        ExtractedEntity(text="PayLah!", label="PRODUCT", start_char=60, end_char=67, confidence=0.7, canonical_form="PayLah!"),
        ExtractedEntity(text="Java", label="TECHNOLOGY", start_char=80, end_char=84, confidence=0.6, canonical_form="Java"),
        ExtractedEntity(text="React", label="TECHNOLOGY", start_char=90, end_char=95, confidence=0.6, canonical_form="React"),
        ExtractedEntity(text="OCBC Bank", label="ORGANIZATION", start_char=100, end_char=109, confidence=0.8, canonical_form="OCBC Bank"),
        ExtractedEntity(text="Mobile Banking", label="SERVICE", start_char=120, end_char=134, confidence=0.7, canonical_form="Mobile Banking"),
        ExtractedEntity(text="AI Platform", label="TECHNOLOGY", start_char=140, end_char=151, confidence=0.9, canonical_form="AI Platform"),
    ]
    return entities

def create_test_relationships_high_volume() -> List[ExtractedRelationship]:
    """Create a high volume of test relationships (simulating 1238 relationships issue)"""
    relationships = []
    
    # Get our test entity names for realistic relationships
    entity_names = ["DBS Bank", "Piyush Gupta", "AWS", "Singapore", "PayLah!", 
                   "Java", "React", "OCBC Bank", "Mobile Banking", "AI Platform"]
    
    # High-quality relationships that should pass filtering
    high_quality_rels = [
        ExtractedRelationship("DBS Bank", "Piyush Gupta", "MANAGED_BY", 0.95, "CEO relationship"),
        ExtractedRelationship("DBS Bank", "AWS", "USES", 0.90, "Cloud infrastructure"),
        ExtractedRelationship("DBS Bank", "Singapore", "LOCATED_IN", 0.85, "Primary market"),
        ExtractedRelationship("PayLah!", "DBS Bank", "OWNED_BY", 0.88, "Digital product"),
        ExtractedRelationship("AI Platform", "Java", "IMPLEMENTS", 0.92, "Technology stack"),
        ExtractedRelationship("DBS Bank", "OCBC Bank", "COMPETES_WITH", 0.80, "Market competition"),
        ExtractedRelationship("Mobile Banking", "PayLah!", "INTEGRATES_WITH", 0.75, "Service integration"),
        ExtractedRelationship("React", "AI Platform", "USED_BY", 0.78, "Frontend technology"),
    ]
    relationships.extend(high_quality_rels)
    
    # Medium-quality relationships connecting our actual entities (some should pass, some filtered out)
    import random
    medium_quality_rels = []
    for i in range(150):  # More relationships between actual entities
        source = random.choice(entity_names)
        target = random.choice(entity_names)
        if source != target:  # Avoid self-references
            # Vary confidence from 0.5 to 0.7
            confidence = 0.5 + (i % 20) * 0.01  
            rel = ExtractedRelationship(
                source, target, "COLLABORATES_WITH", confidence, f"Medium quality {i}"
            )
            medium_quality_rels.append(rel)
    relationships.extend(medium_quality_rels)
    
    # Low-quality relationships that should be filtered out
    low_quality_rels = []
    for i in range(100):
        rel = ExtractedRelationship(
            f"Generic_{i}", f"Generic_{i+1}", "RELATED_TO", 0.3, f"Low quality {i}"
        )
        low_quality_rels.append(rel)
    relationships.extend(low_quality_rels)
    
    # Generic relationship types that should be filtered out
    generic_rels = []
    for i in range(200):
        generic_types = ["RELATED_TO", "ASSOCIATED_WITH", "CONNECTED_TO", "MENTIONED_WITH"]
        rel = ExtractedRelationship(
            f"Item_{i}", f"Item_{i+1}", generic_types[i % len(generic_types)], 0.4, f"Generic {i}"
        )
        generic_rels.append(rel)
    relationships.extend(generic_rels)
    
    # Additional bulk relationships connecting to actual entities
    for i in range(778):  # 8 + 150 + 100 + 200 + 778 = 1236 ≈ 1238
        source = random.choice(entity_names)
        target = random.choice(entity_names + [f"External_{i%100}"])  # Mix of actual and external entities
        if source != target:
            # Vary confidence from 0.4 to 0.7 for bulk relationships  
            confidence = 0.4 + (i % 30) * 0.01
            rel = ExtractedRelationship(
                source, target, "USES", confidence, f"Bulk relationship {i}"
            )
            relationships.append(rel)
    
    return relationships

def test_aggressive_filtering_simulation():
    """Test the aggressive filtering logic with simulated data"""
    
    print("🚨 TESTING AGGRESSIVE FILTERING SIMULATION")
    print("=" * 60)
    
    # Create test data
    entities = create_test_entities()
    relationships = create_test_relationships_high_volume()
    
    print(f"📊 Input Data:")
    print(f"   Entities: {len(entities)}")
    print(f"   Relationships: {len(relationships)}")
    print(f"   Ratio: {len(relationships) / len(entities):.1f} relationships per entity")
    
    # Initialize extractor and test filtering
    extractor = LLMKnowledgeExtractor()
    
    # Test the aggressive filtering method directly
    print(f"\n🚨 Applying aggressive filtering...")
    filtered_relationships = extractor._apply_aggressive_relationship_filtering(relationships, entities)
    
    # Analyze results
    filtered_count = len(filtered_relationships)
    filtered_ratio = filtered_count / len(entities) if entities else 0
    
    print(f"\n📊 Filtering Results:")
    print(f"   Original: {len(relationships)} relationships")
    print(f"   Filtered: {filtered_count} relationships")
    print(f"   Reduction: {((len(relationships) - filtered_count) / len(relationships) * 100):.1f}%")
    print(f"   New ratio: {filtered_ratio:.1f} relationships per entity")
    
    # Validation checks
    success_checks = []
    
    # Check 1: Relationship count within target range (150-300)
    rel_count_ok = 150 <= filtered_count <= 300
    success_checks.append(rel_count_ok)
    status = "✅ PASS" if rel_count_ok else "❌ FAIL"
    print(f"   {status} Relationship count: {filtered_count} (target: 150-300)")
    
    # Check 2: Relationships per entity ratio (2-4)
    ratio_ok = 2.0 <= filtered_ratio <= 4.0
    success_checks.append(ratio_ok)
    status = "✅ PASS" if ratio_ok else "❌ FAIL"
    print(f"   {status} Relationships per entity: {filtered_ratio:.1f} (target: 2.0-4.0)")
    
    # Check 3: No generic relationship types
    generic_types = ['RELATED_TO', 'ASSOCIATED_WITH', 'CONNECTED_TO', 'MENTIONED_WITH']
    generic_found = [r for r in filtered_relationships if r.relationship_type in generic_types]
    no_generic_ok = len(generic_found) == 0
    success_checks.append(no_generic_ok)
    status = "✅ PASS" if no_generic_ok else "❌ FAIL"
    print(f"   {status} No generic relationship types: {len(generic_found)} found")
    
    # Check 4: High confidence relationships (>=0.7)
    high_conf_rels = [r for r in filtered_relationships if float(r.confidence or 0) >= 0.7]
    high_conf_ratio = len(high_conf_rels) / max(filtered_count, 1)
    high_conf_ok = high_conf_ratio >= 0.5  # At least 50% should be high confidence
    success_checks.append(high_conf_ok)
    status = "✅ PASS" if high_conf_ok else "❌ FAIL"
    print(f"   {status} High confidence ratio: {high_conf_ratio:.1%} (target: >=50%)")
    
    # Show relationship type distribution
    print(f"\n📊 Filtered Relationship Type Distribution:")
    rel_type_counts = {}
    for rel in filtered_relationships:
        rel_type = rel.relationship_type
        rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
    
    for rel_type, count in sorted(rel_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {rel_type}: {count}")
    
    # Show sample high-quality relationships
    print(f"\n📊 Sample Filtered Relationships:")
    high_quality_rels = sorted(filtered_relationships, key=lambda r: float(r.confidence or 0), reverse=True)[:5]
    for i, rel in enumerate(high_quality_rels, 1):
        conf = float(rel.confidence or 0)
        print(f"   {i}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity} (conf: {conf:.2f})")
    
    # Overall result
    all_passed = all(success_checks)
    overall_status = "✅ SUCCESS" if all_passed else "❌ FAILED"
    print(f"\n{overall_status}: Aggressive filtering simulation test")
    
    if all_passed:
        print("🎉 Aggressive filtering is working correctly!")
        print("   • Massive relationship reduction achieved (1200+ → ~300)")
        print("   • Relationships per entity optimized for visualization")
        print("   • Generic relationship types eliminated")
        print("   • High confidence relationships prioritized")
    else:
        print("⚠️  Some filtering criteria not met. Review implementation.")
        
    return all_passed

if __name__ == "__main__":
    success = test_aggressive_filtering_simulation()
    sys.exit(0 if success else 1)