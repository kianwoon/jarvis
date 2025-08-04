#!/usr/bin/env python3
"""
Simple test for enhanced technology entity scoring and relationship filtering.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.services.knowledge_graph_types import ExtractedRelationship

def test_technology_entity_scoring():
    """Test that technology entities receive appropriate scores"""
    print("🧪 Testing Technology Entity Scoring")
    print("=" * 50)
    
    extractor = LLMKnowledgeExtractor()
    
    # Test cases: (entity_name, entity_type, expected_to_be_preserved)
    test_cases = [
        # Technology entities that should be preserved
        ("OceanBase", "TECHNOLOGY", True),
        ("SOFAStack", "TECHNOLOGY", True),
        ("Apache Kafka", "TECHNOLOGY", True),
        ("Redis", "TECHNOLOGY", True),
        ("Machine Learning", "TECHNOLOGY", True),
        ("Blockchain", "TECHNOLOGY", True),
        ("Cloud Migration", "TECHNOLOGY", True),
        ("DevOps", "TECHNOLOGY", True),
        ("API Economy", "TECHNOLOGY", True),
        ("Digital Identity", "TECHNOLOGY", True),
        ("Smart Contract", "TECHNOLOGY", True),
        ("Innovation Strategy", "TECHNOLOGY", True),
        ("Mainframe Decommissioning", "TECHNOLOGY", True),
        ("Kubernetes", "TECHNOLOGY", True),
        ("Docker", "TECHNOLOGY", True),
        ("AWS", "TECHNOLOGY", True),
        ("Microsoft Azure", "TECHNOLOGY", True),
        
        # Generic terms that should be filtered
        ("system", "CONCEPT", False),
        ("platform", "CONCEPT", False),
        ("technology", "CONCEPT", False),
        ("solution", "CONCEPT", False),
    ]
    
    preserved_count = 0
    filtered_count = 0
    correct_predictions = 0
    
    for entity_name, entity_type, expected_preserved in test_cases:
        score = extractor._calculate_business_value_score(entity_name, entity_type)
        threshold = 0.45 if entity_type == 'TECHNOLOGY' else 0.6
        actually_preserved = score >= threshold
        
        status = "✅ PRESERVED" if actually_preserved else "❌ FILTERED"
        prediction_correct = actually_preserved == expected_preserved
        correct_indicator = "✓" if prediction_correct else "✗"
        
        print(f"{entity_name:25} | {entity_type:10} | Score: {score:.2f} | {status} | {correct_indicator}")
        
        if actually_preserved:
            preserved_count += 1
        else:
            filtered_count += 1
            
        if prediction_correct:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_cases) * 100
    
    print(f"\n📊 Results:")
    print(f"   Total entities tested: {len(test_cases)}")
    print(f"   Preserved: {preserved_count}")
    print(f"   Filtered: {filtered_count}")
    print(f"   Prediction accuracy: {accuracy:.1f}%")
    
    return accuracy >= 90.0  # Success if 90%+ accuracy

def test_entity_type_classification():
    """Test entity type classification for technology entities"""
    print("\n🔍 Testing Entity Type Classification")
    print("=" * 50)
    
    extractor = LLMKnowledgeExtractor()
    
    # Technology entities that should be classified as TECHNOLOGY
    tech_entities = [
        "OceanBase", "SOFAStack", "Apache Kafka", "Redis", "MongoDB",
        "Machine Learning", "Artificial Intelligence", "Blockchain",
        "Cloud Migration", "DevOps", "API Economy", "Digital Identity",
        "Smart Contract", "Kubernetes", "Docker", "AWS", "Microsoft Azure"
    ]
    
    correctly_classified = 0
    for entity_name in tech_entities:
        entity_type = extractor._infer_entity_type_from_name(entity_name)
        is_correct = entity_type == 'TECHNOLOGY'
        status = "✓" if is_correct else "✗"
        
        print(f"{entity_name:25} | {entity_type:12} | {status}")
        
        if is_correct:
            correctly_classified += 1
    
    accuracy = correctly_classified / len(tech_entities) * 100
    
    print(f"\n📊 Classification Results:")
    print(f"   Total tech entities: {len(tech_entities)}")
    print(f"   Correctly classified: {correctly_classified}")
    print(f"   Classification accuracy: {accuracy:.1f}%")
    
    return accuracy >= 85.0  # Success if 85%+ accuracy

def test_relationship_filtering():
    """Test relationship filtering reduces count by ~25%"""
    print("\n🔗 Testing Relationship Filtering")
    print("=" * 50)
    
    extractor = LLMKnowledgeExtractor()
    
    # Create mock relationships with varying quality
    mock_relationships = []
    
    # High quality relationships
    for i in range(20):
        rel = ExtractedRelationship(
            source_entity=f"TechEntity_{i}",
            target_entity=f"System_{i}",
            relationship_type="USES",
            confidence=0.8 + (i % 3) * 0.05,
            context=f"Detailed context explaining the relationship between entities {i}",
            properties={'fuzzy_matched': False, 'temporal_info': 'Q2 2024'}
        )
        mock_relationships.append(rel)
    
    # Medium quality relationships
    for i in range(15):
        rel = ExtractedRelationship(
            source_entity=f"Entity_{i}",
            target_entity=f"Target_{i}",
            relationship_type="RELATED_TO",
            confidence=0.6,
            context=f"Basic context {i}",
            properties={'fuzzy_matched': True}
        )
        mock_relationships.append(rel)
    
    # Low quality relationships
    for i in range(15):
        rel = ExtractedRelationship(
            source_entity=f"E{i}",  # Very short names
            target_entity=f"T{i}",
            relationship_type="ASSOCIATED_WITH",
            confidence=0.4,
            context="",
            properties={'fuzzy_matched': True}
        )
        mock_relationships.append(rel)
    
    original_count = len(mock_relationships)
    filtered_relationships = extractor._filter_relationships_for_quality(mock_relationships)
    filtered_count = len(filtered_relationships)
    
    reduction_pct = (original_count - filtered_count) / original_count * 100
    target_reduction = 25.0
    
    print(f"Original relationships: {original_count}")
    print(f"Filtered relationships: {filtered_count}")
    print(f"Reduction: {reduction_pct:.1f}% (target: ~{target_reduction}%)")
    
    # Check if reduction is close to target (within 10% tolerance)
    success = abs(reduction_pct - target_reduction) <= 10.0 
    
    if success:
        print("✅ Relationship filtering working correctly")
    else:
        print("⚠️  Relationship filtering needs adjustment")
    
    return success

def main():
    """Run all tests"""
    print("🧪 Enhanced Knowledge Graph System Tests")
    print("=" * 60)
    
    test1_success = test_technology_entity_scoring()
    test2_success = test_entity_type_classification()
    test3_success = test_relationship_filtering()
    
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 30)
    print(f"Entity Scoring Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Type Classification Test: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"Relationship Filtering Test: {'✅ PASS' if test3_success else '❌ FAIL'}")
    
    overall_success = test1_success and test2_success and test3_success
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The enhanced scoring system is working correctly:")
        print("• Technology entities are properly preserved")
        print("• Entity type classification is accurate") 
        print("• Relationship filtering reduces count by ~25%")
    else:
        print("\n⚠️  Some tests failed - system needs review")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)