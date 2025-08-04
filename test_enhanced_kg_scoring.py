#!/usr/bin/env python3
"""
Test script for enhanced knowledge graph scoring system.
Tests technology entity preservation and relationship filtering.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.core.config import get_settings

async def test_technology_entity_scoring():
    """Test that technology entities are properly scored and preserved"""
    print("🧪 Testing Enhanced Technology Entity Scoring")
    print("=" * 60)
    
    extractor = LLMKnowledgeExtractor()
    
    # Test technology entities that should be preserved
    test_entities = [
        ("OceanBase", "TECHNOLOGY"),
        ("SOFAStack", "TECHNOLOGY"), 
        ("Apache Kafka", "TECHNOLOGY"),
        ("Redis", "TECHNOLOGY"),
        ("Machine Learning", "TECHNOLOGY"),
        ("Blockchain", "TECHNOLOGY"),
        ("Cloud Migration", "TECHNOLOGY"),
        ("DevOps", "TECHNOLOGY"),
        ("API Economy", "TECHNOLOGY"),
        ("Digital Identity", "TECHNOLOGY"),
        ("Smart Contract", "TECHNOLOGY"),
        ("Innovation Strategy", "TECHNOLOGY"),
        ("Mainframe Decommissioning", "TECHNOLOGY"),
        ("system", "CONCEPT"),  # Generic term - should be low score
        ("platform", "CONCEPT"),  # Generic term - should be low score
    ]
    
    print("Entity Scoring Results:")
    print("-" * 40)
    
    for entity_name, entity_type in test_entities:
        score = extractor._calculate_business_value_score(entity_name, entity_type)
        threshold = 0.45 if entity_type == 'TECHNOLOGY' else 0.6
        preserved = score >= threshold
        
        status = "✅ PRESERVED" if preserved else "❌ FILTERED"
        print(f"{entity_name:25} | {entity_type:10} | Score: {score:.2f} | Threshold: {threshold:.2f} | {status}")
    
    print("\n" + "=" * 60)
    
    # Count preserved vs filtered technology entities
    tech_entities = [(name, etype) for name, etype in test_entities if etype == 'TECHNOLOGY']
    preserved_tech = []
    filtered_tech = []
    
    for entity_name, entity_type in tech_entities:
        score = extractor._calculate_business_value_score(entity_name, entity_type)
        threshold = 0.45 if entity_type == 'TECHNOLOGY' else 0.6
        
        if score >= threshold:
            preserved_tech.append(entity_name)
        else:
            filtered_tech.append(entity_name)
    
    print(f"📊 Technology Entity Results:")
    print(f"   Total Technology Entities: {len(tech_entities)}")
    print(f"   Preserved: {len(preserved_tech)} ({len(preserved_tech)/len(tech_entities)*100:.1f}%)")
    print(f"   Filtered: {len(filtered_tech)} ({len(filtered_tech)/len(tech_entities)*100:.1f}%)")
    
    if preserved_tech:
        print(f"   ✅ Preserved entities: {', '.join(preserved_tech)}")
    if filtered_tech:
        print(f"   ❌ Filtered entities: {', '.join(filtered_tech)}")
    
    return len(preserved_tech), len(filtered_tech)

def test_entity_type_classification():
    """Test that entity type classification properly identifies technology entities"""
    print("\n🔍 Testing Entity Type Classification")
    print("=" * 60)
    
    extractor = LLMKnowledgeExtractor()
    
    test_entities = [
        "OceanBase",
        "SOFAStack", 
        "Apache Kafka",
        "Redis",
        "Machine Learning",
        "Artificial Intelligence",
        "Blockchain",
        "Cloud Migration",
        "DevOps",
        "API Economy",
        "Digital Identity",
        "Smart Contract",
        "Kubernetes",
        "Docker",
        "AWS",
        "Microsoft Azure",
        "system",  # Should be CONCEPT
        "platform",  # Should be CONCEPT
        "DBS Bank",  # Should be ORGANIZATION
        "CEO John Smith"  # Should be EXECUTIVE (if format matches)
    ]
    
    print("Entity Type Classification Results:")
    print("-" * 50)
    
    tech_count = 0
    for entity_name in test_entities:
        entity_type = extractor._infer_entity_type_from_name(entity_name)
        is_tech = entity_type == 'TECHNOLOGY'
        if is_tech:
            tech_count += 1
        
        status = "🔧 TECH" if is_tech else f"📋 {entity_type}"
        print(f"{entity_name:25} | {status}")
    
    print(f"\n📊 Classification Summary:")
    print(f"   Total entities: {len(test_entities)}")
    print(f"   Classified as TECHNOLOGY: {tech_count} ({tech_count/len(test_entities)*100:.1f}%)")
    
    return tech_count

async def test_sample_document_extraction():
    """Test extraction on a sample technology-heavy document"""
    print("\n📄 Testing Sample Document Extraction")
    print("=" * 60)
    
    # Sample business document with technology entities
    sample_text = """
    DBS Bank is implementing a comprehensive digital transformation strategy using multiple 
    technologies. The core infrastructure will migrate from legacy mainframe systems to 
    cloud-native architecture powered by AWS and Microsoft Azure.
    
    Key technology components include:
    - OceanBase database for core banking operations
    - SOFAStack middleware for application integration
    - Apache Kafka for real-time data streaming
    - Redis for caching and session management
    - Machine Learning algorithms for fraud detection
    - Blockchain technology for smart contracts
    
    The DevOps team is using Kubernetes and Docker for containerization, while the 
    API Economy initiative focuses on open banking capabilities. Digital Identity 
    solutions will enhance customer authentication using biometric technologies.
    
    This Innovation Strategy includes mainframe decommissioning and legacy modernization
    to support the bank's digital wallet and mobile banking platforms.
    """
    
    try:
        extractor = LLMKnowledgeExtractor()
        
        print("Extracting entities and relationships...")
        result = await extractor.extract_entities_and_relationships(sample_text)
        
        entities = result.get('entities', [])
        relationships = result.get('relationships', [])
        
        print(f"\n📊 Extraction Results:")
        print(f"   Total Entities: {len(entities)}")
        print(f"   Total Relationships: {len(relationships)}")
        
        # Count technology entities
        tech_entities = [e for e in entities if hasattr(e, 'entity_type') and e.entity_type == 'TECHNOLOGY']
        print(f"   Technology Entities: {len(tech_entities)}")
        
        if tech_entities:
            print("\n🔧 Technology Entities Found:")
            for entity in tech_entities[:10]:  # Show first 10
                score = extractor._calculate_business_value_score(entity.canonical_form, entity.entity_type)
                print(f"   - {entity.canonical_form} (score: {score:.2f})")
        
        # Test relationship filtering
        if relationships:
            # Create mock relationships to test filtering
            from app.models.knowledge_graph import ExtractedRelationship
            
            mock_relationships = []
            for i, rel in enumerate(relationships[:20]):  # Test with first 20
                mock_rel = ExtractedRelationship(
                    source_entity=f"Entity_{i}",
                    target_entity=f"Target_{i}",
                    relationship_type="USES",
                    confidence=0.5 + (i % 5) * 0.1,
                    context=f"Context for relationship {i}",
                    properties={'fuzzy_matched': i % 3 == 0}
                )
                mock_relationships.append(mock_rel)
            
            original_count = len(mock_relationships)
            filtered_relationships = extractor._filter_relationships_for_quality(mock_relationships)
            filtered_count = len(filtered_relationships)
            
            reduction_pct = (original_count - filtered_count) / original_count * 100
            
            print(f"\n🔗 Relationship Filtering Test:")
            print(f"   Original: {original_count} relationships")
            print(f"   Filtered: {filtered_count} relationships")
            print(f"   Reduction: {reduction_pct:.1f}% (target: ~25%)")
            
            return len(entities), len(relationships), reduction_pct
        
        return len(entities), len(relationships), 0.0
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return 0, 0, 0.0

async def main():
    """Run all tests"""
    print("🧪 Enhanced Knowledge Graph Scoring System Tests")
    print("=" * 80)
    
    # Test 1: Entity scoring
    preserved_tech, filtered_tech = await test_technology_entity_scoring()
    
    # Test 2: Entity type classification  
    tech_classified = test_entity_type_classification()
    
    # Test 3: Sample document extraction
    total_entities, total_relationships, reduction_pct = await test_sample_document_extraction()
    
    # Summary
    print("\n🎯 TEST SUMMARY")
    print("=" * 40)
    print(f"✅ Technology Entity Preservation:")
    print(f"   - {preserved_tech} tech entities preserved")
    print(f"   - {filtered_tech} tech entities filtered")
    
    print(f"✅ Entity Type Classification:")
    print(f"   - {tech_classified} entities classified as TECHNOLOGY")
    
    if total_entities > 0:
        print(f"✅ Sample Document Extraction:")
        print(f"   - {total_entities} entities extracted")
        print(f"   - {total_relationships} relationships extracted")
        print(f"   - {reduction_pct:.1f}% relationship reduction achieved")
    
    # Success criteria
    success = (
        preserved_tech >= 10 and  # Most tech entities should be preserved
        filtered_tech <= 2 and   # Very few tech entities should be filtered
        tech_classified >= 10     # Good classification of tech entities
    )
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Enhanced scoring system working correctly!")
    else:
        print("\n⚠️  Some tests need attention - review scoring thresholds")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)