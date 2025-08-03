#!/usr/bin/env python3
"""
Test Knowledge Graph Extraction Fixes

This script tests the critical fixes made to resolve:
1. Missing _create_enhanced_chunk_windows method
2. Entity name parsing issues (empty entity names)
3. Enhanced debugging and validation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from typing import Dict, Any

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
from app.services.multi_chunk_processor import MultiChunkRelationshipProcessor
from app.document_handlers.base import ExtractedChunk

async def test_entity_parsing_fixes():
    """Test that entity name parsing works correctly"""
    print("🧠 Testing Entity Name Parsing Fixes")
    print("=" * 60)
    
    extractor = LLMKnowledgeExtractor()
    
    # Test with business content that should extract entities
    test_text = """
    DBS Bank is a leading financial services provider in Asia. 
    The bank operates in Singapore, Hong Kong, and China. 
    Tim Cook, CEO of Apple Inc., announced a new partnership with DBS.
    The collaboration will focus on digital banking innovation using AI technology.
    OceanBase database system will be evaluated for implementation.
    """
    
    print(f"📝 Test Text: {test_text}")
    print("=" * 60)
    
    try:
        # Test the complete extraction pipeline
        result = await extractor.extract_knowledge(test_text, context={
            'document_type': 'business_strategy',
            'domain': 'banking'
        })
        
        print(f"✅ Extraction Result:")
        print(f"   🏢 Entities: {len(result.entities)}")
        print(f"   🔗 Relationships: {len(result.relationships)}")
        print(f"   🎯 Confidence: {result.confidence_score:.2f}")
        print(f"   ⚡ Processing Time: {result.processing_time_ms:.0f}ms")
        print(f"   🤖 Model: {result.llm_model_used}")
        print()
        
        # Detailed entity analysis
        if result.entities:
            print("📊 Extracted Entities:")
            for i, entity in enumerate(result.entities, 1):
                print(f"   {i}. '{entity.text}' -> '{entity.canonical_form}' (Type: {entity.label}, Confidence: {entity.confidence:.2f})")
        else:
            print("❌ NO ENTITIES EXTRACTED!")
        
        print()
        
        # Detailed relationship analysis
        if result.relationships:
            print("🔗 Extracted Relationships:")
            for i, rel in enumerate(result.relationships, 1):
                print(f"   {i}. {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity} (Confidence: {rel.confidence:.2f})")
        else:
            print("❌ NO RELATIONSHIPS EXTRACTED!")
        
        print()
        
        # Test specific fixes
        print("🔧 Testing Specific Fixes:")
        
        # 1. Check for empty entity names
        empty_names = [e for e in result.entities if not e.text.strip() or not e.canonical_form.strip()]
        if empty_names:
            print(f"   ❌ Found {len(empty_names)} entities with empty names!")
        else:
            print(f"   ✅ No empty entity names found")
        
        # 2. Check entity-relationship mapping
        entity_names = {e.canonical_form.lower() for e in result.entities}
        orphaned_relationships = []
        for rel in result.relationships:
            if rel.source_entity.lower() not in entity_names or rel.target_entity.lower() not in entity_names:
                orphaned_relationships.append(rel)
        
        if orphaned_relationships:
            print(f"   ❌ Found {len(orphaned_relationships)} orphaned relationships (entities not found)!")
            for rel in orphaned_relationships[:3]:
                print(f"      - {rel.source_entity} -> {rel.target_entity}")
        else:
            print(f"   ✅ All relationships have valid entity mappings")
        
        # 3. Test metadata
        print(f"   📊 Extraction Metadata: {result.extraction_metadata.get('extraction_type', 'unknown')}")
        
        return len(result.entities) > 0 and len(result.relationships) > 0
        
    except Exception as e:
        print(f"❌ Entity parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_multi_chunk_processor_fixes():
    """Test that multi-chunk processor works with the new methods"""
    print("\n🔄 Testing Multi-Chunk Processor Fixes")
    print("=" * 60)
    
    processor = MultiChunkRelationshipProcessor()
    
    # Create test chunks
    test_chunks = [
        ExtractedChunk(
            content="DBS Bank is expanding its digital banking services. The bank has partnered with several technology companies.",
            metadata={'chunk_id': 'chunk_1', 'chunk_index': 0}
        ),
        ExtractedChunk(
            content="Technology companies like Apple and Microsoft are collaborating with DBS. Tim Cook announced the partnership.",
            metadata={'chunk_id': 'chunk_2', 'chunk_index': 1}
        ),
        ExtractedChunk(
            content="The collaboration focuses on AI and machine learning. OceanBase database will be implemented for better performance.",
            metadata={'chunk_id': 'chunk_3', 'chunk_index': 2}
        )
    ]
    
    print(f"📦 Created {len(test_chunks)} test chunks")
    
    try:
        # Test the enhanced multi-chunk processing
        result = await processor.process_document_with_overlap(
            chunks=test_chunks,
            document_id="test_doc_001",
            progressive_storage=False
        )
        
        print(f"✅ Multi-Chunk Processing Result:")
        print(f"   📦 Individual Results: {len(result.individual_results)}")
        print(f"   🏢 Cross-Chunk Entities: {len(result.cross_chunk_entities)}")
        print(f"   🔗 Cross-Chunk Relationships: {len(result.cross_chunk_relationships)}")
        print(f"   ⚡ Total Processing Time: {result.total_processing_time_ms:.0f}ms")
        print()
        
        # Analyze quality metrics
        quality = result.quality_metrics
        if quality:
            print("📊 Quality Metrics:")
            print(f"   🎯 Entity Enhancement Ratio: {quality.get('entity_enhancement_ratio', 0):.2f}x")
            print(f"   🔗 Relationship Enhancement Ratio: {quality.get('relationship_enhancement_ratio', 0):.2f}x")
            print(f"   🏢 High Frequency Entities: {quality.get('high_frequency_entities', 0)}")
            
        # Analyze overlap effectiveness
        overlap = result.overlap_analysis
        if overlap:
            print("🔄 Overlap Analysis:")
            print(f"   📦 Total Windows: {overlap.get('total_windows', 0)}")
            print(f"   🎯 Entity-Aware Windows: {overlap.get('entity_aware_windows', 0)}")
            print(f"   🔥 Entity Hotspot Windows: {overlap.get('entity_hotspot_windows', 0)}")
        
        # Check for successful processing
        success = (len(result.individual_results) > 0 and 
                  result.total_processing_time_ms > 0 and
                  not overlap.get('error'))
        
        if success:
            print("   ✅ Multi-chunk processing completed successfully")
        else:
            print("   ❌ Multi-chunk processing had issues")
            
        return success
        
    except Exception as e:
        print(f"❌ Multi-chunk processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_json_parsing_robustness():
    """Test JSON parsing with various response formats"""
    print("\n🧪 Testing JSON Parsing Robustness")
    print("=" * 60)
    
    extractor = LLMKnowledgeExtractor()
    
    # Test cases with different JSON formats
    test_cases = [
        {
            'name': 'Standard JSON',
            'response': '''
            {
                "entities": [
                    {"text": "DBS Bank", "type": "ORGANIZATION", "confidence": 0.9},
                    {"text": "Singapore", "type": "LOCATION", "confidence": 0.8}
                ],
                "relationships": [
                    {"source_entity": "DBS Bank", "target_entity": "Singapore", "relationship_type": "LOCATED_IN", "confidence": 0.7}
                ]
            }
            '''
        },
        {
            'name': 'JSON with code blocks',
            'response': '''
            Here's the extracted information:
            ```json
            {
                "entities": [
                    {"name": "Apple Inc", "type": "COMPANY", "confidence": 0.95}
                ],
                "relationships": []
            }
            ```
            '''
        },
        {
            'name': 'JSON with thinking tags',
            'response': '''
            <think>
            I need to extract entities from this text about technology companies.
            </think>
            {
                "entities": [
                    {"entity": "Microsoft", "type": "ORGANIZATION", "confidence": 0.85}
                ],
                "relationships": []
            }
            '''
        },
        {
            'name': 'Malformed JSON (trailing comma)',
            'response': '''
            {
                "entities": [
                    {"text": "Tim Cook", "type": "PERSON", "confidence": 0.9},
                ],
                "relationships": []
            }
            '''
        }
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        print(f"🧪 Testing: {test_case['name']}")
        
        try:
            parsed = extractor._parse_llm_response(test_case['response'])
            
            entities = parsed.get('entities', [])
            relationships = parsed.get('relationships', [])
            
            print(f"   ✅ Parsed: {len(entities)} entities, {len(relationships)} relationships")
            
            # Check for entity name extraction
            for entity in entities:
                name = (entity.get('text', '') or entity.get('name', '') or 
                       entity.get('entity', '') or entity.get('canonical_form', '')).strip()
                if not name:
                    print(f"   ❌ Entity with empty name found: {entity}")
                else:
                    print(f"   ✅ Entity name extracted: '{name}'")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n📊 JSON Parsing Test Results: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)

async def main():
    """Run all tests"""
    print("🚀 Testing Knowledge Graph Extraction Fixes")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Entity Parsing Fixes", test_entity_parsing_fixes),
        ("Multi-Chunk Processor Fixes", test_multi_chunk_processor_fixes),
        ("JSON Parsing Robustness", test_json_parsing_robustness)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = await test_func()
            results[test_name] = result
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Knowledge graph extraction fixes are working correctly.")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())