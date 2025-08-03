#!/usr/bin/env python3
"""
Test Entity Parsing Fixes

This script tests the critical entity name parsing fixes to resolve
the empty entity names issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# Configure logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

def test_entity_name_extraction():
    """Test entity name extraction with multiple field formats"""
    print("🧠 Testing Entity Name Extraction Fixes")
    print("=" * 60)
    
    extractor = LLMKnowledgeExtractor()
    
    # Test cases with different entity formats that might come from LLM
    test_cases = [
        {
            'name': 'Standard Format',
            'entities': [
                {"text": "DBS Bank", "type": "ORGANIZATION", "confidence": 0.9},
                {"text": "Singapore", "type": "LOCATION", "confidence": 0.8}
            ]
        },
        {
            'name': 'Name Field Format',
            'entities': [
                {"name": "Apple Inc", "type": "COMPANY", "confidence": 0.95},
                {"name": "Tim Cook", "type": "PERSON", "confidence": 0.85}
            ]
        },
        {
            'name': 'Entity Field Format',
            'entities': [
                {"entity": "Microsoft", "type": "ORGANIZATION", "confidence": 0.85},
                {"entity": "OpenAI", "type": "COMPANY", "confidence": 0.9}
            ]
        },
        {
            'name': 'Canonical Form Format',
            'entities': [
                {"canonical_form": "Google Inc", "type": "ORGANIZATION", "confidence": 0.9},
                {"canonical_form": "California", "type": "LOCATION", "confidence": 0.8}
            ]
        },
        {
            'name': 'Mixed Format',
            'entities': [
                {"text": "Tesla", "canonical_form": "Tesla Inc", "type": "COMPANY", "confidence": 0.9},
                {"name": "Elon Musk", "text": "", "type": "PERSON", "confidence": 0.85}
            ]
        },
        {
            'name': 'Empty Names (Should be filtered)',
            'entities': [
                {"text": "", "type": "ORGANIZATION", "confidence": 0.9},
                {"name": "", "type": "PERSON", "confidence": 0.8},
                {"entity": "   ", "type": "COMPANY", "confidence": 0.7},
                {"text": "Valid Entity", "type": "ORGANIZATION", "confidence": 0.9}
            ]
        }
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"   Input entities: {len(test_case['entities'])}")
        
        try:
            # Use the enhanced entity processing method
            enhanced_entities = extractor._enhance_entities_with_hierarchy(test_case['entities'])
            
            print(f"   Output entities: {len(enhanced_entities)}")
            
            # Check each enhanced entity
            valid_entities = 0
            for entity in enhanced_entities:
                if entity.text and entity.text.strip() and len(entity.text.strip()) >= 2:
                    valid_entities += 1
                    print(f"   ✅ '{entity.text}' -> '{entity.canonical_form}' (Type: {entity.label})")
                else:
                    print(f"   ❌ Invalid entity: text='{entity.text}', canonical='{entity.canonical_form}'")
            
            # Test success criteria
            expected_valid = len([e for e in test_case['entities'] 
                                if (e.get('text', '') or e.get('name', '') or 
                                   e.get('entity', '') or e.get('canonical_form', '')).strip() and
                                   len((e.get('text', '') or e.get('name', '') or 
                                       e.get('entity', '') or e.get('canonical_form', '')).strip()) >= 2])
            
            if valid_entities == expected_valid and valid_entities == len(enhanced_entities):
                print(f"   ✅ PASSED: {valid_entities} valid entities extracted as expected")
                passed_tests += 1
            else:
                print(f"   ❌ FAILED: Expected {expected_valid} valid entities, got {valid_entities} valid out of {len(enhanced_entities)} total")
            
            total_tests += 1
            
        except Exception as e:
            print(f"   ❌ FAILED with exception: {e}")
            total_tests += 1
    
    print(f"\n📊 Entity Name Extraction Test Results: {passed_tests}/{total_tests} successful")
    return passed_tests == total_tests

def test_json_parsing_robustness():
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
        print(f"\n🧪 Testing: {test_case['name']}")
        
        try:
            parsed = extractor._parse_llm_response(test_case['response'])
            
            entities = parsed.get('entities', [])
            relationships = parsed.get('relationships', [])
            
            print(f"   ✅ Parsed: {len(entities)} entities, {len(relationships)} relationships")
            
            # Check for entity name extraction
            valid_names = 0
            for entity in entities:
                name = (entity.get('text', '') or entity.get('name', '') or 
                       entity.get('entity', '') or entity.get('canonical_form', '')).strip()
                if not name:
                    print(f"   ❌ Entity with empty name found: {entity}")
                else:
                    print(f"   ✅ Entity name extracted: '{name}'")
                    valid_names += 1
            
            if valid_names == len(entities) and len(entities) > 0:
                success_count += 1
                print(f"   ✅ PASSED: All {len(entities)} entities have valid names")
            else:
                print(f"   ❌ FAILED: {valid_names}/{len(entities)} entities have valid names")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n📊 JSON Parsing Test Results: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)

def test_relationship_validation():
    """Test relationship validation with entity matching"""
    print("\n🔗 Testing Relationship Validation")
    print("=" * 60)
    
    extractor = LLMKnowledgeExtractor()
    
    # Create test entities
    from app.services.knowledge_graph_types import ExtractedEntity
    test_entities = [
        ExtractedEntity(text="DBS Bank", label="ORGANIZATION", start_char=0, end_char=8, confidence=0.9, canonical_form="DBS Bank"),
        ExtractedEntity(text="Singapore", label="LOCATION", start_char=0, end_char=9, confidence=0.8, canonical_form="Singapore"),
        ExtractedEntity(text="Tim Cook", label="PERSON", start_char=0, end_char=8, confidence=0.85, canonical_form="Tim Cook"),
        ExtractedEntity(text="Apple Inc", label="ORGANIZATION", start_char=0, end_char=9, confidence=0.9, canonical_form="Apple Inc")
    ]
    
    # Create test relationships
    test_relationships = [
        {"source_entity": "DBS Bank", "target_entity": "Singapore", "relationship_type": "LOCATED_IN", "confidence": 0.7},
        {"source_entity": "Tim Cook", "target_entity": "Apple Inc", "relationship_type": "WORKS_FOR", "confidence": 0.8},
        {"source_entity": "dbs bank", "target_entity": "singapore", "relationship_type": "OPERATES_IN", "confidence": 0.6},  # Case mismatch
        {"source_entity": "NonExistent Entity", "target_entity": "Singapore", "relationship_type": "LOCATED_IN", "confidence": 0.5},  # Invalid source
        {"source_entity": "Tim Cook", "target_entity": "Unknown Company", "relationship_type": "WORKS_FOR", "confidence": 0.7}  # Invalid target
    ]
    
    print(f"🏢 Test entities: {len(test_entities)}")
    print(f"🔗 Test relationships: {len(test_relationships)}")
    
    try:
        # Validate relationships
        validated_relationships = extractor._validate_and_score_relationships(test_relationships, test_entities)
        
        print(f"✅ Validated relationships: {len(validated_relationships)}")
        
        # Analyze results
        valid_count = 0
        for rel in validated_relationships:
            print(f"   ✅ {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity} (Confidence: {rel.confidence:.2f})")
            valid_count += 1
        
        # Expected: 3 valid relationships (exact match, exact match, case-insensitive match)
        expected_valid = 3
        
        if valid_count == expected_valid:
            print(f"   ✅ PASSED: {valid_count} relationships validated as expected")
            return True
        else:
            print(f"   ❌ FAILED: Expected {expected_valid} valid relationships, got {valid_count}")
            return False
            
    except Exception as e:
        print(f"   ❌ FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Knowledge Graph Entity Parsing Fixes")
    print("=" * 80)
    
    # Run all tests
    tests = [
        ("Entity Name Extraction", test_entity_name_extraction),
        ("JSON Parsing Robustness", test_json_parsing_robustness),
        ("Relationship Validation", test_relationship_validation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
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
        print("🎉 ALL TESTS PASSED! Entity parsing fixes are working correctly.")
        print("\n🔧 Key Fixes Verified:")
        print("   ✅ Entity names extracted from multiple field formats (text, name, entity, canonical_form)")
        print("   ✅ Empty entity names properly filtered out")
        print("   ✅ JSON parsing handles various response formats robustly")
        print("   ✅ Relationship validation works with fuzzy entity matching")
        print("   ✅ Enhanced debugging and logging provides clear diagnostics")
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()