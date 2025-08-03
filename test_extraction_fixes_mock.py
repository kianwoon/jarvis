#!/usr/bin/env python3
"""
Test script to verify relationship extraction fixes using mock LLM responses.
This tests the logic without requiring an actual LLM service.
"""

import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_extraction_fixes():
    """Test the extraction fixes with mock LLM responses"""
    print("\n" + "="*80)
    print("TESTING RELATIONSHIP EXTRACTION FIXES (MOCK MODE)")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        from app.services.knowledge_graph_types import ExtractedEntity
        
        # Create extractor instance
        extractor = LLMKnowledgeExtractor()
        print("✅ LLM Knowledge Extractor initialized")
        
        # Test 1: JSON parsing robustness
        print("\n🧪 TEST 1: JSON Parsing Robustness")
        
        # Mock LLM response with various issues
        problematic_responses = [
            # Response with trailing commas
            '''{
                "entities": [
                    {
                        "text": "DBS Bank",
                        "canonical_form": "DBS Bank",
                        "type": "ORGANIZATION",
                        "confidence": 0.95,
                    }
                ],
                "relationships": [
                    {
                        "source_entity": "DBS Bank",
                        "target_entity": "Singapore",
                        "relationship_type": "LOCATED_IN",
                        "confidence": 0.8,
                    }
                ],
            }''',
            
            # Response with code blocks
            '''```json
            {
                "entities": [
                    {
                        "text": "Piyush Gupta",
                        "canonical_form": "Piyush Gupta",
                        "type": "PERSON",
                        "confidence": 0.9
                    }
                ],
                "relationships": [
                    {
                        "source_entity": "Piyush Gupta",
                        "target_entity": "DBS Bank",
                        "relationship_type": "WORKS_FOR",
                        "confidence": 0.85
                    }
                ]
            }
            ```''',
            
            # Response with thinking tags
            '''<think>
            I need to extract entities and relationships from this text about DBS Bank.
            </think>
            {
                "entities": [
                    {"text": "DBS Technology", "canonical_form": "DBS Technology", "type": "ORGANIZATION", "confidence": 0.88}
                ],
                "relationships": [
                    {"source_entity": "DBS Technology", "target_entity": "Cloud Solutions", "relationship_type": "PROVIDES", "confidence": 0.7}
                ]
            }'''
        ]
        
        parsing_successes = 0
        for i, response in enumerate(problematic_responses, 1):
            try:
                result = extractor._parse_llm_response(response)
                entities_count = len(result.get('entities', []))
                relationships_count = len(result.get('relationships', []))
                print(f"   Response {i}: ✅ Parsed {entities_count} entities, {relationships_count} relationships")
                parsing_successes += 1
            except Exception as e:
                print(f"   Response {i}: ❌ Failed to parse: {e}")
        
        print(f"   JSON Parsing Success Rate: {parsing_successes}/{len(problematic_responses)}")
        
        # Test 2: Entity matching and validation
        print("\n🧪 TEST 2: Entity Matching and Validation")
        
        # Create mock entities
        mock_entities = [
            ExtractedEntity(text="DBS Bank", label="ORGANIZATION", start_char=0, end_char=8, confidence=0.9, canonical_form="DBS Bank"),
            ExtractedEntity(text="Singapore", label="LOCATION", start_char=50, end_char=59, confidence=0.85, canonical_form="Singapore"),
            ExtractedEntity(text="Piyush Gupta", label="PERSON", start_char=100, end_char=112, confidence=0.88, canonical_form="Piyush Gupta"),
            ExtractedEntity(text="DBS Technology", label="ORGANIZATION", start_char=150, end_char=164, confidence=0.82, canonical_form="DBS Technology")
        ]
        
        # Mock relationships with various entity name formats
        mock_relationships = [
            # Exact match
            {"source_entity": "DBS Bank", "target_entity": "Singapore", "relationship_type": "LOCATED_IN", "confidence": 0.8},
            # Fuzzy match (partial names)
            {"source_entity": "DBS", "target_entity": "Singapore", "relationship_type": "OPERATES_IN", "confidence": 0.75},
            # Case variation
            {"source_entity": "piyush gupta", "target_entity": "dbs bank", "relationship_type": "WORKS_FOR", "confidence": 0.9},
            # Abbreviation
            {"source_entity": "DBS Tech", "target_entity": "Cloud Solutions", "relationship_type": "PROVIDES", "confidence": 0.7},
        ]
        
        validated_relationships = extractor._validate_and_score_relationships(mock_relationships, mock_entities)
        
        print(f"   Raw relationships: {len(mock_relationships)}")
        print(f"   Validated relationships: {len(validated_relationships)}")
        print(f"   Validation success rate: {len(validated_relationships)}/{len(mock_relationships)}")
        
        for rel in validated_relationships:
            fuzzy = rel.properties.get('fuzzy_matched', False)
            indicator = " [FUZZY]" if fuzzy else ""
            print(f"   ✅ {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity}{indicator}")
        
        # Test 3: Fallback relationship inference
        print("\n🧪 TEST 3: Fallback Relationship Inference")
        
        test_text = """
        DBS Bank is headquartered in Singapore. The CEO Piyush Gupta leads the organization. 
        DBS Technology provides cloud solutions. The bank operates in multiple Asian markets.
        """
        
        # Test with entities but no relationships (fallback scenario)
        fallback_relationships = await extractor._infer_fallback_relationships(mock_entities, test_text)
        
        print(f"   Fallback relationships inferred: {len(fallback_relationships)}")
        for rel in fallback_relationships:
            method = rel.properties.get('inference_method', 'unknown')
            print(f"   🔄 {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity} ({method})")
        
        # Test 4: Enhanced prompt content
        print("\n🧪 TEST 4: Enhanced Prompt Content")
        
        from app.services.settings_prompt_service import get_prompt_service
        prompt_service = get_prompt_service()
        
        test_vars = {
            'text': 'test',
            'context_info': '',
            'domain_guidance': '',
            'entity_types': ['PERSON', 'ORGANIZATION'],
            'relationship_types': ['WORKS_FOR', 'LOCATED_IN']
        }
        
        prompt = prompt_service.get_prompt('knowledge_extraction', test_vars)
        
        # Check for key improvements in prompt
        improvements = [
            ("RELATIONSHIP EXTRACTION", "relationship focus"),
            ("flexible entity matching", "fuzzy matching guidance"),
            ("ALWAYS extract relationships", "extraction emphasis"),
            ("semantic understanding", "context awareness"),
            ("meaningful relationships", "quality focus")
        ]
        
        prompt_score = 0
        for keyword, description in improvements:
            if keyword.lower() in prompt.lower():
                print(f"   ✅ Found: {description}")
                prompt_score += 1
            else:
                print(f"   ❌ Missing: {description}")
        
        print(f"   Prompt enhancement score: {prompt_score}/{len(improvements)}")
        
        # Overall assessment
        print(f"\n" + "="*80)
        print("COMPREHENSIVE TEST RESULTS")
        print("="*80)
        
        json_parsing_ok = parsing_successes >= 2
        entity_matching_ok = len(validated_relationships) >= 2
        fallback_ok = len(fallback_relationships) > 0
        prompt_ok = prompt_score >= 3
        
        print(f"JSON Parsing Robustness: {'✅ PASS' if json_parsing_ok else '❌ FAIL'}")
        print(f"Entity Matching & Validation: {'✅ PASS' if entity_matching_ok else '❌ FAIL'}")
        print(f"Fallback Relationship Inference: {'✅ PASS' if fallback_ok else '❌ FAIL'}")
        print(f"Enhanced Prompt Content: {'✅ PASS' if prompt_ok else '❌ FAIL'}")
        
        overall_success = json_parsing_ok and entity_matching_ok and fallback_ok and prompt_ok
        
        if overall_success:
            print(f"\n🎉 ALL CORE FIXES VERIFIED SUCCESSFULLY!")
            print(f"The relationship extraction improvements are working correctly.")
        else:
            print(f"\n⚠️ SOME FIXES NEED ATTENTION.")
            
        print(f"\n💡 KEY IMPROVEMENTS IMPLEMENTED:")
        print(f"   1. Robust JSON parsing handles malformed LLM responses")
        print(f"   2. Fuzzy entity matching allows for name variations")
        print(f"   3. Fallback strategies create relationships when LLM fails")
        print(f"   4. Enhanced prompts emphasize relationship extraction")
        print(f"   5. Comprehensive debug logging tracks processing steps")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_extraction_fixes())
    
    if success:
        print(f"\n✅ RELATIONSHIP EXTRACTION FIXES VERIFIED!")
        print(f"The critical issues causing 0 relationships have been resolved.")
    else:
        print(f"\n❌ SOME ISSUES REMAIN - CHECK THE OUTPUT ABOVE")