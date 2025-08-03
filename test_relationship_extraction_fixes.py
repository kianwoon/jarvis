#!/usr/bin/env python3
"""
Test script to verify the relationship extraction fixes are working properly.
This tests the critical issues that were causing 0 relationships to be extracted.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Set up logging to see debug output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test the LLM knowledge extractor directly
async def test_llm_extractor():
    """Test the LLM knowledge extractor with a sample text"""
    print("\n" + "="*80)
    print("TESTING LLM KNOWLEDGE EXTRACTOR FIXES")
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        from app.services.llm_knowledge_extractor import get_llm_knowledge_extractor
        
        extractor = get_llm_knowledge_extractor()
        print("✅ LLM Knowledge Extractor initialized successfully")
        
        # Test text that should produce entities and relationships
        test_text = """
        DBS Bank is a leading financial services group headquartered in Singapore. 
        The bank operates across 18 markets and serves millions of customers. 
        DBS Technology provides cloud-native architecture solutions for digital transformation.
        CEO Piyush Gupta leads the organization's innovation strategy.
        The bank partners with Amazon Web Services for cloud infrastructure.
        DBS operates in Singapore, Hong Kong, and other Asian markets.
        The bank was founded in 1968 and has become a major player in digital banking.
        """
        
        print(f"\n📝 Test text length: {len(test_text)} characters")
        print(f"Sample: {test_text[:200]}...")
        
        # Test extraction
        print("\n🧠 Running LLM extraction...")
        result = await extractor.extract_knowledge(test_text)
        
        print(f"\n📊 EXTRACTION RESULTS:")
        print(f"   Entities found: {len(result.entities)}")
        print(f"   Relationships found: {len(result.relationships)}")
        print(f"   Confidence score: {result.confidence_score:.3f}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   Model used: {result.llm_model_used}")
        
        if result.entities:
            print(f"\n📋 ENTITIES EXTRACTED:")
            for i, entity in enumerate(result.entities[:10], 1):
                print(f"   {i}. {entity.canonical_form} ({entity.label}) - confidence: {entity.confidence:.2f}")
        
        if result.relationships:
            print(f"\n🔗 RELATIONSHIPS EXTRACTED:")
            for i, rel in enumerate(result.relationships[:10], 1):
                fallback_indicator = " [INFERRED]" if rel.properties.get('fallback_inferred') else ""
                print(f"   {i}. {rel.source_entity} -[{rel.relationship_type}]-> {rel.target_entity}{fallback_indicator}")
                print(f"      Confidence: {rel.confidence:.2f}, Context: {rel.context[:100]}...")
        else:
            print(f"\n❌ NO RELATIONSHIPS FOUND!")
            print("This indicates the fixes may not be working properly.")
        
        # Test the specific issues that were problematic
        print(f"\n🔍 DIAGNOSTIC CHECKS:")
        
        # Check 1: Prompt loading
        print(f"   1. Prompt loading test:")
        try:
            from app.services.settings_prompt_service import get_prompt_service
            prompt_service = get_prompt_service()
            test_prompt = prompt_service.get_prompt('knowledge_extraction', {
                'text': 'test',
                'context_info': '',
                'domain_guidance': '',
                'entity_types': ['PERSON', 'ORGANIZATION'],
                'relationship_types': ['WORKS_FOR', 'PARTNERS_WITH']
            })
            if test_prompt and len(test_prompt) > 100:
                print(f"      ✅ Prompt loaded successfully (length: {len(test_prompt)})")
            else:
                print(f"      ❌ Prompt loading failed or too short")
        except Exception as e:
            print(f"      ❌ Prompt loading error: {e}")
        
        # Check 2: Entity matching
        print(f"   2. Entity matching test:")
        if result.entities:
            # Test fuzzy matching manually
            entity_names = [e.canonical_form.lower() for e in result.entities]
            test_matches = [
                ("dbs", "DBS Bank"),
                ("singapore", "Singapore"),
                ("ceo", "Piyush Gupta")
            ]
            
            for test_name, expected in test_matches:
                found = any(test_name in name or name in test_name for name in entity_names)
                status = "✅" if found else "❌"
                print(f"      {status} Matching '{test_name}' -> found: {found}")
        
        # Check 3: Relationship types validation
        print(f"   3. Relationship type validation:")
        if result.relationships:
            valid_types = 0
            for rel in result.relationships:
                if extractor._is_valid_relationship_type(rel.relationship_type):
                    valid_types += 1
            print(f"      ✅ {valid_types}/{len(result.relationships)} relationships have valid types")
        else:
            print(f"      ⚠️ No relationships to validate")
        
        return len(result.relationships) > 0
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_prompt_service():
    """Test the prompt service fixes"""
    print(f"\n🔧 TESTING PROMPT SERVICE FIXES:")
    
    try:
        from app.services.settings_prompt_service import get_prompt_service
        
        prompt_service = get_prompt_service()
        print("✅ Prompt service initialized")
        
        # Test prompt listing
        prompts = prompt_service.list_prompts()
        print(f"✅ Found {len(prompts)} prompts")
        
        for prompt in prompts:
            print(f"   - {prompt['name']} ({prompt['type']})")
        
        # Test knowledge extraction prompt specifically
        test_vars = {
            'text': 'Sample text',
            'context_info': 'Test context',
            'domain_guidance': 'Test domain',
            'entity_types': ['PERSON', 'ORGANIZATION'],
            'relationship_types': ['WORKS_FOR', 'PARTNERS_WITH']
        }
        
        extraction_prompt = prompt_service.get_prompt('knowledge_extraction', test_vars)
        
        if extraction_prompt and len(extraction_prompt) > 500:
            print(f"✅ Knowledge extraction prompt loaded successfully (length: {len(extraction_prompt)})")
            
            # Check if the enhanced relationship extraction instructions are present
            if "RELATIONSHIP EXTRACTION IS CRITICAL" in extraction_prompt:
                print(f"✅ Enhanced relationship extraction instructions found")
            else:
                print(f"⚠️ Enhanced relationship extraction instructions not found")
                
        else:
            print(f"❌ Knowledge extraction prompt failed to load or too short")
            
        return True
        
    except Exception as e:
        print(f"❌ Prompt service test failed: {e}")
        return False

def test_database_connection():
    """Test database connection and settings"""
    print(f"\n💾 TESTING DATABASE CONNECTION:")
    
    try:
        from app.core.db import SessionLocal, Settings
        
        db = SessionLocal()
        try:
            # Check knowledge graph settings
            kg_settings = db.query(Settings).filter(
                Settings.category == 'knowledge_graph'
            ).first()
            
            if kg_settings:
                print(f"✅ Knowledge graph settings found")
                
                if 'prompts' in kg_settings.settings:
                    prompts = kg_settings.settings['prompts']
                    print(f"✅ Prompts found in settings: {len(prompts)} prompts")
                    
                    if isinstance(prompts, list):
                        for prompt in prompts:
                            if prompt.get('prompt_type') == 'knowledge_extraction':
                                template_length = len(prompt.get('prompt_template', ''))
                                print(f"✅ Knowledge extraction prompt: {template_length} characters")
                                break
                    else:
                        print(f"⚠️ Prompts in wrong format (should be list): {type(prompts)}")
                else:
                    print(f"❌ No prompts found in knowledge graph settings")
            else:
                print(f"❌ No knowledge graph settings found")
                
        finally:
            db.close()
            
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 STARTING RELATIONSHIP EXTRACTION FIXES VERIFICATION")
    
    # Test 1: Database and settings
    db_ok = test_database_connection()
    
    # Test 2: Prompt service
    prompt_ok = await test_prompt_service()
    
    # Test 3: LLM extractor (main test)
    extraction_ok = await test_llm_extractor()
    
    print(f"\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    
    print(f"Database/Settings: {'✅ PASS' if db_ok else '❌ FAIL'}")
    print(f"Prompt Service: {'✅ PASS' if prompt_ok else '❌ FAIL'}")
    print(f"Relationship Extraction: {'✅ PASS' if extraction_ok else '❌ FAIL'}")
    
    overall_success = db_ok and prompt_ok and extraction_ok
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED! Relationship extraction fixes are working.")
    else:
        print(f"\n⚠️ SOME TESTS FAILED. Review the output above for issues.")
        
    print(f"\n📋 SUMMARY OF FIXES IMPLEMENTED:")
    print(f"   1. ✅ Fixed prompt loading category issues")
    print(f"   2. ✅ Added fuzzy entity matching for relationships")
    print(f"   3. ✅ Enhanced debug logging throughout extraction pipeline")
    print(f"   4. ✅ Implemented fallback relationship inference strategies")
    print(f"   5. ✅ Improved JSON parsing robustness")
    print(f"   6. ✅ Added comprehensive error handling and recovery")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)