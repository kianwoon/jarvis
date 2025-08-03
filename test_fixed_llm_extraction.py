#!/usr/bin/env python3
"""
Test LLM Knowledge Extraction After Fixing Async Issues
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_llm_extraction():
    """Test the fixed LLM knowledge extractor"""
    print("🔧 Testing Fixed LLM Knowledge Extraction")
    print("=" * 60)
    
    try:
        # Import after path setup
        from app.services.llm_knowledge_extractor import get_llm_knowledge_extractor
        
        # Get the extractor
        extractor = get_llm_knowledge_extractor()
        print(f"✅ Successfully created LLM extractor")
        print(f"   Model: {extractor.model_config.get('model', 'unknown')}")
        print(f"   Entity types available: {len(extractor.hierarchical_entity_types)}")
        print(f"   Relationship categories: {len(extractor.relationship_taxonomy)}")
        
        # Test with sample text
        test_text = """
        DBS Bank is a leading financial services company in Singapore. 
        The bank uses advanced technology including blockchain and AI to serve customers.
        John Smith is the Chief Technology Officer who oversees digital transformation.
        DBS operates in multiple locations including Hong Kong and India.
        """
        
        print(f"\n🧪 Testing extraction with sample text:")
        print(f"Text length: {len(test_text)} characters")
        
        # Extract knowledge
        result = await extractor.extract_knowledge(
            text=test_text,
            context={
                'document_type': 'test',
                'source': 'test_script',
                'domain': 'financial_services'
            }
        )
        
        print(f"\n📊 Extraction Results:")
        print(f"   Entities found: {len(result.entities)}")
        print(f"   Relationships found: {len(result.relationships)}")
        print(f"   Confidence score: {result.confidence_score:.3f}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   Model used: {result.llm_model_used}")
        
        if result.entities:
            print(f"\n🎯 Entities Extracted:")
            for i, entity in enumerate(result.entities[:10], 1):
                print(f"   {i}. {entity.text} ({entity.label}) - confidence: {entity.confidence:.3f}")
        else:
            print(f"\n⚠️  No entities extracted - this indicates an issue")
            
        if result.relationships:
            print(f"\n🔗 Relationships Extracted:")
            for i, rel in enumerate(result.relationships[:5], 1):
                print(f"   {i}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
                print(f"      Confidence: {rel.confidence:.3f}")
        else:
            print(f"\n⚠️  No relationships extracted")
            
        if result.reasoning:
            print(f"\n💭 Reasoning: {result.reasoning}")
            
        if 'error' in result.extraction_metadata:
            print(f"\n❌ Error occurred: {result.extraction_metadata['error']}")
            return False
            
        # Success if we got at least some entities
        if len(result.entities) > 0:
            print(f"\n✅ SUCCESS: LLM extraction is working and producing entities!")
            return True
        else:
            print(f"\n❌ ISSUE: LLM extraction ran but produced 0 entities")
            print(f"   This suggests the LLM response parsing or entity processing needs investigation")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_extraction())
    if success:
        print(f"\n🎉 Test completed successfully - knowledge extraction is working!")
    else:
        print(f"\n💥 Test failed - knowledge extraction needs further investigation")
        sys.exit(1)