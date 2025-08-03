#!/usr/bin/env python3
"""Test enhanced prompt for direct JSON extraction"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

# Reduce logging noise
logging.getLogger().setLevel(logging.WARNING)

async def test_enhanced_prompt():
    """Test if enhanced prompt produces JSON responses directly"""
    
    print("🧪 Testing Enhanced Prompt for Direct JSON Extraction")
    print("=" * 70)
    
    # Create extractor instance
    extractor = LLMKnowledgeExtractor()
    
    # Test text similar to what the user was processing
    test_text = """DBS Bank is evaluating OceanBase and SOFAStack for its digital banking transformation. 
    OceanBase is a distributed database developed by Alibaba that provides high availability and horizontal scaling. 
    SOFAStack is Ant Group's middleware platform offering microservices architecture and cloud-native solutions.
    TDSQL is Tencent's distributed database solution designed for large-scale applications.
    The bank seeks technologies that support massive scalability and innovation."""
    
    print(f"📄 Test Text:")
    print(f"   Length: {len(test_text)} characters")
    print(f"   Content: {test_text[:100]}...")
    
    try:
        print(f"\n🔧 Model Configuration:")
        print(f"   Model: {extractor.model_config['model']}")
        print(f"   Temperature: {extractor.model_config['temperature']}")
        print(f"   Max Tokens: {extractor.model_config['max_tokens']}")
        
        # Check if system prompt includes the enhanced examples
        prompt = extractor._build_extraction_prompt(test_text)
        if "Example 1 - Simple Business:" in prompt:
            print(f"✅ Enhanced prompt with examples is being used")
        else:
            print(f"❌ Enhanced prompt not detected")
            print(f"   Prompt preview: {prompt[:200]}...")
        
        print(f"\n🚀 Running LLM extraction test...")
        
        # Run the extraction
        result = await extractor.extract_knowledge(test_text)
        
        print(f"\n📊 Extraction Results:")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   Model used: {result.llm_model_used}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Reasoning: {result.reasoning}")
        
        print(f"\n🎯 Entities Found ({len(result.entities)}):")
        for i, entity in enumerate(result.entities[:8], 1):  # Show first 8
            print(f"   {i}. {entity.text} ({entity.label}) - conf: {entity.confidence:.2f}")
        
        print(f"\n🔗 Relationships Found ({len(result.relationships)}):")
        for i, rel in enumerate(result.relationships[:6], 1):  # Show first 6
            print(f"   {i}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity} (conf: {rel.confidence:.2f})")
        
        # Check if this is a natural language fallback
        is_fallback = "natural language" in result.reasoning.lower()
        
        print(f"\n🎯 Analysis:")
        print(f"   JSON parsing success: {'❌ NO (fallback used)' if is_fallback else '✅ YES (direct JSON)'}")
        print(f"   Entity extraction rate: {len(result.entities)} entities")
        print(f"   Relationship extraction rate: {len(result.relationships)} relationships")
        
        if is_fallback:
            print(f"   ⚠️  Still using natural language fallback - enhanced prompt may need refinement")
        else:
            print(f"   🎉 Enhanced prompt working! Direct JSON extraction successful")
        
        # Quality assessment
        if len(result.entities) >= 4 and len(result.relationships) >= 3:
            print(f"   ✅ Good extraction quantity (4+ entities, 3+ relationships)")
        else:
            print(f"   ⚠️  Low extraction quantity - may need prompt tuning")
        
        return not is_fallback
        
    except Exception as e:
        print(f"❌ Extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🔬 Enhanced Prompt Extraction Test")
    print("Testing if new system prompt produces direct JSON responses")
    print("=" * 70)
    
    success = await test_enhanced_prompt()
    
    if success:
        print(f"\n✅ Enhanced prompt test PASSED!")
        print(f"   🎯 LLM producing direct JSON responses")
        print(f"   🚀 Knowledge graph quality should improve")
        print(f"   ⚡ Reduced processing overhead (no fallback parsing)")
    else:
        print(f"\n⚠️  Enhanced prompt test shows issues")
        print(f"   🔧 May need further prompt refinement")
        print(f"   📊 Monitor for continued natural language responses")

if __name__ == "__main__":
    asyncio.run(main())