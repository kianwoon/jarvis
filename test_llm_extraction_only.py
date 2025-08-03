#!/usr/bin/env python3
"""Test LLM knowledge extraction directly without Neo4j dependencies"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
from app.services.llm_knowledge_extractor import get_llm_knowledge_extractor
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

async def test_llm_extraction_direct():
    """Test LLM extraction directly"""
    
    # Test text - simple business document
    test_text = """
    John Smith works at Microsoft as a software engineer. The company is located in Seattle, Washington.
    Microsoft develops Azure cloud platform. John collaborates with Sarah Johnson on the Azure project.
    Sarah is the project manager for Azure development.
    """
    
    print("🧪 Testing LLM extraction directly with smaller model")
    print(f"📝 Text length: {len(test_text)} characters")
    
    # Check current settings
    kg_settings = get_knowledge_graph_settings()
    print(f"🔧 Current model: {kg_settings.get('model', 'unknown')}")
    system_prompt = kg_settings.get('model_config', {}).get('system_prompt', 'Not found')
    if system_prompt != 'Not found':
        print(f"🔧 System prompt (first 100 chars): {system_prompt[:100]}...")
    else:
        print("⚠️  No system_prompt found in settings")
    
    try:
        # Get LLM extractor
        llm_extractor = get_llm_knowledge_extractor()
        
        print("🚀 Starting direct LLM extraction...")
        
        # Extract knowledge with LLM
        result = await llm_extractor.extract_knowledge(test_text)
        
        print(f"✅ LLM extraction completed!")
        print(f"📊 Results:")
        print(f"   • Entities: {len(result.entities)}")
        print(f"   • Relationships: {len(result.relationships)}")
        print(f"   • Processing time: {result.processing_time_ms:.2f}ms")
        print(f"   • Confidence: {result.confidence_score:.3f}")
        print(f"   • Model used: {result.llm_model_used}")
        
        if result.entities:
            print(f"\n📍 Entities found:")
            for i, entity in enumerate(result.entities[:5], 1):  # Show first 5
                print(f"   {i}. {entity.text} ({entity.label}) - confidence: {entity.confidence:.2f}")
        
        if result.relationships:
            print(f"\n🔗 Relationships found:")
            for i, rel in enumerate(result.relationships[:5], 1):  # Show first 5
                print(f"   {i}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity} (confidence: {rel.confidence:.2f})")
        
        if not result.entities and not result.relationships:
            print("⚠️  No entities or relationships extracted")
            print(f"   Reasoning: {result.reasoning}")
            print("   This might indicate JSON parsing issues or model problems")
        
        # Show extraction metadata
        if result.extraction_metadata:
            print(f"\n📋 Extraction metadata:")
            for key, value in result.extraction_metadata.items():
                print(f"   • {key}: {value}")
        
        return result
        
    except Exception as e:
        print(f"❌ LLM extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    print("🔬 Direct LLM Knowledge Extraction Test")
    print("=" * 60)
    
    result = await test_llm_extraction_direct()
    
    if result:
        # Test success criteria
        success = len(result.entities) > 0 and len(result.relationships) > 0
        print(f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}")
        
        if success:
            print("🎉 Small model successfully extracted knowledge graph data!")
            print("   The system_prompt approach is working correctly.")
        else:
            print("🔧 Small model extraction needs further adjustment.")
            print("   Consider checking model response format or prompt configuration.")
            if result:
                print(f"   Last reasoning: {result.reasoning}")
    else:
        print("\n❌ Test FAILED - LLM extraction error occurred")

if __name__ == "__main__":
    asyncio.run(main())