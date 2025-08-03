#!/usr/bin/env python3
"""Test knowledge graph extraction with smaller model using simplified JSON format"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
from app.document_handlers.base import ExtractedChunk
from app.services.knowledge_graph_service import get_knowledge_graph_service
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings

async def test_small_model_extraction():
    """Test extraction with smaller model using simplified format"""
    
    # Test text - simple business document
    test_text = """
    John Smith works at Microsoft as a software engineer. The company is located in Seattle, Washington.
    Microsoft develops Azure cloud platform. John collaborates with Sarah Johnson on the Azure project.
    Sarah is the project manager for Azure development.
    """
    
    # Create test chunk
    test_chunk = ExtractedChunk(
        content=test_text,
        metadata={
            'chunk_id': 'test_small_model_chunk',
            'document_type': 'business_doc',
            'source': 'test',
        },
        quality_score=0.9
    )
    
    print("🧪 Testing small model extraction with simplified JSON format")
    print(f"📝 Text length: {len(test_text)} characters")
    
    # Check current settings
    kg_settings = get_knowledge_graph_settings()
    print(f"🔧 Current model: {kg_settings.get('model', 'unknown')}")
    print(f"🔧 System prompt: {kg_settings.get('model_config', {}).get('system_prompt', 'Not found')[:100]}...")
    
    try:
        # Get knowledge graph service
        kg_service = get_knowledge_graph_service()
        
        print("🚀 Starting extraction...")
        
        # Extract knowledge graph data
        result = await kg_service.extract_from_chunk(test_chunk)
        
        print(f"✅ Extraction completed!")
        print(f"📊 Results:")
        print(f"   • Entities: {len(result.entities)}")
        print(f"   • Relationships: {len(result.relationships)}")
        print(f"   • Processing time: {result.processing_time_ms:.2f}ms")
        
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
            print("   This might indicate JSON parsing issues or model problems")
        
        # Check for any warnings
        if hasattr(result, 'warnings') and result.warnings:
            print(f"\n⚠️  Warnings: {result.warnings}")
        
        return result
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    print("🔬 Small Model Knowledge Graph Extraction Test")
    print("=" * 60)
    
    result = await test_small_model_extraction()
    
    if result:
        # Test success criteria
        success = len(result.entities) > 0 and len(result.relationships) > 0
        print(f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}")
        
        if success:
            print("🎉 Small model successfully extracted knowledge graph data!")
            print("   The simplified JSON format is working correctly.")
        else:
            print("🔧 Small model extraction needs further adjustment.")
            print("   Consider simplifying the prompt further or checking model capabilities.")
    else:
        print("\n❌ Test FAILED - extraction error occurred")

if __name__ == "__main__":
    asyncio.run(main())