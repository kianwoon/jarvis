#!/usr/bin/env python3
"""Test the relationship storage fix"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
from app.document_handlers.base import ExtractedChunk
from app.services.knowledge_graph_service import get_knowledge_graph_service

async def test_relationship_storage_fix():
    """Test that relationships are properly stored using entity ID mapping"""
    
    # Create a test chunk with clear entities and relationships
    test_chunk = ExtractedChunk(
        content="""
        Microsoft Corporation is a technology company. Bill Gates founded Microsoft. 
        The company is headquartered in Seattle, Washington. Microsoft develops Azure cloud platform.
        Bill Gates works at Microsoft as the founder.
        """,
        metadata={
            'chunk_id': 'test_relationship_fix',
            'document_type': 'test',
            'source': 'relationship_test',
        },
        quality_score=0.9
    )
    
    print("🧪 Testing relationship storage fix")
    print("=" * 60)
    
    try:
        # Get knowledge graph service
        kg_service = get_knowledge_graph_service()
        
        print("🚀 Extracting knowledge from test chunk...")
        
        # Extract knowledge graph data
        result = await kg_service.extract_from_chunk(test_chunk, document_id="test_doc_123")
        
        print(f"📊 Extraction Results:")
        print(f"   • Entities: {len(result.entities)}")
        print(f"   • Relationships: {len(result.relationships)}")
        
        if result.entities:
            print(f"\n📍 Extracted Entities:")
            for i, entity in enumerate(result.entities, 1):
                print(f"   {i}. '{entity.text}' (canonical: '{entity.canonical_form}', type: {entity.label})")
        
        if result.relationships:
            print(f"\n🔗 Extracted Relationships:")
            for i, rel in enumerate(result.relationships, 1):
                print(f"   {i}. '{rel.source_entity}' --[{rel.relationship_type}]--> '{rel.target_entity}'")
        
        print(f"\n🏪 Storing in Neo4j...")
        
        # Store in Neo4j using the fixed method
        storage_result = await kg_service.store_in_neo4j(result, document_id="test_doc_123")
        
        print(f"\n📈 Storage Results:")
        print(f"   • Success: {storage_result.get('success')}")
        print(f"   • Entities stored: {storage_result.get('entities_stored', 0)}")
        print(f"   • Relationships stored: {storage_result.get('relationships_stored', 0)}")
        print(f"   • Relationship failures: {storage_result.get('relationship_failures', 0)}")
        
        # Calculate success rates
        if len(result.relationships) > 0:
            relationship_success_rate = (storage_result.get('relationships_stored', 0) / len(result.relationships)) * 100
            print(f"   • Relationship success rate: {relationship_success_rate:.1f}%")
        
        # Test success criteria
        entities_success = storage_result.get('entities_stored', 0) > 0
        relationships_success = storage_result.get('relationships_stored', 0) > 0
        low_failures = storage_result.get('relationship_failures', 0) < len(result.relationships) / 2
        
        overall_success = entities_success and relationships_success and low_failures
        
        print(f"\n{'✅' if overall_success else '❌'} Overall Test: {'PASSED' if overall_success else 'FAILED'}")
        
        if overall_success:
            print("🎉 Relationship storage fix is working!")
            print("   Entities and relationships are being stored successfully.")
        else:
            print("🔧 Relationship storage still needs work:")
            if not entities_success:
                print("   - Entity storage failed")
            if not relationships_success:
                print("   - No relationships were stored")
            if not low_failures:
                print("   - Too many relationship failures")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🔬 Relationship Storage Fix Test")
    print("Testing entity ID mapping and relationship storage")
    print("=" * 60)
    
    success = await test_relationship_storage_fix()
    
    if success:
        print("\n🎊 All tests passed! The relationship storage fix is working.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    asyncio.run(main())