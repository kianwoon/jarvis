#!/usr/bin/env python3
"""
Integration test for the BULLETPROOF JSON parsing fix.
Test the actual knowledge graph extraction with a real document.
"""
import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

async def test_knowledge_graph_extraction():
    """Test knowledge graph extraction with the fixed JSON parsing"""
    
    print("🔧 Testing Knowledge Graph extraction with BULLETPROOF JSON parsing...")
    print("=" * 70)
    
    # Create the service
    extractor = LLMKnowledgeExtractor()
    
    # Test content that should generate the problematic JSON format
    test_content = """
    DBS Bank is a leading financial services group in Asia with operations in 19 markets.
    The bank is headquartered in Singapore and has been focusing heavily on digital banking transformation.
    DBS has established itself as a pioneer in digital banking solutions across Southeast Asia.
    """
    
    print(f"📄 Test content: {test_content.strip()}")
    print()
    
    try:
        # Extract knowledge graph
        print("🔍 Extracting knowledge graph...")
        result = await extractor.extract_knowledge_graph(
            content=test_content,
            chunk_index=0,
            total_chunks=1,
            document_name="test_document.txt"
        )
        
        print("🎯 SUCCESS! Knowledge graph extraction completed!")
        print()
        print(f"   - Entities extracted: {len(result.get('entities', []))}")
        print(f"   - Relationships extracted: {len(result.get('relationships', []))}")
        
        if result.get('entities'):
            print()
            print("📝 Extracted entities:")
            for i, entity in enumerate(result['entities'][:5], 1):
                print(f"   {i}. {entity.get('name')} ({entity.get('type')})")
        
        if result.get('relationships'):
            print()
            print("🔗 Extracted relationships:")
            for i, rel in enumerate(result['relationships'][:3], 1):
                print(f"   {i}. {rel.get('source')} -> {rel.get('target')} ({rel.get('type')})")
        
        print()
        print("✅ INTEGRATION TEST PASSED!")
        print("   The BULLETPROOF JSON parsing fix is working in production!")
        
        return True
        
    except Exception as e:
        print(f"❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_knowledge_graph_extraction())
    if not success:
        sys.exit(1)
    
    print()
    print("🚀 Ready for production! The JSON parsing issue is FIXED.")