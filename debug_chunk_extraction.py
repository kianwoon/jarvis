#!/usr/bin/env python3
"""Debug why DBS document chunks are returning 0 entities"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

async def test_with_business_text():
    """Test extraction with typical business document text"""
    
    extractor = LLMKnowledgeExtractor()
    
    # Sample business text that should contain entities
    business_text = """
    DBS Bank is implementing a comprehensive digital transformation strategy across Singapore and Southeast Asia. 
    The bank is leveraging cloud technologies including Amazon Web Services and Microsoft Azure to modernize 
    their core banking systems. Key initiatives include the adoption of artificial intelligence for customer 
    service and blockchain technology for trade finance. The Chief Technology Officer, John Smith, is leading 
    the digital innovation team based in Singapore headquarters.
    """
    
    print("🧠 Testing LLM Knowledge Extraction with Business Text")
    print(f"📝 Text: {business_text[:200]}...")
    print("=" * 50)
    
    try:
        # Test full extraction
        result = await extractor.extract_knowledge(business_text)
        print(f"🎯 Extraction Result:")
        print(f"   Entities: {len(result.entities)}")
        print(f"   Relationships: {len(result.relationships)}")
        print(f"   Confidence: {result.confidence_score}")
        print(f"   Reasoning: {result.reasoning}")
        
        if result.entities:
            print("\n📋 Entities Found:")
            for i, entity in enumerate(result.entities[:5]):  # Show first 5
                print(f"   {i+1}. {entity.text} ({entity.label}) - {entity.confidence}")
        
        if result.relationships:
            print("\n🔗 Relationships Found:")
            for i, rel in enumerate(result.relationships[:3]):  # Show first 3
                print(f"   {i+1}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity}")
        
        if len(result.entities) == 0:
            print("❌ NO ENTITIES EXTRACTED - This indicates a problem with extraction")
            
            # Test the LLM call directly
            print("\n🔍 Testing direct LLM call...")
            prompt = extractor._build_extraction_prompt(business_text)
            print(f"📋 Prompt preview: {prompt[:300]}...")
            
            raw_response = await extractor._call_llm_for_extraction(prompt)
            print(f"🤖 Raw LLM Response: {raw_response[:500]}...")
            
            parsed = extractor._parse_llm_response(raw_response)
            print(f"✅ Parsed Response: entities={len(parsed.get('entities', []))}, relationships={len(parsed.get('relationships', []))}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_with_business_text())