#!/usr/bin/env python3
"""Test document processing with fixed prompt"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_fixed_document_processing():
    """Test document processing with the fixed enhanced prompt"""
    
    print("🧪 Testing Fixed Document Processing")
    print("=" * 60)
    
    try:
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        
        # Test with realistic banking/technology content
        test_document = """
        DBS Bank is a leading financial services group headquartered in Singapore. 
        The bank operates across 18 markets in Asia and is recognized for its digital banking capabilities.
        
        DBS has been evaluating several database technologies for its digital transformation:
        
        1. OceanBase: A distributed database developed by Alibaba that provides high availability 
           and horizontal scaling capabilities for mission-critical applications.
        
        2. SOFAStack: Ant Group's comprehensive middleware platform that offers microservices 
           architecture and cloud-native solutions for financial institutions.
        
        3. TDSQL: Tencent's distributed database solution designed specifically for large-scale 
           applications in the financial sector.
        
        The bank's technology team, led by their Chief Technology Officer, is focusing on 
        solutions that can support massive transaction volumes while maintaining regulatory 
        compliance across multiple jurisdictions including Singapore, Hong Kong, and Indonesia.
        """
        
        print(f"📄 Test Document:")
        print(f"   Length: {len(test_document)} characters")
        print(f"   Content preview: {test_document[:150]}...")
        
        # Create extractor and test
        extractor = LLMKnowledgeExtractor()
        
        print(f"\n🔧 Extraction Configuration:")
        print(f"   Model: {extractor.model_config['model']}")
        print(f"   Temperature: {extractor.model_config['temperature']}")
        print(f"   Max Tokens: {extractor.model_config['max_tokens']}")
        
        # Check if enhanced prompt is loaded
        prompt = extractor._build_extraction_prompt("test")
        if "Example 1:" in prompt:
            print(f"   ✅ Enhanced prompt loaded ({len(prompt)} chars)")
        else:
            print(f"   ❌ Enhanced prompt not loaded ({len(prompt)} chars)")
            print(f"   Preview: {prompt[:200]}...")
        
        print(f"\n🚀 Running extraction...")
        
        # Extract knowledge
        result = await extractor.extract_knowledge(test_document)
        
        print(f"\n📊 Extraction Results:")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   Model used: {result.llm_model_used}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Reasoning: {result.reasoning}")
        
        print(f"\n🎯 Entities Extracted ({len(result.entities)}):")
        for i, entity in enumerate(result.entities[:10], 1):
            print(f"   {i}. {entity.text} ({entity.label}) - conf: {entity.confidence:.2f}")
        
        print(f"\n🔗 Relationships Extracted ({len(result.relationships)}):")
        for i, rel in enumerate(result.relationships[:8], 1):
            print(f"   {i}. {rel.source_entity} --[{rel.relationship_type}]--> {rel.target_entity} (conf: {rel.confidence:.2f})")
        
        # Quality assessment
        print(f"\n🎯 Quality Assessment:")
        
        # Check for key entities
        entity_names = [e.text.lower() for e in result.entities]
        expected_entities = ['dbs bank', 'singapore', 'oceanbase', 'alibaba', 'sofastack', 'ant group', 'tdsql', 'tencent']
        found_entities = [e for e in expected_entities if e in entity_names]
        
        print(f"   Key entities found: {len(found_entities)}/{len(expected_entities)} ({', '.join(found_entities)})")
        
        # Check for relationships
        has_relationships = len(result.relationships) > 0
        print(f"   Relationships extracted: {'✅ YES' if has_relationships else '❌ NO'} ({len(result.relationships)} found)")
        
        # Check if using fallback parsing
        is_direct_json = "natural language" not in result.reasoning.lower()
        print(f"   Direct JSON parsing: {'✅ YES' if is_direct_json else '❌ NO (fallback used)'}")
        
        # Overall assessment
        if len(found_entities) >= 6 and has_relationships and is_direct_json:
            print(f"\n✅ EXCELLENT: Enhanced prompt working perfectly!")
            print(f"   🎯 Found most key entities ({len(found_entities)}/8)")
            print(f"   🔗 Extracted relationships ({len(result.relationships)})")
            print(f"   ⚡ Direct JSON parsing (no fallback)")
            return True
        elif len(found_entities) >= 4 and has_relationships:
            print(f"\n🔶 GOOD: Enhanced prompt working but could be better")
            print(f"   ⚠️  Some issues detected, but extracting entities and relationships")
            return True
        else:
            print(f"\n❌ POOR: Enhanced prompt not working effectively")
            print(f"   🔧 May need further refinement")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixed_document_processing())
    
    if success:
        print(f"\n🎉 Enhanced prompt test successful!")
        print(f"📋 Ready for production knowledge graph extraction")
    else:
        print(f"\n⚠️  Enhanced prompt needs further work")
        
    sys.exit(0 if success else 1)