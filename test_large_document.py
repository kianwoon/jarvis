#!/usr/bin/env python3
"""Test large document processing"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_document_sizes():
    """Test extraction with different document sizes"""
    
    print("🧪 Testing Document Size Impact")
    print("=" * 60)
    
    try:
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        
        extractor = LLMKnowledgeExtractor()
        
        # Test different document sizes
        base_text = """DBS Bank is a leading financial services group headquartered in Singapore. 
        The bank operates across 18 markets in Asia and is recognized for its digital banking capabilities.
        DBS has been evaluating OceanBase database developed by Alibaba for its digital transformation."""
        
        sizes_to_test = [
            ("Small", base_text),  # ~300 chars
            ("Medium", base_text * 10),  # ~3K chars  
            ("Large", base_text * 50),  # ~15K chars
            ("Very Large", base_text * 100),  # ~30K chars (similar to your PDF)
        ]
        
        for size_name, test_text in sizes_to_test:
            print(f"\n🔬 Testing {size_name} Document:")
            print(f"   Size: {len(test_text):,} characters")
            
            try:
                result = await extractor.extract_knowledge(test_text)
                
                print(f"   ✅ Success: {len(result.entities)} entities, {len(result.relationships)} relationships")
                print(f"   Processing time: {result.processing_time_ms:.1f}ms")
                print(f"   Reasoning: {result.reasoning}")
                
                if len(result.entities) == 0:
                    print(f"   ❌ FAILURE: No entities extracted from {size_name.lower()} document!")
                    break
                    
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                break
        
        # Test with realistic banking content at different sizes
        banking_text = """DBS Bank operates in Singapore, Hong Kong, and Indonesia. 
        The bank evaluates OceanBase distributed database from Alibaba, SOFAStack middleware from Ant Group, 
        and TDSQL distributed database from Tencent. The Chief Technology Officer leads the digital transformation 
        initiative focusing on regulatory compliance across multiple jurisdictions."""
        
        print(f"\n🏦 Testing Banking Content Scaling:")
        for multiplier in [1, 20, 50, 100]:
            scaled_text = banking_text * multiplier
            print(f"\n   Scale x{multiplier}: {len(scaled_text):,} chars")
            
            try:
                result = await extractor.extract_knowledge(scaled_text)
                print(f"   Result: {len(result.entities)} entities, {len(result.relationships)} relationships")
                
                if len(result.entities) == 0:
                    print(f"   ❌ BREAKING POINT: Scale x{multiplier} fails!")
                    break
                    
            except Exception as e:
                print(f"   ❌ ERROR at scale x{multiplier}: {e}")
                break
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_document_sizes())