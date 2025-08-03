#!/usr/bin/env python3
"""Debug raw LLM responses for large documents"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def debug_large_document_llm_response():
    """Debug what the LLM actually returns for large document extraction"""
    
    print("🔍 Debugging Large Document LLM Response")
    print("=" * 70)
    
    try:
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        
        # Create a large document similar to your PDF (31,290 chars)
        base_banking_content = """
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
        
        Key strategic priorities include:
        - Digital transformation across all banking channels
        - Enhanced customer experience through AI and machine learning
        - Regulatory compliance in multiple Asian markets
        - Scalable infrastructure for future growth
        - Partnership with leading technology providers
        
        DBS is particularly interested in cloud-native solutions that can be deployed across 
        multiple regions while maintaining high availability and disaster recovery capabilities.
        The bank has been working with various technology vendors to evaluate their offerings
        and determine the best fit for their specific requirements.
        """
        
        # Scale up to approximately 31K characters (similar to your PDF)
        large_content = (base_banking_content * 25).strip()  # ~31K chars
        
        print(f"📄 Test Document:")
        print(f"   Length: {len(large_content):,} characters")
        print(f"   Content preview: {large_content[:200]}...")
        print(f"   Content suffix: ...{large_content[-200:]}")
        
        # Create extractor and test
        extractor = LLMKnowledgeExtractor()
        
        print(f"\n🔧 Extraction Configuration:")
        print(f"   Model: {extractor.model_config['model']}")
        print(f"   Temperature: {extractor.model_config['temperature']}")
        print(f"   Max Tokens: {extractor.model_config['max_tokens']}")
        
        # Get the prompt that will be sent to LLM
        prompt = extractor._build_extraction_prompt(large_content)
        print(f"\n📝 Prompt Details:")
        print(f"   Total prompt length: {len(prompt):,} characters")
        print(f"   Estimated tokens: {len(prompt)//4:,} tokens")
        print(f"   Prompt preview: {prompt[:300]}...")
        print(f"   Prompt suffix: ...{prompt[-200:]}")
        
        print(f"\n🚀 Calling LLM directly...")
        
        # Call LLM and capture raw response
        raw_response = await extractor._call_llm_for_extraction(prompt)
        
        print(f"\n📊 Raw LLM Response Analysis:")
        print(f"   Response length: {len(raw_response):,} characters")
        print(f"   Contains <think>: {'✅' if '<think>' in raw_response else '❌'}")
        print(f"   Contains </think>: {'✅' if '</think>' in raw_response else '❌'}")
        print(f"   Contains JSON braces: {'✅' if '{' in raw_response and '}' in raw_response else '❌'}")
        
        # Show full response with clear markers
        print(f"\n" + "="*70)
        print(f"FULL RAW LLM RESPONSE:")
        print(f"="*70)
        print(raw_response)
        print(f"="*70)
        
        # Analyze thinking vs output sections
        if '<think>' in raw_response and '</think>' in raw_response:
            parts = raw_response.split('</think>')
            thinking_part = raw_response.split('</think>')[0] + '</think>'
            output_part = parts[1] if len(parts) > 1 else ""
            
            print(f"\n🧠 Thinking Section Analysis:")
            print(f"   Thinking length: {len(thinking_part):,} characters")
            print(f"   Output length: {len(output_part):,} characters")
            print(f"   Output preview: '{output_part[:500]}...'")
            
            if not output_part.strip():
                print(f"   ❌ ISSUE: No content after </think> tag!")
            elif '{' not in output_part:
                print(f"   ❌ ISSUE: No JSON braces in output section!")
            else:
                print(f"   ✅ Output section contains JSON-like content")
        
        # Test the parsing step by step
        print(f"\n🔬 Testing Parsing Logic:")
        try:
            parsed_result = extractor._parse_llm_response(raw_response)
            
            entities = parsed_result.get('entities', [])
            relationships = parsed_result.get('relationships', [])
            reasoning = parsed_result.get('reasoning', '')
            
            print(f"   Parsing result: {len(entities)} entities, {len(relationships)} relationships")
            print(f"   Reasoning: {reasoning}")
            
            if len(entities) == 0:
                print(f"   ❌ PARSING FAILURE: No entities extracted")
            else:
                print(f"   ✅ Parsing success: {entities[:3]}")
                
            if len(relationships) == 0:
                print(f"   ❌ PARSING FAILURE: No relationships extracted")
            else:
                print(f"   ✅ Parsing success: {relationships[:2]}")
                
        except Exception as parse_e:
            print(f"   ❌ PARSING ERROR: {parse_e}")
        
        # Compare with a small working example
        print(f"\n🔄 Comparing with Small Document:")
        small_content = "DBS Bank is located in Singapore. The bank uses OceanBase database developed by Alibaba."
        
        try:
            small_result = await extractor.extract_knowledge(small_content)
            print(f"   Small doc result: {len(small_result.entities)} entities, {len(small_result.relationships)} relationships")
            print(f"   Small doc reasoning: {small_result.reasoning}")
            
            if len(small_result.entities) > 0:
                print(f"   ✅ Small documents work fine")
                print(f"   🔍 Issue is specific to large documents")
            else:
                print(f"   ❌ Even small documents are failing - systematic issue")
                
        except Exception as small_e:
            print(f"   ❌ Small document test failed: {small_e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main debug function"""
    print("🔬 Large Document LLM Response Debug")
    print("Investigating why 31K character documents return 0 entities/relationships")
    print("=" * 70)
    
    success = await debug_large_document_llm_response()
    
    if success:
        print(f"\n✅ Debug completed successfully!")
        print(f"📋 Check the raw response analysis above to identify the issue")
    else:
        print(f"\n❌ Debug failed")

if __name__ == "__main__":
    asyncio.run(main())