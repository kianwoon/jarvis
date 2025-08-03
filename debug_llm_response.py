#!/usr/bin/env python3
"""Debug what the LLM is actually returning"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def debug_llm_raw_response():
    """Debug the actual LLM response to see why extraction fails"""
    
    print("🔍 Debugging LLM Raw Response")
    print("=" * 60)
    
    try:
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        
        # Test with a simple text that should definitely work
        simple_text = """DBS Bank is located in Singapore. The bank uses OceanBase database developed by Alibaba."""
        
        extractor = LLMKnowledgeExtractor()
        
        print(f"📄 Test Text: {simple_text}")
        print(f"🔧 Model: {extractor.model_config['model']}")
        
        # Get the prompt
        prompt = extractor._build_extraction_prompt(simple_text)
        print(f"📝 Prompt length: {len(prompt)} chars")
        print(f"📝 Prompt preview:")
        print(f"   {prompt[:300]}...")
        print(f"   ...{prompt[-200:]}")
        
        # Call LLM directly to see raw response
        print(f"\n🚀 Calling LLM directly...")
        raw_response = await extractor._call_llm_for_extraction(prompt)
        
        print(f"\n📊 Raw LLM Response:")
        print(f"   Length: {len(raw_response)} characters")
        print(f"   Full response:")
        print(f"   {'='*60}")
        print(raw_response)
        print(f"   {'='*60}")
        
        # Test parsing
        print(f"\n🔬 Testing Response Parsing:")
        parsed_result = extractor._parse_llm_response(raw_response)
        
        entities = parsed_result.get('entities', [])
        relationships = parsed_result.get('relationships', [])
        reasoning = parsed_result.get('reasoning', '')
        
        print(f"   Entities found: {len(entities)}")
        print(f"   Relationships found: {len(relationships)}")
        print(f"   Reasoning: {reasoning}")
        
        if len(entities) == 0:
            print(f"\n❌ DIAGNOSIS: LLM response parsing is failing")
            
            # Check for specific issues
            if '<think>' in raw_response:
                think_parts = raw_response.split('</think>')
                if len(think_parts) > 1:
                    after_think = think_parts[1].strip()
                    print(f"   Content after </think>: '{after_think[:200]}...'")
                    if not after_think:
                        print(f"   ⚠️  ISSUE: Nothing after </think> tag")
                    elif '{' not in after_think:
                        print(f"   ⚠️  ISSUE: No JSON braces after </think>")
                else:
                    print(f"   ⚠️  ISSUE: <think> tag found but no </think> closing tag")
            
            if raw_response.strip() == '':
                print(f"   ⚠️  ISSUE: Empty response from LLM")
            elif 'JSON:' in raw_response:
                json_parts = raw_response.split('JSON:')
                if len(json_parts) > 1:
                    json_part = json_parts[1].strip()
                    print(f"   Content after 'JSON:': '{json_part[:200]}...'")
        else:
            print(f"\n✅ Parsing successful!")
            for i, entity in enumerate(entities[:3], 1):
                print(f"   Entity {i}: {entity}")
            for i, rel in enumerate(relationships[:3], 1):
                print(f"   Relationship {i}: {rel}")
                
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_llm_raw_response())