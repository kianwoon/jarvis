#!/usr/bin/env python3
"""Debug LLM extraction to see actual response"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

async def test_llm_extraction():
    """Test LLM extraction and see the raw response"""
    
    extractor = LLMKnowledgeExtractor()
    
    test_text = "Apple Inc. is a technology company based in Cupertino, California. Tim Cook serves as the Chief Executive Officer."
    
    print("🧠 Testing LLM Knowledge Extraction")
    print(f"📝 Text: {test_text}")
    print(f"🔧 Model Config: {extractor.model_config}")
    print("=" * 50)
    
    try:
        # Test the LLM call directly
        prompt = extractor._build_extraction_prompt(test_text)
        print(f"📋 Prompt:\n{prompt[:500]}...")
        print("=" * 50)
        
        # Make the LLM call
        raw_response = await extractor._call_llm_for_extraction(prompt)
        print(f"🤖 Raw LLM Response:\n{raw_response}")
        print("=" * 50)
        
        # Test parsing
        parsed = extractor._parse_llm_response(raw_response)
        print(f"✅ Parsed Response: {parsed}")
        
        # Full extraction
        result = await extractor.extract_knowledge(test_text)
        print(f"🎯 Final Result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm_extraction())