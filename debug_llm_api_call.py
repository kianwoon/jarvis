#!/usr/bin/env python3
"""
Debug LLM API Call to See What's Happening
"""

import asyncio
import sys
import os
import json
import aiohttp

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_llm_api_directly():
    """Test the LLM API call directly to debug the issue"""
    print("🔍 Testing LLM API Call Directly")
    print("=" * 50)
    
    # Test settings
    model_config = {
        'model': 'qwen3:30b-a3b-q4_K_M',
        'temperature': 0.1,
        'max_tokens': 4096,
        'model_server': 'http://localhost:11434'
    }
    
    # Simple test prompt
    test_prompt = """Extract entities and relationships from this text in JSON format:

"DBS Bank is a technology company in Singapore."

Return a JSON object with:
{
  "entities": [{"text": "entity_name", "type": "ENTITY_TYPE", "confidence": 0.9}],
  "relationships": [{"source_entity": "source", "target_entity": "target", "relationship_type": "RELATION", "confidence": 0.8}]
}"""
    
    print(f"Model: {model_config['model']}")
    print(f"Server: {model_config['model_server']}")
    print(f"Prompt length: {len(test_prompt)} characters")
    
    payload = {
        "model": model_config['model'],
        "prompt": test_prompt,
        "temperature": model_config['temperature'],
        "max_tokens": model_config['max_tokens'],
        "stream": False
    }
    
    print(f"\n📦 Payload being sent:")
    print(json.dumps(payload, indent=2))
    
    try:
        async with aiohttp.ClientSession() as session:
            print(f"\n🌐 Making request to {model_config['model_server']}/api/generate")
            
            async with session.post(
                f"{model_config['model_server']}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"\n✅ Raw LLM Response:")
                    print(json.dumps(result, indent=2))
                    
                    # Check what we get back
                    llm_response = result.get('response', '')
                    print(f"\n📝 LLM Response Text:")
                    print(f"Length: {len(llm_response)}")
                    print(f"Content: '{llm_response}'")
                    
                    if llm_response:
                        # Try to parse it
                        try:
                            # Clean response - remove any non-JSON text
                            cleaned = llm_response.strip()
                            if '```json' in cleaned:
                                cleaned = cleaned.split('```json')[1].split('```')[0]
                            elif '```' in cleaned:
                                cleaned = cleaned.split('```')[1].split('```')[0]
                            
                            print(f"\n🧹 Cleaned response:")
                            print(f"'{cleaned}'")
                            
                            parsed = json.loads(cleaned)
                            print(f"\n✅ Successfully parsed JSON:")
                            print(json.dumps(parsed, indent=2))
                            
                            entities = parsed.get('entities', [])
                            relationships = parsed.get('relationships', [])
                            print(f"\n📊 Extracted:")
                            print(f"   {len(entities)} entities")
                            print(f"   {len(relationships)} relationships")
                            
                        except json.JSONDecodeError as e:
                            print(f"\n❌ JSON Parse Error: {e}")
                            print(f"Trying to parse: '{cleaned}'")
                    else:
                        print(f"\n❌ Empty response from LLM")
                        
                else:
                    text = await response.text()
                    print(f"\n❌ LLM API error: {response.status}")
                    print(f"Response text: {text}")
                    
    except Exception as e:
        print(f"\n❌ Request failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm_api_directly())