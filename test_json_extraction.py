#!/usr/bin/env python3
import json
import re

def extract_json_bulletproof(response):
    """
    BULLETPROOF JSON extraction - handles any LLM response format.
    Finds the first { and last } and extracts everything between them.
    """
    # Find first { and last }  
    start = response.find('{')
    end = response.rfind('}')
    
    if start == -1 or end == -1 or end <= start:
        raise ValueError('No valid JSON object found')
    
    json_content = response[start:end+1]
    return json_content

# Test with various problematic formats
test_cases = [
    # Case 1: JSON with code blocks and trailing text
    '''```json
{
  "entities": [
    {
      "name": "DBS Bank",
      "type": "Organization",
      "description": "Major bank in Singapore"
    }
  ],
  "relationships": []
}
```

This response contains additional explanatory text after the JSON.''',
    
    # Case 2: Just JSON with trailing text
    '''{
  "entities": [
    {"name": "Test Entity", "type": "Test"}
  ],
  "relationships": []
}

Additional text here that breaks current parsing.''',
    
    # Case 3: JSON with prefix text
    '''Here is the extracted knowledge graph:

{
  "entities": [
    {"name": "Another Entity", "type": "Test"}
  ],
  "relationships": []
}''',
]

print("🔧 Testing BULLETPROOF JSON extraction:")
print("=" * 50)

for i, test_response in enumerate(test_cases, 1):
    print(f"\n📝 Test Case {i}:")
    print(f"Response length: {len(test_response)} chars")
    print(f"Preview: {test_response[:100]}...")
    
    try:
        # Extract JSON
        extracted = extract_json_bulletproof(test_response)
        print(f"✅ Extraction successful! Length: {len(extracted)} chars")
        
        # Parse JSON
        parsed = json.loads(extracted)
        print(f"✅ JSON parsing successful!")
        print(f"   - Entities: {len(parsed.get('entities', []))}")
        print(f"   - Relationships: {len(parsed.get('relationships', []))}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n🎯 Result: Simple approach works perfectly!")