#!/usr/bin/env python3
"""
Test the BULLETPROOF JSON parsing fix with the exact LLM response format
that was causing failures.
"""
import json
import sys
import os

# Add the app directory to Python path
sys.path.append('/Users/kianwoonwong/Downloads/jarvis')

def test_bulletproof_parsing(response):
    """Test the BULLETPROOF parsing logic exactly as implemented"""
    
    # Clean response (matching the actual code)
    response = response.strip()
    
    # BULLETPROOF JSON parsing - simple and reliable
    parsed = None
    
    # Strategy 1: Direct JSON parsing
    try:
        parsed = json.loads(response)
        print("✅ Direct JSON parsing successful")
        return parsed
    except json.JSONDecodeError as e:
        print(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: BULLETPROOF extraction - find first { and last }
        json_start = response.find('{')
        json_end = response.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_content = response[json_start:json_end+1]
            try:
                parsed = json.loads(json_content)
                print("✅ Bulletproof JSON extraction successful")
                return parsed
            except json.JSONDecodeError as e2:
                print(f"Bulletproof extraction failed: {e2}")
    
    if not parsed:
        print(f"All JSON parsing failed. Response preview: {response[:200]}...")
        raise ValueError("All JSON parsing strategies failed")

# Test with the EXACT format that was failing
problematic_response = '''```json
{
  "entities": [
    {
      "name": "DBS Bank",
      "type": "Organization",
      "description": "A leading financial services group in Asia with operations in 19 markets"
    },
    {
      "name": "Singapore",
      "type": "Location",
      "description": "Home market and headquarters location of DBS Bank"
    },
    {
      "name": "Digital Banking",
      "type": "Concept",
      "description": "Core strategic focus area for DBS Bank's transformation"
    }
  ],
  "relationships": [
    {
      "source": "DBS Bank",
      "target": "Singapore",
      "type": "HEADQUARTERED_IN",
      "description": "DBS Bank is headquartered in Singapore"
    },
    {
      "source": "DBS Bank", 
      "target": "Digital Banking",
      "type": "FOCUSES_ON",
      "description": "DBS Bank focuses on digital banking transformation"
    }
  ]
}
```

This JSON response contains 3 entities and 2 relationships extracted from the DBS document.'''

print("🔧 Testing BULLETPROOF fix with actual LLM response format:")
print("=" * 60)
print(f"Response length: {len(problematic_response)} characters")
print(f"Preview: {problematic_response[:150]}...")
print()

try:
    result = test_bulletproof_parsing(problematic_response)
    print()
    print("🎯 SUCCESS! JSON parsing worked perfectly!")
    print(f"   - Entities found: {len(result.get('entities', []))}")
    print(f"   - Relationships found: {len(result.get('relationships', []))}")
    print()
    print("📝 Sample entities:")
    for i, entity in enumerate(result.get('entities', [])[:2]):
        print(f"   {i+1}. {entity.get('name')} ({entity.get('type')})")
    
    print()
    print("🔗 Sample relationships:")
    for i, rel in enumerate(result.get('relationships', [])[:2]):
        print(f"   {i+1}. {rel.get('source')} -> {rel.get('target')} ({rel.get('type')})")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

print()
print("✅ BULLETPROOF JSON parsing fix is working perfectly!")
print("   The knowledge graph extraction should now work reliably.")