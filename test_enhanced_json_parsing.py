#!/usr/bin/env python3
"""Test enhanced JSON parsing with fallback strategies"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor

def test_json_parsing_strategies():
    """Test various JSON parsing scenarios"""
    
    print("🧪 Testing Enhanced JSON Parsing with Fallback Strategies")
    print("=" * 70)
    
    # Create extractor instance
    extractor = LLMKnowledgeExtractor()
    
    # Test cases
    test_cases = [
        {
            "name": "Valid JSON Response",
            "response": '{"entities": ["Microsoft", "Azure", "Seattle"], "relationships": [{"from": "Microsoft", "to": "Seattle", "type": "LOCATED_IN"}]}',
            "expected_entities": 3,
            "expected_relationships": 1
        },
        {
            "name": "JSON in Code Block",
            "response": '''```json
{"entities": ["DBS Bank", "Singapore"], "relationships": [{"from": "DBS Bank", "to": "Singapore", "type": "LOCATED_IN"}]}
```''',
            "expected_entities": 2,
            "expected_relationships": 1
        },
        {
            "name": "Natural Language Response (Your Issue)",
            "response": """**Key Points from DBS Bank's Interest in OceanBase, SOFAStack, and TDSQL**

### **1. Strategic Drivers**

- **Digital Leadership**: DBS aims to position itself as a global leader in digital banking by adopting cutting-edge technologies to match the agility, scalability, and innovation of tech giants.

- **Scalability & Resiliency**: The bank seeks technologies that support massive scalability, high availability, and fault tolerance to handle growing transaction volumes and global operations.

### **2. Technology Evaluation**

- **OceanBase**: A distributed database system from Alibaba that provides high availability and horizontal scaling capabilities.
- **SOFAStack**: Ant Group's middleware platform offering microservices architecture and cloud-native solutions.
- **TDSQL**: Tencent's distributed database solution designed for large-scale applications.""",
            "expected_entities": 5,  # Should extract DBS Bank, OceanBase, SOFAStack, TDSQL, etc.
            "expected_relationships": 1  # Should find some relationships
        },
        {
            "name": "Mixed Response with JSON",
            "response": '''Here is the analysis:

The key entities are:

```json
{"entities": ["Apple", "iPhone", "California"], "relationships": [{"from": "Apple", "to": "California", "type": "LOCATED_IN"}]}
```

This shows the relationship between Apple and California.''',
            "expected_entities": 3,
            "expected_relationships": 1
        },
        {
            "name": "Thinking Model Response",
            "response": '''<think>
I need to extract entities and relationships from this text about banking technology.
</think>

{"entities": ["Deutsche Bank", "Germany", "Frankfurt"], "relationships": [{"from": "Deutsche Bank", "to": "Germany", "type": "LOCATED_IN"}]}''',
            "expected_entities": 3,
            "expected_relationships": 1
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            # Test the parsing method directly
            result = extractor._parse_llm_response(test_case['response'])
            
            entities_found = len(result.get('entities', []))
            relationships_found = len(result.get('relationships', []))
            reasoning = result.get('reasoning', 'Unknown')
            
            print(f"📊 Results:")
            print(f"   Entities found: {entities_found} (expected: ≥{test_case.get('expected_entities', 0)})")
            print(f"   Relationships found: {relationships_found} (expected: ≥{test_case.get('expected_relationships', 0)})")
            print(f"   Reasoning: {reasoning}")
            
            # Show sample entities
            if result.get('entities'):
                print(f"   Sample entities: {[e['text'] for e in result['entities'][:3]]}")
            
            # Show sample relationships  
            if result.get('relationships'):
                print(f"   Sample relationships: {[(r['source_entity'], r['relationship_type'], r['target_entity']) for r in result['relationships'][:2]]}")
            
            # Check success
            success = (entities_found >= test_case.get('expected_entities', 0) and 
                      relationships_found >= test_case.get('expected_relationships', 0))
            
            print(f"   Status: {'✅ PASS' if success else '❌ FAIL'}")
            
            results.append({
                'name': test_case['name'],
                'success': success,
                'entities_found': entities_found,
                'relationships_found': relationships_found
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'name': test_case['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        if 'error' in result:
            print(f"{status} {result['name']} - Error: {result['error']}")
        else:
            print(f"{status} {result['name']} - {result['entities_found']} entities, {result['relationships_found']} relationships")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced JSON parsing is working correctly.")
        print("   ✅ Valid JSON parsing")
        print("   ✅ Code block extraction") 
        print("   ✅ Natural language fallback")
        print("   ✅ Mixed content handling")
        print("   ✅ Thinking model response parsing")
    else:
        print("⚠️  Some tests failed. The system may need further adjustments.")
    
    return passed == total

if __name__ == "__main__":
    success = test_json_parsing_strategies()
    sys.exit(0 if success else 1)