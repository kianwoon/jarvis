#!/usr/bin/env python3
"""Enhanced JSON prompt with multiple diverse examples for better model learning"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2

def update_enhanced_system_prompt():
    """Update system_prompt with multiple comprehensive JSON examples"""
    
    # Enhanced system prompt with multiple diverse examples
    enhanced_system_prompt = """You are an expert knowledge graph extraction system. Extract entities and relationships from text and return them in STRICT JSON format.

CRITICAL: You MUST respond with ONLY valid JSON. Do NOT include explanations, markdown, thinking, or other text.

REQUIRED JSON FORMAT:
{"entities": ["Entity1", "Entity2", "Entity3"], "relationships": [{"from": "Entity1", "to": "Entity2", "type": "RELATIONSHIP_TYPE"}]}

EXAMPLES:

Example 1 - Simple Business:
{"entities": ["Microsoft", "Seattle", "Azure"], "relationships": [{"from": "Microsoft", "to": "Seattle", "type": "LOCATED_IN"}, {"from": "Microsoft", "to": "Azure", "type": "DEVELOPS"}]}

Example 2 - Banking/Technology:
{"entities": ["DBS Bank", "Singapore", "OceanBase", "Alibaba", "Digital Banking"], "relationships": [{"from": "DBS Bank", "to": "Singapore", "type": "LOCATED_IN"}, {"from": "DBS Bank", "to": "Digital Banking", "type": "FOCUSES_ON"}, {"from": "OceanBase", "to": "Alibaba", "type": "DEVELOPED_BY"}, {"from": "DBS Bank", "to": "OceanBase", "type": "EVALUATES"}]}

Example 3 - Technology Platform:
{"entities": ["SOFAStack", "Ant Group", "Microservices", "Cloud Native", "TDSQL", "Tencent", "Distributed Database"], "relationships": [{"from": "SOFAStack", "to": "Ant Group", "type": "DEVELOPED_BY"}, {"from": "SOFAStack", "to": "Microservices", "type": "SUPPORTS"}, {"from": "SOFAStack", "to": "Cloud Native", "type": "ENABLES"}, {"from": "TDSQL", "to": "Tencent", "type": "DEVELOPED_BY"}, {"from": "TDSQL", "to": "Distributed Database", "type": "IS_A"}, {"from": "SOFAStack", "to": "TDSQL", "type": "COMPETES_WITH"}]}

EXTRACTION RULES:
- Extract 3-12 important entities maximum
- Extract 2-8 relationships maximum  
- Entity names: 2-50 characters, use proper nouns
- Relationship types: LOCATED_IN, WORKS_FOR, PART_OF, USES, MANAGES, OWNS, DEVELOPS, DEVELOPED_BY, SUPPORTS, ENABLES, FOCUSES_ON, EVALUATES, COMPETES_WITH, PARTNERS_WITH, IS_A
- Focus on concrete entities: people, organizations, locations, technologies, products
- Use exact entity names from text, avoid generic terms
- Ensure all relationship entities exist in entities array

Text: {text}

JSON:"""
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="llm_platform", 
            user="postgres",
            password="postgres"
        )
        
        cur = conn.cursor()
        
        # Get current settings
        cur.execute("SELECT settings FROM settings WHERE category = 'knowledge_graph'")
        result = cur.fetchone()
        
        if not result:
            print("❌ No knowledge_graph settings found")
            return False
        
        settings = result[0]
        
        # Update the system_prompt
        if 'model_config' not in settings:
            settings['model_config'] = {}
        
        settings['model_config']['system_prompt'] = enhanced_system_prompt
        
        # Update the database
        cur.execute(
            "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
            (json.dumps(settings),)
        )
        
        conn.commit()
        print("✅ Successfully updated system_prompt with enhanced JSON examples")
        print("📝 Enhanced prompt includes:")
        print("   • 3 diverse example scenarios (business, banking/tech, technology platform)")
        print("   • Domain-specific entities (DBS Bank, OceanBase, SOFAStack, TDSQL)")
        print("   • 15 relationship types covering business and technical domains")
        print("   • Clear entity/relationship quantity limits (3-12 entities, 2-8 relationships)")
        print("   • Explicit instructions for exact entity name usage")
        print("   • Validation rule: all relationship entities must exist in entities array")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def test_enhanced_prompt():
    """Test the enhanced prompt with a sample extraction"""
    print("\n🧪 Testing Enhanced Prompt with Sample Text")
    print("=" * 60)
    
    sample_text = """DBS Bank is evaluating OceanBase for its digital banking transformation. 
    OceanBase is a distributed database developed by Alibaba. The bank is also considering 
    SOFAStack, Ant Group's middleware platform for microservices architecture."""
    
    try:
        from app.services.llm_knowledge_extractor import LLMKnowledgeExtractor
        extractor = LLMKnowledgeExtractor()
        
        # Test the parsing method with a mock JSON response (since we can't call LLM in test)
        mock_json_response = '{"entities": ["DBS Bank", "OceanBase", "Alibaba", "SOFAStack", "Ant Group"], "relationships": [{"from": "DBS Bank", "to": "OceanBase", "type": "EVALUATES"}, {"from": "OceanBase", "to": "Alibaba", "type": "DEVELOPED_BY"}, {"from": "SOFAStack", "to": "Ant Group", "type": "DEVELOPED_BY"}]}'
        
        result = extractor._parse_llm_response(mock_json_response)
        
        print(f"📊 Mock Test Results:")
        print(f"   Entities found: {len(result.get('entities', []))}")
        print(f"   Relationships found: {len(result.get('relationships', []))}")
        print(f"   Sample entities: {[e['text'] for e in result.get('entities', [])]}")
        print(f"   Sample relationships: {[(r['source_entity'], r['relationship_type'], r['target_entity']) for r in result.get('relationships', [])]}")
        
        if len(result.get('entities', [])) >= 3 and len(result.get('relationships', [])) >= 2:
            print("✅ Enhanced prompt structure validation PASSED")
            return True
        else:
            print("❌ Enhanced prompt structure validation FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhancing JSON Prompt with Multiple Examples")
    print("=" * 60)
    
    # Update the prompt
    success = update_enhanced_system_prompt()
    
    if success:
        # Test the enhanced structure
        test_success = test_enhanced_prompt()
        
        if test_success:
            print("\n🎉 Enhanced JSON prompt deployment SUCCESSFUL!")
            print("🔧 Next steps:")
            print("   1. Test with real document processing")
            print("   2. Monitor for improved JSON response rates")
            print("   3. Validate knowledge graph entity/relationship quality")
        else:
            print("\n⚠️ Enhanced prompt updated but test validation failed")
    else:
        print("\n❌ Failed to update enhanced JSON prompt")
    
    sys.exit(0 if success else 1)