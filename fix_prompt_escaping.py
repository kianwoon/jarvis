#!/usr/bin/env python3
"""Fix the enhanced prompt by properly escaping JSON braces"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2

def fix_enhanced_system_prompt():
    """Fix system_prompt by properly escaping JSON braces for Python .format()"""
    
    # Enhanced system prompt with PROPERLY ESCAPED JSON examples
    fixed_prompt = """You are an expert knowledge graph extraction system. Extract entities and relationships from text and return them in STRICT JSON format.

CRITICAL: You MUST respond with ONLY valid JSON. Do NOT include explanations, markdown, thinking, or other text.

REQUIRED JSON FORMAT:
{{"entities": ["Entity1", "Entity2", "Entity3"], "relationships": [{{"from": "Entity1", "to": "Entity2", "type": "RELATIONSHIP_TYPE"}}]}}

EXAMPLES:

Example 1 - Simple Business:
{{"entities": ["Microsoft", "Seattle", "Azure"], "relationships": [{{"from": "Microsoft", "to": "Seattle", "type": "LOCATED_IN"}}, {{"from": "Microsoft", "to": "Azure", "type": "DEVELOPS"}}]}}

Example 2 - Banking/Technology:
{{"entities": ["DBS Bank", "Singapore", "OceanBase", "Alibaba", "Digital Banking"], "relationships": [{{"from": "DBS Bank", "to": "Singapore", "type": "LOCATED_IN"}}, {{"from": "DBS Bank", "to": "Digital Banking", "type": "FOCUSES_ON"}}, {{"from": "OceanBase", "to": "Alibaba", "type": "DEVELOPED_BY"}}, {{"from": "DBS Bank", "to": "OceanBase", "type": "EVALUATES"}}]}}

Example 3 - Technology Platform:
{{"entities": ["SOFAStack", "Ant Group", "Microservices", "Cloud Native", "TDSQL", "Tencent", "Distributed Database"], "relationships": [{{"from": "SOFAStack", "to": "Ant Group", "type": "DEVELOPED_BY"}}, {{"from": "SOFAStack", "to": "Microservices", "type": "SUPPORTS"}}, {{"from": "SOFAStack", "to": "Cloud Native", "type": "ENABLES"}}, {{"from": "TDSQL", "to": "Tencent", "type": "DEVELOPED_BY"}}, {{"from": "TDSQL", "to": "Distributed Database", "type": "IS_A"}}, {{"from": "SOFAStack", "to": "TDSQL", "type": "COMPETES_WITH"}}]}}

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
        
        # Update the system_prompt with properly escaped version
        if 'model_config' not in settings:
            settings['model_config'] = {}
        
        settings['model_config']['system_prompt'] = fixed_prompt
        
        # Update the database
        cur.execute(
            "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
            (json.dumps(settings),)
        )
        
        conn.commit()
        print("✅ Successfully fixed system_prompt with properly escaped JSON examples")
        print("🔧 Fixed issues:")
        print("   • Escaped all JSON braces {{ and }} for Python .format()")
        print("   • Preserved 3 diverse example scenarios")
        print("   • Maintained domain-specific entities and relationships")
        print("   • Kept clear extraction rules and limits")
        
        # Test the fixed prompt
        print("\n🧪 Testing fixed prompt formatting...")
        test_text = "DBS Bank uses OceanBase"
        try:
            formatted_prompt = fixed_prompt.format(text=test_text)
            print(f"✅ Format test PASSED - no KeyError")
            print(f"   Formatted prompt length: {len(formatted_prompt)} characters")
            if "DBS Bank uses OceanBase" in formatted_prompt:
                print(f"   ✅ Text insertion working correctly")
            if '"entities"' in formatted_prompt:
                print(f"   ✅ JSON examples properly escaped")
        except Exception as format_e:
            print(f"❌ Format test FAILED: {format_e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("🔧 Fixing Enhanced Prompt Escaping Issues")
    print("=" * 60)
    
    success = fix_enhanced_system_prompt()
    
    if success:
        print("\n🎉 Enhanced prompt escaping fix SUCCESSFUL!")
        print("🚀 Ready to test enhanced prompt with real LLM extraction")
    else:
        print("\n❌ Failed to fix enhanced prompt escaping")
    
    sys.exit(0 if success else 1)