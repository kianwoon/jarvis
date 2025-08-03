#!/usr/bin/env python3
"""Emergency fix for prompt escaping issue"""

import psycopg2
import json

def emergency_fix_prompt():
    """Directly fix the prompt escaping issue"""
    
    # Properly escaped prompt
    fixed_prompt = """You are an expert knowledge graph extraction system. Extract entities and relationships from text and return them in STRICT JSON format.

CRITICAL: You MUST respond with ONLY valid JSON. Do NOT include explanations, markdown, thinking, or other text.

REQUIRED JSON FORMAT:
{{"entities": ["Entity1", "Entity2"], "relationships": [{{"from": "Entity1", "to": "Entity2", "type": "TYPE"}}]}}

EXAMPLES:

Example 1:
{{"entities": ["Microsoft", "Seattle", "Azure"], "relationships": [{{"from": "Microsoft", "to": "Seattle", "type": "LOCATED_IN"}}, {{"from": "Microsoft", "to": "Azure", "type": "DEVELOPS"}}]}}

Example 2:
{{"entities": ["DBS Bank", "Singapore", "OceanBase"], "relationships": [{{"from": "DBS Bank", "to": "Singapore", "type": "LOCATED_IN"}}, {{"from": "DBS Bank", "to": "OceanBase", "type": "EVALUATES"}}]}}

RULES:
- Extract 3-12 entities maximum
- Extract 2-8 relationships maximum
- Use relationship types: LOCATED_IN, WORKS_FOR, USES, DEVELOPS, EVALUATES, PART_OF, MANAGES

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
        
        if result:
            settings = result[0]
            settings['model_config']['system_prompt'] = fixed_prompt
            
            # Update database
            cur.execute(
                "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
                (json.dumps(settings),)
            )
            
            conn.commit()
            print("✅ Emergency prompt fix applied successfully")
            
            # Test the fix
            try:
                test_formatted = fixed_prompt.format(text="Test text")
                print("✅ Format test passed - no KeyError")
                return True
            except Exception as e:
                print(f"❌ Format test failed: {e}")
                return False
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Emergency fix failed: {e}")
        return False

if __name__ == "__main__":
    emergency_fix_prompt()