#!/usr/bin/env python3
"""Fix the JSON prompt with proper escaping"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2

def update_system_prompt():
    """Update system_prompt with stronger JSON enforcement"""
    
    # Strong JSON-focused system prompt
    new_system_prompt = """You are an expert knowledge graph extraction system. Your ONLY task is to extract entities and relationships from text and return them in STRICT JSON format.

CRITICAL: You MUST respond with ONLY valid JSON. Do NOT include any explanations, markdown, or other text.

REQUIRED JSON FORMAT:
{"entities": ["Entity One", "Entity Two", "Entity Three"], "relationships": [{"from": "Entity One", "to": "Entity Two", "type": "RELATED_TO"}, {"from": "Entity Two", "to": "Entity Three", "type": "LOCATED_IN"}]}

EXTRACTION RULES:
- Extract 3-15 important entities maximum
- Extract 2-10 relationships maximum  
- Entity names: 2-50 characters, use proper nouns
- Relationship types: RELATED_TO, LOCATED_IN, WORKS_FOR, PART_OF, USES, MANAGES, OWNS
- Focus on concrete entities (people, organizations, locations, technologies)
- ONLY output valid JSON, nothing else

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
            print("No knowledge_graph settings found")
            return
        
        settings = result[0]
        
        # Update the system_prompt
        if 'model_config' not in settings:
            settings['model_config'] = {}
        
        settings['model_config']['system_prompt'] = new_system_prompt
        
        # Update the database
        cur.execute(
            "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
            (json.dumps(settings),)
        )
        
        conn.commit()
        print("✅ Successfully updated system_prompt with stronger JSON enforcement")
        print("📝 New prompt focuses on:")
        print("   • STRICT JSON format requirement")
        print("   • Clear entity/relationship limits")
        print("   • Concrete entity types")
        print("   • No natural language responses")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    update_system_prompt()