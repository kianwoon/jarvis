#!/usr/bin/env python3
"""Create a simplified JSON prompt for smaller LLMs"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2

def create_simple_prompt():
    """Create a very simple JSON extraction prompt for smaller models"""
    
    # Much simpler prompt for smaller models
    simple_prompt = """Extract entities and relationships from this text. Return ONLY valid JSON in this exact format:

{"entities": ["entity1", "entity2"], "relationships": [{"from": "entity1", "to": "entity2", "type": "RELATED_TO"}]}

Rules:
- Only extract 2-5 important entities maximum
- Only extract 1-3 relationships maximum  
- Entity names must be 2-20 characters
- Use simple relationship types: RELATED_TO, LOCATED_IN, WORKS_FOR
- Return ONLY the JSON, no other text

Text: {text}

JSON:"""
    
    prompts = [
        {
            "id": "1",
            "name": "entity_discovery",
            "version": 1,
            "is_active": True,
            "parameters": {"format": "json"},
            "description": "Extract entities from text",
            "prompt_type": "entity_discovery",
            "prompt_template": "Find entities in: {text}"
        },
        {
            "id": "2", 
            "name": "relationship_discovery",
            "version": 1,
            "is_active": True,
            "parameters": {"format": "triples"},
            "description": "Find relationships between entities",
            "prompt_type": "relationship_discovery", 
            "prompt_template": "Find relationships in: {text}"
        },
        {
            "id": "3",
            "name": "knowledge_extraction",
            "version": 1,
            "is_active": True,
            "parameters": {"depth": "simple"},
            "description": "Simple knowledge extraction for small models",
            "prompt_type": "knowledge_extraction",
            "prompt_template": simple_prompt
        }
    ]
    
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
        settings['prompts'] = prompts
        
        # Update the database
        cur.execute(
            "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
            (json.dumps(settings),)
        )
        
        conn.commit()
        print("✅ Successfully created simple JSON prompt for smaller LLM")
        print("Simple format: entities list + relationships with from/to/type")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_simple_prompt()