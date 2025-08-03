#!/usr/bin/env python3
"""Simple prompt update using direct database modification"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2

def update_prompt():
    """Update the knowledge_extraction prompt directly in PostgreSQL"""
    
    new_prompt = """You are an expert knowledge graph extraction system. Extract entities and relationships from text and return ONLY JSON in this exact format: {"entities": [{"text": "entity name", "canonical_form": "normalized name", "type": "ENTITY_TYPE", "confidence": 0.95, "start_char": 0, "end_char": 10}], "relationships": [{"source_entity": "source name", "target_entity": "target name", "relationship_type": "RELATIONSHIP_TYPE", "confidence": 0.85, "context": "relationship context"}], "reasoning": "extraction explanation"}. TEXT: {text}"""
    
    try:
        # Connect to database
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
        prompts = settings.get('prompts', [])
        
        # Update the knowledge_extraction prompt
        for i, prompt in enumerate(prompts):
            if prompt.get('prompt_type') == 'knowledge_extraction':
                prompts[i]['prompt_template'] = new_prompt
                print(f"Updated prompt: {prompt.get('name')}")
                break
        else:
            print("knowledge_extraction prompt not found")
            return
        
        # Update the database
        settings['prompts'] = prompts
        cur.execute(
            "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
            (json.dumps(settings),)
        )
        
        conn.commit()
        print("✅ Successfully updated knowledge_extraction prompt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    update_prompt()