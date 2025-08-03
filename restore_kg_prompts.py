#!/usr/bin/env python3
"""Restore the missing prompts section to knowledge_graph settings"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import psycopg2

def restore_prompts():
    """Add the prompts section back to knowledge_graph settings"""
    
    # Define the prompts that should be in the knowledge_graph settings
    prompts = [
        {
            "id": "1",
            "name": "entity_discovery",
            "version": 1,
            "is_active": True,
            "parameters": {"format": "json"},
            "description": "Extract entities from text for knowledge graph construction",
            "prompt_type": "entity_discovery",
            "prompt_template": "Analyze the following text and extract all entities. Return entities in JSON format: {text}"
        },
        {
            "id": "2", 
            "name": "relationship_discovery",
            "version": 1,
            "is_active": True,
            "parameters": {"format": "triples", "confidence_threshold": 0.8},
            "description": "Discover relationships between entities",
            "prompt_type": "relationship_discovery", 
            "prompt_template": "Identify relationships between entities in the text. Return relationships as triples: {text}"
        },
        {
            "id": "3",
            "name": "knowledge_extraction",
            "version": 1,
            "is_active": True,
            "parameters": {"depth": "comprehensive", "include_metadata": True},
            "description": "Extract comprehensive knowledge from documents",
            "prompt_type": "knowledge_extraction",
            "prompt_template": """You are an expert knowledge graph extraction system. Extract entities and relationships from text.

{context_info}
{domain_guidance}

ENTITY TYPES: {entity_types}
RELATIONSHIP TYPES: {relationship_types}

Extract entities and relationships from the following text and return ONLY JSON in this exact format:
{{
    "entities": [
        {{
            "text": "entity name",
            "canonical_form": "normalized name", 
            "type": "ENTITY_TYPE",
            "confidence": 0.95,
            "start_char": 0,
            "end_char": 10
        }}
    ],
    "relationships": [
        {{
            "from": "source entity name",
            "to": "target entity name", 
            "type": "RELATIONSHIP_TYPE",
            "confidence": 0.85
        }}
    ],
    "reasoning": "extraction explanation"
}}

TEXT TO ANALYZE: {text}"""
        }
    ]
    
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
        
        # Add the prompts section
        settings['prompts'] = prompts
        
        # Update the database
        cur.execute(
            "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
            (json.dumps(settings),)
        )
        
        conn.commit()
        print("✅ Successfully restored prompts section to knowledge_graph settings")
        print(f"Added {len(prompts)} prompts")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    restore_prompts()