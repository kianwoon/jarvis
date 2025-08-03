#!/usr/bin/env python3
"""Update knowledge graph extraction prompt with proper JSON format"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.db import SessionLocal, Settings
import json

def update_knowledge_extraction_prompt():
    """Update the knowledge_extraction prompt with a proper JSON extraction template"""
    
    new_prompt_template = """You are an expert knowledge graph extraction system with dynamic schema discovery capabilities. Your task is to extract high-quality entities and relationships from the provided text.

{context_info}
{domain_guidance}

DYNAMIC SCHEMA:
Entity Types: {entity_types}
Relationship Types: {relationship_types}

EXTRACTION GUIDELINES - STRICT QUALITY CONTROL:
1. **Entity Quality**: Only extract proper nouns, names, specific concepts, or clearly identifiable entities
   - ❌ Exclude: "this proposal", "a few steps", "submit the document" (these are phrases/actions)
   - ✅ Include: "Microsoft", "John Smith", "Machine Learning", "New York"
2. **Entity Length**: 2-50 characters, avoid single words unless they are proper nouns
3. **Entity Specificity**: Must be a specific, identifiable thing, not a generic concept
4. **Relationship Quality**: Only create relationships between valid, specific entities
5. **Confidence**: Provide scores based on textual clarity and specificity
6. **Validation**: Ensure entities are not generic terms, actions, or sentence fragments

TEXT TO ANALYZE:
{text}

OUTPUT FORMAT (JSON):
{{
    "entities": [
        {{
            "text": "exact text from source",
            "canonical_form": "normalized name",
            "type": "entity_type",
            "confidence": 0.95,
            "evidence": "supporting text snippet",
            "start_char": 0,
            "end_char": 10,
            "attributes": {{"key": "value"}}
        }}
    ],
    "relationships": [
        {{
            "source_entity": "canonical name of source",
            "target_entity": "canonical name of target",
            "relationship_type": "relationship_type",
            "confidence": 0.85,
            "evidence": "supporting text snippet",
            "context": "broader context of relationship"
        }}
    ],
    "discoveries": {{
        "new_entity_types": [
            {{
                "type": "NewEntityType",
                "description": "What this entity represents",
                "examples": ["example from text"],
                "confidence": 0.8
            }}
        ],
        "new_relationship_types": [
            {{
                "type": "new_relationship",
                "description": "What this relationship represents",
                "inverse": "inverse_type",
                "examples": ["example from text"]
            }}
        ]
    }},
    "reasoning": "Brief explanation of extraction approach and key decisions"
}}

Provide ONLY the JSON output without any additional text or formatting."""
    
    db = SessionLocal()
    try:
        # Get knowledge_graph settings
        kg_settings = db.query(Settings).filter(Settings.category == 'knowledge_graph').first()
        
        if not kg_settings:
            print("No knowledge_graph settings found")
            return
        
        # Update the prompts list
        prompts = kg_settings.settings.get('prompts', [])
        
        # Find and update the knowledge_extraction prompt
        for i, prompt in enumerate(prompts):
            if prompt.get('prompt_type') == 'knowledge_extraction' or prompt.get('name') == 'knowledge_extraction':
                prompts[i]['prompt_template'] = new_prompt_template
                print(f"Updated prompt: {prompt.get('name', prompt.get('prompt_type'))}")
                break
        else:
            print("knowledge_extraction prompt not found")
            return
        
        # Update the settings using SQLAlchemy's mutable JSON handling
        from sqlalchemy.dialects.postgresql import JSONB
        kg_settings.settings = dict(kg_settings.settings)  # Force conversion to dict
        kg_settings.settings['prompts'] = prompts
        
        db.commit()
        print("✅ Successfully updated knowledge_extraction prompt")
        
    except Exception as e:
        print(f"❌ Failed to update prompt: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_knowledge_extraction_prompt()