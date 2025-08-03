#!/usr/bin/env python3
"""
Update the knowledge graph extraction prompt to improve quality
and prevent extracting word fragments like "Ant", "Below", "They"
"""

import psycopg2
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_extraction_prompt():
    """Update the extraction prompt in the knowledge_graph settings"""
    
    # Enhanced prompt that strictly rejects fragments
    enhanced_prompt = """You are an expert knowledge graph extraction system specializing in extracting complete, meaningful entities from business and technical documents. Extract ONLY high-quality, complete entities and relationships.

{context_info}
{domain_guidance}

DYNAMIC SCHEMA:
Entity Types: {entity_types}
Relationship Types: {relationship_types}

STRICT ENTITY EXTRACTION RULES - NO FRAGMENTS ALLOWED:
1. **COMPLETE NAMES ONLY**: Extract full organization names, complete technology names, full person names
   - ✅ GOOD: "DBS Bank", "Cloud-Native Architecture", "Digital Transformation", "Amazon Web Services"  
   - ❌ NEVER: "Ant", "Below", "They", "CEO", "The", "This", "That", "It", "A", "An"

2. **MINIMUM QUALITY THRESHOLDS**:
   - Must be 3+ characters AND meaningful
   - Must be a complete noun phrase, not sentence fragments
   - Must represent a specific entity, not generic words

3. **ENTITY CATEGORIES TO EXTRACT**:
   - Organizations: Full company names, divisions, departments
   - Technologies: Complete technology names, platforms, systems
   - Products: Full product names, services, solutions
   - Concepts: Complete business concepts, methodologies, frameworks
   - Locations: Specific places, regions, offices
   - People: Full names only (not titles alone)

4. **STRICT REJECTIONS - NEVER EXTRACT**:
   - Pronouns: "they", "it", "this", "that", "we", "our"
   - Articles: "a", "an", "the"
   - Prepositions: "in", "on", "at", "by", "for"
   - Generic words: "below", "above", "here", "there"
   - Partial words: "ant" (from "important"), fragments, abbreviations alone
   - Job titles without names: "CEO", "CTO", "Manager" (unless part of complete name)

5. **RELATIONSHIP QUALITY**: Only connect meaningful, complete entities with specific relationships

TEXT TO ANALYZE:
{text}

OUTPUT FORMAT (JSON):
{
    "entities": [
        {
            "text": "exact complete text from source",
            "canonical_form": "normalized complete name",
            "type": "entity_type",
            "confidence": 0.95,
            "evidence": "supporting text snippet showing context",
            "start_char": 0,
            "end_char": 10,
            "attributes": {"key": "value"}
        }
    ],
    "relationships": [
        {
            "source_entity": "complete canonical name of source",
            "target_entity": "complete canonical name of target",
            "relationship_type": "specific_relationship_type",
            "confidence": 0.85,
            "evidence": "supporting text snippet",
            "context": "broader context of relationship"
        }
    ],
    "discoveries": {
        "new_entity_types": [
            {
                "type": "NewEntityType",
                "description": "What this entity represents",
                "examples": ["complete example from text"],
                "confidence": 0.8
            }
        ],
        "new_relationship_types": [
            {
                "type": "specific_relationship",
                "description": "What this relationship represents",
                "inverse": "inverse_type",
                "examples": ["complete example from text"]
            }
        ]
    },
    "reasoning": "Brief explanation focusing on why each entity meets quality thresholds"
}

CRITICAL: Only extract entities that are complete, meaningful names. Reject any fragments, pronouns, or generic words. Quality over quantity - better to extract 20 high-quality entities than 100 meaningless fragments.

Provide ONLY the JSON output without any additional text or formatting."""

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="llm_platform",
            user="postgres",
            password="postgres"
        )
        
        with conn.cursor() as cur:
            # Get current knowledge_graph settings
            cur.execute("SELECT settings FROM settings WHERE category = 'knowledge_graph'")
            result = cur.fetchone()
            
            if result:
                settings = result[0]
                logger.info("Found existing knowledge_graph settings")
                
                # Update the extraction prompt
                if 'prompts' not in settings:
                    settings['prompts'] = []
                
                # Find and update the knowledge_extraction prompt
                updated = False
                for prompt in settings['prompts']:
                    if prompt.get('prompt_type') == 'knowledge_extraction' or prompt.get('name') == 'Knowledge Extraction Prompt':
                        prompt['prompt_template'] = enhanced_prompt
                        updated = True
                        logger.info("Updated existing knowledge_extraction prompt")
                        break
                
                # If not found, add it
                if not updated:
                    settings['prompts'].append({
                        "name": "Knowledge Extraction Prompt",
                        "prompt_type": "knowledge_extraction", 
                        "prompt_template": enhanced_prompt,
                        "description": "Enhanced prompt for high-quality entity extraction",
                        "version": 2,
                        "is_active": True
                    })
                    logger.info("Added new knowledge_extraction prompt")
                
                # Update the database
                cur.execute(
                    "UPDATE settings SET settings = %s WHERE category = 'knowledge_graph'",
                    (json.dumps(settings),)
                )
                
                conn.commit()
                logger.info("✅ Successfully updated extraction prompt in database")
                
            else:
                logger.error("❌ No knowledge_graph settings found")
                
    except Exception as e:
        logger.error(f"❌ Failed to update extraction prompt: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    update_extraction_prompt()