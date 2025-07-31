"""
Settings-based prompt service that uses the existing settings table
instead of creating a new llm_prompts table.
"""

import logging
from typing import Dict, Any, List
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.core.config import get_settings
from app.core.db import SessionLocal
from app.core.db import Settings

logger = logging.getLogger(__name__)

class SettingsPromptService:
    """Service for managing prompts using the existing settings table"""
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_prompt(self, prompt_name: str, variables: Dict[str, Any] = None) -> str:
        """Get a prompt template from settings"""
        try:
            db = SessionLocal()
            try:
                # Get llm settings and extract knowledge_graph section
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'llm'
                ).first()
                
                if not kg_settings or 'knowledge_graph' not in kg_settings.settings:
                    # Initialize default prompts
                    self._initialize_default_prompts()
                    kg_settings = db.query(Settings).filter(
                        Settings.category == 'llm'
                    ).first()
                
                prompts = kg_settings.settings.get('knowledge_graph', {}).get('prompts', {})
                template = prompts.get(prompt_name, {}).get('template', '')
                
                if variables:
                    return template.format(**variables)
                return template
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_name}: {e}")
            return self._get_fallback_prompt(prompt_name, variables)
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts"""
        try:
            db = SessionLocal()
            try:
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'llm'
                ).first()
                
                if not kg_settings or 'knowledge_graph' not in kg_settings.settings:
                    self._initialize_default_prompts()
                    kg_settings = db.query(Settings).filter(
                        Settings.category == 'llm'
                    ).first()
                
                prompts = kg_settings.settings.get('knowledge_graph', {}).get('prompts', {})
                
                return [
                    {
                        "id": name,
                        "name": data.get("name", name),
                        "type": name,
                        "description": data.get("description", ""),
                        "version": data.get("version", 1),
                        "is_active": True
                    }
                    for name, data in prompts.items()
                ]
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return self._get_fallback_prompts()
    
    def update_prompt(self, prompt_name: str, new_template: str) -> bool:
        """Update an existing prompt template"""
        try:
            db = SessionLocal()
            try:
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'llm'
                ).first()
                
                if not kg_settings:
                    kg_settings = Settings(category='llm', settings={'knowledge_graph': {}})
                    db.add(kg_settings)
                
                if 'knowledge_graph' not in kg_settings.settings:
                    kg_settings.settings['knowledge_graph'] = {}
                if 'prompts' not in kg_settings.settings['knowledge_graph']:
                    kg_settings.settings['knowledge_graph']['prompts'] = {}
                
                if prompt_name in kg_settings.settings['knowledge_graph']['prompts']:
                    kg_settings.settings['knowledge_graph']['prompts'][prompt_name]['template'] = new_template
                    kg_settings.settings['knowledge_graph']['prompts'][prompt_name]['version'] = \
                        kg_settings.settings['knowledge_graph']['prompts'][prompt_name].get('version', 1) + 1
                    db.commit()
                    return True
                    
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_name}: {e}")
            return False
    
    def _initialize_default_prompts(self):
        """Initialize default prompts in settings"""
        try:
            db = SessionLocal()
            try:
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'llm'
                ).first()
                
                if not kg_settings:
                    kg_settings = Settings(category='llm', settings={})
                    db.add(kg_settings)
                
                if 'knowledge_graph' not in kg_settings.settings:
                    kg_settings.settings['knowledge_graph'] = {}
                
                if 'prompts' not in kg_settings.settings['knowledge_graph']:
                    kg_settings.settings['knowledge_graph']['prompts'] = {
                        'entity_discovery': {
                            'name': 'Entity Discovery Prompt',
                            'description': 'Prompt for discovering new entity types from text',
                            'template': '''You are an expert knowledge graph architect. Analyze the provided text and discover unique entity types beyond standard categories.

Text: {text}

Current accepted entity types: {existing_types}

Instructions:
1. Identify any unique entity types that don\'t fit standard categories (Person, Organization, Location, Event, Concept)
2. Provide a clear, singular label for each entity type
3. Include a brief description and 1-2 examples from the text
4. Rate confidence (0-1) based on clarity and frequency
5. Group similar entities under unified types
6. Focus on domain-specific or context-specific entities

Return format (JSON):
{
    "discovered_entities": [
        {
            "type": "UniqueEntityType",
            "description": "Brief description of what this entity represents",
            "examples": ["Example from text"],
            "confidence": 0.85,
            "frequency": 3
        }
    ]
}

Important: Only include high-confidence discoveries (confidence >= 0.75).''',
                            'variables': ['text', 'existing_types'],
                            'required': ['text'],
                            'confidence_threshold': 0.75
                        },
                        'relationship_discovery': {
                            'name': 'Relationship Discovery Prompt',
                            'description': 'Prompt for discovering new relationship types between entities',
                            'template': '''You are an expert knowledge graph relationship designer. Discover unique relationship types between entities.

Entities to analyze: {entities}

Current accepted relationship types: {existing_types}

Instructions:
1. Identify relationship types beyond standard ones (works_for, located_in, part_of, related_to, causes)
2. Use clear verb phrases (e.g., "collaborates_with", "influences", "implements")
3. Provide inverse relationships where applicable
4. Rate confidence (0-1) based on semantic clarity
5. Focus on context-specific relationships
6. Consider temporal, causal, and functional relationships

Return format (JSON):
{
    "discovered_relationships": [
        {
            "type": "unique_relationship",
            "description": "What this relationship represents",
            "inverse": "reverse_relationship_type",
            "examples": ["Entity1 -[relationship]-> Entity2"],
            "confidence": 0.82,
            "frequency": 2
        }
    ]
}

Important: Only include high-confidence discoveries (confidence >= 0.7).''',
                            'variables': ['entities', 'existing_types'],
                            'required': ['entities'],
                            'confidence_threshold': 0.7
                        },
                        'knowledge_extraction': {
                            'name': 'Knowledge Extraction Prompt',
                            'description': 'Main prompt for extracting entities and relationships from text',
                            'template': '''You are an expert knowledge graph extraction system with dynamic schema discovery capabilities. Your task is to extract high-quality entities and relationships from the provided text.

{context_info}
{domain_guidance}

DYNAMIC SCHEMA:
Entity Types: {entity_types}
Relationship Types: {relationship_types}

EXTRACTION GUIDELINES - STRICT QUALITY CONTROL:
1. **Entity Quality**: Only extract proper nouns, names, specific concepts, or clearly identifiable entities
   - ❌ Exclude: "this proposal", "a few steps", "submit the document" (these are phrases/actions)
   - ✅ Include: "Microsoft", "John Smith", "Machine Learning", "New York"
2. **Entity Length**: 2-50 characters, avoid single words unless they\'re proper nouns
3. **Entity Specificity**: Must be a specific, identifiable thing, not a generic concept
4. **Relationship Quality**: Only create relationships between valid, specific entities
5. **Confidence**: Provide scores based on textual clarity and specificity
6. **Validation**: Ensure entities are not generic terms, actions, or sentence fragments

TEXT TO ANALYZE:
{text}

OUTPUT FORMAT (JSON):
{
    "entities": [
        {
            "text": "exact text from source",
            "canonical_form": "normalized name",
            "type": "entity_type",
            "confidence": 0.95,
            "evidence": "supporting text snippet",
            "start_char": 0,
            "end_char": 10,
            "attributes": {"key": "value"}
        }
    ],
    "relationships": [
        {
            "source_entity": "canonical name of source",
            "target_entity": "canonical name of target",
            "relationship_type": "relationship_type",
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
                "examples": ["example from text"],
                "confidence": 0.8
            }
        ],
        "new_relationship_types": [
            {
                "type": "new_relationship",
                "description": "What this relationship represents",
                "inverse": "inverse_type",
                "examples": ["example from text"]
            }
        ]
    },
    "reasoning": "Brief explanation of extraction approach and key decisions"
}

Provide ONLY the JSON output without any additional text or formatting.''',
                            'variables': ['text', 'context_info', 'domain_guidance', 'entity_types', 'relationship_types'],
                            'required': ['text']
                        }
                    }
                
                db.commit()
                logger.info("Initialized default prompts in knowledge_graph settings")
                
            except Exception as e:
                logger.error(f"Failed to initialize default prompts: {e}")
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to initialize default prompts: {e}")
    
    def _get_fallback_prompt(self, prompt_name: str, variables: Dict[str, Any] = None) -> str:
        """Fallback prompt for when database is unavailable"""
        fallback_prompts = {
            'entity_discovery': '''You are an expert knowledge graph architect. Analyze the provided text and discover unique entity types beyond standard categories.

Text: {text}

Current accepted entity types: {existing_types}

Return JSON format with discovered entities.''',
            'relationship_discovery': '''You are an expert knowledge graph relationship designer. Discover unique relationship types between entities.

Entities to analyze: {entities}

Current accepted relationship types: {existing_types}

Return JSON format with discovered relationships.''',
            'knowledge_extraction': '''You are an expert knowledge graph extraction system. Extract entities and relationships from the provided text.

Text: {text}

Return JSON format with entities and relationships.'''
        }
        
        template = fallback_prompts.get(prompt_name, '')
        if variables:
            return template.format(**variables)
        return template
    
    def _get_fallback_prompts(self) -> List[Dict[str, Any]]:
        """Fallback prompts when database is unavailable"""
        return [
            {
                "id": "entity_discovery",
                "name": "Entity Discovery Prompt",
                "type": "entity_discovery",
                "description": "Prompt for discovering new entity types",
                "version": 1,
                "is_active": True
            },
            {
                "id": "relationship_discovery",
                "name": "Relationship Discovery Prompt",
                "type": "relationship_discovery",
                "description": "Prompt for discovering new relationship types",
                "version": 1,
                "is_active": True
            },
            {
                "id": "knowledge_extraction",
                "name": "Knowledge Extraction Prompt",
                "type": "knowledge_extraction",
                "description": "Main extraction prompt",
                "version": 1,
                "is_active": True
            }
        ]

# Singleton instance
_prompt_service: SettingsPromptService = None

def get_prompt_service() -> SettingsPromptService:
    """Get or create prompt service singleton"""
    global _prompt_service
    if _prompt_service is None:
        _prompt_service = SettingsPromptService()
    return _prompt_service