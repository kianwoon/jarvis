"""
Settings-based prompt service that uses the existing settings table
with fixed category handling and improved debugging.
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
        """Get a prompt template from settings with improved loading and debugging"""
        try:
            db = SessionLocal()
            try:
                # Get knowledge_graph settings directly first
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'knowledge_graph'
                ).first()
                
                logger.debug(f"Loading prompt '{prompt_name}' from knowledge_graph settings")
                
                if not kg_settings or 'prompts' not in kg_settings.settings:
                    logger.info("Knowledge graph prompts not found, initializing default prompts")
                    # Initialize default prompts
                    self._initialize_default_prompts()
                    kg_settings = db.query(Settings).filter(
                        Settings.category == 'knowledge_graph'
                    ).first()
                
                prompts_data = kg_settings.settings.get('prompts', [])
                
                # Handle both list and dict formats for prompts
                if isinstance(prompts_data, dict):
                    # Convert dict format to list format for compatibility
                    prompts_list = [
                        {
                            'name': key,
                            'prompt_type': key,
                            'prompt_template': value.get('template', value.get('prompt_template', ''))
                        }
                        for key, value in prompts_data.items()
                    ]
                    logger.debug(f"Converted dict prompts to list format: {list(prompts_data.keys())}")
                else:
                    prompts_list = prompts_data
                
                # Find prompt by name or prompt_type
                template = ''
                for prompt in prompts_list:
                    if prompt.get('name') == prompt_name or prompt.get('prompt_type') == prompt_name:
                        template = prompt.get('prompt_template', '')
                        logger.debug(f"Found prompt template for '{prompt_name}', length: {len(template)}")
                        break
                
                if not template:
                    logger.warning(f"No template found for prompt '{prompt_name}', available prompts: {[p.get('name', p.get('prompt_type', 'unknown')) for p in prompts_list]}")
                    return self._get_fallback_prompt(prompt_name, variables)
                
                if variables:
                    try:
                        formatted_template = template.format(**variables)
                        logger.debug(f"Successfully formatted template for '{prompt_name}' with {len(variables)} variables")
                        return formatted_template
                    except KeyError as e:
                        logger.warning(f"Template variable {e} not provided for prompt '{prompt_name}', returning template as-is")
                        return template
                    except ValueError as e:
                        logger.warning(f"Template formatting error for prompt '{prompt_name}': {e}, returning template as-is")
                        return template
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
                # First try knowledge_graph category
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'knowledge_graph'
                ).first()
                
                if not kg_settings or 'prompts' not in kg_settings.settings:
                    # Fallback to llm category
                    kg_settings = db.query(Settings).filter(
                        Settings.category == 'llm'
                    ).first()
                    
                    if not kg_settings or 'knowledge_graph' not in kg_settings.settings:
                        self._initialize_default_prompts()
                        kg_settings = db.query(Settings).filter(
                            Settings.category == 'knowledge_graph'
                        ).first()
                
                # Get prompts from appropriate structure
                if kg_settings.settings.get('prompts'):
                    prompts_data = kg_settings.settings.get('prompts', {})
                else:
                    prompts_data = kg_settings.settings.get('knowledge_graph', {}).get('prompts', {})
                
                # Convert to consistent format
                if isinstance(prompts_data, list):
                    prompts = {p.get('name', p.get('prompt_type', 'unknown')): p for p in prompts_data}
                else:
                    prompts = prompts_data

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
                # Use knowledge_graph category directly
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'knowledge_graph'
                ).first()
                
                if not kg_settings:
                    kg_settings = Settings(category='knowledge_graph', settings={'prompts': []})
                    db.add(kg_settings)
                
                if 'prompts' not in kg_settings.settings:
                    kg_settings.settings['prompts'] = []
                
                prompts_list = kg_settings.settings['prompts']
                
                # Find and update the prompt
                updated = False
                for prompt in prompts_list:
                    if prompt.get('name') == prompt_name or prompt.get('prompt_type') == prompt_name:
                        prompt['prompt_template'] = new_template
                        prompt['version'] = prompt.get('version', 1) + 1
                        updated = True
                        break
                
                if updated:
                    db.commit()
                    return True
                    
                return False
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_name}: {e}")
            return False
    
    def _initialize_default_prompts(self):
        """Initialize default prompts in settings with enhanced relationship extraction"""
        try:
            db = SessionLocal()
            try:
                kg_settings = db.query(Settings).filter(
                    Settings.category == 'knowledge_graph'
                ).first()
                
                if not kg_settings:
                    kg_settings = Settings(category='knowledge_graph', settings={})
                    db.add(kg_settings)
                
                if 'prompts' not in kg_settings.settings:
                    kg_settings.settings['prompts'] = [
                        {
                            'name': 'Entity Discovery Prompt',
                            'prompt_type': 'entity_discovery',
                            'description': 'Prompt for discovering new entity types from text',
                            'prompt_template': '''You are an expert knowledge graph architect. Analyze the provided text and discover unique entity types beyond standard categories.

Text: {text}

Current accepted entity types: {existing_types}

Instructions:
1. Identify any unique entity types that don't fit standard categories (Person, Organization, Location, Event, Concept)
2. Provide a clear, singular label for each entity type
3. Include a brief description and 1-2 examples from the text
4. Rate confidence (0-1) based on clarity and frequency
5. Group similar entities under unified types
6. Focus on domain-specific or context-specific entities

Return format (JSON):
{{
    "discovered_entities": [
        {{
            "type": "UniqueEntityType",
            "description": "Brief description of what this entity represents",
            "examples": ["Example from text"],
            "confidence": 0.85,
            "frequency": 3
        }}
    ]
}}

Important: Only include high-confidence discoveries (confidence >= 0.75).''',
                            'variables': ['text', 'existing_types'],
                            'required': ['text'],
                            'confidence_threshold': 0.75
                        },
                        {
                            'name': 'Relationship Discovery Prompt',
                            'prompt_type': 'relationship_discovery',
                            'description': 'Prompt for discovering new relationship types between entities',
                            'prompt_template': '''You are an expert knowledge graph relationship designer. Discover unique relationship types between entities.

Entities to analyze: {entities}

Current accepted relationship types: {existing_types}

Instructions:
1. Identify meaningful relationship types that emerge naturally from the data
2. Use clear verb phrases (e.g., "collaborates_with", "influences", "implements")
3. Provide inverse relationships where applicable
4. Rate confidence (0-1) based on semantic clarity
5. Focus on context-specific relationships
6. Consider temporal, causal, and functional relationships

Return format (JSON):
{{
    "discovered_relationships": [
        {{
            "type": "unique_relationship",
            "description": "What this relationship represents",
            "inverse": "reverse_relationship_type",
            "examples": ["Entity1 -[relationship]-> Entity2"],
            "confidence": 0.82,
            "frequency": 2
        }}
    ]
}}

Important: Only include high-confidence discoveries (confidence >= 0.7).''',
                            'variables': ['entities', 'existing_types'],
                            'required': ['entities'],
                            'confidence_threshold': 0.7
                        },
                        {
                            'name': 'Knowledge Extraction Prompt',
                            'prompt_type': 'knowledge_extraction',
                            'description': 'Main prompt for extracting entities and relationships from text',
                            'prompt_template': '''You are an expert knowledge graph extraction system specializing in extracting complete, meaningful entities and relationships from business and technical documents.

{context_info}
{domain_guidance}

DYNAMIC SCHEMA:
Entity Types: {entity_types}
Relationship Types: {relationship_types}

EXTRACTION RULES:
1. **COMPLETE ENTITIES**: Extract full names, complete technologies, organizations
   - ✅ GOOD: "DBS Bank", "Cloud-Native Architecture", "Digital Transformation"  
   - ❌ NEVER: "Ant", "Below", "They", "CEO", "The", "This", "That"

2. **RELATIONSHIP EXTRACTION IS CRITICAL**:
   - ALWAYS extract relationships between entities that appear together
   - Use flexible entity matching - slight name variations are acceptable
   - Include organizational relationships (WORKS_FOR, PART_OF, MEMBER_OF)
   - Include business relationships (PARTNERS_WITH, COMPETES_WITH, INVESTS_IN)
   - Include conceptual relationships (IMPLEMENTS, USES, ENABLES, BASED_ON)
   - Include temporal relationships (FOUNDED_IN, ESTABLISHED_IN, OCCURRED_IN)

3. **FLEXIBLE ENTITY MATCHING**:
   - "DBS" can relate to "DBS Bank" or "DBS Technology"
   - "Singapore" can relate to "Singapore market" or "Singapore operations"
   - Use semantic understanding to connect related entities

TEXT TO ANALYZE:
{text}

OUTPUT FORMAT (JSON):
{{
    "entities": [
        {{
            "text": "exact complete text from source",
            "canonical_form": "normalized complete name",
            "type": "entity_type",
            "confidence": 0.95,
            "evidence": "supporting text snippet showing context",
            "start_char": 0,
            "end_char": 10,
            "attributes": {{"key": "value"}}
        }}
    ],
    "relationships": [
        {{
            "source_entity": "canonical name of source entity",
            "target_entity": "canonical name of target entity", 
            "relationship_type": "specific_relationship_type",
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
                "examples": ["complete example from text"],
                "confidence": 0.8
            }}
        ],
        "new_relationship_types": [
            {{
                "type": "specific_relationship",
                "description": "What this relationship represents", 
                "inverse": "inverse_type",
                "examples": ["complete example from text"]
            }}
        ]
    }},
    "reasoning": "Brief explanation of entity quality and relationship identification strategy"
}}

CRITICAL REQUIREMENT: Extract meaningful relationships between entities. If entities exist in the text, there should be relationships connecting them based on context, business logic, and semantic understanding.

Provide ONLY the JSON output without any additional text or formatting.''',
                            'variables': ['text', 'context_info', 'domain_guidance', 'entity_types', 'relationship_types'],
                            'required': ['text']
                        }
                    ]
                
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
            try:
                return template.format(**variables)
            except:
                return template
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