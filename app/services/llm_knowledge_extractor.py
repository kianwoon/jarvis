"""
LLM-Enhanced Knowledge Graph Extractor

Provides LLM-powered entity and relationship extraction with context understanding,
domain awareness, and confidence scoring for improved knowledge graph quality.
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.core.llm_settings_cache import get_llm_settings
from app.services.knowledge_graph_service import ExtractedEntity, ExtractedRelationship

logger = logging.getLogger(__name__)

@dataclass
class LLMExtractionResult:
    """Result of LLM-based entity and relationship extraction"""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    confidence_score: float
    reasoning: str
    processing_time_ms: float
    llm_model_used: str
    extraction_metadata: Dict[str, Any]

class LLMKnowledgeExtractor:
    """LLM-powered knowledge extraction service with advanced reasoning"""
    
    def __init__(self):
        self.kg_settings = get_knowledge_graph_settings()
        self.llm_settings = get_llm_settings()
        self.model_config = self._get_extraction_model_config()
        
        # Enhanced entity types with hierarchical structure
        self.hierarchical_entity_types = {
            'PERSON': {
                'subtypes': ['EXECUTIVE', 'RESEARCHER', 'POLITICIAN', 'ARTIST', 'ENTREPRENEUR'],
                'attributes': ['role', 'organization', 'expertise', 'nationality']
            },
            'ORGANIZATION': {
                'subtypes': ['COMPANY', 'UNIVERSITY', 'GOVERNMENT', 'NONPROFIT', 'STARTUP'],
                'attributes': ['industry', 'size', 'location', 'founded_year']
            },
            'CONCEPT': {
                'subtypes': ['TECHNOLOGY', 'METHODOLOGY', 'THEORY', 'PRODUCT', 'SERVICE'],
                'attributes': ['domain', 'complexity', 'maturity', 'application']
            },
            'LOCATION': {
                'subtypes': ['COUNTRY', 'CITY', 'REGION', 'FACILITY', 'VIRTUAL'],
                'attributes': ['type', 'population', 'coordinates', 'significance']
            },
            'EVENT': {
                'subtypes': ['MEETING', 'LAUNCH', 'ACQUISITION', 'RESEARCH', 'PUBLICATION'],
                'attributes': ['date', 'participants', 'outcome', 'significance']
            }
        }
        
        # Sophisticated relationship taxonomy
        self.relationship_taxonomy = {
            'EMPLOYMENT': ['WORKS_FOR', 'LEADS', 'MANAGES', 'REPORTS_TO', 'ADVISES'],
            'BUSINESS': ['OWNS', 'PARTNERS_WITH', 'COMPETES_WITH', 'SUPPLIES_TO', 'INVESTS_IN'],
            'ACADEMIC': ['RESEARCHES', 'COLLABORATES_ON', 'PUBLISHES_WITH', 'SUPERVISES', 'STUDIES'],
            'TECHNICAL': ['USES', 'DEVELOPS', 'IMPLEMENTS', 'BASED_ON', 'INTEGRATES_WITH'],
            'SPATIAL': ['LOCATED_IN', 'OPERATES_IN', 'HEADQUARTERED_IN', 'SERVES', 'COVERS'],
            'TEMPORAL': ['PRECEDES', 'FOLLOWS', 'CONCURRENT_WITH', 'TRIGGERS', 'ENABLES'],
            'CONCEPTUAL': ['IS_A', 'PART_OF', 'CONTAINS', 'RELATED_TO', 'INFLUENCES']
        }
    
    def _get_extraction_model_config(self) -> Dict[str, Any]:
        """Get LLM configuration optimized for knowledge extraction"""
        config = {
            'model': self.kg_settings.get('model', 'qwen3:30b-a3b-q4_K_M'),
            'temperature': 0.1,  # Low temperature for factual extraction
            'max_tokens': self.kg_settings.get('max_tokens', 4096),
            'model_server': self.kg_settings.get('model_server', 'http://localhost:11434')
        }
        return config
    
    async def extract_with_llm(self, text: str, context: Optional[Dict[str, Any]] = None,
                              domain_hints: Optional[List[str]] = None) -> LLMExtractionResult:
        """Extract entities and relationships using LLM with contextual understanding"""
        start_time = datetime.now()
        
        try:
            # Build sophisticated extraction prompt
            extraction_prompt = self._build_extraction_prompt(text, context, domain_hints)
            
            # Call LLM for extraction
            llm_response = await self._call_llm_for_extraction(extraction_prompt)
            
            # Parse and validate LLM response
            parsed_result = self._parse_llm_response(llm_response)
            
            # Enhance entities with hierarchical classification
            enhanced_entities = self._enhance_entities_with_hierarchy(parsed_result.get('entities', []))
            
            # Validate and score relationships
            validated_relationships = self._validate_and_score_relationships(
                parsed_result.get('relationships', []), enhanced_entities
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Generate overall confidence score
            confidence_score = self._calculate_overall_confidence(
                enhanced_entities, validated_relationships
            )
            
            return LLMExtractionResult(
                entities=enhanced_entities,
                relationships=validated_relationships,
                confidence_score=confidence_score,
                reasoning=parsed_result.get('reasoning', 'LLM-based extraction completed'),
                processing_time_ms=processing_time,
                llm_model_used=self.model_config['model'],
                extraction_metadata={
                    'domain_hints': domain_hints or [],
                    'context_provided': context is not None,
                    'text_length': len(text),
                    'total_extractions': len(enhanced_entities) + len(validated_relationships)
                }
            )
            
        except Exception as e:
            logger.error(f"LLM knowledge extraction failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return LLMExtractionResult(
                entities=[],
                relationships=[],
                confidence_score=0.0,
                reasoning=f"Extraction failed: {str(e)}",
                processing_time_ms=processing_time,
                llm_model_used=self.model_config['model'],
                extraction_metadata={'error': str(e)}
            )
    
    def _build_extraction_prompt(self, text: str, context: Optional[Dict[str, Any]] = None,
                                domain_hints: Optional[List[str]] = None) -> str:
        """Build sophisticated extraction prompt with context and domain awareness"""
        
        # Context information
        context_info = ""
        if context:
            context_info = f"""
CONTEXT INFORMATION:
- Document type: {context.get('document_type', 'unknown')}
- Source: {context.get('source', 'unknown')}
- Date: {context.get('date', 'unknown')}
- Domain: {context.get('domain', 'general')}
"""
        
        # Domain-specific guidance
        domain_guidance = ""
        if domain_hints:
            domain_guidance = f"""
DOMAIN FOCUS: Pay special attention to {', '.join(domain_hints)} related entities and relationships.
"""
        
        # Build comprehensive prompt
        prompt = f"""You are an expert knowledge graph extraction system with deep contextual understanding. Your task is to extract entities and relationships from the provided text with high precision and semantic awareness.

{context_info}
{domain_guidance}

HIERARCHICAL ENTITY TYPES TO CONSIDER:
{json.dumps(self.hierarchical_entity_types, indent=2)}

RELATIONSHIP CATEGORIES:
{json.dumps(self.relationship_taxonomy, indent=2)}

EXTRACTION GUIDELINES:
1. Extract entities with their most specific subtype when possible
2. Include entity attributes when clearly mentioned
3. Focus on explicit relationships with clear evidence in the text
4. Provide confidence scores (0.0-1.0) based on textual evidence
5. Include reasoning for each extraction decision
6. Consider context and implicit relationships
7. Avoid over-extraction of weak or speculative connections

TEXT TO ANALYZE:
{text}

OUTPUT FORMAT (JSON):
{{
    "entities": [
        {{
            "text": "exact text from source",
            "canonical_form": "normalized name",
            "type": "main type (PERSON, ORGANIZATION, etc.)",
            "subtype": "specific subtype if applicable",
            "attributes": {{"key": "value"}},
            "confidence": 0.95,
            "evidence": "supporting text snippet",
            "start_char": 0,
            "end_char": 10
        }}
    ],
    "relationships": [
        {{
            "source_entity": "canonical name of source",
            "target_entity": "canonical name of target", 
            "relationship_type": "specific relationship type",
            "confidence": 0.85,
            "evidence": "supporting text snippet",
            "context": "broader context of relationship",
            "temporal_info": "when this relationship existed/exists",
            "attributes": {{"key": "value"}}
        }}
    ],
    "reasoning": "Brief explanation of extraction approach and key decisions",
    "confidence_notes": "Explanation of confidence scoring rationale"
}}

Provide ONLY the JSON output without any additional text or formatting."""

        return prompt
    
    async def _call_llm_for_extraction(self, prompt: str) -> str:
        """Call LLM API for knowledge extraction"""
        try:
            # This would integrate with your existing LLM service
            # For now, implementing a basic HTTP client call
            import aiohttp
            
            payload = {
                "model": self.model_config['model'],
                "prompt": prompt,
                "temperature": self.model_config['temperature'],
                "max_tokens": self.model_config['max_tokens'],
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.model_config['model_server']}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('response', '')
                    else:
                        raise Exception(f"LLM API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response"""
        try:
            # Clean response - remove any non-JSON text
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            parsed = json.loads(response)
            
            # Validate required fields
            if not isinstance(parsed.get('entities'), list):
                parsed['entities'] = []
            if not isinstance(parsed.get('relationships'), list):
                parsed['relationships'] = []
                
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {'entities': [], 'relationships': [], 'reasoning': 'Parse error'}
    
    def _enhance_entities_with_hierarchy(self, raw_entities: List[Dict[str, Any]]) -> List[ExtractedEntity]:
        """Convert raw entities to ExtractedEntity objects with hierarchical enhancement"""
        enhanced_entities = []
        
        for raw_entity in raw_entities:
            try:
                # Extract basic information
                text = raw_entity.get('text', '')
                canonical_form = raw_entity.get('canonical_form', text).strip().title()
                entity_type = raw_entity.get('type', 'CONCEPT').upper()
                subtype = raw_entity.get('subtype', '').upper()
                confidence = float(raw_entity.get('confidence', 0.7))
                
                # Validate entity type
                if entity_type not in self.hierarchical_entity_types:
                    entity_type = 'CONCEPT'
                
                # Create enhanced entity
                entity = ExtractedEntity(
                    text=text,
                    label=entity_type,
                    start_char=raw_entity.get('start_char', 0),
                    end_char=raw_entity.get('end_char', len(text)),
                    confidence=confidence,
                    canonical_form=canonical_form
                )
                
                enhanced_entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to process entity {raw_entity}: {e}")
                continue
        
        return enhanced_entities
    
    def _validate_and_score_relationships(self, raw_relationships: List[Dict[str, Any]], 
                                        entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Validate relationships and enhance with confidence scoring"""
        validated_relationships = []
        entity_names = {e.canonical_form.lower() for e in entities}
        
        for raw_rel in raw_relationships:
            try:
                source = raw_rel.get('source_entity', '').strip()
                target = raw_rel.get('target_entity', '').strip()
                rel_type = raw_rel.get('relationship_type', '').upper()
                confidence = float(raw_rel.get('confidence', 0.5))
                
                # Skip if entities not found in extracted entities
                if (source.lower() not in entity_names or 
                    target.lower() not in entity_names):
                    logger.debug(f"Skipping relationship - entities not found: {source} -> {target}")
                    continue
                
                # Validate relationship type
                if not self._is_valid_relationship_type(rel_type):
                    rel_type = 'RELATED_TO'
                
                # Create relationship with enhanced properties
                relationship = ExtractedRelationship(
                    source_entity=source,
                    target_entity=target,
                    relationship_type=rel_type,
                    confidence=confidence,
                    context=raw_rel.get('context', raw_rel.get('evidence', '')),
                    properties={
                        'llm_extracted': True,
                        'evidence': raw_rel.get('evidence', ''),
                        'temporal_info': raw_rel.get('temporal_info', ''),
                        'attributes': raw_rel.get('attributes', {}),
                        'reasoning': raw_rel.get('reasoning', '')
                    }
                )
                
                validated_relationships.append(relationship)
                
            except Exception as e:
                logger.warning(f"Failed to process relationship {raw_rel}: {e}")
                continue
        
        return validated_relationships
    
    def _is_valid_relationship_type(self, rel_type: str) -> bool:
        """Check if relationship type is valid in our taxonomy"""
        for category, types in self.relationship_taxonomy.items():
            if rel_type in types:
                return True
        return False
    
    def _calculate_overall_confidence(self, entities: List[ExtractedEntity], 
                                    relationships: List[ExtractedRelationship]) -> float:
        """Calculate overall confidence score for the extraction"""
        if not entities and not relationships:
            return 0.0
        
        total_confidence = 0.0
        total_items = 0
        
        # Entity confidence
        for entity in entities:
            total_confidence += entity.confidence
            total_items += 1
        
        # Relationship confidence
        for rel in relationships:
            total_confidence += rel.confidence
            total_items += 1
        
        return total_confidence / total_items if total_items > 0 else 0.0

# Singleton instance
_llm_extractor: Optional[LLMKnowledgeExtractor] = None

def get_llm_knowledge_extractor() -> LLMKnowledgeExtractor:
    """Get or create LLM knowledge extractor singleton"""
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = LLMKnowledgeExtractor()
    return _llm_extractor