"""
Knowledge Graph Extraction Service

Handles entity and relationship extraction from document chunks using spaCy NLP
and LLM enhancement for building knowledge graphs in Neo4j.
"""

import asyncio
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - falling back to regex-based extraction")

from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings
from app.services.neo4j_service import get_neo4j_service
from app.document_handlers.base import ExtractedChunk

logger = logging.getLogger(__name__)

@dataclass
class ExtractedEntity:
    """Represents an extracted entity from text"""
    text: str
    label: str  # Entity type (PERSON, ORG, CONCEPT, etc.)
    start_char: int
    end_char: int
    confidence: float = 1.0
    canonical_form: str = None  # Normalized form of the entity
    
    def __post_init__(self):
        if self.canonical_form is None:
            self.canonical_form = self.text.strip().title()

@dataclass
class ExtractedRelationship:
    """Represents a relationship between two entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: str  # Sentence or context where relationship was found
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class GraphExtractionResult:
    """Result of knowledge graph extraction from a document chunk"""
    chunk_id: str
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    processing_time_ms: float
    source_metadata: Dict[str, Any]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class KnowledgeGraphExtractionService:
    """Service for extracting entities and relationships from documents"""
    
    def __init__(self):
        self.config = get_knowledge_graph_settings()
        self.nlp = None
        self._initialize_nlp_pipeline()
        
        # Common entity patterns for fallback extraction
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(?:Dr|Mr|Mrs|Ms|Professor|Prof)\. [A-Z][a-z]+ [A-Z][a-z]+\b'  # Title Name
            ],
            'ORG': [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Corporation|University|Institute)\b',
                r'\b(?:Apple|Google|Microsoft|Amazon|Meta|Tesla|SpaceX)\b'  # Known orgs
            ],
            'CONCEPT': [
                r'\b(?:artificial intelligence|machine learning|deep learning|neural network)\b',
                r'\b(?:blockchain|cryptocurrency|quantum computing|cloud computing)\b'
            ]
        }
        
        # Relationship patterns
        self.relationship_patterns = [
            {'pattern': r'(\w+(?:\s+\w+)*)\s+(?:works for|employed by|at)\s+(\w+(?:\s+\w+)*)', 'type': 'WORKS_FOR'},
            {'pattern': r'(\w+(?:\s+\w+)*)\s+(?:founded|created|established)\s+(\w+(?:\s+\w+)*)', 'type': 'FOUNDED'},
            {'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is|was)\s+(?:the|a)\s+(?:CEO|founder|president|director)\s+of\s+(\w+(?:\s+\w+)*)', 'type': 'LEADS'},
            {'pattern': r'(\w+(?:\s+\w+)*)\s+(?:uses|utilizes|implements)\s+(\w+(?:\s+\w+)*)', 'type': 'USES'},
            {'pattern': r'(\w+(?:\s+\w+)*)\s+(?:is related to|connected to|associated with)\s+(\w+(?:\s+\w+)*)', 'type': 'RELATED_TO'}
        ]
    
    def _initialize_nlp_pipeline(self):
        """Initialize the spaCy NLP pipeline"""
        global SPACY_AVAILABLE
        
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available, using fallback extraction methods")
            return
        
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy English model")
        except OSError:
            logger.warning("spaCy English model not found, attempting to use fallback methods")
            self.nlp = None
            SPACY_AVAILABLE = False
    
    async def extract_from_chunk(self, chunk: ExtractedChunk) -> GraphExtractionResult:
        """Extract entities and relationships from a single document chunk"""
        start_time = datetime.now()
        
        try:
            # Extract entities
            entities = await self._extract_entities(chunk.content)
            
            # Extract relationships
            relationships = await self._extract_relationships(chunk.content, entities)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return GraphExtractionResult(
                chunk_id=chunk.metadata.get('chunk_id', 'unknown'),
                entities=entities,
                relationships=relationships,
                processing_time_ms=processing_time,
                source_metadata=chunk.metadata,
                warnings=[]
            )
            
        except Exception as e:
            logger.error(f"Failed to extract graph data from chunk: {e}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return GraphExtractionResult(
                chunk_id=chunk.metadata.get('chunk_id', 'unknown'),
                entities=[],
                relationships=[],
                processing_time_ms=processing_time,
                source_metadata=chunk.metadata,
                warnings=[f"Extraction failed: {str(e)}"]
            )
    
    async def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text using spaCy or fallback methods"""
        entities = []
        
        if self.nlp is not None:
            # Use spaCy for entity extraction
            entities.extend(await self._extract_entities_spacy(text))
        else:
            # Use pattern-based fallback
            entities.extend(await self._extract_entities_patterns(text))
        
        # Remove duplicates and filter by confidence
        entities = self._deduplicate_entities(entities)
        entities = [e for e in entities if e.confidence >= self.config.get('extraction', {}).get('min_entity_confidence', 0.7)]
        
        return entities
    
    async def _extract_entities_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy NER"""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Map spaCy labels to our schema
                entity_type = self._map_spacy_label(ent.label_)
                
                entity = ExtractedEntity(
                    text=ent.text,
                    label=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.9  # High confidence for spaCy entities
                )
                entities.append(entity)
                
        except Exception as e:
            logger.error(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    async def _extract_entities_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns as fallback"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = ExtractedEntity(
                        text=match.group(),
                        label=entity_type,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.6  # Lower confidence for pattern matching
                    )
                    entities.append(entity)
        
        return entities
    
    async def _extract_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships between entities"""
        relationships = []
        
        # Pattern-based relationship extraction
        relationships.extend(await self._extract_relationships_patterns(text))
        
        # Entity co-occurrence relationships
        relationships.extend(await self._extract_cooccurrence_relationships(text, entities))
        
        # Remove duplicates and filter by confidence
        relationships = self._deduplicate_relationships(relationships)
        relationships = [r for r in relationships if r.confidence >= self.config.get('extraction', {}).get('min_relationship_confidence', 0.6)]
        
        return relationships
    
    async def _extract_relationships_patterns(self, text: str) -> List[ExtractedRelationship]:
        """Extract relationships using regex patterns"""
        relationships = []
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            for rel_pattern in self.relationship_patterns:
                matches = re.finditer(rel_pattern['pattern'], sentence, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        source_entity = match.group(1).strip()
                        target_entity = match.group(2).strip()
                        
                        relationship = ExtractedRelationship(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relationship_type=rel_pattern['type'],
                            confidence=0.7,
                            context=sentence
                        )
                        relationships.append(relationship)
        
        return relationships
    
    async def _extract_cooccurrence_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships based on entity co-occurrence in sentences"""
        relationships = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Find entities in this sentence
            sentence_entities = []
            for entity in entities:
                if entity.text.lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # Create relationships between co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    if entity1.label != entity2.label:  # Different types more likely to be related
                        relationship = ExtractedRelationship(
                            source_entity=entity1.canonical_form,
                            target_entity=entity2.canonical_form,
                            relationship_type='MENTIONED_WITH',
                            confidence=0.5,  # Lower confidence for co-occurrence
                            context=sentence,
                            properties={'cooccurrence': True}
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our knowledge graph schema"""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',  # Geopolitical entity
            'LOC': 'LOCATION',
            'PRODUCT': 'CONCEPT',
            'EVENT': 'EVENT',
            'WORK_OF_ART': 'CONCEPT',
            'LAW': 'CONCEPT',
            'LANGUAGE': 'CONCEPT',
            'DATE': 'TEMPORAL',
            'TIME': 'TEMPORAL',
            'PERCENT': 'NUMERIC',
            'MONEY': 'NUMERIC',
            'QUANTITY': 'NUMERIC',
            'CARDINAL': 'NUMERIC',
            'ORDINAL': 'NUMERIC'
        }
        return mapping.get(spacy_label, 'CONCEPT')
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities based on canonical form"""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.canonical_form.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
    def _deduplicate_relationships(self, relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Remove duplicate relationships"""
        seen = set()
        deduplicated = []
        
        for rel in relationships:
            key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)
        
        return deduplicated
    
    async def store_in_neo4j(self, extraction_result: GraphExtractionResult, document_id: str) -> Dict[str, Any]:
        """Store extracted entities and relationships in Neo4j"""
        try:
            neo4j_service = get_neo4j_service()
            
            if not neo4j_service.is_enabled():
                return {'success': False, 'error': 'Neo4j service is not enabled'}
            
            stored_entities = []
            stored_relationships = []
            
            # Store entities
            for entity in extraction_result.entities:
                entity_properties = {
                    'name': entity.canonical_form,
                    'original_text': entity.text,
                    'type': entity.label,
                    'confidence': entity.confidence,
                    'document_id': document_id,
                    'chunk_id': extraction_result.chunk_id,
                    'created_at': datetime.now().isoformat()
                }
                
                entity_id = neo4j_service.create_entity(entity.label, entity_properties)
                if entity_id:
                    stored_entities.append(entity_id)
            
            # Store relationships
            for relationship in extraction_result.relationships:
                # Find entity IDs by name
                source_entities = neo4j_service.find_entities(
                    properties={'name': relationship.source_entity}
                )
                target_entities = neo4j_service.find_entities(
                    properties={'name': relationship.target_entity}
                )
                
                if source_entities and target_entities:
                    source_id = source_entities[0].get('id')
                    target_id = target_entities[0].get('id')
                    
                    if source_id and target_id:
                        rel_properties = {
                            'confidence': relationship.confidence,
                            'context': relationship.context,
                            'document_id': document_id,
                            'chunk_id': extraction_result.chunk_id,
                            'created_at': datetime.now().isoformat()
                        }
                        rel_properties.update(relationship.properties)
                        
                        success = neo4j_service.create_relationship(
                            source_id, target_id, relationship.relationship_type, rel_properties
                        )
                        if success:
                            stored_relationships.append({
                                'source': source_id,
                                'target': target_id,
                                'type': relationship.relationship_type
                            })
            
            return {
                'success': True,
                'entities_stored': len(stored_entities),
                'relationships_stored': len(stored_relationships),
                'processing_time_ms': extraction_result.processing_time_ms
            }
            
        except Exception as e:
            logger.error(f"Failed to store graph data in Neo4j: {e}")
            return {'success': False, 'error': str(e)}

# Singleton instance
_knowledge_graph_service: Optional[KnowledgeGraphExtractionService] = None

def get_knowledge_graph_service() -> KnowledgeGraphExtractionService:
    """Get or create knowledge graph extraction service singleton"""
    global _knowledge_graph_service
    if _knowledge_graph_service is None:
        _knowledge_graph_service = KnowledgeGraphExtractionService()
    return _knowledge_graph_service