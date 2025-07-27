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
        
        # Enhanced relationship patterns with better coverage and entity name handling
        self.relationship_patterns = [
            # Development/Creation (more comprehensive)
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:developed|created|built|designed|founded|established|started|originated)\s+(?:by|from)?\s*([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'DEVELOPED_BY'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:originally\s+)?(?:developed|created|built|designed)\s+by\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'DEVELOPED_BY'},
            
            # Ownership/Control
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:owns|controls|manages|operates)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'OWNS'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:is|was)\s+(?:owned|controlled|managed|operated)\s+by\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'OWNED_BY'},
            
            # Employment/Organization
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:works for|employed by|works at)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'WORKS_FOR'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:is|was)\s+(?:the|a|an)?\s*(?:CEO|founder|president|director|manager|leader|CTO|CIO)\s+(?:of|at)?\s*([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'LEADS'},
            
            # Usage/Implementation  
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:uses|utilizes|implements|employs|applies|adopts)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'USES'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:based on|built on|powered by|using|running on)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'BASED_ON'},
            
            # Business relationships
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:partners with|collaborates with|works with)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'PARTNERS_WITH'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:competes with|rivals)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'COMPETES_WITH'},
            
            # Technical relationships
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:includes|contains|has|features|provides)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'CONTAINS'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:part of|component of|element of|module of)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'PART_OF'},
            
            # Location/Geographic
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:in|at|from|located in|based in|headquartered in)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'LOCATED_IN'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:operates in|serves|covers)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'OPERATES_IN'},
            
            # Temporal
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:during|before|after|since|until)\s+([A-Za-z0-9\s\n\'\u2019]+)', 'type': 'TEMPORAL'},
            
            # Classification/Type relationships
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:is a|is an|is the)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'IS_A'},
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:specializes in|focuses on|known for)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'SPECIALIZES_IN'},
            
            # Association (lower priority, more generic)
            {'pattern': r'([A-Za-z][A-Za-z0-9\s\n\'\u2019]+?)\s+(?:and|with|alongside)\s+([A-Za-z][A-Za-z0-9\s\n\'\u2019]+)', 'type': 'ASSOCIATED_WITH'},
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
        """Extract entities and relationships from a single document chunk with LLM enhancement"""
        start_time = datetime.now()
        
        try:
            # Check if LLM enhancement is enabled
            use_llm_enhancement = self.config.get('extraction', {}).get('enable_llm_enhancement', False)
            
            if use_llm_enhancement:
                # Use LLM-enhanced extraction
                entities, relationships = await self._extract_with_llm_enhancement(chunk)
            else:
                # Use traditional extraction methods
                entities = await self._extract_entities(chunk.content)
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
    
    async def _extract_with_llm_enhancement(self, chunk: ExtractedChunk) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """Extract entities and relationships using LLM enhancement with fallback"""
        try:
            # Import here to avoid circular imports
            from app.services.llm_knowledge_extractor import get_llm_knowledge_extractor
            
            llm_extractor = get_llm_knowledge_extractor()
            
            # Build context from chunk metadata
            context = {
                'document_type': chunk.metadata.get('file_type', 'unknown'),
                'source': chunk.metadata.get('filename', 'unknown'),
                'chunk_id': chunk.metadata.get('chunk_id', 'unknown'),
                'domain': self._infer_domain_from_content(chunk.content)
            }
            
            # Extract domain hints from content
            domain_hints = self._extract_domain_hints(chunk.content)
            
            # Use LLM for extraction
            llm_result = await llm_extractor.extract_with_llm(
                text=chunk.content,
                context=context,
                domain_hints=domain_hints
            )
            
            logger.info(f"🤖 LLM EXTRACTION: {len(llm_result.entities)} entities, {len(llm_result.relationships)} relationships (confidence: {llm_result.confidence_score:.3f})")
            
            # If LLM extraction has low confidence, supplement with traditional methods
            if llm_result.confidence_score < 0.6:
                logger.warning(f"🤖 LLM confidence low ({llm_result.confidence_score:.3f}), supplementing with traditional extraction")
                
                # Get traditional extractions
                traditional_entities = await self._extract_entities(chunk.content)
                traditional_relationships = await self._extract_relationships(chunk.content, traditional_entities)
                
                # Merge results intelligently
                combined_entities = self._merge_entity_extractions(llm_result.entities, traditional_entities)
                combined_relationships = self._merge_relationship_extractions(
                    llm_result.relationships, traditional_relationships
                )
                
                return combined_entities, combined_relationships
            
            return llm_result.entities, llm_result.relationships
            
        except Exception as e:
            logger.error(f"LLM enhancement failed, falling back to traditional extraction: {e}")
            # Fallback to traditional methods
            entities = await self._extract_entities(chunk.content)
            relationships = await self._extract_relationships(chunk.content, entities)
            return entities, relationships
    
    def _infer_domain_from_content(self, content: str) -> str:
        """Infer domain/industry from content for contextual extraction"""
        content_lower = content.lower()
        
        # Domain keywords mapping
        domain_keywords = {
            'technology': ['software', 'ai', 'machine learning', 'algorithm', 'api', 'cloud', 'database'],
            'finance': ['investment', 'funding', 'revenue', 'profit', 'market', 'financial', 'banking'],
            'healthcare': ['medical', 'patient', 'treatment', 'hospital', 'clinical', 'therapeutic'],
            'academic': ['research', 'university', 'study', 'paper', 'academic', 'scholarly', 'publication'],
            'business': ['company', 'corporation', 'management', 'strategy', 'customer', 'market'],
            'legal': ['law', 'legal', 'court', 'regulation', 'compliance', 'contract']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    def _extract_domain_hints(self, content: str) -> List[str]:
        """Extract domain-specific hints from content"""
        hints = []
        content_lower = content.lower()
        
        # Look for specific patterns that indicate domain focus
        if any(term in content_lower for term in ['ceo', 'founder', 'startup', 'company']):
            hints.append('business_leadership')
        
        if any(term in content_lower for term in ['research', 'study', 'analysis', 'findings']):
            hints.append('research_academic')
        
        if any(term in content_lower for term in ['technology', 'software', 'platform', 'system']):
            hints.append('technology')
        
        if any(term in content_lower for term in ['investment', 'funding', 'valuation', 'revenue']):
            hints.append('finance_business')
        
        return hints
    
    def _merge_entity_extractions(self, llm_entities: List[ExtractedEntity], 
                                traditional_entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Intelligently merge LLM and traditional entity extractions"""
        merged = []
        llm_entity_map = {e.canonical_form.lower(): e for e in llm_entities}
        
        # Start with LLM entities (higher quality)
        merged.extend(llm_entities)
        
        # Add traditional entities that don't conflict
        for trad_entity in traditional_entities:
            key = trad_entity.canonical_form.lower()
            if key not in llm_entity_map:
                # Add non-conflicting traditional entity with lower confidence
                trad_entity.confidence *= 0.8  # Reduce confidence for traditional method
                merged.append(trad_entity)
        
        return merged
    
    def _merge_relationship_extractions(self, llm_relationships: List[ExtractedRelationship],
                                      traditional_relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Intelligently merge LLM and traditional relationship extractions"""
        merged = []
        llm_rel_map = set()
        
        # Create map of LLM relationships
        for rel in llm_relationships:
            key = (rel.source_entity.lower(), rel.target_entity.lower(), rel.relationship_type)
            llm_rel_map.add(key)
            merged.append(rel)
        
        # Add non-conflicting traditional relationships
        for trad_rel in traditional_relationships:
            key = (trad_rel.source_entity.lower(), trad_rel.target_entity.lower(), trad_rel.relationship_type)
            if key not in llm_rel_map:
                # Add non-conflicting traditional relationship with lower confidence
                trad_rel.confidence *= 0.7  # Reduce confidence for traditional method
                merged.append(trad_rel)
        
        return merged
    
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
        """Extract relationships between entities with intelligent prioritization"""
        relationships = []
        
        # 1. Pattern-based relationship extraction (HIGHEST PRIORITY)
        pattern_relationships = await self._extract_relationships_patterns(text)
        relationships.extend(pattern_relationships)
        
        # 2. Smart co-occurrence relationships (MEDIUM PRIORITY) 
        cooccurrence_relationships = await self._extract_cooccurrence_relationships(text, entities)
        relationships.extend(cooccurrence_relationships)
        
        # 3. Fallback: Document-level relationships (LOWEST PRIORITY)
        if len(relationships) == 0 and len(entities) > 1:
            logger.warning(f"🔥 FALLBACK: No relationships found, creating document-level relationships between {len(entities)} entities")
            fallback_relationships = await self._create_document_level_relationships(entities)
            relationships.extend(fallback_relationships)
        
        # 4. Intelligent deduplication with priority preservation
        relationships = self._deduplicate_relationships_with_priority(relationships)
        
        # 5. Filter by confidence (lowered threshold for more relationships)
        min_confidence = self.config.get('extraction', {}).get('min_relationship_confidence', 0.3)
        relationships = [r for r in relationships if r.confidence >= min_confidence]
        
        # 6. Log relationship type distribution for analysis
        relationship_types = {}
        for rel in relationships:
            relationship_types[rel.relationship_type] = relationship_types.get(rel.relationship_type, 0) + 1
        
        logger.info(f"🔥 RELATIONSHIP EXTRACTION COMPLETE:")
        logger.info(f"   - Pattern-based: {len(pattern_relationships)}")
        logger.info(f"   - Co-occurrence: {len(cooccurrence_relationships)}")
        logger.info(f"   - Total after deduplication: {len(relationships)}")
        logger.info(f"   - Relationship types: {relationship_types}")
        
        return relationships
    
    async def _extract_relationships_patterns(self, text: str) -> List[ExtractedRelationship]:
        """Extract relationships using regex patterns"""
        relationships = []
        sentences = re.split(r'[.!?]+', text)
        
        logger.debug(f"🔥 PATTERN EXTRACTION: Processing {len(sentences)} sentences with {len(self.relationship_patterns)} patterns")
        
        for sentence_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            for pattern_idx, rel_pattern in enumerate(self.relationship_patterns):
                matches = re.finditer(rel_pattern['pattern'], sentence, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        source_entity = match.group(1).strip()
                        target_entity = match.group(2).strip()
                        
                        # Skip if entities are the same
                        if source_entity.lower() == target_entity.lower():
                            continue
                        
                        relationship = ExtractedRelationship(
                            source_entity=source_entity,
                            target_entity=target_entity,
                            relationship_type=rel_pattern['type'],
                            confidence=0.7,
                            context=sentence[:200],  # Limit context length
                            properties={'pattern_matched': rel_pattern['pattern'][:50]}
                        )
                        relationships.append(relationship)
                        logger.debug(f"🔥 PATTERN MATCH: {source_entity} -> {target_entity} ({rel_pattern['type']})")
        
        logger.info(f"🔥 PATTERN EXTRACTION: Generated {len(relationships)} pattern-based relationships")
        return relationships
    
    async def _extract_cooccurrence_relationships(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Extract relationships based on entity co-occurrence with smart semantic inference"""
        relationships = []
        sentences = re.split(r'[.!?]+', text)
        
        logger.debug(f"🔥 CO-OCCURRENCE: Processing {len(sentences)} sentences with {len(entities)} entities")
        
        for sentence_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 15:  # Reduced from 20 to 15 for more coverage
                continue
            
            # Find entities in this sentence with better matching
            sentence_entities = []
            for entity in entities:
                # Try multiple matching strategies
                entity_found = False
                
                # Exact match
                if entity.text.lower() in sentence.lower():
                    sentence_entities.append(entity)
                    entity_found = True
                # Canonical form match
                elif entity.canonical_form.lower() in sentence.lower():
                    sentence_entities.append(entity)
                    entity_found = True
                # Word boundary match for better precision
                elif re.search(r'\b' + re.escape(entity.text.lower()) + r'\b', sentence.lower()):
                    sentence_entities.append(entity)
                    entity_found = True
                    
                if entity_found:
                    logger.debug(f"🔥 Found entity '{entity.text}' in sentence {sentence_idx}")
            
            # Skip sentences with only one entity
            if len(sentence_entities) < 2:
                continue
            
            # Create smarter relationships between co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    # Infer semantic relationship based on context and entity types
                    relationship_info = self._infer_semantic_relationship(
                        entity1, entity2, sentence, sentence_idx
                    )
                    
                    if relationship_info:
                        relationship = ExtractedRelationship(
                            source_entity=entity1.canonical_form,
                            target_entity=entity2.canonical_form,
                            relationship_type=relationship_info['type'],
                            confidence=relationship_info['confidence'],
                            context=sentence[:200],
                            properties={
                                'cooccurrence': True,
                                'sentence_index': sentence_idx,
                                'inference_reason': relationship_info['reason'],
                                'semantic_context': relationship_info.get('context_clues', [])
                            }
                        )
                        relationships.append(relationship)
                        logger.debug(f"🔥 Created semantic relationship: {entity1.canonical_form} -> {entity2.canonical_form} ({relationship_info['type']}) - {relationship_info['reason']}")
        
        logger.info(f"🔥 CO-OCCURRENCE: Generated {len(relationships)} semantic co-occurrence relationships")
        return relationships
    
    def _infer_semantic_relationship(self, entity1: ExtractedEntity, entity2: ExtractedEntity, 
                                   sentence: str, sentence_idx: int) -> Optional[Dict[str, Any]]:
        """Infer semantic relationship between two entities based on context and types"""
        sentence_lower = sentence.lower()
        entity1_name = entity1.canonical_form.lower()
        entity2_name = entity2.canonical_form.lower()
        
        # Get positions of entities in sentence for proximity analysis
        pos1 = sentence_lower.find(entity1_name)
        pos2 = sentence_lower.find(entity2_name)
        
        if pos1 == -1 or pos2 == -1:
            # Fallback to generic relationship
            return {
                'type': 'MENTIONED_WITH',
                'confidence': 0.3,
                'reason': 'entities co-occur in same sentence',
                'context_clues': []
            }
        
        # Determine order and distance
        first_entity, second_entity = (entity1, entity2) if pos1 < pos2 else (entity2, entity1)
        first_name, second_name = (entity1_name, entity2_name) if pos1 < pos2 else (entity2_name, entity1_name)
        distance = abs(pos2 - pos1)
        
        # Extract context between and around entities
        start_pos = min(pos1, pos2)
        end_pos = max(pos1, pos2) + max(len(entity1_name), len(entity2_name))
        context_snippet = sentence_lower[max(0, start_pos-20):end_pos+20]
        
        # Smart relationship inference based on entity types and context
        type1, type2 = entity1.label, entity2.label
        
        # 1. PERSON + ORGANIZATION relationships
        if (type1 == 'PERSON' and type2 == 'ORGANIZATION') or (type1 == 'ORGANIZATION' and type2 == 'PERSON'):
            person_entity = entity1 if type1 == 'PERSON' else entity2
            org_entity = entity2 if type1 == 'PERSON' else entity1
            
            # Check for employment/leadership indicators
            employment_indicators = ['works for', 'employed by', 'works at', 'employee of', 'staff at', 'team at']
            leadership_indicators = ['ceo of', 'founder of', 'president of', 'director of', 'manager of', 'leads', 'head of']
            
            for indicator in leadership_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'LEADS',
                        'confidence': 0.8,
                        'reason': f'leadership relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
            
            for indicator in employment_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'WORKS_FOR',
                        'confidence': 0.7,
                        'reason': f'employment relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
        
        # 2. ORGANIZATION + ORGANIZATION relationships
        elif type1 == 'ORGANIZATION' and type2 == 'ORGANIZATION':
            partnership_indicators = ['partners with', 'collaborates with', 'alliance with', 'joint venture']
            competition_indicators = ['competes with', 'rival', 'versus', 'vs', 'against']
            ownership_indicators = ['owns', 'acquired', 'subsidiary of', 'parent company']
            
            for indicator in ownership_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'OWNS',
                        'confidence': 0.8,
                        'reason': f'ownership relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
            
            for indicator in partnership_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'PARTNERS_WITH',
                        'confidence': 0.7,
                        'reason': f'partnership relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
            
            for indicator in competition_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'COMPETES_WITH',
                        'confidence': 0.6,
                        'reason': f'competition relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
        
        # 3. CONCEPT + anything relationships (usage, implementation)
        elif type1 == 'CONCEPT' or type2 == 'CONCEPT':
            concept_entity = entity1 if type1 == 'CONCEPT' else entity2
            other_entity = entity2 if type1 == 'CONCEPT' else entity1
            
            usage_indicators = ['uses', 'implements', 'applies', 'adopts', 'based on', 'powered by', 'built with']
            
            for indicator in usage_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'USES',
                        'confidence': 0.7,
                        'reason': f'usage relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
        
        # 4. LOCATION relationships
        elif type1 == 'LOCATION' or type2 == 'LOCATION':
            location_indicators = ['in', 'at', 'from', 'located in', 'based in', 'operates in']
            
            for indicator in location_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'LOCATED_IN',
                        'confidence': 0.7,
                        'reason': f'location relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
        
        # 5. Temporal relationships (events, dates)
        elif type1 == 'TEMPORAL' or type2 == 'TEMPORAL':
            temporal_indicators = ['during', 'before', 'after', 'since', 'until', 'when']
            
            for indicator in temporal_indicators:
                if indicator in context_snippet:
                    return {
                        'type': 'TEMPORAL',
                        'confidence': 0.6,
                        'reason': f'temporal relationship inferred from "{indicator}"',
                        'context_clues': [indicator]
                    }
        
        # 6. Proximity-based relationships (very close entities likely more related)
        if distance < 30:  # Very close entities
            # Check for connecting words that imply relationships
            connecting_words = ['and', 'with', 'alongside', 'together', 'both']
            
            for word in connecting_words:
                if word in context_snippet:
                    return {
                        'type': 'ASSOCIATED_WITH',
                        'confidence': 0.5,
                        'reason': f'close association inferred from proximity and "{word}"',
                        'context_clues': [word]
                    }
        
        # 7. Same type entities (often related in some way)
        if type1 == type2 and type1 != 'CONCEPT':
            return {
                'type': 'RELATED_TO',
                'confidence': 0.4,
                'reason': f'same type entities ({type1}) likely related',
                'context_clues': [f'both_{type1}']
            }
        
        # 8. Fallback: Create a more informative generic relationship
        return {
            'type': 'CONTEXTUALLY_RELATED',
            'confidence': 0.3,
            'reason': f'{type1}-{type2} co-occurrence, distance: {distance}',
            'context_clues': [f'type_pair_{type1}_{type2}']
        }
    
    async def _create_document_level_relationships(self, entities: List[ExtractedEntity]) -> List[ExtractedRelationship]:
        """Create basic relationships between all entities in the same document as fallback"""
        relationships = []
        
        # Create relationships between every pair of entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Skip if entities are too similar
                if entity1.canonical_form.lower() == entity2.canonical_form.lower():
                    continue
                
                # Create bidirectional relationships for better connectivity
                relationship = ExtractedRelationship(
                    source_entity=entity1.canonical_form,
                    target_entity=entity2.canonical_form,
                    relationship_type='DOCUMENT_RELATED',
                    confidence=0.4,  # Lower confidence since this is a fallback
                    context=f"Both entities appear in the same document",
                    properties={'fallback': True, 'document_level': True}
                )
                relationships.append(relationship)
                
                logger.debug(f"🔥 FALLBACK: Created document-level relationship: {entity1.canonical_form} -> {entity2.canonical_form}")
        
        logger.info(f"🔥 FALLBACK: Created {len(relationships)} document-level relationships")
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
    
    def _deduplicate_relationships_with_priority(self, relationships: List[ExtractedRelationship]) -> List[ExtractedRelationship]:
        """Remove duplicate relationships keeping higher priority ones"""
        # Define relationship type priorities (higher number = higher priority)
        type_priorities = {
            # Pattern-based relationships (highest priority)
            'DEVELOPED_BY': 90, 'OWNS': 88, 'OWNED_BY': 88, 'WORKS_FOR': 85, 'LEADS': 87, 
            'USES': 80, 'BASED_ON': 82, 'PARTNERS_WITH': 75, 'COMPETES_WITH': 75,
            'CONTAINS': 70, 'PART_OF': 72, 'LOCATED_IN': 78, 'OPERATES_IN': 76,
            'TEMPORAL': 65, 'IS_A': 68, 'SPECIALIZES_IN': 68,
            
            # Smart co-occurrence relationships (medium priority)
            'CONTEXTUALLY_RELATED': 40, 'RELATED_TO': 35, 'ASSOCIATED_WITH': 45,
            
            # Generic relationships (lower priority)
            'MENTIONED_WITH': 20, 'DOCUMENT_RELATED': 10
        }
        
        # Group relationships by entity pair
        entity_pairs = {}
        for rel in relationships:
            # Create bidirectional key to handle both directions
            key1 = (rel.source_entity.lower(), rel.target_entity.lower())
            key2 = (rel.target_entity.lower(), rel.source_entity.lower())
            
            # Use consistent key (alphabetically sorted)
            key = key1 if key1[0] <= key1[1] else key2
            
            if key not in entity_pairs:
                entity_pairs[key] = []
            entity_pairs[key].append(rel)
        
        # For each entity pair, keep the highest priority relationship
        deduplicated = []
        for pair_key, pair_relationships in entity_pairs.items():
            if len(pair_relationships) == 1:
                deduplicated.append(pair_relationships[0])
            else:
                # Sort by priority (pattern match flag, then type priority, then confidence)
                def get_priority(rel):
                    type_priority = type_priorities.get(rel.relationship_type, 0)
                    pattern_bonus = 20 if rel.properties.get('pattern_matched') else 0
                    confidence_bonus = rel.confidence * 10
                    return type_priority + pattern_bonus + confidence_bonus
                
                best_relationship = max(pair_relationships, key=get_priority)
                deduplicated.append(best_relationship)
                
                # Log deduplication decisions
                if len(pair_relationships) > 1:
                    logger.debug(f"🔥 DEDUP: Kept {best_relationship.relationship_type} over {[r.relationship_type for r in pair_relationships if r != best_relationship]} for {pair_key}")
        
        logger.info(f"🔥 PRIORITY DEDUPLICATION: {len(relationships)} -> {len(deduplicated)} relationships")
        return deduplicated
    
    async def store_in_neo4j(self, extraction_result: GraphExtractionResult, document_id: str) -> Dict[str, Any]:
        """Store extracted entities and relationships in Neo4j with cross-document linking"""
        try:
            neo4j_service = get_neo4j_service()
            
            if not neo4j_service.is_enabled():
                return {'success': False, 'error': 'Neo4j service is not enabled'}
            
            logger.info(f"🔥 STORING: {len(extraction_result.entities)} entities, {len(extraction_result.relationships)} relationships for doc {document_id}")
            
            stored_entities = []
            stored_relationships = []
            entity_name_to_id = {}
            
            # Step 1: Cross-document entity linking (if enabled)
            enable_linking = self.config.get('extraction', {}).get('enable_cross_document_linking', True)
            linking_results = []
            
            if enable_linking:
                try:
                    from app.services.entity_linking_service import get_entity_linking_service
                    linking_service = get_entity_linking_service()
                    linking_results = await linking_service.link_entities_in_document(
                        extraction_result.entities, document_id
                    )
                    logger.info(f"🔗 ENTITY LINKING: Processed {len(linking_results)} entities")
                except Exception as e:
                    logger.error(f"Entity linking failed, proceeding without linking: {e}")
                    linking_results = []
            
            # Step 2: Store entities with linking information
            for i, entity in enumerate(extraction_result.entities):
                linking_result = linking_results[i] if i < len(linking_results) else None
                
                if linking_result and not linking_result.is_new_entity:
                    # Entity linked to existing entity
                    entity_id = linking_result.linked_entity_id
                    logger.debug(f"🔗 Using linked entity: {entity.canonical_form} -> {entity_id}")
                    
                    # Update existing entity with new document reference
                    await self._update_linked_entity(entity_id, entity, document_id, extraction_result.chunk_id)
                    
                else:
                    # Create new entity (either no linking or new entity)
                    entity_properties = {
                        'name': entity.canonical_form,
                        'original_text': entity.text,
                        'type': entity.label,
                        'confidence': entity.confidence,
                        'document_id': document_id,
                        'chunk_id': extraction_result.chunk_id,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    # Add linking metadata if available
                    if linking_result:
                        entity_properties.update({
                            'linking_confidence': linking_result.linking_confidence,
                            'similarity_score': linking_result.similarity_score,
                            'linking_reasoning': linking_result.reasoning,
                            'alternative_candidates_count': len(linking_result.alternative_candidates)
                        })
                    
                    logger.debug(f"🔥 Creating new entity: {entity.canonical_form} ({entity.label})")
                    entity_id = neo4j_service.create_entity(entity.label, entity_properties)
                
                if entity_id:
                    stored_entities.append(entity_id)
                    
                    # Build comprehensive mapping for relationship creation
                    canonical_lower = entity.canonical_form.lower()
                    original_lower = entity.text.lower()
                    
                    # Map canonical form (primary)
                    entity_name_to_id[canonical_lower] = entity_id
                    # Map original text (fallback)
                    entity_name_to_id[original_lower] = entity_id
                    # Map without spaces (for matching flexibility)
                    entity_name_to_id[canonical_lower.replace(' ', '')] = entity_id
                    entity_name_to_id[original_lower.replace(' ', '')] = entity_id
                    
                    logger.debug(f"🔥 Entity stored: {entity.canonical_form} -> {entity_id}")
                else:
                    logger.error(f"🔥 Failed to store entity: {entity.canonical_form}")
            
            logger.info(f"🔥 Entity mapping created: {len(entity_name_to_id)} mappings")
            
            # Step 3: Store relationships using enhanced mapping with flexible matching
            for relationship in extraction_result.relationships:
                logger.debug(f"🔥 Processing relationship: {relationship.source_entity} -> {relationship.target_entity} ({relationship.relationship_type})")
                
                # Enhanced mapping lookup with multiple strategies
                source_id = self._find_entity_id_flexible(relationship.source_entity, entity_name_to_id)
                target_id = self._find_entity_id_flexible(relationship.target_entity, entity_name_to_id)
                
                if not source_id:
                    logger.warning(f"🔥 Source entity not found: '{relationship.source_entity}' (tried all variations)")
                    continue
                    
                if not target_id:
                    logger.warning(f"🔥 Target entity not found: '{relationship.target_entity}' (tried all variations)")
                    continue
                
                # Create relationship with found IDs
                rel_properties = {
                    'confidence': relationship.confidence,
                    'context': relationship.context,
                    'document_id': document_id,
                    'chunk_id': extraction_result.chunk_id,
                    'created_at': datetime.now().isoformat()
                }
                rel_properties.update(relationship.properties)
                
                logger.debug(f"🔥 Creating relationship: {source_id} -> {target_id} ({relationship.relationship_type})")
                success = neo4j_service.create_relationship(
                    source_id, target_id, relationship.relationship_type, rel_properties
                )
                if success:
                    stored_relationships.append({
                        'source': source_id,
                        'target': target_id,
                        'type': relationship.relationship_type
                    })
                    logger.debug(f"🔥 Relationship stored successfully: {relationship.source_entity} -> {relationship.target_entity}")
                else:
                    logger.error(f"🔥 Failed to store relationship: {relationship.source_entity} -> {relationship.target_entity}")
            
            logger.info(f"🔥 STORAGE COMPLETE: {len(stored_entities)} entities, {len(stored_relationships)} relationships stored")
            
            return {
                'success': True,
                'entities_stored': len(stored_entities),
                'relationships_stored': len(stored_relationships),
                'processing_time_ms': extraction_result.processing_time_ms,
                'entity_mappings_created': len(entity_name_to_id),
                'debug_info': {
                    'entity_names': list(entity_name_to_id.keys()),
                    'relationship_attempts': len(extraction_result.relationships),
                    'relationship_successes': len(stored_relationships)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to store graph data in Neo4j: {e}")
            return {'success': False, 'error': str(e)}
    
    def _find_entity_id_flexible(self, entity_name: str, entity_mapping: Dict[str, str]) -> Optional[str]:
        """Find entity ID using flexible matching strategies"""
        if not entity_name:
            return None
        
        # Strategy 1: Exact lowercase match
        exact_match = entity_mapping.get(entity_name.lower())
        if exact_match:
            logger.debug(f"🔥 Found entity via exact match: {entity_name} -> {exact_match}")
            return exact_match
        
        # Strategy 2: Remove spaces and match
        no_spaces = entity_name.lower().replace(' ', '')
        no_spaces_match = entity_mapping.get(no_spaces)
        if no_spaces_match:
            logger.debug(f"🔥 Found entity via no-spaces match: {entity_name} -> {no_spaces_match}")
            return no_spaces_match
        
        # Strategy 3: Partial match (entity name contains or is contained in mapping key)
        entity_lower = entity_name.lower()
        for mapped_name, mapped_id in entity_mapping.items():
            # Check if entity name is contained in mapped name or vice versa
            if (entity_lower in mapped_name) or (mapped_name in entity_lower):
                # Ensure it's a reasonable partial match (at least 3 characters)
                if len(entity_lower) >= 3 and len(mapped_name) >= 3:
                    logger.debug(f"🔥 Found entity via partial match: {entity_name} ~ {mapped_name} -> {mapped_id}")
                    return mapped_id
        
        # Strategy 4: Title case variations
        title_case = entity_name.title().lower()
        title_match = entity_mapping.get(title_case)
        if title_match:
            logger.debug(f"🔥 Found entity via title case: {entity_name} -> {title_match}")
            return title_match
        
        logger.debug(f"🔥 No entity ID found for: {entity_name} (tried all strategies)")
        return None
    
    async def _update_linked_entity(self, entity_id: str, entity: ExtractedEntity, 
                                   document_id: str, chunk_id: str) -> bool:
        """Update an existing linked entity with new document reference"""
        try:
            neo4j_service = get_neo4j_service()
            
            # Update entity with new document reference and confidence boost
            update_query = """
            MATCH (e {id: $entity_id})
            SET e.document_count = COALESCE(e.document_count, 0) + 1,
                e.last_updated = datetime(),
                e.confidence = CASE 
                    WHEN $new_confidence > COALESCE(e.confidence, 0) THEN $new_confidence 
                    ELSE COALESCE(e.confidence, $new_confidence)
                END,
                e.cross_document_references = COALESCE(e.cross_document_references, []) + [$document_id]
            RETURN e.id as entity_id
            """
            
            result = neo4j_service.execute_cypher(update_query, {
                'entity_id': entity_id,
                'new_confidence': entity.confidence,
                'document_id': document_id
            })
            
            return len(result) > 0
            
        except Exception as e:
            logger.error(f"Failed to update linked entity {entity_id}: {e}")
            return False

# Singleton instance
_knowledge_graph_service: Optional[KnowledgeGraphExtractionService] = None

def get_knowledge_graph_service() -> KnowledgeGraphExtractionService:
    """Get or create knowledge graph extraction service singleton"""
    global _knowledge_graph_service
    if _knowledge_graph_service is None:
        _knowledge_graph_service = KnowledgeGraphExtractionService()
    return _knowledge_graph_service