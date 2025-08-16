"""
Accuracy Checker

Verifies entity extraction accuracy, validates relationship correctness,
cross-references with existing knowledge, and validates confidence thresholds.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import re
from difflib import SequenceMatcher

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.neo4j_service import get_neo4j_service
from app.core.redis_client import get_redis_client
from app.core.radiating_settings_cache import get_radiating_settings

logger = logging.getLogger(__name__)


class AccuracyLevel(Enum):
    """Accuracy levels"""
    HIGH = "high"           # > 90% accurate
    MEDIUM = "medium"       # 70-90% accurate
    LOW = "low"            # 50-70% accurate
    UNVERIFIED = "unverified"  # < 50% or not checked


@dataclass
class EntityAccuracy:
    """Entity extraction accuracy result"""
    entity_id: str
    entity_name: str
    accuracy_level: AccuracyLevel
    confidence_score: float
    extraction_score: float  # How well the entity was extracted
    validation_score: float  # Cross-reference validation
    issues: List[str] = field(default_factory=list)
    corrections: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipAccuracy:
    """Relationship accuracy result"""
    relationship_id: str
    source_entity: str
    target_entity: str
    relationship_type: str
    accuracy_level: AccuracyLevel
    confidence_score: float
    validation_score: float
    issues: List[str] = field(default_factory=list)
    suggested_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossReferenceResult:
    """Cross-reference validation result"""
    entity_id: str
    found_in_knowledge_base: bool
    knowledge_base_matches: List[Dict[str, Any]]
    similarity_scores: List[float]
    discrepancies: List[str]
    confidence_adjustment: float  # Multiplier for confidence


class AccuracyChecker:
    """
    Verifies accuracy of entity extraction and relationship identification
    in the radiating system.
    """
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': 0.9,
        'medium': 0.7,
        'low': 0.5,
        'minimum': 0.3
    }
    
    # Extraction quality indicators
    QUALITY_INDICATORS = {
        'proper_noun': r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',
        'organization': r'^[A-Z][A-Z0-9\s&\-\.]+$',
        'valid_email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'valid_url': r'^https?://[^\s]+$',
        'valid_phone': r'^\+?[\d\s\-\(\)]+$',
        'alphanumeric_id': r'^[A-Z0-9\-]+$'
    }
    
    # Common extraction errors
    EXTRACTION_ERRORS = {
        'truncated': 'Entity name appears truncated',
        'merged': 'Multiple entities merged together',
        'partial': 'Partial entity extraction',
        'noise': 'Contains noise or artifacts',
        'format': 'Incorrect format for entity type',
        'case': 'Incorrect capitalization'
    }
    
    def __init__(self):
        """Initialize AccuracyChecker"""
        self.neo4j_service = get_neo4j_service()
        self.redis_client = get_redis_client()
        
        # Knowledge base cache
        self.kb_cache: Dict[str, List[Dict]] = {}
        self.kb_cache_ttl = 3600  # 1 hour
        
        # Validation patterns
        self.validation_patterns = self._compile_validation_patterns()
        
        # Statistics
        self.stats = {
            'entities_checked': 0,
            'relationships_checked': 0,
            'high_accuracy': 0,
            'medium_accuracy': 0,
            'low_accuracy': 0,
            'unverified': 0,
            'corrections_suggested': 0,
            'kb_matches': 0
        }
    
    def _compile_validation_patterns(self) -> Dict[str, re.Pattern]:
        """Compile validation regex patterns"""
        return {
            name: re.compile(pattern)
            for name, pattern in self.QUALITY_INDICATORS.items()
        }
    
    async def check_entity_accuracy(
        self,
        entity: RadiatingEntity,
        source_text: Optional[str] = None
    ) -> EntityAccuracy:
        """
        Check accuracy of entity extraction
        
        Args:
            entity: Entity to check
            source_text: Original text entity was extracted from
            
        Returns:
            EntityAccuracy result
        """
        self.stats['entities_checked'] += 1
        
        issues = []
        corrections = {}
        
        # Check extraction quality
        extraction_score = self._check_extraction_quality(entity, issues)
        
        # Validate entity format
        format_valid = self._validate_entity_format(entity, issues, corrections)
        
        # Check for common extraction errors
        self._check_extraction_errors(entity, source_text, issues, corrections)
        
        # Cross-reference with knowledge base
        cross_ref = await self._cross_reference_entity(entity)
        validation_score = self._calculate_validation_score(cross_ref)
        
        # Apply cross-reference corrections
        if cross_ref.knowledge_base_matches:
            best_match = cross_ref.knowledge_base_matches[0]
            if best_match.get('name') != entity.name:
                corrections['suggested_name'] = best_match['name']
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence(
            entity.confidence,
            extraction_score,
            validation_score
        )
        
        # Determine accuracy level
        accuracy_level = self._determine_accuracy_level(confidence_score)
        
        # Update statistics
        self._update_accuracy_stats(accuracy_level)
        
        # Create accuracy result
        accuracy = EntityAccuracy(
            entity_id=entity.id,
            entity_name=entity.name,
            accuracy_level=accuracy_level,
            confidence_score=confidence_score,
            extraction_score=extraction_score,
            validation_score=validation_score,
            issues=issues,
            corrections=corrections,
            metadata={
                'entity_type': entity.type,
                'original_confidence': entity.confidence,
                'kb_matched': cross_ref.found_in_knowledge_base,
                'checked_at': datetime.now().isoformat()
            }
        )
        
        # Suggest corrections if needed
        if corrections:
            self.stats['corrections_suggested'] += 1
        
        return accuracy
    
    async def check_relationship_accuracy(
        self,
        relationship: RadiatingRelationship,
        source_entity: RadiatingEntity,
        target_entity: RadiatingEntity
    ) -> RelationshipAccuracy:
        """
        Check accuracy of relationship identification
        
        Args:
            relationship: Relationship to check
            source_entity: Source entity
            target_entity: Target entity
            
        Returns:
            RelationshipAccuracy result
        """
        self.stats['relationships_checked'] += 1
        
        issues = []
        
        # Validate relationship type
        type_valid = self._validate_relationship_type(
            relationship.type,
            source_entity.type,
            target_entity.type,
            issues
        )
        
        # Check relationship consistency
        consistency_score = self._check_relationship_consistency(
            relationship,
            source_entity,
            target_entity
        )
        
        # Cross-reference with knowledge base
        kb_validation = await self._validate_relationship_in_kb(
            source_entity.name,
            target_entity.name,
            relationship.type
        )
        
        # Calculate validation score
        validation_score = (
            (1.0 if type_valid else 0.5) * 0.3 +
            consistency_score * 0.3 +
            kb_validation * 0.4
        )
        
        # Suggest alternative relationship type if needed
        suggested_type = None
        if not type_valid or validation_score < 0.5:
            suggested_type = await self._suggest_relationship_type(
                source_entity,
                target_entity
            )
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence(
            relationship.confidence,
            validation_score,
            consistency_score
        )
        
        # Determine accuracy level
        accuracy_level = self._determine_accuracy_level(confidence_score)
        
        # Create accuracy result
        accuracy = RelationshipAccuracy(
            relationship_id=relationship.id,
            source_entity=source_entity.name,
            target_entity=target_entity.name,
            relationship_type=relationship.type,
            accuracy_level=accuracy_level,
            confidence_score=confidence_score,
            validation_score=validation_score,
            issues=issues,
            suggested_type=suggested_type,
            metadata={
                'original_confidence': relationship.confidence,
                'consistency_score': consistency_score,
                'kb_validated': kb_validation > 0.5,
                'checked_at': datetime.now().isoformat()
            }
        )
        
        return accuracy
    
    async def cross_reference_with_knowledge(
        self,
        entities: List[RadiatingEntity],
        relationships: List[RadiatingRelationship]
    ) -> Dict[str, Any]:
        """
        Cross-reference entities and relationships with existing knowledge base
        
        Args:
            entities: List of entities to check
            relationships: List of relationships to check
            
        Returns:
            Cross-reference results
        """
        results = {
            'entities': {},
            'relationships': {},
            'summary': {
                'total_entities': len(entities),
                'matched_entities': 0,
                'total_relationships': len(relationships),
                'validated_relationships': 0
            }
        }
        
        # Cross-reference entities
        for entity in entities:
            cross_ref = await self._cross_reference_entity(entity)
            results['entities'][entity.id] = {
                'found': cross_ref.found_in_knowledge_base,
                'matches': len(cross_ref.knowledge_base_matches),
                'best_similarity': max(cross_ref.similarity_scores) if cross_ref.similarity_scores else 0,
                'discrepancies': cross_ref.discrepancies
            }
            
            if cross_ref.found_in_knowledge_base:
                results['summary']['matched_entities'] += 1
        
        # Validate relationships
        for rel in relationships:
            # Find source and target entities
            source = next((e for e in entities if e.id == rel.source_id), None)
            target = next((e for e in entities if e.id == rel.target_id), None)
            
            if source and target:
                kb_score = await self._validate_relationship_in_kb(
                    source.name,
                    target.name,
                    rel.type
                )
                
                results['relationships'][rel.id] = {
                    'validated': kb_score > 0.5,
                    'confidence': kb_score,
                    'source': source.name,
                    'target': target.name,
                    'type': rel.type
                }
                
                if kb_score > 0.5:
                    results['summary']['validated_relationships'] += 1
        
        return results
    
    async def validate_confidence_thresholds(
        self,
        entities: List[RadiatingEntity],
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Validate that entities meet confidence thresholds
        
        Args:
            entities: Entities to validate
            min_confidence: Minimum confidence threshold
            
        Returns:
            Validation results
        """
        results = {
            'total': len(entities),
            'above_threshold': 0,
            'below_threshold': 0,
            'low_confidence_entities': [],
            'confidence_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0,
                'very_low': 0
            }
        }
        
        for entity in entities:
            if entity.confidence >= min_confidence:
                results['above_threshold'] += 1
            else:
                results['below_threshold'] += 1
                results['low_confidence_entities'].append({
                    'id': entity.id,
                    'name': entity.name,
                    'confidence': entity.confidence,
                    'type': entity.type
                })
            
            # Categorize confidence levels
            if entity.confidence >= self.CONFIDENCE_THRESHOLDS['high']:
                results['confidence_distribution']['high'] += 1
            elif entity.confidence >= self.CONFIDENCE_THRESHOLDS['medium']:
                results['confidence_distribution']['medium'] += 1
            elif entity.confidence >= self.CONFIDENCE_THRESHOLDS['low']:
                results['confidence_distribution']['low'] += 1
            else:
                results['confidence_distribution']['very_low'] += 1
        
        # Calculate statistics
        if entities:
            confidences = [e.confidence for e in entities]
            results['statistics'] = {
                'mean': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'threshold_pass_rate': results['above_threshold'] / len(entities)
            }
        
        return results
    
    def _check_extraction_quality(
        self,
        entity: RadiatingEntity,
        issues: List[str]
    ) -> float:
        """Check quality of entity extraction"""
        score = 1.0
        
        # Check entity name quality
        if not entity.name or len(entity.name) < 2:
            issues.append("Entity name too short")
            score *= 0.5
        
        if len(entity.name) > 100:
            issues.append("Entity name unusually long")
            score *= 0.8
        
        # Check for special characters or artifacts
        if re.search(r'[<>{}[\]\\|`]', entity.name):
            issues.append("Entity name contains suspicious characters")
            score *= 0.7
        
        # Check for incomplete extraction
        if entity.name.endswith('...') or entity.name.startswith('...'):
            issues.append("Entity name appears truncated")
            score *= 0.6
        
        # Check entity type validity
        valid_types = ['person', 'organization', 'location', 'product', 
                      'technology', 'concept', 'event', 'document']
        
        if entity.type.lower() not in valid_types:
            issues.append(f"Unusual entity type: {entity.type}")
            score *= 0.9
        
        return max(0.1, score)
    
    def _validate_entity_format(
        self,
        entity: RadiatingEntity,
        issues: List[str],
        corrections: Dict[str, Any]
    ) -> bool:
        """Validate entity format based on type"""
        valid = True
        
        # Check format based on entity type
        if entity.type.lower() == 'person':
            if not self.validation_patterns['proper_noun'].match(entity.name):
                issues.append("Person name format incorrect")
                # Suggest correction
                corrected = self._correct_person_name(entity.name)
                if corrected != entity.name:
                    corrections['name_format'] = corrected
                valid = False
        
        elif entity.type.lower() == 'organization':
            # Organizations can have various formats, be lenient
            if len(entity.name) < 2:
                issues.append("Organization name too short")
                valid = False
        
        elif entity.type.lower() == 'email':
            if not self.validation_patterns['valid_email'].match(entity.name):
                issues.append("Invalid email format")
                valid = False
        
        elif entity.type.lower() == 'url':
            if not self.validation_patterns['valid_url'].match(entity.name):
                issues.append("Invalid URL format")
                valid = False
        
        return valid
    
    def _check_extraction_errors(
        self,
        entity: RadiatingEntity,
        source_text: Optional[str],
        issues: List[str],
        corrections: Dict[str, Any]
    ):
        """Check for common extraction errors"""
        # Check for merged entities (multiple capitals in unexpected places)
        capital_count = sum(1 for c in entity.name if c.isupper())
        if capital_count > len(entity.name.split()) * 2:
            issues.append(self.EXTRACTION_ERRORS['merged'])
        
        # Check for partial extraction
        if source_text:
            # Look for entity in source text
            if entity.name in source_text:
                # Check if there's more context around it
                index = source_text.index(entity.name)
                context_before = source_text[max(0, index-10):index]
                context_after = source_text[index+len(entity.name):index+len(entity.name)+10]
                
                # Check if extraction was cut off
                if context_before and not context_before[-1].isspace():
                    issues.append(self.EXTRACTION_ERRORS['partial'])
                
                if context_after and not context_after[0] in '.,;:!? ':
                    issues.append(self.EXTRACTION_ERRORS['partial'])
    
    async def _cross_reference_entity(
        self,
        entity: RadiatingEntity
    ) -> CrossReferenceResult:
        """Cross-reference entity with knowledge base"""
        # Check cache first
        cache_key = f"kb_ref:{entity.name}:{entity.type}"
        if cache_key in self.kb_cache:
            kb_matches = self.kb_cache[cache_key]
        else:
            # Query knowledge base
            kb_matches = await self._query_knowledge_base(entity)
            self.kb_cache[cache_key] = kb_matches
        
        # Calculate similarity scores
        similarity_scores = []
        discrepancies = []
        
        for match in kb_matches:
            # Calculate name similarity
            similarity = SequenceMatcher(
                None,
                entity.name.lower(),
                match.get('name', '').lower()
            ).ratio()
            similarity_scores.append(similarity)
            
            # Check for discrepancies
            if match.get('type') != entity.type:
                discrepancies.append(
                    f"Type mismatch: expected {match.get('type')}, got {entity.type}"
                )
        
        # Calculate confidence adjustment
        if kb_matches:
            self.stats['kb_matches'] += 1
            max_similarity = max(similarity_scores)
            if max_similarity > 0.9:
                confidence_adjustment = 1.2  # Boost confidence
            elif max_similarity > 0.7:
                confidence_adjustment = 1.0  # No change
            else:
                confidence_adjustment = 0.8  # Reduce confidence
        else:
            confidence_adjustment = 0.9  # Slight reduction for no matches
        
        return CrossReferenceResult(
            entity_id=entity.id,
            found_in_knowledge_base=len(kb_matches) > 0,
            knowledge_base_matches=kb_matches,
            similarity_scores=similarity_scores,
            discrepancies=discrepancies,
            confidence_adjustment=confidence_adjustment
        )
    
    async def _query_knowledge_base(
        self,
        entity: RadiatingEntity
    ) -> List[Dict[str, Any]]:
        """Query knowledge base for entity matches"""
        try:
            with self.neo4j_service.driver.session() as session:
                # Search for similar entities
                query = """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($name)
                   OR toLower($name) CONTAINS toLower(n.name)
                RETURN n.name as name, 
                       labels(n)[0] as type,
                       n.confidence as confidence
                LIMIT 5
                """
                
                result = session.run(query, name=entity.name)
                
                matches = []
                for record in result:
                    matches.append({
                        'name': record['name'],
                        'type': record['type'],
                        'confidence': record.get('confidence', 0.5)
                    })
                
                return matches
        
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return []
    
    def _calculate_validation_score(
        self,
        cross_ref: CrossReferenceResult
    ) -> float:
        """Calculate validation score from cross-reference"""
        if not cross_ref.found_in_knowledge_base:
            return 0.5  # Neutral score for new entities
        
        # Use best similarity score
        if cross_ref.similarity_scores:
            best_similarity = max(cross_ref.similarity_scores)
        else:
            best_similarity = 0.0
        
        # Penalize for discrepancies
        discrepancy_penalty = min(0.3, len(cross_ref.discrepancies) * 0.1)
        
        return max(0.1, best_similarity - discrepancy_penalty)
    
    def _calculate_confidence(
        self,
        original_confidence: float,
        extraction_score: float,
        validation_score: float
    ) -> float:
        """Calculate overall confidence score"""
        # Weighted average
        confidence = (
            original_confidence * 0.4 +
            extraction_score * 0.3 +
            validation_score * 0.3
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _determine_accuracy_level(self, confidence: float) -> AccuracyLevel:
        """Determine accuracy level from confidence score"""
        if confidence >= self.CONFIDENCE_THRESHOLDS['high']:
            return AccuracyLevel.HIGH
        elif confidence >= self.CONFIDENCE_THRESHOLDS['medium']:
            return AccuracyLevel.MEDIUM
        elif confidence >= self.CONFIDENCE_THRESHOLDS['low']:
            return AccuracyLevel.LOW
        else:
            return AccuracyLevel.UNVERIFIED
    
    def _update_accuracy_stats(self, accuracy_level: AccuracyLevel):
        """Update accuracy statistics"""
        if accuracy_level == AccuracyLevel.HIGH:
            self.stats['high_accuracy'] += 1
        elif accuracy_level == AccuracyLevel.MEDIUM:
            self.stats['medium_accuracy'] += 1
        elif accuracy_level == AccuracyLevel.LOW:
            self.stats['low_accuracy'] += 1
        else:
            self.stats['unverified'] += 1
    
    def _validate_relationship_type(
        self,
        rel_type: str,
        source_type: str,
        target_type: str,
        issues: List[str]
    ) -> bool:
        """Validate relationship type compatibility"""
        # Define valid relationship patterns
        valid_patterns = {
            ('person', 'organization'): ['works_for', 'owns', 'founded', 'manages'],
            ('person', 'person'): ['knows', 'reports_to', 'married_to', 'related_to'],
            ('organization', 'organization'): ['owns', 'partners_with', 'competes_with', 'acquires'],
            ('person', 'location'): ['lives_in', 'born_in', 'visited'],
            ('organization', 'location'): ['headquartered_in', 'operates_in'],
            ('product', 'organization'): ['manufactured_by', 'sold_by', 'developed_by']
        }
        
        # Check if relationship type is valid for entity types
        pattern_key = (source_type.lower(), target_type.lower())
        reverse_key = (target_type.lower(), source_type.lower())
        
        valid_types = (
            valid_patterns.get(pattern_key, []) +
            valid_patterns.get(reverse_key, [])
        )
        
        if not valid_types:
            # No specific pattern, allow generic relationships
            return True
        
        if rel_type.lower() not in [t.lower() for t in valid_types]:
            issues.append(
                f"Relationship type '{rel_type}' unusual for "
                f"{source_type} -> {target_type}"
            )
            return False
        
        return True
    
    def _check_relationship_consistency(
        self,
        relationship: RadiatingRelationship,
        source: RadiatingEntity,
        target: RadiatingEntity
    ) -> float:
        """Check relationship consistency"""
        score = 1.0
        
        # Check if relationship makes semantic sense
        # (simplified check based on confidence scores)
        avg_entity_confidence = (source.confidence + target.confidence) / 2
        
        if relationship.confidence < avg_entity_confidence * 0.5:
            score *= 0.7  # Relationship much less confident than entities
        
        # Check depth consistency
        depth_diff = abs(source.depth - target.depth)
        if depth_diff > 2:
            score *= 0.8  # Large depth difference
        
        return score
    
    async def _validate_relationship_in_kb(
        self,
        source_name: str,
        target_name: str,
        rel_type: str
    ) -> float:
        """Validate relationship exists in knowledge base"""
        try:
            with self.neo4j_service.driver.session() as session:
                # Check if relationship exists
                query = """
                MATCH (s)-[r]->(t)
                WHERE toLower(s.name) = toLower($source)
                  AND toLower(t.name) = toLower($target)
                RETURN type(r) as rel_type, count(r) as count
                """
                
                result = session.run(
                    query,
                    source=source_name,
                    target=target_name
                )
                
                for record in result:
                    if record['rel_type'].lower() == rel_type.lower():
                        return 1.0  # Exact match
                    else:
                        return 0.5  # Different relationship exists
                
                return 0.0  # No relationship found
        
        except Exception as e:
            logger.error(f"Error validating relationship: {e}")
            return 0.5  # Return neutral score on error
    
    async def _suggest_relationship_type(
        self,
        source: RadiatingEntity,
        target: RadiatingEntity
    ) -> Optional[str]:
        """Suggest appropriate relationship type"""
        # Define suggestions based on entity types
        suggestions = {
            ('person', 'organization'): 'works_for',
            ('person', 'person'): 'knows',
            ('organization', 'organization'): 'partners_with',
            ('person', 'location'): 'lives_in',
            ('organization', 'location'): 'headquartered_in',
            ('product', 'organization'): 'manufactured_by'
        }
        
        pattern_key = (source.type.lower(), target.type.lower())
        return suggestions.get(pattern_key)
    
    def _correct_person_name(self, name: str) -> str:
        """Correct person name formatting"""
        # Simple title case correction
        parts = name.split()
        corrected_parts = []
        
        for part in parts:
            if len(part) > 0:
                # Handle special cases (e.g., "van", "de", "Jr.")
                if part.lower() in ['van', 'de', 'von', 'der', 'la', 'le']:
                    corrected_parts.append(part.lower())
                elif part.upper() in ['II', 'III', 'IV', 'JR', 'SR']:
                    corrected_parts.append(part.upper())
                else:
                    corrected_parts.append(part.capitalize())
        
        return ' '.join(corrected_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get accuracy checking statistics"""
        total_checked = self.stats['entities_checked'] + self.stats['relationships_checked']
        
        return {
            'entities_checked': self.stats['entities_checked'],
            'relationships_checked': self.stats['relationships_checked'],
            'total_checked': total_checked,
            'accuracy_distribution': {
                'high': self.stats['high_accuracy'],
                'medium': self.stats['medium_accuracy'],
                'low': self.stats['low_accuracy'],
                'unverified': self.stats['unverified']
            },
            'accuracy_rate': {
                'high': (
                    self.stats['high_accuracy'] / total_checked * 100
                    if total_checked > 0 else 0
                ),
                'medium': (
                    self.stats['medium_accuracy'] / total_checked * 100
                    if total_checked > 0 else 0
                ),
                'low': (
                    self.stats['low_accuracy'] / total_checked * 100
                    if total_checked > 0 else 0
                )
            },
            'corrections_suggested': self.stats['corrections_suggested'],
            'kb_match_rate': (
                self.stats['kb_matches'] / self.stats['entities_checked'] * 100
                if self.stats['entities_checked'] > 0 else 0
            )
        }