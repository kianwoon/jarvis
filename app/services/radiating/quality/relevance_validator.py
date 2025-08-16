"""
Relevance Validator

Validates entity relevance scores, relationship strength, coherent expansion,
and detects topic drift in the radiating system.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import asyncio

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.radiating.models.radiating_context import RadiatingContext
from app.services.radiating.models.radiating_graph import RadiatingGraph
from app.core.redis_client import get_redis_client
from app.core.radiating_settings_cache import get_radiating_settings

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result types"""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"


@dataclass
class RelevanceValidation:
    """Relevance validation result"""
    entity_id: str
    validation_result: ValidationResult
    relevance_score: float
    adjusted_score: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpansionCoherence:
    """Expansion coherence analysis"""
    depth: int
    coherence_score: float  # 0-1 scale
    topic_consistency: float  # 0-1 scale
    semantic_drift: float  # 0-1 scale (lower is better)
    entity_distribution: Dict[str, int]
    relationship_distribution: Dict[str, int]


@dataclass
class TopicDrift:
    """Topic drift detection result"""
    detected: bool
    drift_score: float  # 0-1 scale
    original_topics: List[str]
    current_topics: List[str]
    drift_points: List[Tuple[int, str]]  # (depth, entity_id) where drift occurred
    severity: str  # low, medium, high


class RelevanceValidator:
    """
    Validates relevance of entities and relationships in the radiating system,
    ensuring coherent expansion and detecting topic drift.
    """
    
    # Validation thresholds
    THRESHOLDS = {
        'min_relevance': 0.3,
        'min_relationship_strength': 0.2,
        'max_semantic_drift': 0.5,
        'min_topic_consistency': 0.6,
        'min_coherence': 0.5
    }
    
    # Validation weights for composite scoring
    WEIGHTS = {
        'relevance': 0.3,
        'relationship_strength': 0.25,
        'semantic_similarity': 0.2,
        'structural_importance': 0.15,
        'temporal_relevance': 0.1
    }
    
    def __init__(self):
        """Initialize RelevanceValidator"""
        self.redis_client = get_redis_client()
        
        # Validation cache
        self.validation_cache: Dict[str, RelevanceValidation] = {}
        
        # Topic tracking
        self.topic_embeddings: Dict[str, np.ndarray] = {}
        self.topic_history: List[Tuple[int, List[str]]] = []
        
        # Statistics
        self.stats = {
            'total_validations': 0,
            'valid_entities': 0,
            'warning_entities': 0,
            'invalid_entities': 0,
            'drift_detections': 0,
            'coherence_checks': 0
        }
    
    async def validate_entity(
        self,
        entity: RadiatingEntity,
        context: RadiatingContext,
        graph: Optional[RadiatingGraph] = None
    ) -> RelevanceValidation:
        """
        Validate entity relevance
        
        Args:
            entity: Entity to validate
            context: Radiating context
            graph: Optional graph for structural validation
            
        Returns:
            RelevanceValidation result
        """
        self.stats['total_validations'] += 1
        
        issues = []
        recommendations = []
        
        # Check basic relevance score
        if entity.relevance_score < self.THRESHOLDS['min_relevance']:
            issues.append(f"Low relevance score: {entity.relevance_score:.2f}")
            recommendations.append("Consider filtering or adjusting relevance threshold")
        
        # Validate semantic similarity if embeddings available
        semantic_score = await self._validate_semantic_similarity(entity, context)
        if semantic_score < 0.5:
            issues.append(f"Low semantic similarity: {semantic_score:.2f}")
            recommendations.append("Entity may be semantically unrelated to query")
        
        # Check structural importance if graph provided
        structural_score = 0.5  # Default
        if graph:
            structural_score = self._calculate_structural_importance(entity, graph)
            if structural_score < 0.3:
                issues.append(f"Low structural importance: {structural_score:.2f}")
                recommendations.append("Entity has few connections in graph")
        
        # Calculate composite validation score
        composite_score = self._calculate_composite_score(
            entity.relevance_score,
            semantic_score,
            structural_score
        )
        
        # Determine validation result
        if composite_score >= 0.7:
            result = ValidationResult.VALID
            self.stats['valid_entities'] += 1
        elif composite_score >= 0.5:
            result = ValidationResult.WARNING
            self.stats['warning_entities'] += 1
        else:
            result = ValidationResult.INVALID
            self.stats['invalid_entities'] += 1
        
        # Adjust score if needed
        adjusted_score = None
        if result == ValidationResult.WARNING:
            # Apply penalty for warnings
            adjusted_score = entity.relevance_score * 0.8
        elif result == ValidationResult.INVALID:
            # Significantly reduce score for invalid entities
            adjusted_score = entity.relevance_score * 0.3
        
        validation = RelevanceValidation(
            entity_id=entity.id,
            validation_result=result,
            relevance_score=entity.relevance_score,
            adjusted_score=adjusted_score,
            issues=issues,
            recommendations=recommendations,
            metadata={
                'semantic_score': semantic_score,
                'structural_score': structural_score,
                'composite_score': composite_score,
                'depth': entity.depth
            }
        )
        
        # Cache validation result
        self.validation_cache[entity.id] = validation
        
        return validation
    
    async def validate_relationship(
        self,
        relationship: RadiatingRelationship,
        source_entity: RadiatingEntity,
        target_entity: RadiatingEntity,
        context: RadiatingContext
    ) -> Dict[str, Any]:
        """
        Validate relationship strength and relevance
        
        Args:
            relationship: Relationship to validate
            source_entity: Source entity
            target_entity: Target entity
            context: Radiating context
            
        Returns:
            Validation results
        """
        issues = []
        recommendations = []
        
        # Check relationship strength
        if relationship.strength < self.THRESHOLDS['min_relationship_strength']:
            issues.append(f"Weak relationship strength: {relationship.strength:.2f}")
            recommendations.append("Consider filtering weak relationships")
        
        # Validate relationship type relevance
        type_relevance = self._validate_relationship_type(
            relationship.type,
            context.domain_context
        )
        
        if type_relevance < 0.5:
            issues.append(f"Relationship type '{relationship.type}' may not be relevant")
            recommendations.append("Focus on domain-specific relationship types")
        
        # Check entity coherence
        entity_coherence = await self._check_entity_coherence(
            source_entity,
            target_entity,
            context
        )
        
        if entity_coherence < 0.6:
            issues.append("Low coherence between connected entities")
            recommendations.append("Entities may not be meaningfully related")
        
        # Calculate overall validation score
        validation_score = (
            relationship.strength * 0.4 +
            type_relevance * 0.3 +
            entity_coherence * 0.3
        )
        
        return {
            'valid': validation_score >= 0.5,
            'score': validation_score,
            'strength': relationship.strength,
            'type_relevance': type_relevance,
            'entity_coherence': entity_coherence,
            'issues': issues,
            'recommendations': recommendations
        }
    
    async def check_expansion_coherence(
        self,
        graph: RadiatingGraph,
        context: RadiatingContext
    ) -> ExpansionCoherence:
        """
        Check coherence of graph expansion
        
        Args:
            graph: Radiating graph
            context: Radiating context
            
        Returns:
            ExpansionCoherence analysis
        """
        self.stats['coherence_checks'] += 1
        
        # Analyze entity distribution by depth
        entity_distribution = self._analyze_entity_distribution(graph)
        
        # Analyze relationship distribution
        relationship_distribution = self._analyze_relationship_distribution(graph)
        
        # Calculate topic consistency
        topic_consistency = await self._calculate_topic_consistency(graph, context)
        
        # Calculate semantic drift
        semantic_drift = await self._calculate_semantic_drift(graph, context)
        
        # Calculate overall coherence score
        coherence_score = self._calculate_coherence_score(
            topic_consistency,
            semantic_drift,
            entity_distribution,
            relationship_distribution
        )
        
        # Determine expansion depth with best coherence
        optimal_depth = self._find_optimal_depth(graph, coherence_score)
        
        return ExpansionCoherence(
            depth=optimal_depth,
            coherence_score=coherence_score,
            topic_consistency=topic_consistency,
            semantic_drift=semantic_drift,
            entity_distribution=entity_distribution,
            relationship_distribution=relationship_distribution
        )
    
    async def detect_topic_drift(
        self,
        graph: RadiatingGraph,
        context: RadiatingContext
    ) -> TopicDrift:
        """
        Detect topic drift in graph expansion
        
        Args:
            graph: Radiating graph
            context: Radiating context
            
        Returns:
            TopicDrift detection result
        """
        # Extract topics at each depth
        topics_by_depth = await self._extract_topics_by_depth(graph)
        
        # Get original topics from context
        original_topics = self._extract_context_topics(context)
        
        # Track topic changes
        drift_points = []
        max_drift = 0.0
        
        for depth, topics in topics_by_depth.items():
            # Calculate drift from original
            drift_score = self._calculate_topic_drift_score(
                original_topics,
                topics
            )
            
            if drift_score > max_drift:
                max_drift = drift_score
            
            # Check for significant drift
            if drift_score > self.THRESHOLDS['max_semantic_drift']:
                # Find entities causing drift
                drift_entities = self._identify_drift_entities(
                    graph,
                    depth,
                    original_topics,
                    topics
                )
                
                for entity_id in drift_entities:
                    drift_points.append((depth, entity_id))
        
        # Determine drift severity
        if max_drift < 0.3:
            severity = "low"
        elif max_drift < 0.6:
            severity = "medium"
        else:
            severity = "high"
        
        # Get current topics (at max depth)
        max_depth = max(topics_by_depth.keys()) if topics_by_depth else 0
        current_topics = topics_by_depth.get(max_depth, [])
        
        drift_detected = max_drift > self.THRESHOLDS['max_semantic_drift']
        
        if drift_detected:
            self.stats['drift_detections'] += 1
            logger.warning(f"Topic drift detected: {max_drift:.2f} ({severity})")
        
        return TopicDrift(
            detected=drift_detected,
            drift_score=max_drift,
            original_topics=original_topics,
            current_topics=current_topics,
            drift_points=drift_points,
            severity=severity
        )
    
    async def _validate_semantic_similarity(
        self,
        entity: RadiatingEntity,
        context: RadiatingContext
    ) -> float:
        """Calculate semantic similarity between entity and context"""
        # This would use embeddings in a real implementation
        # For now, return a simulated score based on depth
        base_similarity = 0.9
        depth_penalty = 0.1 * entity.depth
        return max(0.3, base_similarity - depth_penalty)
    
    def _calculate_structural_importance(
        self,
        entity: RadiatingEntity,
        graph: RadiatingGraph
    ) -> float:
        """Calculate structural importance of entity in graph"""
        # Count connections
        connections = 0
        
        for rel in graph.relationships:
            if rel.source_id == entity.id or rel.target_id == entity.id:
                connections += 1
        
        # Normalize by total relationships
        if len(graph.relationships) == 0:
            return 0.5
        
        importance = connections / max(1, len(graph.relationships) * 0.1)
        return min(1.0, importance)
    
    def _calculate_composite_score(
        self,
        relevance: float,
        semantic: float,
        structural: float
    ) -> float:
        """Calculate composite validation score"""
        return (
            relevance * self.WEIGHTS['relevance'] +
            semantic * self.WEIGHTS['semantic_similarity'] +
            structural * self.WEIGHTS['structural_importance']
        )
    
    def _validate_relationship_type(
        self,
        relationship_type: str,
        domain_context: Optional[str]
    ) -> float:
        """Validate relationship type relevance for domain"""
        # Domain-specific relationship types
        domain_relationships = {
            'financial': ['owns', 'invests_in', 'funds', 'regulates', 'audits'],
            'medical': ['treats', 'diagnoses', 'prescribes', 'symptoms_of', 'causes'],
            'technology': ['uses', 'implements', 'depends_on', 'integrates', 'extends'],
            'business': ['employs', 'partners_with', 'competes_with', 'supplies', 'acquires']
        }
        
        if not domain_context:
            return 0.5  # Neutral score if no domain specified
        
        domain = domain_context.lower()
        if domain in domain_relationships:
            if relationship_type.lower() in domain_relationships[domain]:
                return 1.0
            else:
                return 0.3
        
        return 0.5
    
    async def _check_entity_coherence(
        self,
        entity1: RadiatingEntity,
        entity2: RadiatingEntity,
        context: RadiatingContext
    ) -> float:
        """Check coherence between two entities"""
        # Simple coherence based on type and depth difference
        type_match = 1.0 if entity1.type == entity2.type else 0.5
        depth_diff = abs(entity1.depth - entity2.depth)
        depth_penalty = min(1.0, depth_diff * 0.2)
        
        coherence = type_match * (1 - depth_penalty)
        
        # Adjust for relevance scores
        avg_relevance = (entity1.relevance_score + entity2.relevance_score) / 2
        coherence *= avg_relevance
        
        return coherence
    
    def _analyze_entity_distribution(self, graph: RadiatingGraph) -> Dict[str, int]:
        """Analyze entity distribution in graph"""
        distribution = {}
        
        for entity in graph.entities:
            depth_key = f"depth_{entity.depth}"
            distribution[depth_key] = distribution.get(depth_key, 0) + 1
            
            type_key = f"type_{entity.type}"
            distribution[type_key] = distribution.get(type_key, 0) + 1
        
        return distribution
    
    def _analyze_relationship_distribution(self, graph: RadiatingGraph) -> Dict[str, int]:
        """Analyze relationship distribution in graph"""
        distribution = {}
        
        for rel in graph.relationships:
            type_key = f"rel_{rel.type}"
            distribution[type_key] = distribution.get(type_key, 0) + 1
        
        return distribution
    
    async def _calculate_topic_consistency(
        self,
        graph: RadiatingGraph,
        context: RadiatingContext
    ) -> float:
        """Calculate topic consistency across graph"""
        # Extract topics at each depth
        topics_by_depth = await self._extract_topics_by_depth(graph)
        
        if not topics_by_depth:
            return 1.0
        
        # Calculate consistency between consecutive depths
        consistencies = []
        depths = sorted(topics_by_depth.keys())
        
        for i in range(len(depths) - 1):
            topics1 = set(topics_by_depth[depths[i]])
            topics2 = set(topics_by_depth[depths[i + 1]])
            
            if topics1 or topics2:
                overlap = len(topics1 & topics2)
                total = len(topics1 | topics2)
                consistency = overlap / total if total > 0 else 0
                consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 1.0
    
    async def _calculate_semantic_drift(
        self,
        graph: RadiatingGraph,
        context: RadiatingContext
    ) -> float:
        """Calculate semantic drift from original context"""
        # Get original embedding (simulated)
        original_embedding = np.random.rand(100)  # Would use real embeddings
        
        # Calculate drift at each depth
        drifts = []
        
        for depth in range(1, context.max_depth + 1):
            # Get entities at this depth
            depth_entities = [e for e in graph.entities if e.depth == depth]
            
            if depth_entities:
                # Calculate average embedding (simulated)
                depth_embedding = np.random.rand(100)
                
                # Calculate cosine distance as drift
                similarity = cosine_similarity(
                    original_embedding.reshape(1, -1),
                    depth_embedding.reshape(1, -1)
                )[0, 0]
                
                drift = 1 - similarity
                drifts.append(drift)
        
        return np.mean(drifts) if drifts else 0.0
    
    def _calculate_coherence_score(
        self,
        topic_consistency: float,
        semantic_drift: float,
        entity_distribution: Dict[str, int],
        relationship_distribution: Dict[str, int]
    ) -> float:
        """Calculate overall coherence score"""
        # Base coherence from topic consistency
        coherence = topic_consistency * 0.4
        
        # Penalize for semantic drift
        coherence += (1 - semantic_drift) * 0.3
        
        # Check for balanced distribution
        entity_balance = self._calculate_distribution_balance(entity_distribution)
        rel_balance = self._calculate_distribution_balance(relationship_distribution)
        
        coherence += entity_balance * 0.15
        coherence += rel_balance * 0.15
        
        return min(1.0, coherence)
    
    def _calculate_distribution_balance(self, distribution: Dict[str, int]) -> float:
        """Calculate balance of a distribution"""
        if not distribution:
            return 0.0
        
        values = list(distribution.values())
        if len(values) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean = np.mean(values)
        std = np.std(values)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        
        # Convert to balance score (lower CV = better balance)
        balance = max(0, 1 - cv)
        
        return balance
    
    def _find_optimal_depth(self, graph: RadiatingGraph, coherence_score: float) -> int:
        """Find depth with optimal coherence"""
        # For now, return the depth that maintains minimum coherence
        for depth in range(1, max(e.depth for e in graph.entities) + 1):
            depth_entities = [e for e in graph.entities if e.depth <= depth]
            
            # Check if coherence drops significantly
            if len(depth_entities) > 0:
                avg_relevance = np.mean([e.relevance_score for e in depth_entities])
                
                if avg_relevance < self.THRESHOLDS['min_relevance']:
                    return depth - 1
        
        return max(e.depth for e in graph.entities)
    
    async def _extract_topics_by_depth(
        self,
        graph: RadiatingGraph
    ) -> Dict[int, List[str]]:
        """Extract topics at each depth level"""
        topics_by_depth = {}
        
        for depth in range(1, max(e.depth for e in graph.entities) + 1):
            depth_entities = [e for e in graph.entities if e.depth == depth]
            
            # Extract topics from entity names and types (simplified)
            topics = []
            for entity in depth_entities:
                # Extract key terms from entity name
                terms = entity.name.lower().split()
                topics.extend(terms)
                topics.append(entity.type.lower())
            
            # Get unique topics
            topics_by_depth[depth] = list(set(topics))
        
        return topics_by_depth
    
    def _extract_context_topics(self, context: RadiatingContext) -> List[str]:
        """Extract topics from context"""
        topics = []
        
        # Extract from query
        if context.query:
            topics.extend(context.query.lower().split())
        
        # Add domain context
        if context.domain_context:
            topics.append(context.domain_context.lower())
        
        # Add entity types if specified
        if context.entity_types:
            topics.extend([t.lower() for t in context.entity_types])
        
        return list(set(topics))
    
    def _calculate_topic_drift_score(
        self,
        original_topics: List[str],
        current_topics: List[str]
    ) -> float:
        """Calculate drift score between topic sets"""
        if not original_topics and not current_topics:
            return 0.0
        
        if not original_topics or not current_topics:
            return 1.0
        
        original_set = set(original_topics)
        current_set = set(current_topics)
        
        # Calculate Jaccard distance
        intersection = len(original_set & current_set)
        union = len(original_set | current_set)
        
        if union == 0:
            return 0.0
        
        jaccard_similarity = intersection / union
        drift = 1 - jaccard_similarity
        
        return drift
    
    def _identify_drift_entities(
        self,
        graph: RadiatingGraph,
        depth: int,
        original_topics: List[str],
        current_topics: List[str]
    ) -> List[str]:
        """Identify entities causing topic drift"""
        drift_entities = []
        
        # Get entities at this depth
        depth_entities = [e for e in graph.entities if e.depth == depth]
        
        # Find entities with topics not in original
        original_set = set(original_topics)
        
        for entity in depth_entities:
            entity_terms = set(entity.name.lower().split())
            
            # Check if entity introduces new topics
            new_topics = entity_terms - original_set
            
            if len(new_topics) > 0:
                drift_entities.append(entity.id)
        
        return drift_entities
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.stats['total_validations']
        
        return {
            'total_validations': total,
            'valid_entities': self.stats['valid_entities'],
            'warning_entities': self.stats['warning_entities'],
            'invalid_entities': self.stats['invalid_entities'],
            'valid_percentage': (
                self.stats['valid_entities'] / total * 100
                if total > 0 else 0
            ),
            'drift_detections': self.stats['drift_detections'],
            'coherence_checks': self.stats['coherence_checks']
        }