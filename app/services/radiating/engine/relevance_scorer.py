"""
Relevance Scorer

Scoring engine for calculating relevance of entities and relationships
in the radiating traversal context.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.radiating.models.radiating_context import RadiatingContext, DomainContext
from app.services.radiating.models.radiating_graph import RadiatingGraph

logger = logging.getLogger(__name__)


class RelevanceScorer:
    """
    Calculates relevance scores for entities and relationships
    based on various factors including distance, context, and domain.
    """
    
    def __init__(self):
        """Initialize RelevanceScorer"""
        # Scoring weights
        self.weights = {
            'distance_decay': 0.3,
            'semantic_similarity': 0.25,
            'domain_relevance': 0.2,
            'connectivity': 0.15,
            'confidence': 0.1
        }
        
        # Decay functions
        self.decay_functions = {
            'exponential': lambda d: math.exp(-0.5 * d),
            'linear': lambda d: max(0, 1 - 0.2 * d),
            'logarithmic': lambda d: 1 / (1 + math.log(1 + d)),
            'inverse': lambda d: 1 / (1 + d)
        }
        
        # Domain-specific scoring adjustments
        self.domain_adjustments = {
            DomainContext.FINANCIAL: {
                'MONEY': 1.5,
                'PERCENTAGE': 1.3,
                'ORGANIZATION': 1.2,
                'DATE': 1.1
            },
            DomainContext.TECHNOLOGY: {
                'TECHNOLOGY': 1.5,
                'PRODUCT': 1.3,
                'CONCEPT': 1.2,
                'ORGANIZATION': 1.1
            },
            DomainContext.HEALTHCARE: {
                'CONDITION': 1.5,
                'TREATMENT': 1.4,
                'MEDICATION': 1.3,
                'PERSON': 1.1
            },
            DomainContext.RESEARCH: {
                'CONCEPT': 1.4,
                'METHODOLOGY': 1.3,
                'PERSON': 1.2,
                'ORGANIZATION': 1.1
            }
        }
    
    async def calculate_entity_relevance(self, entity: RadiatingEntity,
                                        context: RadiatingContext,
                                        graph: RadiatingGraph) -> float:
        """
        Calculate relevance score for an entity
        
        Args:
            entity: Entity to score
            context: Traversal context
            graph: Current graph
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        scores = {}
        
        # 1. Distance decay score
        scores['distance'] = self._calculate_distance_score(
            entity.traversal_depth,
            context.depth_limit
        )
        
        # 2. Semantic similarity score
        scores['semantic'] = await self._calculate_semantic_similarity(
            entity,
            context
        )
        
        # 3. Domain relevance score
        scores['domain'] = self._calculate_domain_relevance(
            entity,
            context
        )
        
        # 4. Connectivity score
        scores['connectivity'] = self._calculate_connectivity_score(
            entity,
            graph
        )
        
        # 5. Confidence score
        scores['confidence'] = entity.confidence
        
        # Apply weights
        weighted_score = sum(
            scores[key] * self.weights.get(key.replace('_score', ''), 0.1)
            for key in ['distance', 'semantic', 'domain', 'connectivity', 'confidence']
        )
        
        # Apply domain-specific adjustments
        if context.query_domain in self.domain_adjustments:
            adjustments = self.domain_adjustments[context.query_domain]
            if entity.label in adjustments:
                weighted_score *= adjustments[entity.label]
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, weighted_score))
        
        logger.debug(f"Entity {entity.canonical_form} relevance: {final_score:.3f} "
                    f"(dist:{scores['distance']:.2f}, sem:{scores['semantic']:.2f}, "
                    f"dom:{scores['domain']:.2f}, conn:{scores['connectivity']:.2f})")
        
        return final_score
    
    async def calculate_relationship_relevance(self, relationship: RadiatingRelationship,
                                              context: RadiatingContext,
                                              graph: RadiatingGraph) -> float:
        """
        Calculate relevance score for a relationship
        
        Args:
            relationship: Relationship to score
            context: Traversal context
            graph: Current graph
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        scores = {}
        
        # 1. Base confidence score
        scores['confidence'] = relationship.confidence
        
        # 2. Entity relevance scores
        source_node = graph.nodes.get(relationship.source_entity)
        target_node = graph.nodes.get(relationship.target_entity)
        
        if source_node and target_node:
            scores['entity_relevance'] = (
                source_node.entity.relevance_score + 
                target_node.entity.relevance_score
            ) / 2
        else:
            scores['entity_relevance'] = 0.5
        
        # 3. Relationship type relevance
        scores['type_relevance'] = self._calculate_relationship_type_relevance(
            relationship.relationship_type,
            context
        )
        
        # 4. Path importance
        scores['path_importance'] = self._calculate_path_importance(
            relationship,
            graph,
            context
        )
        
        # Calculate weighted score
        weighted_score = (
            scores['confidence'] * 0.3 +
            scores['entity_relevance'] * 0.3 +
            scores['type_relevance'] * 0.2 +
            scores['path_importance'] * 0.2
        )
        
        # Apply bidirectional bonus
        if relationship.bidirectional:
            weighted_score *= 1.1
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, weighted_score))
        
        return final_score
    
    def _calculate_distance_score(self, depth: int, max_depth: int,
                                 decay_type: str = "exponential") -> float:
        """
        Calculate distance-based decay score
        
        Args:
            depth: Current traversal depth
            max_depth: Maximum traversal depth
            decay_type: Type of decay function
            
        Returns:
            Distance score (0.0 to 1.0)
        """
        if depth == 0:
            return 1.0
        
        normalized_depth = depth / max(1, max_depth)
        decay_func = self.decay_functions.get(decay_type, self.decay_functions['exponential'])
        
        return decay_func(normalized_depth * 2)  # Scale for stronger decay
    
    async def _calculate_semantic_similarity(self, entity: RadiatingEntity,
                                           context: RadiatingContext) -> float:
        """
        Calculate semantic similarity between entity and query
        
        Args:
            entity: Entity to compare
            context: Query context
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple text-based similarity for now
        # In production, would use embeddings
        
        if not context.original_query:
            return 0.5
        
        query_words = set(context.original_query.lower().split())
        entity_words = set(entity.canonical_form.lower().split())
        
        if not query_words or not entity_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(entity_words))
        union = len(query_words.union(entity_words))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Boost if entity contains query terms
        containment_score = 0.0
        for query_word in query_words:
            if query_word in entity.canonical_form.lower():
                containment_score += 0.2
        
        containment_score = min(1.0, containment_score)
        
        # Combine scores
        return 0.6 * jaccard_score + 0.4 * containment_score
    
    def _calculate_domain_relevance(self, entity: RadiatingEntity,
                                   context: RadiatingContext) -> float:
        """
        Calculate domain-specific relevance
        
        Args:
            entity: Entity to score
            context: Query context
            
        Returns:
            Domain relevance score (0.0 to 1.0)
        """
        base_score = 0.5  # Neutral score
        
        # Check if entity type is preferred for domain
        if entity.label in context.entity_type_preferences:
            base_score += 0.3
        
        # Check domain weights
        if entity.label in context.domain_weights:
            weight = context.domain_weights[entity.label]
            base_score *= weight
        
        # Check domain metadata
        if entity.domain_types:
            # Bonus if entity has relevant domain classification
            if context.query_domain.value in entity.domain_types:
                base_score += 0.2
        
        # Use domain importance if available
        if entity.domain_importance > 0:
            base_score = 0.7 * base_score + 0.3 * entity.domain_importance
        
        return min(1.0, base_score)
    
    def _calculate_connectivity_score(self, entity: RadiatingEntity,
                                     graph: RadiatingGraph) -> float:
        """
        Calculate connectivity-based importance
        
        Args:
            entity: Entity to score
            graph: Current graph
            
        Returns:
            Connectivity score (0.0 to 1.0)
        """
        entity_id = entity.get_entity_id()
        
        if entity_id not in graph.nodes:
            return 0.0
        
        # Get degree centrality
        neighbors = graph.get_neighbors(entity_id, direction="both")
        degree = len(neighbors)
        
        # Normalize by graph size (with minimum to avoid division by zero)
        max_possible_degree = max(1, graph.total_nodes - 1)
        normalized_degree = degree / max_possible_degree
        
        # Apply logarithmic scaling for better distribution
        if degree > 0:
            log_score = math.log(1 + degree) / math.log(1 + max_possible_degree)
        else:
            log_score = 0.0
        
        # Combine linear and logarithmic scores
        connectivity_score = 0.6 * normalized_degree + 0.4 * log_score
        
        # Boost for hub nodes (high connectivity)
        if degree > graph.total_nodes * 0.1:  # Top 10% by degree
            connectivity_score *= 1.2
        
        return min(1.0, connectivity_score)
    
    def _calculate_relationship_type_relevance(self, rel_type: str,
                                              context: RadiatingContext) -> float:
        """
        Calculate relevance of relationship type
        
        Args:
            rel_type: Relationship type
            context: Query context
            
        Returns:
            Type relevance score (0.0 to 1.0)
        """
        base_score = 0.5
        
        # Check if type is preferred
        if rel_type in context.relationship_type_preferences:
            base_score += 0.4
        
        # Domain-specific relationship importance
        domain_important_rels = {
            DomainContext.FINANCIAL: ['INVESTS_IN', 'OWNS', 'FUNDS', 'PAYS'],
            DomainContext.TECHNOLOGY: ['USES', 'IMPLEMENTS', 'DEVELOPS', 'INTEGRATES'],
            DomainContext.HEALTHCARE: ['TREATS', 'DIAGNOSES', 'PRESCRIBES', 'CAUSES'],
            DomainContext.RESEARCH: ['CITES', 'BUILDS_ON', 'VALIDATES', 'CONTRADICTS']
        }
        
        if context.query_domain in domain_important_rels:
            if rel_type in domain_important_rels[context.query_domain]:
                base_score += 0.3
        
        return min(1.0, base_score)
    
    def _calculate_path_importance(self, relationship: RadiatingRelationship,
                                  graph: RadiatingGraph,
                                  context: RadiatingContext) -> float:
        """
        Calculate importance of relationship in paths
        
        Args:
            relationship: Relationship to score
            graph: Current graph
            context: Query context
            
        Returns:
            Path importance score (0.0 to 1.0)
        """
        # Check if relationship appears in discovered paths
        path_count = 0
        total_path_score = 0.0
        
        for path in context.traversal_history:
            if relationship.get_relationship_id() in path.relationship_ids:
                path_count += 1
                total_path_score += path.total_score
        
        if path_count == 0:
            return 0.3  # Base score for relationships not in paths
        
        # Average path score for paths containing this relationship
        avg_path_score = total_path_score / path_count
        
        # Normalize by number of paths
        path_frequency = path_count / max(1, len(context.traversal_history))
        
        # Combine scores
        importance = 0.6 * avg_path_score + 0.4 * path_frequency
        
        return min(1.0, importance)
    
    def calculate_path_relevance(self, path: List[str],
                                graph: RadiatingGraph,
                                context: RadiatingContext) -> float:
        """
        Calculate relevance score for an entire path
        
        Args:
            path: List of entity IDs in path
            graph: Current graph
            context: Query context
            
        Returns:
            Path relevance score (0.0 to 1.0)
        """
        if not path:
            return 0.0
        
        # Calculate average entity relevance in path
        entity_scores = []
        for entity_id in path:
            node = graph.nodes.get(entity_id)
            if node:
                entity_scores.append(node.entity.relevance_score)
        
        if not entity_scores:
            return 0.0
        
        avg_entity_score = sum(entity_scores) / len(entity_scores)
        
        # Apply length penalty (prefer shorter paths)
        length_penalty = 1.0 / (1 + 0.1 * len(path))
        
        # Check path coherence (entities should be related)
        coherence_score = self._calculate_path_coherence(path, graph)
        
        # Combine scores
        path_score = (
            avg_entity_score * 0.5 +
            length_penalty * 0.3 +
            coherence_score * 0.2
        )
        
        return min(1.0, path_score)
    
    def _calculate_path_coherence(self, path: List[str],
                                 graph: RadiatingGraph) -> float:
        """
        Calculate coherence of entities in path
        
        Args:
            path: List of entity IDs
            graph: Current graph
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if len(path) < 2:
            return 1.0
        
        # Check if consecutive entities are connected
        connected_pairs = 0
        for i in range(len(path) - 1):
            neighbors = graph.get_neighbors(path[i], direction="both")
            if path[i + 1] in neighbors:
                connected_pairs += 1
        
        coherence = connected_pairs / (len(path) - 1)
        
        return coherence
    
    def adjust_scores_for_feedback(self, entity_scores: Dict[str, float],
                                  positive_examples: List[str],
                                  negative_examples: List[str]) -> Dict[str, float]:
        """
        Adjust relevance scores based on user feedback
        
        Args:
            entity_scores: Current entity scores
            positive_examples: Entity IDs marked as relevant
            negative_examples: Entity IDs marked as irrelevant
            
        Returns:
            Adjusted scores
        """
        adjusted_scores = entity_scores.copy()
        
        # Boost scores for positive examples
        for entity_id in positive_examples:
            if entity_id in adjusted_scores:
                adjusted_scores[entity_id] = min(1.0, adjusted_scores[entity_id] * 1.5)
        
        # Reduce scores for negative examples
        for entity_id in negative_examples:
            if entity_id in adjusted_scores:
                adjusted_scores[entity_id] = max(0.0, adjusted_scores[entity_id] * 0.5)
        
        return adjusted_scores