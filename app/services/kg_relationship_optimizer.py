"""
Knowledge Graph Relationship Optimizer

Enforces strict relationship limits and quality controls to maintain
optimal performance (≤4 relationships per entity).
"""

import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class RelationshipOptimizer:
    """Optimize relationships to maintain performance targets"""
    
    def __init__(self, target_ratio: float = 4.0):
        self.target_ratio = target_ratio
        self.type_priorities = {
            # Higher priority = more likely to keep
            'USES': 90,
            'USED_BY': 85,
            'OFFERS': 80,
            'OFFERED_BY': 80,
            'PARTNERS_WITH': 70,
            'IMPLEMENTS': 65,
            'IMPLEMENTED_BY': 65,
            'OPERATES_IN': 60,
            'COMPETES_WITH': 40,  # Lower priority - too many of these
            'RELATED_TECHNOLOGY': 35,  # Lower priority - too many of these
            'CONTEXTUALLY_RELATED': 20,  # Lowest - generic relationships
        }
        
    def optimize_relationships(self, entities: List[Dict], relationships: List[Dict]) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Optimize relationships to achieve target ratio.
        
        Returns:
            Tuple of (optimized_relationships, optimization_stats)
        """
        logger.info(f"Starting optimization: {len(entities)} entities, {len(relationships)} relationships")
        
        if not entities:
            return relationships, {'error': 'No entities provided'}
        
        # Calculate current state
        current_ratio = len(relationships) / len(entities)
        target_count = int(len(entities) * self.target_ratio)
        
        stats = {
            'initial_count': len(relationships),
            'initial_ratio': current_ratio,
            'target_count': target_count,
            'target_ratio': self.target_ratio
        }
        
        # If already within target, return as-is
        if current_ratio <= self.target_ratio:
            stats['optimized_count'] = len(relationships)
            stats['optimized_ratio'] = current_ratio
            stats['removed_count'] = 0
            logger.info(f"Already within target ratio: {current_ratio:.2f} <= {self.target_ratio}")
            return relationships, stats
        
        # Need to optimize
        relationships_to_remove = len(relationships) - target_count
        logger.info(f"Need to remove {relationships_to_remove} relationships")
        
        # Score and sort relationships
        scored_relationships = self._score_relationships(relationships)
        
        # Keep only the highest scoring relationships up to target
        optimized = scored_relationships[:target_count]
        
        # Update stats
        stats['optimized_count'] = len(optimized)
        stats['optimized_ratio'] = len(optimized) / len(entities)
        stats['removed_count'] = len(relationships) - len(optimized)
        
        # Log what was removed
        removed_by_type = defaultdict(int)
        for rel in relationships:
            if rel not in optimized:
                removed_by_type[rel.get('relationship_type', 'UNKNOWN')] += 1
        
        stats['removed_by_type'] = dict(removed_by_type)
        
        logger.info(f"Optimization complete: {stats['initial_count']} -> {stats['optimized_count']} relationships")
        logger.info(f"Ratio: {stats['initial_ratio']:.2f} -> {stats['optimized_ratio']:.2f}")
        
        return optimized, stats
    
    def _score_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Score relationships based on quality metrics"""
        scored = []
        
        for rel in relationships:
            score = 0.0
            
            # Factor 1: Confidence (0-100 points)
            confidence = rel.get('confidence', 0.5)
            score += confidence * 100
            
            # Factor 2: Relationship type priority (0-100 points)
            rel_type = rel.get('relationship_type', 'UNKNOWN')
            type_priority = self.type_priorities.get(rel_type, 30)
            score += type_priority
            
            # Factor 3: Creation source penalty
            created_by = rel.get('created_by', '')
            if 'anti_silo' in created_by:
                score -= 50  # Penalize anti-silo relationships
            elif 'nuclear' in created_by:
                score -= 80  # Heavily penalize nuclear relationships
            elif 'llm' in created_by:
                score += 20  # Prefer LLM-discovered relationships
            
            # Factor 4: Specificity bonus
            if rel.get('reasoning') and len(rel.get('reasoning', '')) > 50:
                score += 10  # Bonus for well-reasoned relationships
            
            # Store score for sorting
            rel['_optimization_score'] = score
            scored.append(rel)
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: x['_optimization_score'], reverse=True)
        
        # Remove temporary score field
        for rel in scored:
            del rel['_optimization_score']
        
        return scored
    
    def optimize_by_entity(self, entities: List[Dict], relationships: List[Dict], 
                          max_per_entity: int = 4) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Optimize relationships with per-entity limits.
        
        Ensures no entity has more than max_per_entity relationships.
        """
        logger.info(f"Optimizing with per-entity limit: {max_per_entity}")
        
        # Build entity relationship map
        entity_rels = defaultdict(list)
        for rel in relationships:
            source = rel.get('source_entity')
            target = rel.get('target_entity')
            if source:
                entity_rels[source].append(rel)
            if target:
                entity_rels[target].append(rel)
        
        # Score all relationships first
        scored_relationships = self._score_relationships(relationships)
        
        # Track which relationships to keep
        kept_relationships = []
        entity_counts = defaultdict(int)
        
        # Process relationships in score order
        for rel in scored_relationships:
            source = rel.get('source_entity')
            target = rel.get('target_entity')
            
            # Check if both entities are under the limit
            source_ok = not source or entity_counts[source] < max_per_entity
            target_ok = not target or entity_counts[target] < max_per_entity
            
            if source_ok and target_ok:
                kept_relationships.append(rel)
                if source:
                    entity_counts[source] += 1
                if target:
                    entity_counts[target] += 1
        
        stats = {
            'initial_count': len(relationships),
            'optimized_count': len(kept_relationships),
            'removed_count': len(relationships) - len(kept_relationships),
            'max_per_entity': max_per_entity,
            'entities_at_limit': sum(1 for count in entity_counts.values() if count >= max_per_entity)
        }
        
        logger.info(f"Per-entity optimization: {stats['initial_count']} -> {stats['optimized_count']} relationships")
        
        return kept_relationships, stats
    
    def remove_low_quality_relationships(self, relationships: List[Dict], 
                                       min_confidence: float = 0.7) -> Tuple[List[Dict], int]:
        """Remove relationships below confidence threshold"""
        filtered = []
        removed = 0
        
        for rel in relationships:
            confidence = rel.get('confidence', 0.5)
            if confidence >= min_confidence:
                filtered.append(rel)
            else:
                removed += 1
                logger.debug(f"Removing low confidence ({confidence:.2f}) relationship: {rel.get('relationship_type')}")
        
        logger.info(f"Removed {removed} relationships with confidence < {min_confidence}")
        return filtered, removed
    
    def remove_duplicate_relationships(self, relationships: List[Dict]) -> Tuple[List[Dict], int]:
        """Remove duplicate relationships between same entities"""
        seen = set()
        filtered = []
        removed = 0
        
        for rel in relationships:
            source = rel.get('source_entity', '')
            target = rel.get('target_entity', '')
            rel_type = rel.get('relationship_type', '')
            
            # Create unique key (bidirectional)
            key = tuple(sorted([source, target]) + [rel_type])
            
            if key not in seen:
                seen.add(key)
                filtered.append(rel)
            else:
                removed += 1
                logger.debug(f"Removing duplicate: {source} -{rel_type}-> {target}")
        
        logger.info(f"Removed {removed} duplicate relationships")
        return filtered, removed
    
    def enforce_type_limits(self, relationships: List[Dict], 
                           type_limits: Dict[str, int]) -> Tuple[List[Dict], Dict[str, int]]:
        """Enforce maximum count per relationship type"""
        type_counts = defaultdict(int)
        filtered = []
        removed_by_type = defaultdict(int)
        
        # First pass: count types
        for rel in relationships:
            rel_type = rel.get('relationship_type', 'UNKNOWN')
            type_counts[rel_type] += 1
        
        # Score relationships within each type
        scored_by_type = defaultdict(list)
        for rel in relationships:
            rel_type = rel.get('relationship_type', 'UNKNOWN')
            scored_by_type[rel_type].append(rel)
        
        # Sort each type by confidence
        for rel_type in scored_by_type:
            scored_by_type[rel_type].sort(
                key=lambda x: x.get('confidence', 0.5), 
                reverse=True
            )
        
        # Apply limits
        type_kept_counts = defaultdict(int)
        for rel_type, rels in scored_by_type.items():
            limit = type_limits.get(rel_type, float('inf'))
            
            for rel in rels:
                if type_kept_counts[rel_type] < limit:
                    filtered.append(rel)
                    type_kept_counts[rel_type] += 1
                else:
                    removed_by_type[rel_type] += 1
        
        logger.info(f"Type limit enforcement removed: {dict(removed_by_type)}")
        return filtered, dict(removed_by_type)