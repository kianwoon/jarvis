"""
Radiating Context Model

Query context model for managing radiating traversal state, history,
and configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum


class TraversalStrategy(Enum):
    """Traversal strategy options"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"  # Priority-based
    HYBRID = "hybrid"  # Combines strategies


class DomainContext(Enum):
    """Domain context types for specialized traversal"""
    GENERAL = "general"
    FINANCIAL = "financial"
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    RESEARCH = "research"
    BUSINESS = "business"
    LEGAL = "legal"
    SCIENTIFIC = "scientific"


@dataclass
class TraversalPath:
    """Represents a path taken during traversal"""
    path_id: str
    entity_ids: List[str]
    relationship_ids: List[str]
    total_score: float
    depth: int
    discovered_at: datetime
    
    def get_path_string(self) -> str:
        """Get string representation of path"""
        return " -> ".join(self.entity_ids)


@dataclass
class RadiatingContext:
    """
    Query context for radiating traversal system.
    Manages traversal configuration, history, and discovered entities.
    """
    
    # Core query information
    original_query: str
    query_embedding: Optional[List[float]] = None
    query_intent: Optional[str] = None
    query_domain: DomainContext = DomainContext.GENERAL
    
    # Discovered entities and relationships
    expanded_entities: List['RadiatingEntity'] = field(default_factory=list)
    discovered_relationships: List['RadiatingRelationship'] = field(default_factory=list)
    
    # Traversal configuration
    depth_limit: int = 3
    relevance_threshold: float = 0.1
    max_entities_per_level: int = 50
    max_total_entities: int = 500
    traversal_strategy: TraversalStrategy = TraversalStrategy.BEST_FIRST
    
    # Traversal state
    traversal_history: List[TraversalPath] = field(default_factory=list)
    visited_entity_ids: Set[str] = field(default_factory=set)
    visited_relationship_ids: Set[str] = field(default_factory=set)
    current_depth: int = 0
    traversal_queue: List[Tuple[str, int, float]] = field(default_factory=list)  # (entity_id, depth, priority)
    
    # Performance tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_entities_discovered: int = 0
    total_relationships_discovered: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Domain-specific parameters
    domain_weights: Dict[str, float] = field(default_factory=dict)
    entity_type_preferences: List[str] = field(default_factory=list)
    relationship_type_preferences: List[str] = field(default_factory=list)
    
    # Filtering and constraints
    excluded_entity_types: Set[str] = field(default_factory=set)
    excluded_relationship_types: Set[str] = field(default_factory=set)
    required_entity_properties: Dict[str, Any] = field(default_factory=dict)
    
    # Results configuration
    min_entity_confidence: float = 0.3
    min_relationship_confidence: float = 0.3
    aggregate_similar_entities: bool = True
    include_inverse_relationships: bool = False
    
    # LLM Discovery configuration
    enable_llm_discovery: bool = True  # Enable LLM fallback when Neo4j returns empty
    llm_discovery_entities: List['RadiatingEntity'] = field(default_factory=list)  # Pool of entities for LLM discovery
    llm_max_entities_for_discovery: int = 30  # Maximum entities to use for LLM discovery
    
    def __post_init__(self):
        """Initialize context defaults"""
        if self.start_time is None:
            self.start_time = datetime.now()
        
        # Set default domain weights if not provided
        if not self.domain_weights:
            self.domain_weights = self._get_default_domain_weights()
    
    def _get_default_domain_weights(self) -> Dict[str, float]:
        """Get default domain weights based on query domain"""
        if self.query_domain == DomainContext.FINANCIAL:
            return {
                'ORGANIZATION': 1.2,
                'MONEY': 1.5,
                'PERCENTAGE': 1.3,
                'DATE': 1.1,
                'PERSON': 0.9
            }
        elif self.query_domain == DomainContext.TECHNOLOGY:
            return {
                'TECHNOLOGY': 1.5,
                'PRODUCT': 1.3,
                'ORGANIZATION': 1.1,
                'CONCEPT': 1.2,
                'PERSON': 0.8
            }
        elif self.query_domain == DomainContext.HEALTHCARE:
            return {
                'CONDITION': 1.5,
                'TREATMENT': 1.4,
                'MEDICATION': 1.3,
                'ORGANIZATION': 1.0,
                'PERSON': 1.1
            }
        else:
            # General balanced weights
            return {
                'PERSON': 1.0,
                'ORGANIZATION': 1.0,
                'LOCATION': 1.0,
                'CONCEPT': 1.0,
                'TECHNOLOGY': 1.0
            }
    
    def add_discovered_entity(self, entity: 'RadiatingEntity'):
        """
        Add a discovered entity to the context
        
        Args:
            entity: RadiatingEntity to add
        """
        # Don't add to visited_entity_ids here - that's only for processed entities
        # visited_entity_ids is managed by the traverser's _process_entity method
        entity_id = entity.get_entity_id()
        
        # Check if we've already discovered this entity
        existing_ids = {e.get_entity_id() for e in self.expanded_entities}
        if entity_id not in existing_ids:
            self.expanded_entities.append(entity)
            self.total_entities_discovered += 1
    
    def add_discovered_relationship(self, relationship: 'RadiatingRelationship'):
        """
        Add a discovered relationship to the context
        
        Args:
            relationship: RadiatingRelationship to add
        """
        rel_id = relationship.get_relationship_id()
        if rel_id not in self.visited_relationship_ids:
            self.discovered_relationships.append(relationship)
            self.visited_relationship_ids.add(rel_id)
            self.total_relationships_discovered += 1
    
    def add_traversal_path(self, entity_ids: List[str], 
                          relationship_ids: List[str],
                          total_score: float):
        """
        Add a traversal path to history
        
        Args:
            entity_ids: List of entity IDs in path
            relationship_ids: List of relationship IDs in path
            total_score: Total score of path
        """
        import hashlib
        
        path_string = "_".join(entity_ids)
        path_id = hashlib.md5(path_string.encode()).hexdigest()[:8]
        
        path = TraversalPath(
            path_id=path_id,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            total_score=total_score,
            depth=len(entity_ids) - 1,
            discovered_at=datetime.now()
        )
        
        self.traversal_history.append(path)
    
    def should_continue_traversal(self) -> bool:
        """
        Determine if traversal should continue
        
        Returns:
            True if traversal should continue, False otherwise
        """
        # Check depth limit
        if self.current_depth >= self.depth_limit:
            return False
        
        # Check entity count limit
        if self.total_entities_discovered >= self.max_total_entities:
            return False
        
        # Check if queue is empty (for breadth-first or best-first)
        if self.traversal_strategy in [TraversalStrategy.BREADTH_FIRST, 
                                       TraversalStrategy.BEST_FIRST]:
            if not self.traversal_queue:
                return False
        
        return True
    
    def get_next_entity_to_traverse(self) -> Optional[Tuple[str, int, float]]:
        """
        Get next entity to traverse based on strategy
        
        Returns:
            Tuple of (entity_id, depth, priority) or None if queue is empty
        """
        if not self.traversal_queue:
            return None
        
        if self.traversal_strategy == TraversalStrategy.BREADTH_FIRST:
            # FIFO queue
            return self.traversal_queue.pop(0)
        
        elif self.traversal_strategy == TraversalStrategy.DEPTH_FIRST:
            # LIFO stack
            return self.traversal_queue.pop()
        
        elif self.traversal_strategy == TraversalStrategy.BEST_FIRST:
            # Priority queue - sort by priority and take highest
            self.traversal_queue.sort(key=lambda x: x[2], reverse=True)
            return self.traversal_queue.pop(0)
        
        elif self.traversal_strategy == TraversalStrategy.HYBRID:
            # Hybrid approach - alternate between strategies
            # Use best-first for first half of depth, then breadth-first
            if self.current_depth < self.depth_limit // 2:
                self.traversal_queue.sort(key=lambda x: x[2], reverse=True)
                return self.traversal_queue.pop(0)
            else:
                return self.traversal_queue.pop(0)
        
        return None
    
    def add_to_traversal_queue(self, entity_id: str, depth: int, priority: float):
        """
        Add entity to traversal queue
        
        Args:
            entity_id: Entity ID to add
            depth: Depth of entity
            priority: Priority score for traversal
        """
        if entity_id not in self.visited_entity_ids:
            self.traversal_queue.append((entity_id, depth, priority))
    
    def update_cache_stats(self, hit: bool):
        """
        Update cache statistics
        
        Args:
            hit: True for cache hit, False for cache miss
        """
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def set_domain_preferences(self, domain: DomainContext,
                              entity_types: Optional[List[str]] = None,
                              relationship_types: Optional[List[str]] = None):
        """
        Set domain-specific preferences
        
        Args:
            domain: Domain context
            entity_types: Preferred entity types for domain
            relationship_types: Preferred relationship types for domain
        """
        self.query_domain = domain
        self.domain_weights = self._get_default_domain_weights()
        
        if entity_types:
            self.entity_type_preferences = entity_types
        
        if relationship_types:
            self.relationship_type_preferences = relationship_types
    
    def apply_entity_filter(self, entity: 'RadiatingEntity') -> bool:
        """
        Apply filters to determine if entity should be included
        
        Args:
            entity: Entity to filter
            
        Returns:
            True if entity passes filters, False otherwise
        """
        # Check confidence threshold
        if entity.confidence < self.min_entity_confidence:
            return False
        
        # Check excluded types
        if entity.label in self.excluded_entity_types:
            return False
        
        # Check required properties
        for prop_key, prop_value in self.required_entity_properties.items():
            if prop_key not in entity.properties:
                return False
            if entity.properties[prop_key] != prop_value:
                return False
        
        # Check relevance threshold
        if entity.relevance_score < self.relevance_threshold:
            return False
        
        return True
    
    def apply_relationship_filter(self, relationship: 'RadiatingRelationship') -> bool:
        """
        Apply filters to determine if relationship should be included
        
        Args:
            relationship: Relationship to filter
            
        Returns:
            True if relationship passes filters, False otherwise
        """
        # Check confidence threshold
        if relationship.confidence < self.min_relationship_confidence:
            return False
        
        # Check excluded types
        if relationship.relationship_type in self.excluded_relationship_types:
            return False
        
        return True
    
    def finalize(self):
        """Finalize context after traversal completion"""
        self.end_time = datetime.now()
        
        # Sort entities by relevance
        self.expanded_entities.sort(key=lambda e: e.relevance_score, reverse=True)
        
        # Sort paths by score
        self.traversal_history.sort(key=lambda p: p.total_score, reverse=True)
    
    def get_traversal_summary(self) -> Dict[str, Any]:
        """Get summary of traversal results"""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'query': self.original_query,
            'domain': self.query_domain.value,
            'strategy': self.traversal_strategy.value,
            'depth_limit': self.depth_limit,
            'relevance_threshold': self.relevance_threshold,
            'total_entities': self.total_entities_discovered,
            'total_relationships': self.total_relationships_discovered,
            'unique_paths': len(self.traversal_history),
            'max_depth_reached': self.current_depth,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'duration_seconds': duration,
            'top_entities': [e.canonical_form for e in self.expanded_entities[:10]],
            'entity_type_distribution': self._get_entity_type_distribution()
        }
    
    def _get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types discovered"""
        distribution = {}
        for entity in self.expanded_entities:
            entity_type = entity.label
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        return distribution