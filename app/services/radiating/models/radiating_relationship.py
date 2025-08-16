"""
Radiating Relationship Model

Enhanced relationship model for the radiating traversal system with
traversal weights, discovery context, and bidirectional support.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from app.services.knowledge_graph_types import ExtractedRelationship


@dataclass
class RadiatingRelationship(ExtractedRelationship):
    """
    Enhanced relationship model for radiating traversal system.
    Extends the base ExtractedRelationship with radiating-specific properties.
    """
    
    # Radiating-specific properties
    traversal_weight: float = 1.0
    discovery_context: Dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = False
    strength_score: float = 1.0
    
    # Traversal metadata
    traversal_count: int = 0
    last_traversed: Optional[datetime] = None
    traversal_direction: str = "forward"  # forward, backward, or both
    
    # Path tracking
    discovery_depth: int = 0
    discovery_timestamp: Optional[datetime] = None
    discovered_via_path: List[str] = field(default_factory=list)
    
    # Relationship quality metrics
    confidence_adjusted: float = 1.0
    semantic_similarity: float = 0.0
    co_occurrence_count: int = 1
    
    # Domain-specific attributes
    domain_type: Optional[str] = None
    domain_specific_weight: float = 1.0
    relationship_subtype: Optional[str] = None
    
    # Caching and optimization
    cached_at: Optional[datetime] = None
    skip_in_traversal: bool = False
    priority_level: int = 0  # 0 = normal, 1 = high, 2 = critical
    
    def __post_init__(self):
        """Initialize inherited fields and set defaults"""
        super().__post_init__()
        
        # Set discovery timestamp if not provided
        if self.discovery_timestamp is None:
            self.discovery_timestamp = datetime.now()
        
        # Adjust confidence based on strength
        self.confidence_adjusted = self.confidence * self.strength_score
    
    def get_relationship_id(self) -> str:
        """Generate unique relationship identifier"""
        import hashlib
        
        # Create deterministic ID from source, target, and type
        rel_string = f"{self.source_entity}_{self.relationship_type}_{self.target_entity}"
        rel_hash = hashlib.md5(rel_string.encode()).hexdigest()[:8]
        return f"rel_{rel_hash}"
    
    def calculate_traversal_score(self, direction: str = "forward") -> float:
        """
        Calculate traversal score for this relationship
        
        Args:
            direction: Direction of traversal ('forward' or 'backward')
            
        Returns:
            Traversal score combining various factors
        """
        # Base score from weights
        base_score = self.traversal_weight * self.strength_score
        
        # Apply directional modifier
        if direction == "backward" and not self.bidirectional:
            base_score *= 0.7  # Penalty for traversing non-bidirectional backwards
        
        # Apply confidence adjustment
        base_score *= self.confidence_adjusted
        
        # Apply domain-specific weight
        if self.domain_specific_weight != 1.0:
            base_score *= self.domain_specific_weight
        
        # Apply priority boost
        if self.priority_level > 0:
            base_score *= (1 + 0.5 * self.priority_level)
        
        # Apply traversal count decay (diminishing returns)
        if self.traversal_count > 0:
            base_score *= (1.0 / (1 + 0.2 * self.traversal_count))
        
        return base_score
    
    def mark_traversed(self, direction: str = "forward"):
        """
        Mark relationship as traversed
        
        Args:
            direction: Direction of traversal
        """
        self.traversal_count += 1
        self.last_traversed = datetime.now()
        
        # Update traversal direction
        if self.traversal_direction == "both":
            pass  # Already traversed in both directions
        elif self.traversal_direction != direction:
            if self.traversal_direction in ["forward", "backward"]:
                self.traversal_direction = "both"
        else:
            self.traversal_direction = direction
    
    def should_traverse(self, max_traversals: int = 5, 
                       min_weight: float = 0.1) -> bool:
        """
        Determine if relationship should be traversed
        
        Args:
            max_traversals: Maximum number of traversals allowed
            min_weight: Minimum traversal weight threshold
            
        Returns:
            True if relationship should be traversed
        """
        if self.skip_in_traversal:
            return False
        
        return (
            self.traversal_count < max_traversals and
            self.calculate_traversal_score() >= min_weight
        )
    
    def get_inverse_relationship(self) -> 'RadiatingRelationship':
        """
        Create inverse relationship (swap source and target)
        
        Returns:
            New RadiatingRelationship with swapped direction
        """
        inverse = RadiatingRelationship(
            source_entity=self.target_entity,
            target_entity=self.source_entity,
            relationship_type=f"inverse_{self.relationship_type}",
            confidence=self.confidence,
            context=self.context,
            properties=self.properties.copy() if self.properties else {}
        )
        
        # Copy radiating-specific properties
        inverse.traversal_weight = self.traversal_weight
        inverse.discovery_context = self.discovery_context.copy()
        inverse.bidirectional = self.bidirectional
        inverse.strength_score = self.strength_score
        inverse.discovery_depth = self.discovery_depth
        inverse.domain_type = self.domain_type
        inverse.domain_specific_weight = self.domain_specific_weight
        inverse.relationship_subtype = self.relationship_subtype
        inverse.priority_level = self.priority_level
        
        return inverse
    
    def add_discovery_context(self, key: str, value: Any):
        """
        Add context about how this relationship was discovered
        
        Args:
            key: Context key
            value: Context value
        """
        self.discovery_context[key] = value
    
    def set_domain_classification(self, domain: str, 
                                 subtype: Optional[str] = None,
                                 weight: float = 1.0):
        """
        Set domain-specific classification for the relationship
        
        Args:
            domain: Domain identifier
            subtype: Optional relationship subtype within domain
            weight: Domain-specific weight modifier
        """
        self.domain_type = domain
        self.relationship_subtype = subtype
        self.domain_specific_weight = weight
    
    def update_strength(self, new_strength: float, 
                       co_occurrence_boost: bool = False):
        """
        Update relationship strength score
        
        Args:
            new_strength: New strength value
            co_occurrence_boost: Whether to boost based on co-occurrence
        """
        self.strength_score = new_strength
        
        if co_occurrence_boost:
            self.co_occurrence_count += 1
            # Boost strength based on co-occurrence (with diminishing returns)
            boost = min(1.5, 1 + (0.1 * self.co_occurrence_count))
            self.strength_score *= boost
        
        # Recalculate adjusted confidence
        self.confidence_adjusted = self.confidence * self.strength_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation"""
        return {
            'id': self.get_relationship_id(),
            'source_entity': self.source_entity,
            'target_entity': self.target_entity,
            'relationship_type': self.relationship_type,
            'confidence': self.confidence,
            'context': self.context,
            'properties': self.properties,
            'traversal_weight': self.traversal_weight,
            'discovery_context': self.discovery_context,
            'bidirectional': self.bidirectional,
            'strength_score': self.strength_score,
            'traversal_count': self.traversal_count,
            'last_traversed': self.last_traversed.isoformat() if self.last_traversed else None,
            'traversal_direction': self.traversal_direction,
            'discovery_depth': self.discovery_depth,
            'discovery_timestamp': self.discovery_timestamp.isoformat() if self.discovery_timestamp else None,
            'discovered_via_path': self.discovered_via_path,
            'confidence_adjusted': self.confidence_adjusted,
            'semantic_similarity': self.semantic_similarity,
            'co_occurrence_count': self.co_occurrence_count,
            'domain_type': self.domain_type,
            'domain_specific_weight': self.domain_specific_weight,
            'relationship_subtype': self.relationship_subtype,
            'priority_level': self.priority_level,
            'traversal_score': self.calculate_traversal_score()
        }
    
    @classmethod
    def from_extracted_relationship(cls, relationship: ExtractedRelationship,
                                   discovery_depth: int = 0,
                                   bidirectional: bool = False) -> 'RadiatingRelationship':
        """
        Create RadiatingRelationship from base ExtractedRelationship
        
        Args:
            relationship: Base ExtractedRelationship
            discovery_depth: Depth at which relationship was discovered
            bidirectional: Whether relationship is bidirectional
            
        Returns:
            RadiatingRelationship instance
        """
        return cls(
            source_entity=relationship.source_entity,
            target_entity=relationship.target_entity,
            relationship_type=relationship.relationship_type,
            confidence=relationship.confidence,
            context=relationship.context,
            properties=relationship.properties,
            discovery_depth=discovery_depth,
            bidirectional=bidirectional
        )