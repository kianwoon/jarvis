"""
Radiating Entity Model

Universal entity model that extends the existing ExtractedEntity with
radiating-specific properties for traversal and discovery.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from app.services.knowledge_graph_types import ExtractedEntity


@dataclass
class RadiatingEntity(ExtractedEntity):
    """
    Enhanced entity model for radiating traversal system.
    Extends the base ExtractedEntity with radiating-specific properties.
    """
    
    # Radiating-specific properties
    domain_metadata: Dict[str, Any] = field(default_factory=dict)
    traversal_depth: int = 0
    discovery_source: str = "initial"
    relevance_score: float = 1.0
    
    # Discovery tracking
    discovered_at: Optional[datetime] = None
    discovered_from_entity_id: Optional[str] = None
    discovery_path: List[str] = field(default_factory=list)
    
    # Traversal metadata
    visit_count: int = 0
    last_visited: Optional[datetime] = None
    connected_entity_ids: Set[str] = field(default_factory=set)
    
    # Domain-specific classifications
    domain_types: List[str] = field(default_factory=list)
    domain_importance: float = 0.5
    
    # Caching and optimization
    cached_neighbors: Optional[List[str]] = None
    neighbor_cache_timestamp: Optional[datetime] = None
    traversal_priority: float = 0.5
    
    def __post_init__(self):
        """Initialize inherited fields and set defaults"""
        super().__post_init__()
        
        # Set discovery timestamp if not provided
        if self.discovered_at is None:
            self.discovered_at = datetime.now()
        
        # Initialize discovery path if starting entity
        if not self.discovery_path and self.traversal_depth == 0:
            self.discovery_path = [self.get_entity_id()]
    
    def get_entity_id(self) -> str:
        """Generate unique entity identifier"""
        # Use properties id if available, otherwise create from canonical form and type
        if self.properties and 'id' in self.properties:
            return self.properties['id']
        
        import hashlib
        canonical_lower = self.canonical_form.lower().strip()
        entity_hash = hashlib.md5(f"{self.label}_{canonical_lower}".encode()).hexdigest()[:8]
        return f"{self.label}_{canonical_lower.replace(' ', '_')}_{entity_hash}"
    
    def update_relevance(self, new_score: float, decay_factor: float = 0.9):
        """
        Update relevance score with decay based on traversal depth
        
        Args:
            new_score: New relevance score to apply
            decay_factor: Factor to decay score by depth (default 0.9)
        """
        # Apply depth-based decay
        depth_decay = decay_factor ** self.traversal_depth
        self.relevance_score = new_score * depth_decay
    
    def mark_visited(self):
        """Mark entity as visited during traversal"""
        self.visit_count += 1
        self.last_visited = datetime.now()
    
    def add_connected_entity(self, entity_id: str):
        """Add a connected entity ID for tracking relationships"""
        self.connected_entity_ids.add(entity_id)
    
    def set_domain_classification(self, domain: str, importance: float = 0.5):
        """
        Set domain-specific classification for the entity
        
        Args:
            domain: Domain identifier (e.g., 'finance', 'technology', 'healthcare')
            importance: Domain-specific importance score (0.0 to 1.0)
        """
        if domain not in self.domain_types:
            self.domain_types.append(domain)
        self.domain_importance = max(self.domain_importance, importance)
    
    def add_domain_metadata(self, key: str, value: Any):
        """
        Add domain-specific metadata
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.domain_metadata[key] = value
    
    def get_traversal_weight(self) -> float:
        """
        Calculate traversal weight for prioritization
        
        Returns:
            Weight score combining relevance, importance, and priority
        """
        return (
            self.relevance_score * 0.4 +
            self.domain_importance * 0.3 +
            self.traversal_priority * 0.3
        )
    
    def should_traverse(self, min_relevance: float = 0.1, max_visits: int = 3) -> bool:
        """
        Determine if entity should be traversed further
        
        Args:
            min_relevance: Minimum relevance score threshold
            max_visits: Maximum number of visits allowed
            
        Returns:
            True if entity should be traversed, False otherwise
        """
        return (
            self.relevance_score >= min_relevance and
            self.visit_count < max_visits
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation"""
        return {
            'id': self.get_entity_id(),
            'text': self.text,
            'label': self.label,
            'canonical_form': self.canonical_form,
            'confidence': self.confidence,
            'properties': self.properties,
            'domain_metadata': self.domain_metadata,
            'traversal_depth': self.traversal_depth,
            'discovery_source': self.discovery_source,
            'relevance_score': self.relevance_score,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            'discovered_from_entity_id': self.discovered_from_entity_id,
            'discovery_path': self.discovery_path,
            'visit_count': self.visit_count,
            'last_visited': self.last_visited.isoformat() if self.last_visited else None,
            'connected_entity_ids': list(self.connected_entity_ids),
            'domain_types': self.domain_types,
            'domain_importance': self.domain_importance,
            'traversal_priority': self.traversal_priority,
            'traversal_weight': self.get_traversal_weight()
        }
    
    @classmethod
    def from_extracted_entity(cls, entity: ExtractedEntity, 
                             traversal_depth: int = 0,
                             discovery_source: str = "initial") -> 'RadiatingEntity':
        """
        Create RadiatingEntity from base ExtractedEntity
        
        Args:
            entity: Base ExtractedEntity
            traversal_depth: Initial traversal depth
            discovery_source: Source of discovery
            
        Returns:
            RadiatingEntity instance
        """
        return cls(
            text=entity.text,
            label=entity.label,
            start_char=entity.start_char,
            end_char=entity.end_char,
            confidence=entity.confidence,
            canonical_form=entity.canonical_form,
            properties=entity.properties,
            traversal_depth=traversal_depth,
            discovery_source=discovery_source
        )