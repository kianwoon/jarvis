"""
Knowledge Graph Type Definitions

Shared dataclasses and types for pure LLM-driven knowledge graph extraction.
Separated to avoid circular imports between services.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class ExtractedEntity:
    """Represents an LLM-extracted entity from text"""
    text: str
    label: str  # Entity type discovered by LLM
    start_char: int
    end_char: int
    confidence: float = 1.0
    canonical_form: str = None  # LLM-normalized form
    properties: Dict[str, Any] = None  # Additional properties discovered by LLM
    
    def __post_init__(self):
        if self.canonical_form is None:
            self.canonical_form = self.text.strip()
        if self.properties is None:
            self.properties = {}

@dataclass
class ExtractedRelationship:
    """Represents an LLM-discovered relationship between entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float
    context: str  # Context where relationship was discovered
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class GraphExtractionResult:
    """Result of LLM-driven knowledge graph extraction"""
    chunk_id: str
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    processing_time_ms: float
    source_metadata: Dict[str, Any]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []