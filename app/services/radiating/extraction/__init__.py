"""
Universal Extraction System for Radiating Coverage

Provides domain-agnostic entity and relationship extraction capabilities
using LLM intelligence without hardcoded patterns or types.
"""

from .universal_entity_extractor import UniversalEntityExtractor
from .relationship_discoverer import RelationshipDiscoverer

__all__ = [
    'UniversalEntityExtractor',
    'RelationshipDiscoverer'
]