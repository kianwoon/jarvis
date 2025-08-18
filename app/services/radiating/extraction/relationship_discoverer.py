"""
Relationship Discoverer for Radiating Coverage System

Discovers relationships between entities using LLM intelligence.
Supports pattern-based detection and bidirectional relationship discovery.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib

from app.core.radiating_settings_cache import get_extraction_config, get_radiating_settings, get_prompt
from .universal_entity_extractor import ExtractedEntity

logger = logging.getLogger(__name__)

@dataclass
class DiscoveredRelationship:
    """Represents a discovered relationship between entities"""
    source_entity: ExtractedEntity
    target_entity: ExtractedEntity
    relationship_type: str
    confidence: float
    bidirectional: bool
    context: str
    metadata: Dict[str, Any]
    relationship_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.relationship_id:
            self.relationship_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID for relationship"""
        hash_input = f"{self.source_entity.entity_id}_{self.relationship_type}_{self.target_entity.entity_id}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

class RelationshipDiscoverer:
    """
    Discovers relationships between entities using LLM and pattern detection.
    Adapts to any domain without predefined relationship types.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the relationship discoverer.
        
        Args:
            llm_client: Optional LLM client for discovery
        """
        self.config = get_extraction_config()
        self.llm_client = llm_client
        self._init_llm_if_needed()
        
        # Cache for discovered relationship types
        self.discovered_rel_types: Dict[str, Dict[str, Any]] = {}
        
        # Pattern templates for common relationships
        self.relationship_patterns = self._init_relationship_patterns()
    
    def _init_llm_if_needed(self):
        """Initialize LLM client if not provided"""
        if not self.llm_client:
            try:
                from app.llm.ollama import JarvisLLM
                settings = get_radiating_settings()
                model_config = settings.get('model_config', {})
                
                # Initialize JarvisLLM with max_tokens from config
                max_tokens = model_config.get('max_tokens', 4096)
                llm_mode = model_config.get('llm_mode', 'non-thinking')  # Get mode from config
                self.llm_client = JarvisLLM(mode=llm_mode, max_tokens=max_tokens)
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self.llm_client = None
    
    def _init_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Initialize pattern templates for relationship detection"""
        return [
            {
                'pattern': '{source} is {relationship} {target}',
                'type': 'IS_A',
                'bidirectional': False
            },
            {
                'pattern': '{source} has {target}',
                'type': 'HAS',
                'bidirectional': False
            },
            {
                'pattern': '{source} belongs to {target}',
                'type': 'BELONGS_TO',
                'bidirectional': False
            },
            {
                'pattern': '{source} works with {target}',
                'type': 'WORKS_WITH',
                'bidirectional': True
            },
            {
                'pattern': '{source} causes {target}',
                'type': 'CAUSES',
                'bidirectional': False
            },
            {
                'pattern': '{source} leads to {target}',
                'type': 'LEADS_TO',
                'bidirectional': False
            },
            {
                'pattern': '{source} depends on {target}',
                'type': 'DEPENDS_ON',
                'bidirectional': False
            },
            {
                'pattern': '{source} is part of {target}',
                'type': 'PART_OF',
                'bidirectional': False
            },
            {
                'pattern': '{source} and {target}',
                'type': 'ASSOCIATED_WITH',
                'bidirectional': True
            }
        ]
    
    async def discover_relationships(
        self,
        text: str,
        entities: List[ExtractedEntity],
        domain_hints: Optional[List[str]] = None
    ) -> List[DiscoveredRelationship]:
        """
        Discover relationships between entities in text.
        
        Args:
            text: Source text containing entities
            entities: List of extracted entities
            domain_hints: Optional domain hints
            
        Returns:
            List of discovered relationships
        """
        
        if len(entities) < 2:
            return []  # Need at least 2 entities for relationships
        
        relationships = []
        
        try:
            # Use LLM to discover relationships
            if self.llm_client:
                llm_relationships = await self._discover_with_llm(
                    text, entities, domain_hints
                )
                relationships.extend(llm_relationships)
            
            # Use pattern detection if enabled
            if self.config.get('enable_pattern_detection', True):
                pattern_relationships = self._discover_with_patterns(text, entities)
                relationships.extend(pattern_relationships)
            
            # Deduplicate relationships
            unique_relationships = self._deduplicate_relationships(relationships)
            
            # Apply bidirectional expansion if enabled
            if self.config.get('bidirectional_relationships', True):
                unique_relationships = self._expand_bidirectional(unique_relationships)
            
            # Filter by confidence threshold
            min_confidence = self.config.get('relationship_confidence_threshold', 0.65)
            filtered = [r for r in unique_relationships if r.confidence >= min_confidence]
            
            # Limit to max relationships
            max_relationships = self.config.get('max_relationships_per_query', 30)
            return filtered[:max_relationships]
            
        except Exception as e:
            logger.error(f"Error discovering relationships: {e}")
            return []
    
    async def _discover_with_llm(
        self,
        text: str,
        entities: List[ExtractedEntity],
        domain_hints: Optional[List[str]] = None
    ) -> List[DiscoveredRelationship]:
        """Discover relationships using LLM"""
        
        if not self.llm_client:
            return []
        
        # Prepare entity list for prompt
        entity_list = [
            f"{i+1}. {e.text} ({e.entity_type})"
            for i, e in enumerate(entities[:20])  # Limit to prevent prompt overflow
        ]
        
        domain_context = f"in the context of {', '.join(domain_hints)}" if domain_hints else ""
        
        # Get prompt template from settings
        prompt_template = get_prompt(
            'relationship_discovery',
            'relationship_analysis',
            # Fallback prompt if not in database
            """Analyze the relationships between these entities {domain_context}:
        
        Entities:
        {entity_list}
        
        Text:
        {text}
        
        Discover all meaningful relationships between the entities.
        Be specific about relationship types - avoid generic "RELATED_TO".
        
        Return as JSON array..."""
        )
        
        prompt = prompt_template.format(
            domain_context=domain_context,
            entity_list=chr(10).join(entity_list),
            text=text[:2000]
        )
        
        try:
            response = await self.llm_client.invoke(prompt)
            relationships_data = json.loads(response)
            
            relationships = []
            for rel_data in relationships_data:
                source_idx = rel_data['source_index'] - 1
                target_idx = rel_data['target_index'] - 1
                
                if 0 <= source_idx < len(entities) and 0 <= target_idx < len(entities):
                    relationship = DiscoveredRelationship(
                        source_entity=entities[source_idx],
                        target_entity=entities[target_idx],
                        relationship_type=rel_data['relationship_type'],
                        confidence=rel_data.get('confidence', 0.7),
                        bidirectional=rel_data.get('bidirectional', False),
                        context=rel_data.get('context', ''),
                        metadata={
                            'reasoning': rel_data.get('reasoning', ''),
                            'discovery_method': 'llm'
                        }
                    )
                    relationships.append(relationship)
                    
                    # Update discovered types cache
                    self._update_discovered_types(rel_data['relationship_type'])
            
            return relationships
            
        except Exception as e:
            logger.error(f"LLM relationship discovery failed: {e}")
            return []
    
    def _discover_with_patterns(
        self,
        text: str,
        entities: List[ExtractedEntity]
    ) -> List[DiscoveredRelationship]:
        """Discover relationships using pattern matching"""
        
        relationships = []
        text_lower = text.lower()
        
        # Check each pair of entities
        for i, source in enumerate(entities):
            for j, target in enumerate(entities):
                if i >= j:  # Skip self and already processed pairs
                    continue
                
                # Look for patterns between entities
                source_text = source.text.lower()
                target_text = target.text.lower()
                
                # Find text between entities
                between_text = self._extract_text_between(
                    text_lower, source_text, target_text
                )
                
                if between_text:
                    # Check against patterns
                    for pattern_info in self.relationship_patterns:
                        if self._matches_pattern(
                            between_text, 
                            pattern_info['pattern']
                        ):
                            relationship = DiscoveredRelationship(
                                source_entity=source,
                                target_entity=target,
                                relationship_type=pattern_info['type'],
                                confidence=0.6,  # Base confidence for pattern match
                                bidirectional=pattern_info['bidirectional'],
                                context=between_text[:200],
                                metadata={
                                    'discovery_method': 'pattern',
                                    'pattern': pattern_info['pattern']
                                }
                            )
                            relationships.append(relationship)
                            break
                
                # Check for proximity-based relationships
                if self._are_proximate(text_lower, source_text, target_text):
                    relationship = DiscoveredRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type='ASSOCIATED_WITH',
                        confidence=0.5,  # Lower confidence for proximity
                        bidirectional=True,
                        context=self._get_proximity_context(
                            text, source.text, target.text
                        ),
                        metadata={
                            'discovery_method': 'proximity'
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_text_between(
        self,
        text: str,
        entity1: str,
        entity2: str,
        max_distance: int = 100
    ) -> Optional[str]:
        """Extract text between two entities"""
        
        # Find positions of both entities
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return None
        
        # Ensure reasonable distance
        distance = abs(pos2 - pos1)
        if distance > max_distance:
            return None
        
        # Extract text between
        if pos1 < pos2:
            start = pos1 + len(entity1)
            end = pos2
        else:
            start = pos2 + len(entity2)
            end = pos1
        
        between = text[start:end].strip()
        
        # Include entities in context
        if pos1 < pos2:
            return f"{entity1} {between} {entity2}"
        else:
            return f"{entity2} {between} {entity1}"
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a relationship pattern"""
        
        # Simple pattern matching - could be enhanced with regex
        pattern_words = pattern.replace('{source}', '').replace('{target}', '').strip().split()
        
        for word in pattern_words:
            if word.lower() in text.lower():
                return True
        
        return False
    
    def _are_proximate(
        self,
        text: str,
        entity1: str,
        entity2: str,
        max_distance: int = 50
    ) -> bool:
        """Check if two entities are in close proximity"""
        
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return False
        
        # Count words between entities
        start = min(pos1, pos2)
        end = max(pos1, pos2)
        between = text[start:end]
        word_count = len(between.split())
        
        return word_count <= max_distance
    
    def _get_proximity_context(
        self,
        text: str,
        entity1: str,
        entity2: str,
        context_size: int = 50
    ) -> str:
        """Get context around two proximate entities"""
        
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return ""
        
        start = max(0, min(pos1, pos2) - context_size)
        end = min(len(text), max(pos1 + len(entity1), pos2 + len(entity2)) + context_size)
        
        return text[start:end]
    
    def _deduplicate_relationships(
        self,
        relationships: List[DiscoveredRelationship]
    ) -> List[DiscoveredRelationship]:
        """Remove duplicate relationships, keeping highest confidence"""
        
        unique = {}
        
        for rel in relationships:
            # Create key for deduplication
            key = (
                rel.source_entity.entity_id,
                rel.target_entity.entity_id,
                rel.relationship_type
            )
            
            if key in unique:
                # Keep relationship with higher confidence
                if rel.confidence > unique[key].confidence:
                    unique[key] = rel
            else:
                unique[key] = rel
        
        return list(unique.values())
    
    def _expand_bidirectional(
        self,
        relationships: List[DiscoveredRelationship]
    ) -> List[DiscoveredRelationship]:
        """Expand bidirectional relationships to include reverse direction"""
        
        expanded = list(relationships)
        
        for rel in relationships:
            if rel.bidirectional:
                # Create reverse relationship
                reverse = DiscoveredRelationship(
                    source_entity=rel.target_entity,
                    target_entity=rel.source_entity,
                    relationship_type=rel.relationship_type,
                    confidence=rel.confidence * 0.9,  # Slightly lower confidence
                    bidirectional=True,
                    context=rel.context,
                    metadata={
                        **rel.metadata,
                        'reversed_from': rel.relationship_id
                    }
                )
                expanded.append(reverse)
        
        return expanded
    
    def _update_discovered_types(self, relationship_type: str):
        """Update cache of discovered relationship types"""
        
        if relationship_type not in self.discovered_rel_types:
            self.discovered_rel_types[relationship_type] = {
                'count': 1,
                'first_seen': 'now',
                'confidence_sum': 0.7
            }
        else:
            self.discovered_rel_types[relationship_type]['count'] += 1
    
    async def discover_implicit_relationships(
        self,
        entities: List[ExtractedEntity],
        domain_hints: Optional[List[str]] = None
    ) -> List[DiscoveredRelationship]:
        """
        Discover implicit relationships based on entity properties.
        
        Args:
            entities: List of entities to analyze
            domain_hints: Optional domain context
            
        Returns:
            List of implicit relationships discovered
        """
        
        if not self.config.get('extract_implicit_relationships', False):
            return []
        
        if not self.llm_client or len(entities) < 2:
            return []
        
        # Group entities by type for type-based relationships
        entities_by_type = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)
        
        domain_context = f"in {', '.join(domain_hints)}" if domain_hints else ""
        
        # Get prompt template from settings
        entities_json = json.dumps([{'text': e.text, 'type': e.entity_type} for e in entities[:15]], indent=2)
        
        prompt_template = get_prompt(
            'relationship_discovery',
            'implicit_relationships',
            # Fallback prompt if not in database
            """Analyze these entities and infer implicit relationships {domain_context}:
        
        {entities_json}
        
        Infer relationships based on:
        1. Entity types (e.g., all people might work at same company)
        2. Naming patterns (e.g., similar names might indicate versions)
        3. Logical connections (e.g., cause and effect)
        4. Domain knowledge
        
        Return as JSON array of implicit relationships.
        Only include high-confidence inferences."""
        )
        
        prompt = prompt_template.format(
            domain_context=domain_context,
            entities_json=entities_json
        )
        
        try:
            response = await self.llm_client.invoke(prompt)
            implicit_rels = json.loads(response)
            
            # Convert to DiscoveredRelationship objects
            relationships = []
            for rel_data in implicit_rels:
                # Find matching entities
                source = next(
                    (e for e in entities if e.text == rel_data['source']),
                    None
                )
                target = next(
                    (e for e in entities if e.text == rel_data['target']),
                    None
                )
                
                if source and target:
                    relationship = DiscoveredRelationship(
                        source_entity=source,
                        target_entity=target,
                        relationship_type=rel_data['type'],
                        confidence=rel_data.get('confidence', 0.6),
                        bidirectional=rel_data.get('bidirectional', False),
                        context="Implicit relationship",
                        metadata={
                            'discovery_method': 'implicit',
                            'reasoning': rel_data.get('reasoning', '')
                        }
                    )
                    relationships.append(relationship)
            
            return relationships
            
        except Exception as e:
            logger.debug(f"Implicit relationship discovery failed: {e}")
            return []
    
    def get_discovered_types_summary(self) -> Dict[str, Any]:
        """Get summary of discovered relationship types"""
        
        sorted_types = sorted(
            self.discovered_rel_types.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        return {
            'total_types': len(self.discovered_rel_types),
            'most_common': [
                {
                    'type': rel_type,
                    'count': info['count'],
                    'avg_confidence': info['confidence_sum'] / info['count']
                }
                for rel_type, info in sorted_types[:10]
            ]
        }
    
    def reset_discovered_types(self):
        """Reset the cache of discovered relationship types"""
        self.discovered_rel_types = {}
        logger.info("Reset discovered relationship types cache")