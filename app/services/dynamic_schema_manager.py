"""
Dynamic Schema Manager for LLM-driven knowledge graph schema discovery.
Replaces static entity/relationship types with intelligent, context-aware discovery.
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from app.core.redis_client import get_redis_client
from app.core.knowledge_graph_settings_cache import get_knowledge_graph_settings


@dataclass
class DiscoveredEntityType:
    """Represents a dynamically discovered entity type"""
    type: str
    description: str
    examples: List[str]
    confidence: float
    frequency: int
    first_seen: datetime
    last_seen: datetime
    status: str  # 'pending', 'accepted', 'rejected', 'deprecated'
    domain: Optional[str] = None
    parent_type: Optional[str] = None
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class DiscoveredRelationshipType:
    """Represents a dynamically discovered relationship type"""
    type: str
    description: str
    inverse: Optional[str]
    examples: List[str]
    confidence: float
    frequency: int
    first_seen: datetime
    last_seen: datetime
    status: str  # 'pending', 'accepted', 'rejected', 'deprecated'
    domain_types: List[str] = None
    range_types: List[str] = None
    
    def __post_init__(self):
        if self.domain_types is None:
            self.domain_types = []
        if self.range_types is None:
            self.range_types = []


class DynamicSchemaManager:
    """
    Manages dynamic discovery and evolution of knowledge graph schemas
    using LLM-driven analysis of document content.
    """
    
    def __init__(self):
        self.redis = get_redis_client()
        self.entity_cache_key = "kg:discovered_entities"
        self.relationship_cache_key = "kg:discovered_relationships"
        self.schema_cache_key = "kg:schema_metadata"
        self.discovery_stats_key = "kg:discovery_stats"
        
    async def discover_entity_types(self, text: str, context: Dict[str, Any] = None) -> List[DiscoveredEntityType]:
        """
        Discover new entity types from text using LLM analysis
        """
        settings = get_knowledge_graph_settings()
        discovery_config = settings.get('entity_discovery', {})
        
        if not discovery_config.get('enabled', False):
            return []
            
        # Get existing entities for context
        existing_entities = await self.get_current_entities()
        
        # Build LLM prompt
        prompt = self._build_entity_discovery_prompt(text, existing_entities, context)
        
        # Call LLM for discovery
        llm_response = await self._call_llm_for_discovery(prompt, settings)
        
        # Parse and validate response
        discovered_entities = self._parse_entity_discovery_response(llm_response)
        
        # Filter by confidence threshold
        filtered_entities = [
            entity for entity in discovered_entities
            if entity.confidence >= discovery_config.get('confidence_threshold', 0.75)
        ]
        
        # Update cache
        await self._update_entity_cache(filtered_entities)
        
        return filtered_entities
    
    async def discover_relationship_types(self, entities: List[str], context: Dict[str, Any] = None) -> List[DiscoveredRelationshipType]:
        """
        Discover new relationship types from entity interactions
        """
        settings = get_knowledge_graph_settings()
        discovery_config = settings.get('relationship_discovery', {})
        
        if not discovery_config.get('enabled', False):
            return []
            
        # Get existing relationships for context
        existing_relationships = await self.get_current_relationships()
        
        # Build LLM prompt
        prompt = self._build_relationship_discovery_prompt(entities, existing_relationships, context)
        
        # Call LLM for discovery
        llm_response = await self._call_llm_for_discovery(prompt, settings)
        
        # Parse and validate response
        discovered_relationships = self._parse_relationship_discovery_response(llm_response)
        
        # Filter by confidence threshold
        filtered_relationships = [
            rel for rel in discovered_relationships
            if rel.confidence >= discovery_config.get('confidence_threshold', 0.7)
        ]
        
        # Update cache
        await self._update_relationship_cache(filtered_relationships)
        
        return filtered_relationships
    
    async def get_dynamic_schema(self, domain: str = None) -> Dict[str, Any]:
        """
        Get current dynamic schema for a specific domain
        """
        entities = await self.get_current_entities(domain)
        relationships = await self.get_current_relationships(domain)
        stats = await self.get_discovery_stats()
        
        return {
            'entities': entities,
            'relationships': relationships,
            'stats': stats,
            'last_updated': stats.get('last_updated'),
            'version': stats.get('version', '1.0.0')
        }
    
    async def get_current_entities(self, domain: str = None) -> List[DiscoveredEntityType]:
        """Get current discovered entities"""
        try:
            entities_data = self.redis.hgetall(self.entity_cache_key)
            entities = []
            
            for entity_json in entities_data.values():
                entity_dict = json.loads(entity_json)
                if domain is None or entity_dict.get('domain') == domain:
                    entities.append(DiscoveredEntityType(**entity_dict))
            
            return sorted(entities, key=lambda x: (x.confidence, x.frequency), reverse=True)
        except Exception:
            return []
    
    async def get_current_relationships(self, domain: str = None) -> List[DiscoveredRelationshipType]:
        """Get current discovered relationships"""
        try:
            relationships_data = self.redis.hgetall(self.relationship_cache_key)
            relationships = []
            
            for rel_json in relationships_data.values():
                rel_dict = json.loads(rel_json)
                if domain is None or rel_dict.get('domain') == domain:
                    relationships.append(DiscoveredRelationshipType(**rel_dict))
            
            return sorted(relationships, key=lambda x: (x.confidence, x.frequency), reverse=True)
        except Exception:
            return []
    
    async def approve_entity_type(self, entity_type: str) -> bool:
        """Approve a discovered entity type for use"""
        try:
            entity_json = self.redis.hget(self.entity_cache_key, entity_type)
            if entity_json:
                entity = DiscoveredEntityType(**json.loads(entity_json))
                entity.status = 'accepted'
                entity.last_seen = datetime.utcnow()
                
                self.redis.hset(self.entity_cache_key, entity_type, json.dumps(asdict(entity)))
                return True
            return False
        except Exception:
            return False
    
    async def reject_entity_type(self, entity_type: str) -> bool:
        """Reject a discovered entity type"""
        try:
            entity_json = self.redis.hget(self.entity_cache_key, entity_type)
            if entity_json:
                entity = DiscoveredEntityType(**json.loads(entity_json))
                entity.status = 'rejected'
                entity.last_seen = datetime.utcnow()
                
                self.redis.hset(self.entity_cache_key, entity_type, json.dumps(asdict(entity)))
                return True
            return False
        except Exception:
            return False
    
    async def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        try:
            stats = self.redis.get(self.discovery_stats_key)
            if stats:
                return json.loads(stats)
            
            # Initialize with default stats
            default_stats = {
                'total_entities_discovered': 0,
                'total_relationships_discovered': 0,
                'entities_accepted': 0,
                'entities_rejected': 0,
                'relationships_accepted': 0,
                'relationships_rejected': 0,
                'last_discovery': None,
                'last_updated': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            }
            
            self.redis.set(self.discovery_stats_key, json.dumps(default_stats))
            return default_stats
        except Exception:
            return {}
    
    def _build_entity_discovery_prompt(self, text: str, existing_entities: List[DiscoveredEntityType], context: Dict[str, Any] = None) -> str:
        """Build LLM prompt for entity type discovery"""
        from app.services.settings_prompt_service import get_prompt_service as get_settings_prompt_service
        
        prompt_service = get_settings_prompt_service()
        existing_types = [e.type for e in existing_entities if e.status == 'accepted'][:20]
        
        return prompt_service.get_prompt(
            "entity_discovery",
            variables={
                "text": text[:2000],  # Truncate for performance
                "existing_types": existing_types
            }
        )
    
    def _build_relationship_discovery_prompt(self, entities: List[str], existing_relationships: List[DiscoveredRelationshipType], context: Dict[str, Any] = None) -> str:
        """Build LLM prompt for relationship type discovery"""
        from app.services.settings_prompt_service import get_prompt_service as get_settings_prompt_service
        
        prompt_service = get_settings_prompt_service()
        existing_types = [r.type for r in existing_relationships if r.status == 'accepted'][:15]
        
        return prompt_service.get_prompt(
            "relationship_discovery",
            variables={
                "entities": entities[:10],
                "existing_types": existing_types
            }
        )
    
    async def _call_llm_for_discovery(self, prompt: str, settings: Dict[str, Any]) -> str:
        """Call LLM for schema discovery"""
        # This would integrate with your existing LLM service
        # For now, return a mock response structure
        return await self._mock_llm_response(prompt)
    
    async def _mock_llm_response(self, prompt: str) -> str:
        """Mock LLM response for testing"""
        if "entity" in prompt.lower():
            return json.dumps({
                "discovered_entities": [
                    {
                        "type": "Technology",
                        "description": "Software, hardware, or technological concepts",
                        "examples": ["machine learning", "blockchain", "API"],
                        "confidence": 0.85,
                        "frequency": 5
                    },
                    {
                        "type": "Process",
                        "description": "Business or technical processes",
                        "examples": ["data processing", "user authentication"],
                        "confidence": 0.78,
                        "frequency": 3
                    }
                ]
            })
        else:
            return json.dumps({
                "discovered_relationships": [
                    {
                        "type": "implements",
                        "description": "One entity implements or uses another",
                        "inverse": "implemented_by",
                        "examples": ["System implements API"],
                        "confidence": 0.82,
                        "frequency": 4
                    },
                    {
                        "type": "depends_on",
                        "description": "One entity depends on another for functionality",
                        "inverse": "required_by",
                        "examples": ["Application depends_on Database"],
                        "confidence": 0.75,
                        "frequency": 3
                    }
                ]
            })
    
    def _parse_entity_discovery_response(self, response: str) -> List[DiscoveredEntityType]:
        """Parse LLM response for entity discovery"""
        try:
            data = json.loads(response)
            entities = []
            
            for entity_data in data.get('discovered_entities', []):
                entity = DiscoveredEntityType(
                    type=entity_data['type'],
                    description=entity_data['description'],
                    examples=entity_data['examples'],
                    confidence=entity_data['confidence'],
                    frequency=entity_data['frequency'],
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    status='pending'
                )
                entities.append(entity)
            
            return entities
        except Exception as e:
            print(f"Error parsing entity discovery response: {e}")
            return []
    
    def _parse_relationship_discovery_response(self, response: str) -> List[DiscoveredRelationshipType]:
        """Parse LLM response for relationship discovery"""
        try:
            data = json.loads(response)
            relationships = []
            
            for rel_data in data.get('discovered_relationships', []):
                relationship = DiscoveredRelationshipType(
                    type=rel_data['type'],
                    description=rel_data['description'],
                    inverse=rel_data.get('inverse'),
                    examples=rel_data['examples'],
                    confidence=rel_data['confidence'],
                    frequency=rel_data['frequency'],
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    status='pending'
                )
                relationships.append(relationship)
            
            return relationships
        except Exception as e:
            print(f"Error parsing relationship discovery response: {e}")
            return []
    
    async def _update_entity_cache(self, entities: List[DiscoveredEntityType]) -> None:
        """Update Redis cache with discovered entities"""
        for entity in entities:
            key = entity.type
            existing = self.redis.hget(self.entity_cache_key, key)
            
            if existing:
                # Update existing entity
                existing_entity = DiscoveredEntityType(**json.loads(existing))
                existing_entity.frequency += entity.frequency
                existing_entity.last_seen = datetime.utcnow()
                existing_entity.confidence = max(existing_entity.confidence, entity.confidence)
                entity = existing_entity
            
            self.redis.hset(self.entity_cache_key, key, json.dumps(asdict(entity)))
    
    async def _update_relationship_cache(self, relationships: List[DiscoveredRelationshipType]) -> None:
        """Update Redis cache with discovered relationships"""
        for rel in relationships:
            key = rel.type
            existing = self.redis.hget(self.relationship_cache_key, key)
            
            if existing:
                # Update existing relationship
                existing_rel = DiscoveredRelationshipType(**json.loads(existing))
                existing_rel.frequency += rel.frequency
                existing_rel.last_seen = datetime.utcnow()
                existing_rel.confidence = max(existing_rel.confidence, rel.confidence)
                rel = existing_rel
            
            self.redis.hset(self.relationship_cache_key, key, json.dumps(asdict(rel)))
    
    async def get_combined_schema(self) -> Dict[str, List[str]]:
        """Get pure LLM-discovered schema - no static fallbacks"""
        # Always use dynamic (LLM-discovered) schema - no static fallbacks
        dynamic_entities = await self.get_current_entities()
        dynamic_relationships = await self.get_current_relationships()
        
        return {
            'entity_types': [e.type for e in dynamic_entities if e.status == 'accepted'],
            'relationship_types': [r.type for r in dynamic_relationships if r.status == 'accepted']
        }


# Global instance for easy access
dynamic_schema_manager = DynamicSchemaManager()