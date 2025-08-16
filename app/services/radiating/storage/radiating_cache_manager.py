"""
Radiating Cache Manager

Redis-based caching for the radiating traversal system.
Manages caching of traversal patterns, entity neighbors, and path results.
"""

import logging
import json
import pickle
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import hashlib

from app.core.redis_client import get_redis_client
from app.services.radiating.models.radiating_entity import RadiatingEntity
from app.services.radiating.models.radiating_relationship import RadiatingRelationship
from app.services.radiating.models.radiating_graph import RadiatingGraph

logger = logging.getLogger(__name__)


class RadiatingCacheManager:
    """
    Cache manager for radiating traversal system using Redis.
    Provides efficient caching of traversal patterns and results.
    """
    
    def __init__(self, cache_prefix: str = "radiating"):
        """
        Initialize RadiatingCacheManager
        
        Args:
            cache_prefix: Prefix for all cache keys
        """
        self.redis_client = get_redis_client()
        self.cache_prefix = cache_prefix
        
        # Cache TTLs (in seconds)
        self.entity_ttl = 3600  # 1 hour
        self.relationship_ttl = 3600  # 1 hour
        self.path_ttl = 1800  # 30 minutes
        self.pattern_ttl = 7200  # 2 hours
        self.graph_ttl = 900  # 15 minutes
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
    
    def _get_key(self, key_type: str, identifier: str) -> str:
        """
        Generate cache key with prefix
        
        Args:
            key_type: Type of cache key
            identifier: Unique identifier
            
        Returns:
            Formatted cache key
        """
        return f"{self.cache_prefix}:{key_type}:{identifier}"
    
    def _generate_hash(self, data: Any) -> str:
        """
        Generate hash for complex data structures
        
        Args:
            data: Data to hash
            
        Returns:
            Hash string
        """
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    async def get_entity_neighbors(self, entity_id: str, 
                                  depth: int) -> Optional[List[RadiatingEntity]]:
        """
        Get cached entity neighbors
        
        Args:
            entity_id: Entity ID
            depth: Traversal depth
            
        Returns:
            List of RadiatingEntity objects or None if not cached
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._get_key("neighbors", f"{entity_id}:{depth}")
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                self.stats['hits'] += 1
                # Deserialize entities
                entities_data = json.loads(cached_data)
                entities = []
                
                for entity_dict in entities_data:
                    entity = RadiatingEntity(
                        text=entity_dict['text'],
                        label=entity_dict['label'],
                        start_char=entity_dict['start_char'],
                        end_char=entity_dict['end_char'],
                        confidence=entity_dict['confidence'],
                        canonical_form=entity_dict['canonical_form'],
                        properties=entity_dict.get('properties', {}),
                        traversal_depth=entity_dict.get('traversal_depth', depth),
                        discovery_source=entity_dict.get('discovery_source', 'cache'),
                        relevance_score=entity_dict.get('relevance_score', 1.0)
                    )
                    entities.append(entity)
                
                logger.debug(f"Cache hit for neighbors of {entity_id} at depth {depth}")
                return entities
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached neighbors: {e}")
            return None
    
    async def set_entity_neighbors(self, entity_id: str, depth: int,
                                  neighbors: List[RadiatingEntity]):
        """
        Cache entity neighbors
        
        Args:
            entity_id: Entity ID
            depth: Traversal depth
            neighbors: List of RadiatingEntity objects
        """
        if not self.redis_client:
            return
        
        try:
            key = self._get_key("neighbors", f"{entity_id}:{depth}")
            
            # Serialize entities
            entities_data = []
            for entity in neighbors:
                entities_data.append({
                    'text': entity.text,
                    'label': entity.label,
                    'start_char': entity.start_char,
                    'end_char': entity.end_char,
                    'confidence': entity.confidence,
                    'canonical_form': entity.canonical_form,
                    'properties': entity.properties,
                    'traversal_depth': entity.traversal_depth,
                    'discovery_source': entity.discovery_source,
                    'relevance_score': entity.relevance_score
                })
            
            self.redis_client.setex(
                key,
                self.entity_ttl,
                json.dumps(entities_data)
            )
            self.stats['sets'] += 1
            
            logger.debug(f"Cached {len(neighbors)} neighbors for {entity_id} at depth {depth}")
            
        except Exception as e:
            logger.error(f"Error caching neighbors: {e}")
    
    async def get_entity_relationships(self, entity_id: str,
                                      direction: str = "both") -> Optional[List[RadiatingRelationship]]:
        """
        Get cached entity relationships
        
        Args:
            entity_id: Entity ID
            direction: Relationship direction
            
        Returns:
            List of RadiatingRelationship objects or None if not cached
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._get_key("relationships", f"{entity_id}:{direction}")
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                self.stats['hits'] += 1
                # Deserialize relationships
                rels_data = json.loads(cached_data)
                relationships = []
                
                for rel_dict in rels_data:
                    rel = RadiatingRelationship(
                        source_entity=rel_dict['source_entity'],
                        target_entity=rel_dict['target_entity'],
                        relationship_type=rel_dict['relationship_type'],
                        confidence=rel_dict['confidence'],
                        context=rel_dict.get('context', ''),
                        properties=rel_dict.get('properties', {}),
                        traversal_weight=rel_dict.get('traversal_weight', 1.0),
                        bidirectional=rel_dict.get('bidirectional', False),
                        strength_score=rel_dict.get('strength_score', 1.0)
                    )
                    relationships.append(rel)
                
                return relationships
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached relationships: {e}")
            return None
    
    async def set_entity_relationships(self, entity_id: str, direction: str,
                                      relationships: List[RadiatingRelationship]):
        """
        Cache entity relationships
        
        Args:
            entity_id: Entity ID
            direction: Relationship direction
            relationships: List of RadiatingRelationship objects
        """
        if not self.redis_client:
            return
        
        try:
            key = self._get_key("relationships", f"{entity_id}:{direction}")
            
            # Serialize relationships
            rels_data = []
            for rel in relationships:
                rels_data.append({
                    'source_entity': rel.source_entity,
                    'target_entity': rel.target_entity,
                    'relationship_type': rel.relationship_type,
                    'confidence': rel.confidence,
                    'context': rel.context,
                    'properties': rel.properties,
                    'traversal_weight': rel.traversal_weight,
                    'bidirectional': rel.bidirectional,
                    'strength_score': rel.strength_score
                })
            
            self.redis_client.setex(
                key,
                self.relationship_ttl,
                json.dumps(rels_data)
            )
            self.stats['sets'] += 1
            
        except Exception as e:
            logger.error(f"Error caching relationships: {e}")
    
    async def get_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """
        Get cached path between entities
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            
        Returns:
            List of entity IDs in path or None if not cached
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._get_key("path", f"{start_id}:{end_id}")
            cached_path = self.redis_client.get(key)
            
            if cached_path:
                self.stats['hits'] += 1
                return json.loads(cached_path)
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached path: {e}")
            return None
    
    async def set_path(self, start_id: str, end_id: str, path: List[str]):
        """
        Cache path between entities
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            path: List of entity IDs in path
        """
        if not self.redis_client:
            return
        
        try:
            key = self._get_key("path", f"{start_id}:{end_id}")
            self.redis_client.setex(
                key,
                self.path_ttl,
                json.dumps(path)
            )
            self.stats['sets'] += 1
            
            # Also cache reverse path
            reverse_key = self._get_key("path", f"{end_id}:{start_id}")
            self.redis_client.setex(
                reverse_key,
                self.path_ttl,
                json.dumps(list(reversed(path)))
            )
            
        except Exception as e:
            logger.error(f"Error caching path: {e}")
    
    async def get_traversal_pattern(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached traversal pattern for a query
        
        Args:
            query_hash: Hash of the query
            
        Returns:
            Traversal pattern dictionary or None if not cached
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._get_key("pattern", query_hash)
            cached_pattern = self.redis_client.get(key)
            
            if cached_pattern:
                self.stats['hits'] += 1
                return json.loads(cached_pattern)
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached pattern: {e}")
            return None
    
    async def set_traversal_pattern(self, query_hash: str, pattern: Dict[str, Any]):
        """
        Cache traversal pattern for a query
        
        Args:
            query_hash: Hash of the query
            pattern: Traversal pattern dictionary
        """
        if not self.redis_client:
            return
        
        try:
            key = self._get_key("pattern", query_hash)
            self.redis_client.setex(
                key,
                self.pattern_ttl,
                json.dumps(pattern)
            )
            self.stats['sets'] += 1
            
        except Exception as e:
            logger.error(f"Error caching pattern: {e}")
    
    async def get_subgraph(self, graph_id: str) -> Optional[RadiatingGraph]:
        """
        Get cached subgraph
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            RadiatingGraph object or None if not cached
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._get_key("graph", graph_id)
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                self.stats['hits'] += 1
                # Use pickle for complex graph structure
                return pickle.loads(cached_data.encode('latin-1'))
            else:
                self.stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached graph: {e}")
            return None
    
    async def set_subgraph(self, graph_id: str, graph: RadiatingGraph):
        """
        Cache subgraph
        
        Args:
            graph_id: Graph identifier
            graph: RadiatingGraph object
        """
        if not self.redis_client:
            return
        
        try:
            key = self._get_key("graph", graph_id)
            # Use pickle for complex graph structure
            graph_data = pickle.dumps(graph).decode('latin-1')
            self.redis_client.setex(
                key,
                self.graph_ttl,
                graph_data
            )
            self.stats['sets'] += 1
            
        except Exception as e:
            logger.error(f"Error caching graph: {e}")
    
    async def invalidate_entity_cache(self, entity_id: str):
        """
        Invalidate all cache entries for an entity
        
        Args:
            entity_id: Entity ID to invalidate
        """
        if not self.redis_client:
            return
        
        try:
            # Find and delete all keys related to this entity
            patterns = [
                f"{self.cache_prefix}:neighbors:{entity_id}:*",
                f"{self.cache_prefix}:relationships:{entity_id}:*",
                f"{self.cache_prefix}:path:{entity_id}:*",
                f"{self.cache_prefix}:path:*:{entity_id}"
            ]
            
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    self.stats['evictions'] += len(keys)
            
            logger.debug(f"Invalidated cache for entity {entity_id}")
            
        except Exception as e:
            logger.error(f"Error invalidating entity cache: {e}")
    
    async def warm_cache_for_entities(self, entity_ids: List[str]):
        """
        Pre-warm cache for a list of entities
        
        Args:
            entity_ids: List of entity IDs to warm cache for
        """
        # This would typically fetch data from Neo4j and cache it
        # Implementation depends on specific warming strategy
        logger.info(f"Cache warming requested for {len(entity_ids)} entities")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary of cache statistics
        """
        hit_rate = 0.0
        total_requests = self.stats['hits'] + self.stats['misses']
        if total_requests > 0:
            hit_rate = self.stats['hits'] / total_requests
        
        return {
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'sets': self.stats['sets'],
            'evictions': self.stats['evictions'],
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def reset_statistics(self):
        """Reset cache statistics"""
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
    
    async def clear_all_cache(self):
        """Clear all radiating cache entries"""
        if not self.redis_client:
            return
        
        try:
            pattern = f"{self.cache_prefix}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} radiating cache entries")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


# Singleton instance
_radiating_cache_manager: Optional[RadiatingCacheManager] = None


def get_radiating_cache_manager() -> RadiatingCacheManager:
    """Get or create RadiatingCacheManager singleton"""
    global _radiating_cache_manager
    if _radiating_cache_manager is None:
        _radiating_cache_manager = RadiatingCacheManager()
    return _radiating_cache_manager