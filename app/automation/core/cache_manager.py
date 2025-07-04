"""
Universal Cache Manager for Workflow Automation
Handles all cache operations across different workflow patterns
"""
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from app.automation.integrations.redis_bridge import workflow_redis

logger = logging.getLogger(__name__)


class CachePosition(Enum):
    """Defines where a cache node sits relative to other nodes"""
    INTERCEPTOR = "interceptor"     # A → Cache → B (in-line)
    STORAGE = "storage"             # A → B, A → Cache (side-branch)
    SHARED = "shared"               # A → Cache ← B (multi-source)
    HIERARCHICAL = "hierarchical"   # Cache1 → Cache2 (chained)


@dataclass
class CacheNodeInfo:
    """Information about a cache node and its relationships"""
    node_id: str
    position: CachePosition
    upstream_nodes: List[str]      # Nodes that feed into this cache
    downstream_nodes: List[str]    # Nodes that read from this cache
    config: Dict[str, Any]
    is_terminal: bool = False      # True if cache has no downstream nodes


@dataclass
class CacheCheckResult:
    """Result of checking cache for a node"""
    cache_hit: bool
    cached_data: Optional[Any] = None
    cache_key: Optional[str] = None
    cache_node_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UniversalCacheManager:
    """Manages all cache operations across workflow execution"""
    
    def __init__(self):
        self.cache_map: Dict[str, CacheNodeInfo] = {}
        self.node_cache_relationships: Dict[str, Set[str]] = {}
        
    def analyze_workflow(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """
        Analyze workflow to build complete cache dependency map
        Returns a comprehensive cache analysis
        """
        # Reset state
        self.cache_map.clear()
        self.node_cache_relationships.clear()
        
        # First pass: identify all cache nodes
        cache_nodes = {}
        for node in nodes:
            if node.get("type") == "cachenode" or node.get("data", {}).get("type") == "CacheNode":
                node_id = node["id"]
                cache_nodes[node_id] = {
                    "node": node,
                    "upstream": [],
                    "downstream": [],
                    "config": node.get("data", {}).get("node", {})
                }
        
        # Second pass: analyze edges to determine relationships
        for edge in edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            
            # Check if source or target is a cache node
            if target_id in cache_nodes:
                cache_nodes[target_id]["upstream"].append(source_id)
            
            if source_id in cache_nodes:
                cache_nodes[source_id]["downstream"].append(target_id)
        
        # Third pass: determine cache positions and create CacheNodeInfo
        for cache_id, cache_data in cache_nodes.items():
            position = self._determine_cache_position(
                cache_data["upstream"],
                cache_data["downstream"],
                nodes,
                edges
            )
            
            cache_info = CacheNodeInfo(
                node_id=cache_id,
                position=position,
                upstream_nodes=cache_data["upstream"],
                downstream_nodes=cache_data["downstream"],
                config=cache_data["config"],
                is_terminal=len(cache_data["downstream"]) == 0
            )
            
            self.cache_map[cache_id] = cache_info
            
            # Build reverse relationships (which nodes have which caches)
            for upstream_node in cache_data["upstream"]:
                if upstream_node not in self.node_cache_relationships:
                    self.node_cache_relationships[upstream_node] = set()
                self.node_cache_relationships[upstream_node].add(cache_id)
        
        # Return analysis
        return {
            "cache_nodes": len(self.cache_map),
            "cache_positions": {
                position.value: sum(1 for c in self.cache_map.values() if c.position == position)
                for position in CachePosition
            },
            "node_cache_map": {
                node_id: list(cache_ids)
                for node_id, cache_ids in self.node_cache_relationships.items()
            },
            "cache_details": {
                cache_id: {
                    "position": info.position.value,
                    "upstream_count": len(info.upstream_nodes),
                    "downstream_count": len(info.downstream_nodes),
                    "is_terminal": info.is_terminal
                }
                for cache_id, info in self.cache_map.items()
            }
        }
    
    def _determine_cache_position(
        self,
        upstream: List[str],
        downstream: List[str],
        nodes: List[Dict],
        edges: List[Dict]
    ) -> CachePosition:
        """Determine the position type of a cache node"""
        # Multiple upstream nodes = shared cache
        if len(upstream) > 1:
            return CachePosition.SHARED
        
        # Check if cache is in-line (interceptor)
        if len(upstream) == 1 and len(downstream) == 1:
            # Check if there's a direct edge from upstream to downstream
            direct_edge_exists = any(
                e.get("source") == upstream[0] and e.get("target") == downstream[0]
                for e in edges
            )
            if not direct_edge_exists:
                return CachePosition.INTERCEPTOR
        
        # Check if downstream is another cache (hierarchical)
        if downstream:
            downstream_types = []
            for node in nodes:
                if node["id"] in downstream:
                    node_type = node.get("type") or node.get("data", {}).get("type", "")
                    downstream_types.append(node_type)
            
            if any("cache" in t.lower() for t in downstream_types):
                return CachePosition.HIERARCHICAL
        
        # Default to storage (side-branch)
        return CachePosition.STORAGE
    
    async def check_all_caches(
        self,
        node_id: str,
        input_data: str,
        workflow_id: int,
        check_upstream: bool = True
    ) -> Optional[CacheCheckResult]:
        """
        Check all relevant caches for a node before execution
        Returns the first cache hit found, or None if no hits
        """
        # Get all cache nodes connected to this node
        connected_caches = self.node_cache_relationships.get(node_id, set())
        
        if not connected_caches:
            logger.debug(f"[CACHE] No cache nodes connected to {node_id}")
            return None
        
        # Check each connected cache
        for cache_id in connected_caches:
            cache_info = self.cache_map.get(cache_id)
            if not cache_info:
                continue
            
            # Check cache based on position type
            result = await self._check_single_cache(
                cache_info,
                node_id,
                input_data,
                workflow_id
            )
            
            if result and result.cache_hit:
                logger.info(f"[CACHE] Cache HIT for node {node_id} from cache {cache_id}")
                return result
        
        # If check_upstream is True, also check caches of upstream nodes
        if check_upstream:
            # TODO: Implement upstream cache checking for interceptor patterns
            pass
        
        logger.debug(f"[CACHE] No cache hits found for node {node_id}")
        return None
    
    async def _check_single_cache(
        self,
        cache_info: CacheNodeInfo,
        node_id: str,
        input_data: str,
        workflow_id: int
    ) -> Optional[CacheCheckResult]:
        """Check a single cache node for cached data"""
        try:
            # Generate cache key based on cache configuration
            cache_key = self._generate_cache_key(
                cache_info.config,
                workflow_id,
                cache_info.node_id,
                node_id,
                input_data
            )
            
            # Check cache
            cached_data = workflow_redis.get_value(cache_key)
            
            if cached_data:
                # Calculate metadata
                metadata = {
                    "cache_position": cache_info.position.value,
                    "cache_node_id": cache_info.node_id,
                    "retrieved_at": datetime.utcnow().isoformat()
                }
                
                return CacheCheckResult(
                    cache_hit=True,
                    cached_data=cached_data,
                    cache_key=cache_key,
                    cache_node_id=cache_info.node_id,
                    metadata=metadata
                )
            
            return CacheCheckResult(
                cache_hit=False,
                cache_key=cache_key,
                cache_node_id=cache_info.node_id
            )
            
        except Exception as e:
            logger.error(f"[CACHE] Error checking cache {cache_info.node_id}: {e}")
            return None
    
    async def store_in_all_caches(
        self,
        node_id: str,
        result_data: Any,
        workflow_id: int,
        input_data: str
    ) -> List[str]:
        """
        Store result in all connected cache nodes
        Returns list of cache keys where data was stored
        """
        stored_keys = []
        connected_caches = self.node_cache_relationships.get(node_id, set())
        
        for cache_id in connected_caches:
            cache_info = self.cache_map.get(cache_id)
            if not cache_info:
                continue
            
            # Don't store in upstream caches (they should cache their own upstream)
            if cache_info.position == CachePosition.INTERCEPTOR and node_id in cache_info.downstream_nodes:
                continue
            
            # Generate cache key
            cache_key = self._generate_cache_key(
                cache_info.config,
                workflow_id,
                cache_info.node_id,
                node_id,
                input_data
            )
            
            # Store in cache
            ttl = cache_info.config.get("ttl", 3600)
            success = workflow_redis.set_value(cache_key, result_data, expire=ttl)
            
            if success:
                stored_keys.append(cache_key)
                logger.info(f"[CACHE] Stored result for node {node_id} in cache {cache_id}")
            
        return stored_keys
    
    def _generate_cache_key(
        self,
        cache_config: Dict[str, Any],
        workflow_id: int,
        cache_node_id: str,
        source_node_id: str,
        input_data: str
    ) -> str:
        """Generate cache key using unified logic"""
        pattern = cache_config.get("cache_key_pattern", "auto")
        namespace = cache_config.get("cache_namespace", "default")
        custom_key = cache_config.get("cache_key", "")
        
        if pattern == "custom" and custom_key:
            return f"{namespace}:custom:{custom_key}"
        
        elif pattern == "node_only":
            return f"{namespace}:node:{cache_node_id}"
        
        elif pattern == "input_hash":
            input_hash = hashlib.md5(input_data.encode()).hexdigest()[:8]
            return f"{namespace}:input:{input_hash}"
        
        else:  # auto pattern (default)
            # Use consistent key format for all workflow types
            input_hash = hashlib.md5(input_data.encode()).hexdigest()[:8]
            return f"{namespace}:w{workflow_id}:c{cache_node_id}:s{source_node_id}:i{input_hash}"
    
    def get_cache_dependencies(self, node_id: str) -> Dict[str, List[str]]:
        """Get all cache dependencies for a node"""
        dependencies = {
            "direct_caches": [],
            "upstream_caches": [],
            "downstream_caches": []
        }
        
        # Direct caches
        direct_cache_ids = self.node_cache_relationships.get(node_id, set())
        dependencies["direct_caches"] = list(direct_cache_ids)
        
        # TODO: Implement upstream and downstream cache discovery
        
        return dependencies


# Global instance for reuse
_cache_manager = None

def get_cache_manager() -> UniversalCacheManager:
    """Get or create the global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = UniversalCacheManager()
    return _cache_manager