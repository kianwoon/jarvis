"""
Cache Strategy

Multi-tier caching strategy for the radiating system with intelligent
invalidation, pattern-based warming, and hit rate tracking.
"""

import logging
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque, defaultdict

from app.core.redis_client import get_redis_client
from app.core.db import get_db_session
from app.core.radiating_settings_cache import get_radiating_settings

logger = logging.getLogger(__name__)


class CacheTier(Enum):
    """Cache tier levels"""
    MEMORY = "memory"      # L1 - In-memory cache
    REDIS = "redis"        # L2 - Redis cache
    DATABASE = "database"  # L3 - Database cache


@dataclass
class CacheEntry:
    """Represents a cache entry"""
    key: str
    value: Any
    tier: CacheTier
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: int = 3600
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    size_bytes: int = 0


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    tier: CacheTier
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_response_time_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheStrategy:
    """
    Multi-tier caching strategy with smart invalidation and warming.
    Implements LRU with frequency tracking for optimal performance.
    """
    
    # Cache configuration
    MEMORY_CACHE_SIZE = 1000  # Maximum entries in memory
    MEMORY_CACHE_SIZE_MB = 100  # Maximum memory cache size in MB
    
    # Cache key prefixes
    PREFIX_ENTITY = "radiating:cache:entity:"
    PREFIX_RELATIONSHIP = "radiating:cache:rel:"
    PREFIX_TRAVERSAL = "radiating:cache:traversal:"
    PREFIX_QUERY = "radiating:cache:query:"
    PREFIX_PATTERN = "radiating:cache:pattern:"
    
    # TTL configurations (in seconds)
    TTL_CONFIG = {
        CacheTier.MEMORY: {
            'entity': 300,      # 5 minutes
            'relationship': 300,
            'traversal': 600,   # 10 minutes
            'query': 1800,      # 30 minutes
            'pattern': 3600     # 1 hour
        },
        CacheTier.REDIS: {
            'entity': 3600,     # 1 hour
            'relationship': 3600,
            'traversal': 7200,  # 2 hours
            'query': 14400,     # 4 hours
            'pattern': 86400    # 24 hours
        },
        CacheTier.DATABASE: {
            'entity': 86400,    # 24 hours
            'relationship': 86400,
            'traversal': 172800, # 48 hours
            'query': 604800,    # 7 days
            'pattern': 2592000  # 30 days
        }
    }
    
    def __init__(self):
        """Initialize CacheStrategy"""
        self.redis_client = get_redis_client()
        
        # Memory cache (L1)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_order = deque(maxlen=self.MEMORY_CACHE_SIZE)
        self.memory_cache_size_bytes = 0
        
        # Access patterns tracking
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.pattern_cache: Dict[str, Set[str]] = defaultdict(set)
        
        # Cache statistics
        self.statistics = {
            CacheTier.MEMORY: CacheStatistics(CacheTier.MEMORY),
            CacheTier.REDIS: CacheStatistics(CacheTier.REDIS),
            CacheTier.DATABASE: CacheStatistics(CacheTier.DATABASE)
        }
        
        # Invalidation tracking
        self.invalidation_queue: asyncio.Queue = asyncio.Queue()
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Start background tasks
        asyncio.create_task(self._background_maintenance())
        asyncio.create_task(self._pattern_analyzer())
    
    async def get(
        self,
        key: str,
        cache_type: str = "entity",
        check_tiers: List[CacheTier] = None
    ) -> Optional[Any]:
        """
        Get value from cache, checking multiple tiers
        
        Args:
            key: Cache key
            cache_type: Type of cached data
            check_tiers: Tiers to check (default: all)
            
        Returns:
            Cached value or None
        """
        if check_tiers is None:
            check_tiers = [CacheTier.MEMORY, CacheTier.REDIS, CacheTier.DATABASE]
        
        start_time = time.time()
        
        # Check L1 - Memory cache
        if CacheTier.MEMORY in check_tiers:
            value = self._get_from_memory(key)
            if value is not None:
                self.statistics[CacheTier.MEMORY].hits += 1
                self._track_access(key, cache_type)
                return value
            self.statistics[CacheTier.MEMORY].misses += 1
        
        # Check L2 - Redis cache
        if CacheTier.REDIS in check_tiers:
            value = await self._get_from_redis(key, cache_type)
            if value is not None:
                self.statistics[CacheTier.REDIS].hits += 1
                # Promote to memory cache
                await self._promote_to_memory(key, value, cache_type)
                self._track_access(key, cache_type)
                return value
            self.statistics[CacheTier.REDIS].misses += 1
        
        # Check L3 - Database cache
        if CacheTier.DATABASE in check_tiers:
            value = await self._get_from_database(key, cache_type)
            if value is not None:
                self.statistics[CacheTier.DATABASE].hits += 1
                # Promote to higher tiers
                await self._promote_to_redis(key, value, cache_type)
                await self._promote_to_memory(key, value, cache_type)
                self._track_access(key, cache_type)
                return value
            self.statistics[CacheTier.DATABASE].misses += 1
        
        # Update response time statistics
        response_time = (time.time() - start_time) * 1000
        for tier in check_tiers:
            stats = self.statistics[tier]
            stats.avg_response_time_ms = (
                (stats.avg_response_time_ms * (stats.hits + stats.misses - 1) + response_time) /
                (stats.hits + stats.misses)
            ) if (stats.hits + stats.misses) > 0 else response_time
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        cache_type: str = "entity",
        ttl: Optional[int] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
        tiers: Optional[List[CacheTier]] = None
    ):
        """
        Set value in cache across multiple tiers
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cached data
            ttl: Time to live in seconds
            tags: Tags for grouping/invalidation
            dependencies: Keys this entry depends on
            tiers: Tiers to cache in (default: all)
        """
        if tiers is None:
            tiers = [CacheTier.MEMORY, CacheTier.REDIS]
        
        # Determine TTL if not specified
        if ttl is None:
            ttl = self.TTL_CONFIG[CacheTier.REDIS].get(cache_type, 3600)
        
        # Cache in specified tiers
        if CacheTier.MEMORY in tiers:
            await self._set_in_memory(key, value, cache_type, ttl, tags, dependencies)
        
        if CacheTier.REDIS in tiers:
            await self._set_in_redis(key, value, cache_type, ttl, tags, dependencies)
        
        if CacheTier.DATABASE in tiers:
            await self._set_in_database(key, value, cache_type, ttl, tags, dependencies)
        
        # Track dependencies
        if dependencies:
            for dep_key in dependencies:
                self.dependency_graph[dep_key].add(key)
        
        # Track patterns
        if tags:
            for tag in tags:
                self.pattern_cache[tag].add(key)
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            
            # Check if expired
            if datetime.now() > entry.created_at + timedelta(seconds=entry.ttl):
                del self.memory_cache[key]
                return None
            
            # Update access info
            entry.accessed_at = datetime.now()
            entry.access_count += 1
            
            # Move to end of access order (LRU)
            if key in self.memory_cache_order:
                self.memory_cache_order.remove(key)
            self.memory_cache_order.append(key)
            
            return entry.value
        
        return None
    
    async def _get_from_redis(self, key: str, cache_type: str) -> Optional[Any]:
        """Get value from Redis cache"""
        try:
            prefix = self._get_prefix(cache_type)
            cached = await self.redis_client.get(f"{prefix}{key}")
            
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.debug(f"Error getting from Redis: {e}")
        
        return None
    
    async def _get_from_database(self, key: str, cache_type: str) -> Optional[Any]:
        """Get value from database cache"""
        try:
            async with get_db_session() as session:
                # Query cache table
                query = """
                    SELECT value, expires_at
                    FROM radiating_cache
                    WHERE key = %s AND cache_type = %s
                """
                result = await session.execute(query, (key, cache_type))
                row = result.fetchone()
                
                if row:
                    value, expires_at = row
                    
                    # Check expiration
                    if expires_at and datetime.now() > expires_at:
                        # Delete expired entry
                        await session.execute(
                            "DELETE FROM radiating_cache WHERE key = %s",
                            (key,)
                        )
                        await session.commit()
                        return None
                    
                    return json.loads(value)
        except Exception as e:
            logger.debug(f"Error getting from database: {e}")
        
        return None
    
    async def _set_in_memory(
        self,
        key: str,
        value: Any,
        cache_type: str,
        ttl: int,
        tags: Optional[Set[str]],
        dependencies: Optional[Set[str]]
    ):
        """Set value in memory cache"""
        # Calculate size
        size_bytes = len(json.dumps(value))
        
        # Check memory limits
        if self.memory_cache_size_bytes + size_bytes > self.MEMORY_CACHE_SIZE_MB * 1024 * 1024:
            await self._evict_from_memory()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            tier=CacheTier.MEMORY,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl=ttl,
            tags=tags or set(),
            dependencies=dependencies or set(),
            size_bytes=size_bytes
        )
        
        # Store in cache
        self.memory_cache[key] = entry
        self.memory_cache_order.append(key)
        self.memory_cache_size_bytes += size_bytes
        
        # Update statistics
        self.statistics[CacheTier.MEMORY].total_size_bytes = self.memory_cache_size_bytes
    
    async def _set_in_redis(
        self,
        key: str,
        value: Any,
        cache_type: str,
        ttl: int,
        tags: Optional[Set[str]],
        dependencies: Optional[Set[str]]
    ):
        """Set value in Redis cache"""
        try:
            prefix = self._get_prefix(cache_type)
            
            # Store main value
            await self.redis_client.setex(
                f"{prefix}{key}",
                ttl,
                json.dumps(value)
            )
            
            # Store metadata
            metadata = {
                'tags': list(tags) if tags else [],
                'dependencies': list(dependencies) if dependencies else [],
                'created_at': datetime.now().isoformat(),
                'cache_type': cache_type
            }
            
            await self.redis_client.setex(
                f"{prefix}meta:{key}",
                ttl,
                json.dumps(metadata)
            )
        except Exception as e:
            logger.error(f"Error setting in Redis: {e}")
    
    async def _set_in_database(
        self,
        key: str,
        value: Any,
        cache_type: str,
        ttl: int,
        tags: Optional[Set[str]],
        dependencies: Optional[Set[str]]
    ):
        """Set value in database cache"""
        try:
            async with get_db_session() as session:
                expires_at = datetime.now() + timedelta(seconds=ttl)
                
                # Upsert cache entry
                query = """
                    INSERT INTO radiating_cache (key, value, cache_type, expires_at, tags, dependencies)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        cache_type = EXCLUDED.cache_type,
                        expires_at = EXCLUDED.expires_at,
                        tags = EXCLUDED.tags,
                        dependencies = EXCLUDED.dependencies,
                        updated_at = NOW()
                """
                
                await session.execute(
                    query,
                    (
                        key,
                        json.dumps(value),
                        cache_type,
                        expires_at,
                        json.dumps(list(tags)) if tags else '[]',
                        json.dumps(list(dependencies)) if dependencies else '[]'
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Error setting in database: {e}")
    
    async def _promote_to_memory(self, key: str, value: Any, cache_type: str):
        """Promote value to memory cache"""
        ttl = self.TTL_CONFIG[CacheTier.MEMORY].get(cache_type, 300)
        await self._set_in_memory(key, value, cache_type, ttl, None, None)
    
    async def _promote_to_redis(self, key: str, value: Any, cache_type: str):
        """Promote value to Redis cache"""
        ttl = self.TTL_CONFIG[CacheTier.REDIS].get(cache_type, 3600)
        await self._set_in_redis(key, value, cache_type, ttl, None, None)
    
    async def _evict_from_memory(self):
        """Evict entries from memory cache using LRU"""
        # Remove 10% of cache when full
        evict_count = max(1, len(self.memory_cache) // 10)
        
        for _ in range(evict_count):
            if self.memory_cache_order:
                # Get least recently used key
                key = self.memory_cache_order.popleft()
                
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    self.memory_cache_size_bytes -= entry.size_bytes
                    del self.memory_cache[key]
                    
                    self.statistics[CacheTier.MEMORY].evictions += 1
    
    def _get_prefix(self, cache_type: str) -> str:
        """Get cache key prefix based on type"""
        prefix_map = {
            'entity': self.PREFIX_ENTITY,
            'relationship': self.PREFIX_RELATIONSHIP,
            'traversal': self.PREFIX_TRAVERSAL,
            'query': self.PREFIX_QUERY,
            'pattern': self.PREFIX_PATTERN
        }
        return prefix_map.get(cache_type, self.PREFIX_ENTITY)
    
    def _track_access(self, key: str, cache_type: str):
        """Track access patterns for intelligent warming"""
        now = datetime.now()
        self.access_patterns[key].append(now)
        
        # Keep only recent accesses (last hour)
        cutoff = now - timedelta(hours=1)
        self.access_patterns[key] = [
            t for t in self.access_patterns[key]
            if t > cutoff
        ]
    
    async def invalidate(
        self,
        key: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        pattern: Optional[str] = None,
        cascade: bool = True
    ):
        """
        Invalidate cache entries
        
        Args:
            key: Specific key to invalidate
            tags: Invalidate all entries with these tags
            pattern: Invalidate keys matching pattern
            cascade: Invalidate dependent entries
        """
        keys_to_invalidate = set()
        
        # Collect keys to invalidate
        if key:
            keys_to_invalidate.add(key)
        
        if tags:
            for tag in tags:
                keys_to_invalidate.update(self.pattern_cache.get(tag, set()))
        
        if pattern:
            # Match pattern against all cached keys
            for cache_key in list(self.memory_cache.keys()):
                if self._matches_pattern(cache_key, pattern):
                    keys_to_invalidate.add(cache_key)
        
        # Add dependent keys if cascading
        if cascade:
            for key in list(keys_to_invalidate):
                keys_to_invalidate.update(self.dependency_graph.get(key, set()))
        
        # Invalidate from all tiers
        for key in keys_to_invalidate:
            await self._invalidate_key(key)
    
    async def _invalidate_key(self, key: str):
        """Invalidate a specific key from all tiers"""
        # Remove from memory
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            self.memory_cache_size_bytes -= entry.size_bytes
            del self.memory_cache[key]
            
            if key in self.memory_cache_order:
                self.memory_cache_order.remove(key)
        
        # Remove from Redis
        for prefix in [self.PREFIX_ENTITY, self.PREFIX_RELATIONSHIP, 
                      self.PREFIX_TRAVERSAL, self.PREFIX_QUERY, self.PREFIX_PATTERN]:
            await self.redis_client.delete(f"{prefix}{key}")
            await self.redis_client.delete(f"{prefix}meta:{key}")
        
        # Mark for database removal (async)
        await self.invalidation_queue.put(key)
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcards)"""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
    
    async def warm_cache(
        self,
        patterns: List[str],
        loader_func: Any,
        cache_type: str = "entity"
    ):
        """
        Warm cache with commonly accessed patterns
        
        Args:
            patterns: List of key patterns to warm
            loader_func: Function to load data for keys
            cache_type: Type of data being cached
        """
        logger.info(f"Warming cache for {len(patterns)} patterns")
        
        for pattern in patterns:
            try:
                # Check access frequency
                access_count = len(self.access_patterns.get(pattern, []))
                
                if access_count > 5:  # Frequently accessed
                    # Load data
                    data = await loader_func(pattern)
                    
                    if data:
                        # Cache in memory and Redis
                        await self.set(
                            pattern,
                            data,
                            cache_type,
                            tiers=[CacheTier.MEMORY, CacheTier.REDIS]
                        )
            except Exception as e:
                logger.error(f"Error warming cache for {pattern}: {e}")
    
    async def _background_maintenance(self):
        """Background task for cache maintenance"""
        while True:
            try:
                # Clean expired entries
                await self._clean_expired_entries()
                
                # Process invalidation queue
                await self._process_invalidation_queue()
                
                # Log statistics
                self._log_statistics()
                
                # Sleep for maintenance interval
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in cache maintenance: {e}")
                await asyncio.sleep(5)
    
    async def _clean_expired_entries(self):
        """Remove expired entries from memory cache"""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self.memory_cache.items():
            if now > entry.created_at + timedelta(seconds=entry.ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._invalidate_key(key)
    
    async def _process_invalidation_queue(self):
        """Process database invalidation queue"""
        batch = []
        
        # Collect up to 100 keys
        while not self.invalidation_queue.empty() and len(batch) < 100:
            try:
                key = self.invalidation_queue.get_nowait()
                batch.append(key)
            except asyncio.QueueEmpty:
                break
        
        if batch:
            try:
                async with get_db_session() as session:
                    query = "DELETE FROM radiating_cache WHERE key = ANY(%s)"
                    await session.execute(query, (batch,))
                    await session.commit()
            except Exception as e:
                logger.error(f"Error processing invalidation queue: {e}")
    
    async def _pattern_analyzer(self):
        """Analyze access patterns for optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Identify hot keys
                hot_keys = self._identify_hot_keys()
                
                # Suggest optimizations
                if hot_keys:
                    logger.info(f"Hot keys detected: {hot_keys[:5]}")
                    
                    # Pre-warm hot keys if not in memory
                    for key in hot_keys:
                        if key not in self.memory_cache:
                            # Attempt to load from lower tier
                            value = await self._get_from_redis(key, "entity")
                            if value:
                                await self._promote_to_memory(key, value, "entity")
                
            except Exception as e:
                logger.error(f"Error in pattern analyzer: {e}")
    
    def _identify_hot_keys(self, threshold: int = 10) -> List[str]:
        """Identify frequently accessed keys"""
        hot_keys = []
        now = datetime.now()
        window = timedelta(minutes=5)
        
        for key, accesses in self.access_patterns.items():
            recent_accesses = [a for a in accesses if now - a < window]
            
            if len(recent_accesses) >= threshold:
                hot_keys.append((key, len(recent_accesses)))
        
        # Sort by access count
        hot_keys.sort(key=lambda x: x[1], reverse=True)
        
        return [k for k, _ in hot_keys]
    
    def _log_statistics(self):
        """Log cache statistics"""
        for tier, stats in self.statistics.items():
            if stats.hits + stats.misses > 0:
                logger.debug(
                    f"Cache {tier.value} - "
                    f"Hit rate: {stats.hit_rate:.2%}, "
                    f"Hits: {stats.hits}, "
                    f"Misses: {stats.misses}, "
                    f"Evictions: {stats.evictions}, "
                    f"Avg response: {stats.avg_response_time_ms:.2f}ms"
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            tier.value: {
                'hit_rate': stats.hit_rate,
                'hits': stats.hits,
                'misses': stats.misses,
                'evictions': stats.evictions,
                'total_size_bytes': stats.total_size_bytes,
                'avg_response_time_ms': stats.avg_response_time_ms
            }
            for tier, stats in self.statistics.items()
        }