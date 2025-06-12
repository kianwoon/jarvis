"""
Enterprise-grade cache base class with auto-reload, circuit breaker, and fallback strategies
"""
import asyncio
import time
import redis
import json
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CacheState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Redis down, using DB fallback
    CIRCUIT_OPEN = "circuit_open"  # Multiple failures, using defaults

@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    errors: int = 0
    last_reload: Optional[float] = None
    state: CacheState = CacheState.HEALTHY

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    def should_attempt(self) -> bool:
        if not self.is_open:
            return True
            
        # Check if timeout has passed
        if time.time() - self.last_failure_time > self.timeout:
            self.is_open = False
            self.failure_count = 0
            return True
        return False
    
    def record_success(self):
        self.failure_count = 0
        self.is_open = False
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class EnterpriseCache:
    """
    Enterprise-grade cache with:
    - Auto-reload from DB when Redis fails
    - Circuit breaker pattern
    - Graceful degradation
    - Background refresh
    - Metrics and monitoring
    """
    
    def __init__(self, 
                 cache_key: str,
                 redis_ttl: int = 3600,
                 auto_reload_interval: int = 300,
                 db_loader: Optional[Callable] = None,
                 default_value_factory: Optional[Callable] = None):
        self.cache_key = cache_key
        self.redis_ttl = redis_ttl
        self.auto_reload_interval = auto_reload_interval
        self.db_loader = db_loader
        self.default_value_factory = default_value_factory
        
        self.circuit_breaker = CircuitBreaker()
        self.metrics = CacheMetrics()
        self._redis_client = None
        self._last_value = None
        self._background_task = None
        
    def _get_redis_client(self):
        """Lazy Redis connection"""
        if self._redis_client is None:
            try:
                from app.core.config import get_settings
                config = get_settings()
                self._redis_client = redis.Redis(
                    host=config.REDIS_HOST, 
                    port=config.REDIS_PORT, 
                    password=config.REDIS_PASSWORD, 
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                self._redis_client.ping()  # Test connection
                logger.info(f"Redis connected for cache {self.cache_key}")
            except Exception as e:
                logger.warning(f"Redis connection failed for {self.cache_key}: {e}")
                self._redis_client = None
        return self._redis_client
    
    def get(self, force_reload: bool = False) -> Any:
        """
        Get value with enterprise-grade fallback strategy:
        1. Try Redis cache
        2. If Redis fails, try database (with circuit breaker)
        3. If database fails, use default value
        4. Auto-reload in background
        """
        
        # Step 1: Try Redis cache
        if not force_reload:
            try:
                redis_client = self._get_redis_client()
                if redis_client:
                    cached = redis_client.get(self.cache_key)
                    if cached:
                        value = json.loads(cached)
                        self.metrics.hits += 1
                        self.circuit_breaker.record_success()
                        self._last_value = value
                        return value
            except Exception as e:
                logger.warning(f"Redis get failed for {self.cache_key}: {e}")
                self.circuit_breaker.record_failure()
                self.metrics.errors += 1
        
        # Step 2: Try database with circuit breaker
        if self.db_loader and self.circuit_breaker.should_attempt():
            try:
                value = self.db_loader()
                self.metrics.misses += 1
                self.metrics.last_reload = time.time()
                self._last_value = value
                
                # Try to cache the result
                self._cache_value(value)
                
                self.circuit_breaker.record_success()
                self.metrics.state = CacheState.DEGRADED if not self._get_redis_client() else CacheState.HEALTHY
                
                return value
            except Exception as e:
                logger.error(f"Database reload failed for {self.cache_key}: {e}")
                self.circuit_breaker.record_failure()
                self.metrics.errors += 1
        
        # Step 3: Use cached value or default
        if self._last_value is not None:
            logger.info(f"Using last known value for {self.cache_key}")
            return self._last_value
        
        if self.default_value_factory:
            logger.info(f"Using default value for {self.cache_key}")
            default = self.default_value_factory()
            self._last_value = default
            return default
        
        # Step 4: Ultimate fallback
        self.metrics.state = CacheState.CIRCUIT_OPEN
        raise RuntimeError(f"No value available for {self.cache_key}")
    
    def _cache_value(self, value: Any):
        """Cache value in Redis with error handling"""
        try:
            redis_client = self._get_redis_client()
            if redis_client:
                redis_client.setex(self.cache_key, self.redis_ttl, json.dumps(value))
                logger.debug(f"Cached value for {self.cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache value for {self.cache_key}: {e}")
    
    def set(self, value: Any):
        """Set value in cache and update last known value"""
        self._last_value = value
        self._cache_value(value)
        
        # Also update database if loader supports it
        if hasattr(self.db_loader, 'save'):
            try:
                self.db_loader.save(value)
            except Exception as e:
                logger.error(f"Failed to save to database for {self.cache_key}: {e}")
    
    def start_background_refresh(self):
        """Start background task for periodic cache refresh"""
        if self._background_task is None:
            self._background_task = asyncio.create_task(self._background_refresh())
    
    async def _background_refresh(self):
        """Background task to refresh cache periodically"""
        while True:
            try:
                await asyncio.sleep(self.auto_reload_interval)
                
                # Only refresh if circuit breaker allows and we have a loader
                if self.db_loader and self.circuit_breaker.should_attempt():
                    logger.debug(f"Background refresh for {self.cache_key}")
                    self.get(force_reload=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background refresh error for {self.cache_key}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        hit_rate = self.metrics.hits / (self.metrics.hits + self.metrics.misses) if (self.metrics.hits + self.metrics.misses) > 0 else 0
        
        return {
            "cache_key": self.cache_key,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses, 
            "errors": self.metrics.errors,
            "hit_rate": hit_rate,
            "state": self.metrics.state.value,
            "circuit_breaker_open": self.circuit_breaker.is_open,
            "last_reload": self.metrics.last_reload
        }