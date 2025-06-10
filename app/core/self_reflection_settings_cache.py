"""
Self-Reflection Settings Cache

This module manages caching of self-reflection settings
from the database to improve performance.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import json

from app.core.redis_client import get_redis_client
from app.core.db import get_db

logger = logging.getLogger(__name__)

# Cache key prefixes
CACHE_PREFIX = "self_reflection:"
SETTINGS_KEY = f"{CACHE_PREFIX}settings"
DIMENSIONS_KEY = f"{CACHE_PREFIX}dimensions"
CACHE_TTL = 3600  # 1 hour


class SelfReflectionSettingsCache:
    """Cache manager for self-reflection settings"""
    
    def __init__(self):
        self.redis_client = None
        self._local_cache = {}
        self._cache_timestamp = {}
        
    def _get_redis(self):
        """Get Redis client lazily"""
        if self.redis_client is None:
            self.redis_client = get_redis_client()
        return self.redis_client
    
    async def get_reflection_settings(self, model_name: str) -> Dict[str, Any]:
        """
        Get reflection settings for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of reflection settings
        """
        cache_key = f"{SETTINGS_KEY}:{model_name}"
        
        # Check local cache first
        if self._is_local_cache_valid(cache_key):
            return self._local_cache[cache_key]
        
        # Try Redis cache
        try:
            redis = self._get_redis()
            if redis:
                cached = await redis.get(cache_key)
                if cached:
                    settings = json.loads(cached)
                    self._update_local_cache(cache_key, settings)
                    return settings
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")
        
        # Load from database
        settings = await self._load_settings_from_db(model_name)
        
        # Update caches
        await self._update_caches(cache_key, settings)
        
        return settings
    
    async def get_quality_dimensions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get quality dimension settings
        
        Returns:
            Dictionary of dimension settings
        """
        cache_key = DIMENSIONS_KEY
        
        # Check local cache first
        if self._is_local_cache_valid(cache_key):
            return self._local_cache[cache_key]
        
        # Try Redis cache
        try:
            redis = self._get_redis()
            if redis:
                cached = await redis.get(cache_key)
                if cached:
                    dimensions = json.loads(cached)
                    self._update_local_cache(cache_key, dimensions)
                    return dimensions
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")
        
        # Load from database
        dimensions = await self._load_dimensions_from_db()
        
        # Update caches
        await self._update_caches(cache_key, dimensions)
        
        return dimensions
    
    async def update_reflection_settings(
        self, model_name: str, settings: Dict[str, Any]
    ) -> bool:
        """
        Update reflection settings for a model
        
        Args:
            model_name: Name of the model
            settings: New settings
            
        Returns:
            True if successful
        """
        try:
            # Update database
            async for db in get_db():
                await db.execute("""
                    INSERT INTO self_reflection_settings 
                    (model_name, enabled, reflection_mode, quality_threshold, 
                     max_iterations, min_improvement_threshold, enable_caching, 
                     cache_ttl_seconds, timeout_seconds)
                    VALUES (:model_name, :enabled, :reflection_mode, :quality_threshold,
                            :max_iterations, :min_improvement_threshold, :enable_caching,
                            :cache_ttl_seconds, :timeout_seconds)
                    ON CONFLICT (model_name) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        reflection_mode = EXCLUDED.reflection_mode,
                        quality_threshold = EXCLUDED.quality_threshold,
                        max_iterations = EXCLUDED.max_iterations,
                        min_improvement_threshold = EXCLUDED.min_improvement_threshold,
                        enable_caching = EXCLUDED.enable_caching,
                        cache_ttl_seconds = EXCLUDED.cache_ttl_seconds,
                        timeout_seconds = EXCLUDED.timeout_seconds,
                        updated_at = CURRENT_TIMESTAMP
                """, {
                    "model_name": model_name,
                    **settings
                })
                await db.commit()
            
            # Invalidate caches
            cache_key = f"{SETTINGS_KEY}:{model_name}"
            await self._invalidate_cache(cache_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update reflection settings: {e}")
            return False
    
    async def log_reflection_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Log reflection metrics to database
        
        Args:
            metrics: Reflection metrics to log
            
        Returns:
            True if successful
        """
        try:
            async for db in get_db():
                await db.execute("""
                    INSERT INTO reflection_metrics
                    (query_hash, model_name, initial_quality, final_quality,
                     quality_improvement, iterations_performed, total_time_ms,
                     evaluation_time_ms, refinement_time_ms, strategies_used, success)
                    VALUES (:query_hash, :model_name, :initial_quality, :final_quality,
                            :quality_improvement, :iterations_performed, :total_time_ms,
                            :evaluation_time_ms, :refinement_time_ms, :strategies_used, :success)
                """, metrics)
                await db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log reflection metrics: {e}")
            return False
    
    async def get_reflection_stats(
        self, model_name: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get reflection statistics
        
        Args:
            model_name: Optional model name filter
            hours: Number of hours to look back
            
        Returns:
            Dictionary of statistics
        """
        try:
            async for db in get_db():
                query = """
                    SELECT 
                        COUNT(*) as total_reflections,
                        AVG(quality_improvement) as avg_improvement,
                        AVG(total_time_ms) as avg_time_ms,
                        AVG(iterations_performed) as avg_iterations,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate,
                        MAX(final_quality) as max_quality,
                        MIN(initial_quality) as min_initial_quality
                    FROM reflection_metrics
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                """
                params = [hours]
                
                if model_name:
                    query += " AND model_name = %s"
                    params.append(model_name)
                
                result = await db.fetch_one(query, params)
                
                return {
                    "total_reflections": result["total_reflections"] or 0,
                    "avg_improvement": float(result["avg_improvement"] or 0),
                    "avg_time_ms": float(result["avg_time_ms"] or 0),
                    "avg_iterations": float(result["avg_iterations"] or 0),
                    "success_rate": float(result["success_rate"] or 0),
                    "max_quality": float(result["max_quality"] or 0),
                    "min_initial_quality": float(result["min_initial_quality"] or 0)
                }
                
        except Exception as e:
            logger.error(f"Failed to get reflection stats: {e}")
            return {}
    
    async def _load_settings_from_db(self, model_name: str) -> Dict[str, Any]:
        """Load reflection settings from database"""
        try:
            async for db in get_db():
                # Try specific model first
                result = await db.fetch_one(
                    "SELECT * FROM self_reflection_settings WHERE model_name = :model_name",
                    {"model_name": model_name}
                )
                
                # Fall back to default if not found
                if not result:
                    result = await db.fetch_one(
                        "SELECT * FROM self_reflection_settings WHERE model_name = 'default'"
                    )
                
                if result:
                    return {
                        "enabled": result["enabled"],
                        "reflection_mode": result["reflection_mode"],
                        "quality_threshold": result["quality_threshold"],
                        "max_iterations": result["max_iterations"],
                        "min_improvement_threshold": result["min_improvement_threshold"],
                        "enable_caching": result["enable_caching"],
                        "cache_ttl_seconds": result["cache_ttl_seconds"],
                        "timeout_seconds": result["timeout_seconds"]
                    }
                    
        except Exception as e:
            logger.error(f"Failed to load settings from DB: {e}")
        
        # Return defaults
        return {
            "enabled": True,
            "reflection_mode": "balanced",
            "quality_threshold": 0.8,
            "max_iterations": 3,
            "min_improvement_threshold": 0.05,
            "enable_caching": True,
            "cache_ttl_seconds": 3600,
            "timeout_seconds": 30
        }
    
    async def _load_dimensions_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Load quality dimension settings from database"""
        try:
            async for db in get_db():
                results = await db.fetch_all(
                    "SELECT * FROM quality_dimension_settings WHERE enabled = TRUE"
                )
                
                dimensions = {}
                for row in results:
                    dimensions[row["dimension_name"]] = {
                        "weight": row["weight"],
                        "min_acceptable_score": row["min_acceptable_score"],
                        "enabled": row["enabled"]
                    }
                
                return dimensions
                
        except Exception as e:
            logger.error(f"Failed to load dimensions from DB: {e}")
        
        # Return defaults
        return {
            "completeness": {"weight": 0.25, "min_acceptable_score": 0.7, "enabled": True},
            "relevance": {"weight": 0.25, "min_acceptable_score": 0.7, "enabled": True},
            "accuracy": {"weight": 0.20, "min_acceptable_score": 0.8, "enabled": True},
            "coherence": {"weight": 0.10, "min_acceptable_score": 0.6, "enabled": True},
            "specificity": {"weight": 0.10, "min_acceptable_score": 0.6, "enabled": True},
            "confidence": {"weight": 0.10, "min_acceptable_score": 0.6, "enabled": True}
        }
    
    def _is_local_cache_valid(self, key: str) -> bool:
        """Check if local cache is valid"""
        if key not in self._cache_timestamp:
            return False
        
        age = datetime.now() - self._cache_timestamp[key]
        return age < timedelta(seconds=300)  # 5 minutes local cache
    
    def _update_local_cache(self, key: str, value: Any):
        """Update local cache"""
        self._local_cache[key] = value
        self._cache_timestamp[key] = datetime.now()
    
    async def _update_caches(self, key: str, value: Any):
        """Update both Redis and local caches"""
        # Update local cache
        self._update_local_cache(key, value)
        
        # Update Redis cache
        try:
            redis = self._get_redis()
            if redis:
                await redis.setex(key, CACHE_TTL, json.dumps(value))
        except Exception as e:
            logger.warning(f"Failed to update Redis cache: {e}")
    
    async def _invalidate_cache(self, key: str):
        """Invalidate cache entries"""
        # Remove from local cache
        self._local_cache.pop(key, None)
        self._cache_timestamp.pop(key, None)
        
        # Remove from Redis
        try:
            redis = self._get_redis()
            if redis:
                await redis.delete(key)
        except Exception as e:
            logger.warning(f"Failed to invalidate Redis cache: {e}")


# Global instance
_cache_instance = None


def get_self_reflection_cache() -> SelfReflectionSettingsCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SelfReflectionSettingsCache()
    return _cache_instance


# Convenience functions
async def get_reflection_settings(model_name: str) -> Dict[str, Any]:
    """Get reflection settings for a model"""
    cache = get_self_reflection_cache()
    return await cache.get_reflection_settings(model_name)


async def get_quality_dimensions() -> Dict[str, Dict[str, Any]]:
    """Get quality dimension settings"""
    cache = get_self_reflection_cache()
    return await cache.get_quality_dimensions()


async def update_reflection_settings(
    model_name: str, settings: Dict[str, Any]
) -> bool:
    """Update reflection settings"""
    cache = get_self_reflection_cache()
    return await cache.update_reflection_settings(model_name, settings)


async def log_reflection_metrics(metrics: Dict[str, Any]) -> bool:
    """Log reflection metrics"""
    cache = get_self_reflection_cache()
    return await cache.log_reflection_metrics(metrics)