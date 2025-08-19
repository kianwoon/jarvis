"""
Meta-Task Settings Cache
Redis-based caching for meta-task configuration settings
"""

import json
import logging
from typing import Dict, Any, Optional
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class MetaTaskSettingsCache:
    """Manages caching for meta-task settings"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.cache_prefix = "meta_task_settings"
        self.cache_ttl = 3600  # 1 hour TTL
        self.logger = logger
    
    def get_cache_key(self, key: str) -> str:
        """Generate cache key with prefix"""
        return f"{self.cache_prefix}:{key}"
    
    async def get_settings(self, category: str = "general") -> Optional[Dict[str, Any]]:
        """Get cached meta-task settings"""
        try:
            cache_key = self.get_cache_key(category)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                self.logger.debug(f"Cache hit for meta-task settings: {category}")
                return json.loads(cached_data)
            
            self.logger.debug(f"Cache miss for meta-task settings: {category}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting meta-task settings from cache: {e}")
            return None
    
    async def set_settings(self, category: str, settings: Dict[str, Any]) -> bool:
        """Cache meta-task settings"""
        try:
            cache_key = self.get_cache_key(category)
            serialized_data = json.dumps(settings, default=str)
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                serialized_data
            )
            
            self.logger.debug(f"Cached meta-task settings for: {category}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching meta-task settings: {e}")
            return False
    
    async def delete_settings(self, category: str = None) -> bool:
        """Delete cached meta-task settings"""
        try:
            if category:
                cache_key = self.get_cache_key(category)
                await self.redis_client.delete(cache_key)
                self.logger.debug(f"Deleted cache for meta-task settings: {category}")
            else:
                # Delete all meta-task cache entries
                pattern = f"{self.cache_prefix}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    self.logger.debug(f"Deleted all meta-task cache entries: {len(keys)} keys")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting meta-task cache: {e}")
            return False
    
    async def get_template_cache(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get cached template data"""
        try:
            cache_key = self.get_cache_key(f"template:{template_id}")
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting template cache: {e}")
            return None
    
    async def set_template_cache(self, template_id: str, template_data: Dict[str, Any]) -> bool:
        """Cache template data"""
        try:
            cache_key = self.get_cache_key(f"template:{template_id}")
            serialized_data = json.dumps(template_data, default=str)
            
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                serialized_data
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching template: {e}")
            return False
    
    async def get_workflow_cache(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get cached workflow data"""
        try:
            cache_key = self.get_cache_key(f"workflow:{workflow_id}")
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting workflow cache: {e}")
            return None
    
    async def set_workflow_cache(self, workflow_id: str, workflow_data: Dict[str, Any]) -> bool:
        """Cache workflow data"""
        try:
            cache_key = self.get_cache_key(f"workflow:{workflow_id}")
            serialized_data = json.dumps(workflow_data, default=str)
            
            # Use shorter TTL for active workflows
            workflow_ttl = 1800 if workflow_data.get('status') == 'running' else self.cache_ttl
            
            await self.redis_client.setex(
                cache_key,
                workflow_ttl,
                serialized_data
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching workflow: {e}")
            return False
    
    async def get_execution_cache(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get cached execution data"""
        try:
            cache_key = self.get_cache_key(f"execution:{execution_id}")
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting execution cache: {e}")
            return None
    
    async def set_execution_cache(self, execution_id: str, execution_data: Dict[str, Any]) -> bool:
        """Cache execution data"""
        try:
            cache_key = self.get_cache_key(f"execution:{execution_id}")
            serialized_data = json.dumps(execution_data, default=str)
            
            # Shorter TTL for execution data
            execution_ttl = 900  # 15 minutes
            
            await self.redis_client.setex(
                cache_key,
                execution_ttl,
                serialized_data
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching execution: {e}")
            return False
    
    async def invalidate_all(self) -> bool:
        """Invalidate all meta-task caches"""
        return await self.delete_settings()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            pattern = f"{self.cache_prefix}:*"
            keys = await self.redis_client.keys(pattern)
            
            stats = {
                'total_entries': len(keys),
                'cache_prefix': self.cache_prefix,
                'cache_ttl': self.cache_ttl,
                'categories': {}
            }
            
            # Categorize keys
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(':')
                if len(parts) >= 3:
                    category = parts[2]
                    stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {
                'total_entries': 0,
                'error': str(e)
            }

# Global instance
meta_task_cache = MetaTaskSettingsCache()

# Convenience functions
async def get_meta_task_settings(category: str = "general") -> Dict[str, Any]:
    """Get meta-task settings with fallback to defaults"""
    try:
        settings = await meta_task_cache.get_settings(category)
        
        if settings:
            return settings
        
        # Return default settings if not cached
        default_settings = get_default_meta_task_settings()
        
        # Cache the defaults
        await meta_task_cache.set_settings(category, default_settings)
        
        return default_settings
        
    except Exception as e:
        logger.error(f"Error getting meta-task settings: {e}")
        return get_default_meta_task_settings()

def get_default_meta_task_settings() -> Dict[str, Any]:
    """Get default meta-task settings"""
    return {
        "enabled": True,
        "execution": {
            "max_phases": 10,
            "phase_timeout_minutes": 30,
            "retry_attempts": 3,
            "parallel_execution": False
        },
        "models": {
            "analyzer_model": "qwen3:30b-a3b",
            "generator_model": "qwen3:30b-a3b", 
            "reviewer_model": "qwen3:30b-a3b",
            "assembler_model": "qwen3:30b-a3b"
        },
        "quality_control": {
            "minimum_quality_score": 0.8,
            "require_human_review": False,
            "auto_retry_on_low_quality": True
        },
        "output": {
            "default_format": "markdown",
            "include_metadata": True,
            "max_output_size_mb": 10
        },
        "caching": {
            "cache_templates": True,
            "cache_workflows": True,
            "cache_ttl_hours": 1
        }
    }

async def invalidate_meta_task_cache() -> bool:
    """Invalidate all meta-task caches"""
    return await meta_task_cache.invalidate_all()