"""
Redis Bridge for Langflow Integration
Provides Redis read/write capabilities for Langflow workflows
"""
import logging
from typing import Dict, Any, Optional, List
import json

from app.core.redis_base import RedisCache
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)

class RedisBridge:
    """Bridge between Langflow and Redis infrastructure"""
    
    def __init__(self, key_prefix: str = "langflow_"):
        self.cache = RedisCache(key_prefix=key_prefix)
        self.key_prefix = key_prefix
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get value from Redis"""
        try:
            result = self.cache.get(key, default)
            logger.info(f"[REDIS BRIDGE] Retrieved key: {key}")
            return result
        except Exception as e:
            logger.error(f"[REDIS BRIDGE] Failed to get key {key}: {e}")
            return default
    
    def set_value(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration"""
        try:
            success = self.cache.set(key, value, expire=expire)
            if success:
                logger.info(f"[REDIS BRIDGE] Set key: {key} (expire: {expire})")
            else:
                logger.warning(f"[REDIS BRIDGE] Failed to set key: {key}")
            return success
        except Exception as e:
            logger.error(f"[REDIS BRIDGE] Error setting key {key}: {e}")
            return False
    
    def delete_value(self, key: str) -> bool:
        """Delete value from Redis"""
        try:
            success = self.cache.delete(key)
            if success:
                logger.info(f"[REDIS BRIDGE] Deleted key: {key}")
            return success
        except Exception as e:
            logger.error(f"[REDIS BRIDGE] Error deleting key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            return self.cache.exists(key)
        except Exception as e:
            logger.error(f"[REDIS BRIDGE] Error checking key existence {key}: {e}")
            return False
    
    def get_workflow_state(self, workflow_id: str, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution state from Redis"""
        state_key = f"workflow_state:{workflow_id}:{execution_id}"
        return self.get_value(state_key, {})
    
    def set_workflow_state(
        self, 
        workflow_id: str, 
        execution_id: str, 
        state: Dict[str, Any],
        expire: int = 3600
    ) -> bool:
        """Set workflow execution state in Redis with TTL"""
        state_key = f"workflow_state:{workflow_id}:{execution_id}"
        return self.set_value(state_key, state, expire=expire)
    
    def update_workflow_state(
        self, 
        workflow_id: str, 
        execution_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update specific fields in workflow state"""
        current_state = self.get_workflow_state(workflow_id, execution_id)
        current_state.update(updates)
        return self.set_workflow_state(workflow_id, execution_id, current_state)
    
    def get_shared_data(self, namespace: str, key: str) -> Any:
        """Get shared data between workflow steps"""
        shared_key = f"shared:{namespace}:{key}"
        return self.get_value(shared_key)
    
    def set_shared_data(
        self, 
        namespace: str, 
        key: str, 
        value: Any, 
        expire: Optional[int] = 1800
    ) -> bool:
        """Set shared data between workflow steps"""
        shared_key = f"shared:{namespace}:{key}"
        return self.set_value(shared_key, value, expire=expire)
    
    def get_cache_keys_pattern(self, pattern: str) -> List[str]:
        """Get Redis keys matching a pattern"""
        try:
            client = get_redis_client()
            if not client:
                return []
            
            full_pattern = f"{self.key_prefix}{pattern}"
            keys = client.keys(full_pattern)
            
            # Remove prefix from returned keys
            cleaned_keys = []
            for key in keys:
                if key.startswith(self.key_prefix):
                    cleaned_keys.append(key[len(self.key_prefix):])
                else:
                    cleaned_keys.append(key)
            
            logger.info(f"[REDIS BRIDGE] Found {len(cleaned_keys)} keys matching pattern: {pattern}")
            return cleaned_keys
        except Exception as e:
            logger.error(f"[REDIS BRIDGE] Error getting keys with pattern {pattern}: {e}")
            return []
    
    def increment_counter(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in Redis"""
        try:
            client = get_redis_client()
            if not client:
                return None
            
            full_key = f"{self.key_prefix}{key}"
            new_value = client.incr(full_key, amount)
            logger.info(f"[REDIS BRIDGE] Incremented counter {key} by {amount}, new value: {new_value}")
            return new_value
        except Exception as e:
            logger.error(f"[REDIS BRIDGE] Error incrementing counter {key}: {e}")
            return None
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get Redis connection status"""
        client = get_redis_client()
        if client:
            try:
                client.ping()
                return {
                    "connected": True,
                    "key_prefix": self.key_prefix,
                    "host": client.connection_pool.connection_kwargs.get("host", "unknown"),
                    "port": client.connection_pool.connection_kwargs.get("port", "unknown")
                }
            except Exception as e:
                return {
                    "connected": False,
                    "error": str(e),
                    "key_prefix": self.key_prefix
                }
        else:
            return {
                "connected": False,
                "error": "Redis client not available",
                "key_prefix": self.key_prefix
            }

# Global instances for different use cases
workflow_redis = RedisBridge(key_prefix="workflow_")
shared_redis = RedisBridge(key_prefix="shared_")
general_redis = RedisBridge(key_prefix="langflow_")