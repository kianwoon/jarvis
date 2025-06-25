"""
Base Redis cache module with proper error handling
"""
import redis
import os
import json
from typing import Optional, Any, Dict
from functools import wraps
from datetime import datetime, date
from decimal import Decimal

class RedisCache:
    """Redis cache with fallback support"""
    
    def __init__(self, key_prefix: str = ""):
        self.key_prefix = key_prefix
        self._client: Optional[redis.Redis] = None
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_password = os.getenv("REDIS_PASSWORD", None)
        
    def _get_client(self) -> Optional[redis.Redis]:
        """Get Redis client with lazy initialization"""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    password=self.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                self._client.ping()
                return self._client
            except (redis.ConnectionError, redis.TimeoutError) as e:
                print(f"Redis connection failed: {e}. Running without cache.")
                self._client = None
        return self._client
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with fallback"""
        client = self._get_client()
        if not client:
            return default
            
        try:
            full_key = f"{self.key_prefix}{key}"
            value = client.get(full_key)
            if value is not None:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return default
        except Exception as e:
            print(f"Redis get error: {e}")
            return default
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Type {type(obj)} not serializable")
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in cache with automatic JSON serialization"""
        client = self._get_client()
        if not client:
            return False
            
        try:
            full_key = f"{self.key_prefix}{key}"
            if isinstance(value, (dict, list)):
                # Use custom serializer for datetime and other objects
                value = json.dumps(value, default=self._json_serializer)
            elif isinstance(value, (datetime, date)):
                # Handle datetime objects directly
                value = value.isoformat()
            
            if expire:
                return bool(client.setex(full_key, expire, value))
            else:
                return bool(client.set(full_key, value))
        except Exception as e:
            print(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        client = self._get_client()
        if not client:
            return False
            
        try:
            full_key = f"{self.key_prefix}{key}"
            return bool(client.delete(full_key))
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        client = self._get_client()
        if not client:
            return False
            
        try:
            full_key = f"{self.key_prefix}{key}"
            return bool(client.exists(full_key))
        except Exception as e:
            print(f"Redis exists error: {e}")
            return False
    
    def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set TTL for existing key"""
        client = self._get_client()
        if not client:
            return False
            
        try:
            full_key = f"{self.key_prefix}{key}"
            return bool(client.expire(full_key, ttl_seconds))
        except Exception as e:
            print(f"Redis expire error: {e}")
            return False

def cache_fallback(func):
    """Decorator that provides fallback when Redis is not available"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except redis.RedisError as e:
            print(f"Redis error in {func.__name__}: {e}")
            # Return None or implement fallback logic
            return None
    return wrapper