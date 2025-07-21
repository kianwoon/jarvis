"""
Centralized Redis client with connection pooling for automation workflows
"""
import redis
import redis.asyncio as redis_async
import os
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

# Global connection pool instance
_redis_pool = None

def get_redis_pool() -> Optional[redis.ConnectionPool]:
    """
    Get or create Redis connection pool optimized for automation workflows
    """
    global _redis_pool
    
    if _redis_pool is None:
        from app.core.config import get_settings
        settings = get_settings()
        redis_host = settings.REDIS_HOST
        redis_port = settings.REDIS_PORT
        
        # In development, if Redis host is localhost but we're in Docker, try 'redis' first
        if redis_host == "localhost" and os.path.exists("/.dockerenv"):
            redis_host = "redis"
        
        try:
            # Enhanced connection pool for automation workflows with 20+ agents
            _redis_pool = redis.ConnectionPool(
                host=redis_host,
                port=redis_port,
                max_connections=50,        # Allow up to 50 connections for high concurrency
                retry_on_timeout=True,     # Retry on connection timeout
                socket_connect_timeout=5,  # 5 second connection timeout
                socket_timeout=5,          # 5 second socket timeout
                socket_keepalive=True,     # Enable TCP keepalive
                health_check_interval=30   # Health check every 30 seconds
            )
            logger.info(f"Redis connection pool created: {redis_host}:{redis_port} (max_connections=50)")
        except Exception as e:
            logger.error(f"Failed to create Redis connection pool: {e}")
            _redis_pool = None
    
    return _redis_pool

def get_redis_client(max_retries: int = 3, retry_delay: float = 1.0, decode_responses: bool = True) -> Optional[redis.Redis]:
    """
    Get Redis client using connection pool with retry logic and better error handling
    
    Args:
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        decode_responses: Whether to decode responses to strings (default True)
    """
    pool = get_redis_pool()
    if not pool:
        return None
    
    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                connection_pool=pool,
                decode_responses=decode_responses
            )
            client.ping()
            if attempt > 0:
                logger.info(f"Redis connected after {attempt + 1} attempts")
            return client
        except (redis.ConnectionError, redis.TimeoutError) as e:
            if attempt < max_retries - 1:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Redis connection failed after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {e}")
            return None
    
    return None

def get_redis_client_for_langgraph() -> Optional[redis.Redis]:
    """
    Get Redis client specifically configured for LangGraph RedisSaver
    (requires decode_responses=False for binary data handling)
    """
    return get_redis_client(decode_responses=False)

def get_redis_pool_info() -> dict:
    """Get Redis connection pool information for monitoring"""
    pool = get_redis_pool()
    if not pool:
        return {"status": "no_pool"}
    
    return {
        "status": "active",
        "max_connections": pool.max_connections,
        "created_connections": pool.created_connections,
        "available_connections": len(pool._available_connections),
        "in_use_connections": len(pool._in_use_connections)
    }

async def get_async_redis_client(decode_responses: bool = True) -> Optional[redis_async.Redis]:
    """
    Get async Redis client for async operations
    Used by components that require async Redis operations (e.g., conversation managers)
    
    Args:
        decode_responses: Whether to decode responses to strings (default True)
    
    Returns:
        Async Redis client or None if connection fails
    """
    from app.core.config import get_settings
    settings = get_settings()
    redis_host = settings.REDIS_HOST
    redis_port = settings.REDIS_PORT
    
    # In development, if Redis host is localhost but we're in Docker, try 'redis' first
    if redis_host == "localhost" and os.path.exists("/.dockerenv"):
        redis_host = "redis"
    
    try:
        # Create async Redis client with connection URL
        redis_url = f"redis://{redis_host}:{redis_port}"
        client = redis_async.from_url(redis_url, decode_responses=decode_responses)
        
        # Test connection
        await client.ping()
        logger.info(f"Async Redis client connected: {redis_host}:{redis_port}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create async Redis client: {e}")
        return None