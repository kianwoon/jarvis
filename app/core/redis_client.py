"""
Centralized Redis client with better error handling
"""
import redis
import os
from typing import Optional
import time

def get_redis_client(max_retries: int = 3, retry_delay: float = 1.0) -> Optional[redis.Redis]:
    """
    Get Redis client with retry logic and better error handling
    """
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    # In development, if Redis host is localhost but we're in Docker, try 'redis' first
    if redis_host == "localhost" and os.path.exists("/.dockerenv"):
        redis_host = "redis"
    
    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            client.ping()
            if attempt > 0:
                print(f"Redis connected after {attempt + 1} attempts")
            return client
        except (redis.ConnectionError, redis.TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"Redis connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"Redis connection failed after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            print(f"Unexpected error connecting to Redis: {e}")
            return None
    
    return None