"""
Health check utilities for external services
"""
import time
import redis
import psycopg2
from app.core.config import get_settings

def wait_for_redis(max_retries: int = 30, retry_delay: float = 1.0) -> bool:
    """Wait for Redis to be ready"""
    config = get_settings()
    
    for attempt in range(max_retries):
        try:
            r = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                socket_connect_timeout=2
            )
            r.ping()
            print(f"✓ Redis is ready at {config.REDIS_HOST}:{config.REDIS_PORT}")
            return True
        except (redis.ConnectionError, redis.TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"Waiting for Redis... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"✗ Redis not available after {max_retries} attempts: {e}")
                return False
    return False

def wait_for_postgres(max_retries: int = 30, retry_delay: float = 1.0) -> bool:
    """Wait for PostgreSQL to be ready"""
    config = get_settings()
    
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=config.POSTGRES_HOST,
                port=config.POSTGRES_PORT,
                user=config.POSTGRES_USER,
                password=config.POSTGRES_PASSWORD,
                database=config.POSTGRES_DB,
                connect_timeout=2
            )
            conn.close()
            print(f"✓ PostgreSQL is ready at {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")
            return True
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                print(f"Waiting for PostgreSQL... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"✗ PostgreSQL not available after {max_retries} attempts: {e}")
                return False
    return False

def check_all_services():
    """Check all required services"""
    print("Checking service dependencies...")
    
    # These are warnings, not failures - the app can run without them
    postgres_ready = wait_for_postgres(max_retries=10)
    redis_ready = wait_for_redis(max_retries=10)
    
    if not postgres_ready:
        print("⚠️  Warning: PostgreSQL is not available. Using SQLite fallback.")
    
    if not redis_ready:
        print("⚠️  Warning: Redis is not available. Running without cache.")
    
    return True  # Always return True so the app starts

if __name__ == "__main__":
    check_all_services()