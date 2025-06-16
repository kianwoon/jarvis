import redis
import json
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
LANGFUSE_SETTINGS_KEY = 'langfuse_settings_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client with lazy initialization"""
    global r
    if r is None:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
            r.ping()  # Test connection
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"Redis connection failed: {e}")
            r = None
    return r

def get_langfuse_settings():
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(LANGFUSE_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                return settings
        except Exception as e:
            print(f"Redis error: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_langfuse_settings()

def set_langfuse_settings(settings_dict):
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(LANGFUSE_SETTINGS_KEY, json.dumps(settings_dict))
        except Exception as e:
            print(f"Failed to cache settings in Redis: {e}")

# Call this after updating settings in DB
def reload_langfuse_settings():
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'langfuse').first()
            if row:
                settings = row.settings
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(LANGFUSE_SETTINGS_KEY, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache settings in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, return default settings
                return get_default_langfuse_settings()
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to load Langfuse settings from database: {e}")
        # Return default settings to prevent complete failure
        return get_default_langfuse_settings()

def get_default_langfuse_settings():
    """Return default Langfuse settings"""
    return {
        "enabled": False,
        "host": "http://localhost:3000",
        "public_key": "",
        "secret_key": "",
        "project_id": "",
        "trace_sampling_rate": 1.0,
        "debug_mode": False,
        "flush_at": 50,
        "flush_interval": 10,
        "timeout": 10,
        # Cloudflare R2 Storage Configuration
        "s3_enabled": False,
        "s3_bucket_name": "",
        "s3_endpoint_url": "",
        "s3_access_key_id": "",
        "s3_secret_access_key": "",
        "s3_region": "auto"
    }