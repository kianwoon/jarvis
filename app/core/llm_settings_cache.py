import redis
import json
from app.core.db import SessionLocal, Settings as SettingsModel
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
LLM_SETTINGS_KEY = 'llm_settings_cache'

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

def get_llm_settings():
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(LLM_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                # Validate structure
                if 'thinking_mode' not in settings or 'non_thinking_mode' not in settings:
                    raise RuntimeError('LLM settings missing thinking_mode or non_thinking_mode')
                return settings
        except Exception as e:
            print(f"Redis error: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_llm_settings()

def set_llm_settings(settings_dict):
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(LLM_SETTINGS_KEY, json.dumps(settings_dict))
        except Exception as e:
            print(f"Failed to cache settings in Redis: {e}")

# Call this after updating settings in DB
def reload_llm_settings():
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
        if row:
            settings = row.settings
            if 'thinking_mode' not in settings or 'non_thinking_mode' not in settings:
                raise RuntimeError('LLM settings missing thinking_mode or non_thinking_mode')
            
            # Try to cache in Redis
            redis_client = _get_redis_client()
            if redis_client:
                try:
                    redis_client.set(LLM_SETTINGS_KEY, json.dumps(settings))
                except Exception as e:
                    print(f"Failed to cache settings in Redis: {e}")
            
            return settings
        else:
            # No user-defined settings, raise error
            raise RuntimeError('No LLM settings found in database')
    finally:
        db.close() 