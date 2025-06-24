import json
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
LLM_SETTINGS_KEY = 'llm_settings_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
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
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
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
    except Exception as e:
        print(f"Failed to load LLM settings from database: {e}")
        # Return default settings to prevent complete failure
        return {
            "model": "llama3.1:8b",
            "thinking_mode": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4000},
            "non_thinking_mode": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 4000},
            "max_tokens": 4000,
            "query_classifier": {
                "min_confidence_threshold": 0.1,
                "max_classifications": 3,
                "classifier_max_tokens": 10,
                "enable_hybrid_detection": True,
                "confidence_decay_factor": 0.8,
                "pattern_combination_bonus": 0.15
            }
        } 