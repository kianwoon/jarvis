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
                
                # Merge in latest query classifier settings with new LLM fields
                try:
                    from app.core.query_classifier_settings_cache import get_query_classifier_settings
                    latest_query_classifier = get_query_classifier_settings()
                    settings['query_classifier'] = latest_query_classifier
                except Exception as e:
                    print(f"Warning: Failed to merge query classifier settings: {e}")
                
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
        raise

def get_main_llm_full_config(settings=None):
    """Construct full main_llm configuration by merging base config with mode parameters"""
    if settings is None:
        settings = get_llm_settings()
    
    main_llm = settings.get('main_llm', {})
    mode = main_llm.get('mode', 'thinking')
    
    # Get the appropriate mode parameters
    if mode == 'thinking':
        mode_params = settings.get('thinking_mode_params', {})
    else:
        mode_params = settings.get('non_thinking_mode_params', {})
    
    # Merge base config with mode parameters
    full_config = main_llm.copy()
    full_config.update(mode_params)
    
    return full_config

def get_query_classifier_full_config(settings=None):
    """Construct full query_classifier configuration by merging base config with mode parameters"""
    if settings is None:
        settings = get_llm_settings()
    
    query_classifier = settings.get('query_classifier', {})
    # Query classifier should default to non-thinking mode for simple classification responses
    mode = query_classifier.get('mode', 'non-thinking')
    
    # Get the appropriate mode parameters
    if mode == 'thinking':
        mode_params = settings.get('thinking_mode_params', {})
    else:
        mode_params = settings.get('non_thinking_mode_params', {})
    
    # Merge base config with mode parameters
    full_config = query_classifier.copy()
    full_config.update(mode_params)
    
    return full_config 