import redis
import json
import yaml
from pathlib import Path
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
CLASSIFIER_CONFIG_KEY = 'enhanced_query_classifier_config'
CLASSIFIER_SETTINGS_KEY = 'enhanced_query_classifier_settings'

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

def get_enhanced_classifier_config():
    """Get enhanced query classifier configuration from cache or file"""
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(CLASSIFIER_CONFIG_KEY)
            if cached:
                config = json.loads(cached)
                return config
        except Exception as e:
            print(f"Redis error: {e}, falling back to file")
    
    # If not cached or Redis failed, load from file
    return reload_classifier_config()

def set_enhanced_classifier_config(config_dict):
    """Cache enhanced query classifier configuration in Redis"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.setex(CLASSIFIER_CONFIG_KEY, 3600, json.dumps(config_dict))  # 1 hour TTL
        except Exception as e:
            print(f"Failed to cache classifier config in Redis: {e}")

def reload_classifier_config():
    """Load configuration from YAML file and cache it"""
    try:
        config_path = Path(__file__).parent.parent / "langchain" / "query_patterns_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Cache in Redis
        set_enhanced_classifier_config(config)
        
        return config
    except Exception as e:
        print(f"Failed to load classifier config from file: {e}")
        # Return default config
        return {
            "tool_patterns": {},
            "rag_patterns": {},
            "code_patterns": {},
            "multi_agent_patterns": {},
            "direct_llm_patterns": {},
            "hybrid_indicators": {},
            "settings": {
                "min_confidence_threshold": 0.1,
                "max_classifications": 3,
                "enable_hybrid_detection": True,
                "confidence_decay_factor": 0.8,
                "pattern_combination_bonus": 0.15
            }
        }

def get_enhanced_classifier_settings():
    """Get enhanced query classifier settings from database/cache"""
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(CLASSIFIER_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                return settings
        except Exception as e:
            print(f"Redis error: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_classifier_settings()

def reload_classifier_settings():
    """Load settings from database and cache them"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row and row.settings:
                classifier_settings = row.settings.get('query_classifier', {})
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.setex(CLASSIFIER_SETTINGS_KEY, 3600, json.dumps(classifier_settings))  # 1 hour TTL
                    except Exception as e:
                        print(f"Failed to cache classifier settings in Redis: {e}")
                
                return classifier_settings
            else:
                # Return default settings
                default_settings = {
                    "min_confidence_threshold": 0.1,
                    "max_classifications": 3,
                    "classifier_max_tokens": 10,
                    "enable_hybrid_detection": True,
                    "confidence_decay_factor": 0.8,
                    "pattern_combination_bonus": 0.15
                }
                return default_settings
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to load classifier settings from database: {e}")
        # Return default settings on any error
        return {
            "min_confidence_threshold": 0.1,
            "max_classifications": 3,
            "classifier_max_tokens": 10,
            "enable_hybrid_detection": True,
            "confidence_decay_factor": 0.8,
            "pattern_combination_bonus": 0.15
        }

def set_enhanced_classifier_settings(settings_dict):
    """Cache enhanced query classifier settings in Redis"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.setex(CLASSIFIER_SETTINGS_KEY, 3600, json.dumps(settings_dict))  # 1 hour TTL
        except Exception as e:
            print(f"Failed to cache classifier settings in Redis: {e}")