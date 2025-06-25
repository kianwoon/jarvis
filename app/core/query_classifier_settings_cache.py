"""
Query Classifier Settings Cache
Manages query classifier configuration with Redis caching and database persistence
"""

import redis
import json
import logging
from typing import Dict, Optional
from app.core.config import get_settings

logger = logging.getLogger(__name__)

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
CACHE_KEY = "query_classifier_settings"

# Don't create connection at import time
r = None

# Default query classifier settings
DEFAULT_QUERY_CLASSIFIER_SETTINGS = {
    "min_confidence_threshold": 0.1,
    "direct_execution_threshold": 0.55,  # For TOOL queries
    "llm_direct_threshold": 0.8,  # For LLM queries
    "multi_agent_threshold": 0.6,  # For MULTI_AGENT queries
    "max_classifications": 3,
    "enable_hybrid_detection": True,
    "confidence_decay_factor": 0.8,
    "pattern_combination_bonus": 0.15
}

def _get_redis_client():
    """Get Redis client with lazy initialization"""
    global r
    if r is None:
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)
            r.ping()  # Test connection
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection failed: {e}")
            r = None
    return r

def get_query_classifier_settings() -> Dict:
    """Get query classifier settings from cache or database"""
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(CACHE_KEY)
            if cached:
                settings = json.loads(cached)
                return settings
        except Exception as e:
            logger.warning(f"Redis error: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_query_classifier_settings()

def set_query_classifier_settings(settings_dict):
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(CACHE_KEY, json.dumps(settings_dict))
        except Exception as e:
            logger.warning(f"Failed to cache settings in Redis: {e}")

def reload_query_classifier_settings():
    """Call this after updating settings in DB"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row and 'query_classifier' in row.settings:
                settings = row.settings['query_classifier']
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(settings))
                    except Exception as e:
                        logger.warning(f"Failed to cache settings in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, return defaults
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(DEFAULT_QUERY_CLASSIFIER_SETTINGS))
                    except Exception as e:
                        logger.warning(f"Failed to cache default settings in Redis: {e}")
                return DEFAULT_QUERY_CLASSIFIER_SETTINGS
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to load query classifier settings from database: {e}")
        # Return default settings to prevent complete failure
        return DEFAULT_QUERY_CLASSIFIER_SETTINGS

def update_query_classifier_settings(settings: Dict) -> bool:
    """Update query classifier settings in database and cache"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row:
                current_settings = row.settings
                current_settings['query_classifier'] = settings
                row.settings = current_settings
                db.commit()
                
                # Update Redis cache
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(settings))
                    except Exception as e:
                        logger.warning(f"Failed to update Redis cache: {e}")
                
                logger.info("Query classifier settings updated successfully")
                return True
            return False
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to update query classifier settings: {e}")
        return False

def clear_cache():
    """Clear the query classifier settings cache"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.delete(CACHE_KEY)
            logger.info("Query classifier settings cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")