"""
Query Classifier Settings Cache
Manages query classifier configuration with Redis caching and database persistence
"""

import json
import logging
from typing import Dict, Optional
from app.core.db import get_db
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)

# Default query classifier settings
DEFAULT_QUERY_CLASSIFIER_SETTINGS = {
    "min_confidence_threshold": 0.1,
    "max_classifications": 3,
    "enable_hybrid_detection": True,
    "confidence_decay_factor": 0.8,
    "pattern_combination_bonus": 0.15
}

CACHE_KEY = "query_classifier_settings"
CACHE_TTL = 3600  # 1 hour

def get_query_classifier_settings() -> Dict:
    """Get query classifier settings from cache or database"""
    # Try Redis first
    redis_client = get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(CACHE_KEY)
            if cached:
                logger.debug(f"Retrieved query classifier settings from Redis cache")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis error when getting query classifier settings: {e}")
    
    # Fallback to database
    try:
        session = next(get_db())
        from sqlalchemy import text
        
        result = session.execute(
            text("SELECT settings FROM settings WHERE category = 'llm'")
        ).fetchone()
        
        if result and result[0]:
            settings = result[0] if isinstance(result[0], dict) else json.loads(result[0])
            query_classifier_settings = settings.get('query_classifier', DEFAULT_QUERY_CLASSIFIER_SETTINGS)
            
            # Cache in Redis
            if redis_client:
                try:
                    redis_client.setex(
                        CACHE_KEY,
                        CACHE_TTL,
                        json.dumps(query_classifier_settings)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache query classifier settings in Redis: {e}")
            
            return query_classifier_settings
        
        session.close()
    except Exception as e:
        logger.error(f"Database error when getting query classifier settings: {e}")
    
    logger.info("Using default query classifier settings")
    return DEFAULT_QUERY_CLASSIFIER_SETTINGS

def update_query_classifier_settings(settings: Dict) -> bool:
    """Update query classifier settings in database and cache"""
    try:
        session = next(get_db())
        from sqlalchemy import text
        
        # Get current LLM settings
        result = session.execute(
            text("SELECT settings FROM settings WHERE category = 'llm'")
        ).fetchone()
        
        if result:
            current_settings = result[0] if isinstance(result[0], dict) else json.loads(result[0])
            current_settings['query_classifier'] = settings
            
            # Update database
            session.execute(
                text("UPDATE settings SET settings = :settings WHERE category = 'llm'"),
                {"settings": json.dumps(current_settings)}
            )
            session.commit()
            
            # Update Redis cache
            redis_client = get_redis_client()
            if redis_client:
                try:
                    redis_client.setex(
                        CACHE_KEY,
                        CACHE_TTL,
                        json.dumps(settings)
                    )
                except Exception as e:
                    logger.warning(f"Failed to update Redis cache: {e}")
            
            logger.info("Query classifier settings updated successfully")
            return True
        
        session.close()
    except Exception as e:
        logger.error(f"Failed to update query classifier settings: {e}")
        return False

def clear_cache():
    """Clear the query classifier settings cache"""
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_client.delete(CACHE_KEY)
            logger.info("Query classifier settings cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")