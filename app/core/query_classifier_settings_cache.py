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
    # Classification thresholds
    "min_confidence_threshold": 0.1,
    "direct_execution_threshold": 0.55,  # For TOOL queries
    "llm_direct_threshold": 0.8,  # For LLM queries
    "multi_agent_threshold": 0.6,  # For MULTI_AGENT queries
    
    # Pattern-based classification settings
    "max_classifications": 3,
    "enable_hybrid_detection": True,
    "confidence_decay_factor": 0.8,
    "pattern_combination_bonus": 0.15,
    
    # LLM-based classification settings
    "enable_llm_classification": False,  # Feature toggle for LLM-based classification
    "llm_model": "",  # LLM model for classification (e.g., "qwen2.5:0.5b")
    "context_length": 0,  # Model context length (auto-updated when model changes)
    "llm_temperature": 0.1,  # Lower temperature for consistent classification
    "llm_max_tokens": 10,  # Token limit for classification responses (auto-updated to 75% of context)
    "llm_timeout_seconds": 5,  # Timeout for LLM classification calls
    "llm_system_prompt": "You are a query classifier. Classify the user query into exactly one of these types: RAG (for questions requiring document search), TOOL (for actions requiring tools), LLM (for general questions), MULTI_AGENT (for complex tasks). Respond with ONLY the type and confidence in this exact format: TYPE|CONFIDENCE (e.g., 'rag|0.85'). Do not include explanations, thinking, or other text.",
    "fallback_to_patterns": True,  # Fallback to pattern-based classification if LLM fails
    "llm_classification_priority": False  # If true, use LLM first; if false, use patterns first
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
                
                # Merge with new defaults to ensure all fields are present
                merged_settings = DEFAULT_QUERY_CLASSIFIER_SETTINGS.copy()
                merged_settings.update(settings)
                
                # If new fields were added, save back to database
                if merged_settings != settings:
                    current_settings = row.settings
                    current_settings['query_classifier'] = merged_settings
                    row.settings = current_settings
                    db.commit()
                    logger.info("Updated query classifier settings with new LLM fields")
                    settings = merged_settings
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(settings))
                    except Exception as e:
                        logger.warning(f"Failed to cache settings in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, initialize with defaults and save to DB
                if row:
                    # LLM row exists but no query_classifier, add it
                    current_settings = row.settings or {}
                    current_settings['query_classifier'] = DEFAULT_QUERY_CLASSIFIER_SETTINGS
                    row.settings = current_settings
                    db.commit()
                    logger.info("Initialized query classifier settings with LLM defaults")
                else:
                    # No LLM row at all, create one with query_classifier
                    new_row = SettingsModel(
                        category='llm', 
                        settings={'query_classifier': DEFAULT_QUERY_CLASSIFIER_SETTINGS}
                    )
                    db.add(new_row)
                    db.commit()
                    logger.info("Created new LLM settings with query classifier defaults")
                
                # Cache the defaults
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

def validate_query_classifier_settings(settings: Dict) -> Dict:
    """Validate and sanitize query classifier settings"""
    validated = settings.copy()
    
    # Validate numeric ranges
    validated["min_confidence_threshold"] = max(0.0, min(1.0, float(validated.get("min_confidence_threshold", 0.1))))
    validated["direct_execution_threshold"] = max(0.0, min(1.0, float(validated.get("direct_execution_threshold", 0.55))))
    validated["llm_direct_threshold"] = max(0.0, min(1.0, float(validated.get("llm_direct_threshold", 0.8))))
    validated["multi_agent_threshold"] = max(0.0, min(1.0, float(validated.get("multi_agent_threshold", 0.6))))
    validated["llm_temperature"] = max(0.0, min(2.0, float(validated.get("llm_temperature", 0.1))))
    validated["confidence_decay_factor"] = max(0.0, min(1.0, float(validated.get("confidence_decay_factor", 0.8))))
    validated["pattern_combination_bonus"] = max(0.0, min(1.0, float(validated.get("pattern_combination_bonus", 0.15))))
    
    # Validate integers
    validated["max_classifications"] = max(1, min(10, int(validated.get("max_classifications", 3))))
    validated["llm_max_tokens"] = max(1, min(100, int(validated.get("llm_max_tokens", 10))))
    validated["llm_timeout_seconds"] = max(1, min(30, int(validated.get("llm_timeout_seconds", 5))))
    
    # Validate booleans
    validated["enable_hybrid_detection"] = bool(validated.get("enable_hybrid_detection", True))
    validated["enable_llm_classification"] = bool(validated.get("enable_llm_classification", False))
    validated["fallback_to_patterns"] = bool(validated.get("fallback_to_patterns", True))
    validated["llm_classification_priority"] = bool(validated.get("llm_classification_priority", False))
    
    # Validate strings
    validated["llm_model"] = str(validated.get("llm_model", "")).strip()
    validated["llm_system_prompt"] = str(validated.get("llm_system_prompt", DEFAULT_QUERY_CLASSIFIER_SETTINGS["llm_system_prompt"])).strip()
    
    return validated

def update_query_classifier_settings(settings: Dict) -> bool:
    """Update query classifier settings in database and cache"""
    try:
        # Validate settings first
        validated_settings = validate_query_classifier_settings(settings)
        
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row:
                current_settings = row.settings
                current_settings['query_classifier'] = validated_settings
                row.settings = current_settings
                db.commit()
                
                # Update Redis cache
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(validated_settings))
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