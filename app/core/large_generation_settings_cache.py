"""
Cache management for large generation settings
"""

import redis
import json
import os
from typing import Dict, Any, Optional
from app.core.redis_client import get_redis_client
from dataclasses import dataclass, field
from app.core.db import SessionLocal, Settings

# Default configuration values
DEFAULT_LARGE_GENERATION_CONFIG = {
    "detection_thresholds": {
        "strong_number_threshold": 30,
        "medium_number_threshold": 20,
        "small_number_threshold": 20,
        "min_items_for_chunking": 20
    },
    "scoring_parameters": {
        "min_score_for_keywords": 3,
        "min_score_for_medium_numbers": 2,
        "score_multiplier": 15,
        "default_comprehensive_items": 30,
        "min_estimated_items": 10,
        "pattern_score_weight": 2
    },
    "confidence_calculation": {
        "max_score_for_confidence": 5.0,
        "max_number_for_confidence": 100.0
    },
    "processing_parameters": {
        "default_chunk_size": 15,
        "max_target_count": 500,
        "estimated_seconds_per_chunk": 45
    },
    "memory_management": {
        "redis_conversation_ttl": 7 * 24 * 3600,  # 7 days
        "max_redis_messages": 50,
        "max_memory_messages": 20,
        "conversation_history_display": 10
    },
    "keywords_and_patterns": {
        "large_output_indicators": [
            "generate", "create", "list", "write", "develop", "design", "build",
            "comprehensive", "detailed", "complete", "full", "extensive", "thorough",
            "step by step", "step-by-step", "all", "many", "multiple", "various",
            "questions", "examples", "ideas", "recommendations", "strategies", "options",
            "points", "items", "factors", "aspects", "benefits", "advantages", "features"
        ],
        "comprehensive_keywords": [
            "comprehensive", "detailed", "all", "many"
        ],
        "large_patterns": [
            r'\b(\d+)\s+(questions|examples|items|points|ideas|strategies|options|factors|aspects|benefits|features)',
            r'(comprehensive|detailed|complete|full|extensive|thorough)\s+(list|guide|analysis|overview|breakdown)',
            r'(all|many|multiple|various)\s+(ways|methods|approaches|techniques|strategies|options)',
            r'generate.*\b(\d+)',
            r'create.*\b(\d+)',
            r'list.*\b(\d+)'
        ]
    }
}

# Cache key for Redis
LARGE_GENERATION_CACHE_KEY = "large_generation_settings"

def get_large_generation_settings_from_db() -> Dict[str, Any]:
    """Get large generation settings from database"""
    try:
        db = SessionLocal()
        try:
            setting_row = db.query(Settings).filter(Settings.category == "large_generation").first()
            if setting_row and setting_row.settings:
                return setting_row.settings
            else:
                # Return default configuration if no settings found
                return DEFAULT_LARGE_GENERATION_CONFIG
        finally:
            db.close()
    except Exception as e:
        print(f"[ERROR] Failed to get large generation settings from DB: {e}")
        return DEFAULT_LARGE_GENERATION_CONFIG

def cache_large_generation_settings(settings: Dict[str, Any]) -> bool:
    """Cache large generation settings in Redis"""
    try:
        redis_client = get_redis_client()
        if redis_client:
            redis_client.set(
                LARGE_GENERATION_CACHE_KEY, 
                json.dumps(settings), 
                ex=3600  # Cache for 1 hour
            )
            print(f"[DEBUG] Cached large generation settings in Redis")
            return True
    except Exception as e:
        print(f"[ERROR] Failed to cache large generation settings: {e}")
    return False

def get_large_generation_settings_from_cache() -> Optional[Dict[str, Any]]:
    """Get large generation settings from Redis cache"""
    try:
        redis_client = get_redis_client()
        if redis_client:
            cached_data = redis_client.get(LARGE_GENERATION_CACHE_KEY)
            if cached_data:
                settings = json.loads(cached_data)
                print(f"[DEBUG] Retrieved large generation settings from Redis cache")
                return settings
    except Exception as e:
        print(f"[ERROR] Failed to get large generation settings from cache: {e}")
    return None

def get_large_generation_settings() -> Dict[str, Any]:
    """Get large generation settings (cache first, then DB, then defaults)"""
    # Try cache first
    settings = get_large_generation_settings_from_cache()
    if settings:
        return settings
    
    # Try database
    settings = get_large_generation_settings_from_db()
    
    # Cache the settings
    cache_large_generation_settings(settings)
    
    return settings

def reload_large_generation_settings() -> Dict[str, Any]:
    """Reload large generation settings from database and update cache"""
    print("[DEBUG] Reloading large generation settings from database")
    
    # Get fresh settings from database
    settings = get_large_generation_settings_from_db()
    
    # Update cache
    cache_large_generation_settings(settings)
    
    return settings

def validate_large_generation_config(config: Dict[str, Any]) -> tuple[bool, str]:
    """Validate large generation configuration"""
    try:
        # Check required sections
        required_sections = [
            "detection_thresholds",
            "scoring_parameters", 
            "confidence_calculation",
            "processing_parameters",
            "memory_management",
            "keywords_and_patterns"
        ]
        
        for section in required_sections:
            if section not in config:
                return False, f"Missing required section: {section}"
        
        # Validate detection thresholds
        detection = config["detection_thresholds"]
        if detection["strong_number_threshold"] < detection["medium_number_threshold"]:
            return False, "Strong threshold must be >= medium threshold"
        
        if detection["min_items_for_chunking"] < 1:
            return False, "Minimum items for chunking must be >= 1"
        
        # Validate scoring parameters
        scoring = config["scoring_parameters"]
        if scoring["min_score_for_keywords"] < 1:
            return False, "Minimum score for keywords must be >= 1"
        
        if scoring["score_multiplier"] < 1:
            return False, "Score multiplier must be >= 1"
        
        # Validate processing parameters
        processing = config["processing_parameters"]
        if processing["default_chunk_size"] < 1:
            return False, "Default chunk size must be >= 1"
        
        if processing["max_target_count"] < 10:
            return False, "Max target count must be >= 10"
        
        # Validate memory management
        memory = config["memory_management"]
        if memory["redis_conversation_ttl"] < 3600:  # At least 1 hour
            return False, "Redis TTL must be >= 3600 seconds (1 hour)"
        
        if memory["max_redis_messages"] < 10:
            return False, "Max Redis messages must be >= 10"
        
        # Validate keywords and patterns
        keywords = config["keywords_and_patterns"]
        if not keywords["large_output_indicators"]:
            return False, "Large output indicators cannot be empty"
        
        if not keywords["comprehensive_keywords"]:
            return False, "Comprehensive keywords cannot be empty"
        
        return True, "Configuration is valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge provided config with defaults, ensuring all required fields exist"""
    merged = DEFAULT_LARGE_GENERATION_CONFIG.copy()
    
    def deep_merge(default_dict: Dict, user_dict: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        result = default_dict.copy()
        for key, value in user_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return deep_merge(merged, config)

# Export the global getter for use in other modules
_cached_settings = None

def get_config() -> Dict[str, Any]:
    """Get the global large generation configuration"""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = get_large_generation_settings()
    return _cached_settings

def reload_config() -> Dict[str, Any]:
    """Reload configuration and update global cache"""
    global _cached_settings
    _cached_settings = reload_large_generation_settings()
    return _cached_settings