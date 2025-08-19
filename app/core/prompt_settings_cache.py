"""
Prompt Settings Cache Service
Manages all prompt templates and formatting settings from the database
"""

import json
from typing import Dict, Any, Optional
from app.core.config import get_settings

config = get_settings()
SYNTHESIS_PROMPTS_KEY = 'synthesis_prompts_cache'
FORMATTING_TEMPLATES_KEY = 'formatting_templates_cache'
SYSTEM_BEHAVIORS_KEY = 'system_behaviors_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
    return r

def get_synthesis_prompts() -> Dict[str, str]:
    """
    Get synthesis prompt templates from database with caching
    
    Returns:
        Dict of prompt templates for synthesis operations
    """
    # Layer 1: Try Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(SYNTHESIS_PROMPTS_KEY)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"[PROMPT_SETTINGS] Redis error: {e}, falling back to database")
    
    # Layer 2: Load from database
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'synthesis_prompts').first()
            if row and row.settings:
                settings = row.settings
                
                # Cache in Redis
                if redis_client:
                    try:
                        from app.core.timeout_settings_cache import get_settings_cache_ttl
                        ttl = get_settings_cache_ttl()
                        redis_client.setex(SYNTHESIS_PROMPTS_KEY, ttl, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache synthesis prompts in Redis: {e}")
                
                return settings
        finally:
            db.close()
    except Exception as e:
        print(f"[PROMPT_SETTINGS] Database error: {e}, using fallback")
    
    # Layer 3: Emergency fallback
    return {
        'knowledge_base_synthesis': "Answer the question based on the provided context.\n\nQuestion: {enhanced_question}\n\nContext: {documents_text}",
        'default_synthesis': "Answer based on: {context}",
        'radiating_synthesis': "Answer based on analysis: {analysis}"
    }

def get_formatting_templates() -> Dict[str, str]:
    """
    Get formatting templates from database with caching
    
    Returns:
        Dict of formatting templates
    """
    # Layer 1: Try Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(FORMATTING_TEMPLATES_KEY)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"[FORMATTING_TEMPLATES] Redis error: {e}, falling back to database")
    
    # Layer 2: Load from database
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'formatting_templates').first()
            if row and row.settings:
                settings = row.settings
                
                # Cache in Redis
                if redis_client:
                    try:
                        from app.core.timeout_settings_cache import get_settings_cache_ttl
                        ttl = get_settings_cache_ttl()
                        redis_client.setex(FORMATTING_TEMPLATES_KEY, ttl, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache formatting templates in Redis: {e}")
                
                return settings
        finally:
            db.close()
    except Exception as e:
        print(f"[FORMATTING_TEMPLATES] Database error: {e}, using fallback")
    
    # Layer 3: Emergency fallback
    return {
        'document_context': "{file_info}: {content}",
        'search_result': "Source: {source}\nContent: {content}",
        'entity_reference': "{entity_name} ({entity_type})"
    }

def get_system_behaviors() -> Dict[str, Any]:
    """
    Get system behavior settings from database with caching
    
    Returns:
        Dict of system behavior configurations
    """
    # Layer 1: Try Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(SYSTEM_BEHAVIORS_KEY)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"[SYSTEM_BEHAVIORS] Redis error: {e}, falling back to database")
    
    # Layer 2: Load from database
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'system_behaviors').first()
            if row and row.settings:
                settings = row.settings
                
                # Cache in Redis
                if redis_client:
                    try:
                        from app.core.timeout_settings_cache import get_settings_cache_ttl
                        ttl = get_settings_cache_ttl()
                        redis_client.setex(SYSTEM_BEHAVIORS_KEY, ttl, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache system behaviors in Redis: {e}")
                
                return settings
        finally:
            db.close()
    except Exception as e:
        print(f"[SYSTEM_BEHAVIORS] Database error: {e}, using fallback")
    
    # Layer 3: Emergency fallback
    return {
        'answer_first_approach': True,
        'include_sources': True,
        'format_with_markdown': True,
        'max_source_attributions': 5
    }

def reload_prompt_settings():
    """Reload all prompt settings from database and clear cache"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            # Clear all prompt-related caches
            for key in [SYNTHESIS_PROMPTS_KEY, FORMATTING_TEMPLATES_KEY, SYSTEM_BEHAVIORS_KEY]:
                redis_client.delete(key)
            print("Cleared all prompt settings caches")
        except Exception as e:
            print(f"Failed to clear prompt caches: {e}")
    
    # Force reload each category
    get_synthesis_prompts()
    get_formatting_templates()
    get_system_behaviors()
    
    return True