"""
Enterprise LLM Settings Cache with auto-reload and fallback
"""
from app.core.enterprise_cache_base import EnterpriseCache
import logging

logger = logging.getLogger(__name__)

def _load_llm_settings_from_db():
    """Database loader function"""
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row and row.settings:
                settings = row.settings
                if 'thinking_mode' not in settings or 'non_thinking_mode' not in settings:
                    raise RuntimeError('LLM settings missing thinking_mode or non_thinking_mode')
                return settings
            else:
                raise RuntimeError('No LLM settings found in database')
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to load LLM settings from database: {e}")
        raise

def _get_default_llm_settings():
    """Default LLM settings factory"""
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

# Create enterprise cache instance
llm_cache = EnterpriseCache(
    cache_key='llm_settings_cache',
    redis_ttl=3600,  # 1 hour
    auto_reload_interval=300,  # 5 minutes background refresh
    db_loader=_load_llm_settings_from_db,
    default_value_factory=_get_default_llm_settings
)

def get_llm_settings():
    """
    Get LLM settings with enterprise-grade reliability:
    - Redis cache first
    - Database fallback with circuit breaker
    - Default values as ultimate fallback
    - Auto-reload every 5 minutes
    """
    return llm_cache.get()

def set_llm_settings(settings_dict):
    """Set LLM settings in cache and database"""
    llm_cache.set(settings_dict)

def reload_llm_settings():
    """Force reload from database"""
    return llm_cache.get(force_reload=True)

def get_llm_cache_metrics():
    """Get cache performance metrics for monitoring"""
    return llm_cache.get_metrics()

# Start background refresh (call this from startup)
def start_llm_cache_background_refresh():
    llm_cache.start_background_refresh()