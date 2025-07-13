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
                # Validate structure - check for new schema first, then old schema
                has_new_schema = 'thinking_mode_params' in settings and 'non_thinking_mode_params' in settings
                has_old_schema = 'thinking_mode' in settings and 'non_thinking_mode' in settings
                if not has_new_schema and not has_old_schema:
                    raise RuntimeError('LLM settings missing mode parameters (expecting thinking_mode_params/non_thinking_mode_params or thinking_mode/non_thinking_mode)')
                return settings
            else:
                raise RuntimeError('No LLM settings found in database')
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to load LLM settings from database: {e}")
        raise

def _get_default_llm_settings():
    """Default LLM settings factory - using new schema"""
    return {
        "main_llm": {
            "mode": "thinking",
            "model": "llama3.1:8b",
            "max_tokens": 4000,
            "model_server": "http://localhost:11434",
            "system_prompt": "You are Jarvis, an AI assistant."
        },
        "thinking_mode_params": {"temperature": 0.7, "top_p": 0.9, "min_p": 0, "top_k": 20},
        "non_thinking_mode_params": {"temperature": 0.7, "top_p": 0.9, "min_p": 0, "top_k": 20},
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