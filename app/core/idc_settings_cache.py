"""
IDC Settings Cache Module
Follows the exact pattern from llm_settings_cache.py with 3-layer failsafe system.
NO HARDCODING - everything comes from environment variables and database.
"""

import json
import logging
from typing import Dict, Any, Optional
from app.core.config import get_settings
from app.core.timeout_settings_cache import get_settings_cache_ttl

logger = logging.getLogger(__name__)

config = get_settings()
IDC_SETTINGS_KEY = 'idc_settings_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
    return r

def get_idc_settings() -> Dict[str, Any]:
    """
    Get IDC settings with comprehensive failsafe system to prevent cascading failures.
    
    Failsafe hierarchy:
    1. Redis cache (fastest)
    2. Database reload (reliable)
    3. Emergency fallback settings (prevents total failure)
    """
    # Layer 1: Try Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(IDC_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                # Validate settings contain required keys
                if _validate_idc_settings(settings):
                    return settings
                else:
                    logger.warning("[IDC_SETTINGS] Cached settings invalid, falling back to database")
        except Exception as e:
            logger.warning(f"[IDC_SETTINGS] Redis error: {e}, falling back to database")
    
    # Layer 2: Try database reload
    try:
        return reload_idc_settings()
    except Exception as e:
        logger.error(f"[IDC_SETTINGS] Database reload failed: {e}, using emergency fallback")
        
        # Layer 3: Emergency fallback to prevent total system failure
        return _get_emergency_fallback_settings()

def set_idc_settings(settings_dict: Dict[str, Any]):
    """Cache IDC settings in Redis"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(IDC_SETTINGS_KEY, json.dumps(settings_dict), ex=get_settings_cache_ttl())
            logger.info("[IDC_SETTINGS] Settings cached in Redis")
        except Exception as e:
            logger.error(f"[IDC_SETTINGS] Failed to cache settings in Redis: {e}")

def reload_idc_settings() -> Dict[str, Any]:
    """Reload IDC settings from database"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'idc').first()
            if row:
                settings = row.settings
                
                # Ensure all required fields are present
                settings = _ensure_complete_settings(settings)
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(IDC_SETTINGS_KEY, json.dumps(settings), ex=get_settings_cache_ttl())
                        logger.info("[IDC_SETTINGS] Settings cached in Redis")
                    except Exception as e:
                        logger.warning(f"[IDC_SETTINGS] Failed to cache settings in Redis: {e}")
                
                return settings
            else:
                # No IDC settings in database, create default and save
                logger.info("[IDC_SETTINGS] No IDC settings found, creating defaults")
                default_settings = _get_emergency_fallback_settings()
                
                # Save to database
                new_row = SettingsModel(category='idc', settings=default_settings)
                db.add(new_row)
                db.commit()
                
                return default_settings
        finally:
            db.close()
    except Exception as e:
        logger.error(f"[IDC_SETTINGS] Failed to load IDC settings from database: {e}")
        raise RuntimeError(f"Database configuration retrieval failed: {e}")

def get_extraction_config(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get extraction configuration from IDC settings"""
    if settings is None:
        settings = get_idc_settings()
    
    extraction_config = settings.get('extraction', {})
    
    # Merge with LLM settings for model configuration
    try:
        from app.core.llm_settings_cache import get_llm_settings
        llm_settings = get_llm_settings()
        main_llm = llm_settings.get('main_llm', {})
        
        # Use main LLM model if not specified - align with Jarvis system default
        if 'model' not in extraction_config:
            extraction_config['model'] = main_llm.get('model', 'qwen3:30b-a3b-q4_K_M')
        if 'model_server' not in extraction_config:
            extraction_config['model_server'] = main_llm.get('model_server', '')
    except Exception as e:
        logger.warning(f"[IDC_SETTINGS] Could not merge LLM settings: {e}")
    
    return extraction_config

def get_validation_config(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Get validation configuration from IDC settings"""
    if settings is None:
        settings = get_idc_settings()
    
    validation_config = settings.get('validation', {})
    
    # Merge with LLM settings for model configuration
    try:
        from app.core.llm_settings_cache import get_llm_settings
        llm_settings = get_llm_settings()
        main_llm = llm_settings.get('main_llm', {})
        
        # Use main LLM model if not specified - align with Jarvis system default
        if 'model' not in validation_config:
            validation_config['model'] = main_llm.get('model', 'qwen3:30b-a3b-q4_K_M')
        if 'model_server' not in validation_config:
            validation_config['model_server'] = main_llm.get('model_server', '')
    except Exception as e:
        logger.warning(f"[IDC_SETTINGS] Could not merge LLM settings: {e}")
    
    return validation_config

def get_extraction_system_prompt(settings: Optional[Dict[str, Any]] = None) -> str:
    """Get extraction system prompt from IDC settings or SettingsPromptService"""
    if settings is None:
        settings = get_idc_settings()
    
    # First try to get from IDC settings
    extraction_prompt = settings.get('extraction', {}).get('system_prompt', '')
    
    if extraction_prompt:
        return extraction_prompt
    
    # Fallback to SettingsPromptService
    try:
        from app.services.settings_prompt_service import get_prompt_service
        prompt_service = get_prompt_service()
        extraction_prompt = prompt_service.get_prompt('idc_extraction')
        if extraction_prompt:
            return extraction_prompt
    except Exception as e:
        logger.warning(f"[IDC_SETTINGS] Could not get prompt from service: {e}")
    
    # Ultimate fallback
    return _get_default_extraction_prompt()

def get_validation_system_prompt(settings: Optional[Dict[str, Any]] = None) -> str:
    """Get validation system prompt from IDC settings or SettingsPromptService"""
    if settings is None:
        settings = get_idc_settings()
    
    # First try to get from IDC settings
    validation_prompt = settings.get('validation', {}).get('system_prompt', '')
    
    if validation_prompt:
        return validation_prompt
    
    # Fallback to SettingsPromptService
    try:
        from app.services.settings_prompt_service import get_prompt_service
        prompt_service = get_prompt_service()
        validation_prompt = prompt_service.get_prompt('idc_validation')
        if validation_prompt:
            return validation_prompt
    except Exception as e:
        logger.warning(f"[IDC_SETTINGS] Could not get prompt from service: {e}")
    
    # Ultimate fallback
    return _get_default_validation_prompt()

def _validate_idc_settings(settings: Dict[str, Any]) -> bool:
    """Validate that IDC settings contain required keys"""
    if not isinstance(settings, dict):
        return False
    
    required_keys = ['extraction', 'validation', 'comparison']
    for key in required_keys:
        if key not in settings or not isinstance(settings[key], dict):
            logger.warning(f"[IDC_SETTINGS] Missing or invalid key: {key}")
            return False
    
    return True

def _ensure_complete_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure settings have all required fields with defaults"""
    defaults = _get_emergency_fallback_settings()
    
    # Merge defaults with existing settings
    for key in ['extraction', 'validation', 'comparison']:
        if key not in settings:
            settings[key] = defaults[key]
        else:
            # Merge nested dictionaries
            for subkey, value in defaults[key].items():
                if subkey not in settings[key]:
                    settings[key][subkey] = value
    
    return settings

def _get_default_extraction_prompt() -> str:
    """Default extraction prompt when all else fails"""
    return """Extract and convert the following document to clean, structured markdown format.

REQUIREMENTS:
1. Preserve all important information and structure
2. Use appropriate markdown headers (# ## ###)
3. Format lists, tables, and code blocks properly
4. Remove formatting artifacts and noise
5. Maintain logical document flow
6. Extract key-value pairs as tables where appropriate

Return ONLY the clean markdown content without explanations."""

def _get_default_validation_prompt() -> str:
    """Default validation prompt when all else fails"""
    return """Analyze the following comparison result and identify the key differences.

Focus on:
1. Content additions or removals
2. Structural changes
3. Value modifications
4. Formatting differences

Provide a clear, concise summary of the changes."""

def _get_emergency_fallback_settings() -> Dict[str, Any]:
    """
    Emergency fallback settings to prevent total system failure.
    Uses safe, known-working configuration from environment.
    """
    logger.warning("[IDC_SETTINGS] EMERGENCY FALLBACK ACTIVATED - Using minimal safe configuration")
    
    import os
    # Get model server from environment - NO HARDCODING
    model_server = os.environ.get("OLLAMA_BASE_URL", "")
    
    if not model_server:
        logger.critical("[IDC_SETTINGS] CRITICAL: No model server configured in environment (OLLAMA_BASE_URL)")
        logger.critical("[IDC_SETTINGS] System will not function without proper model server configuration")
    
    # Align with Jarvis system default model
    default_model = "qwen3:30b-a3b-q4_K_M"
    
    emergency_settings = {
        "extraction": {
            "model": default_model,  # Use Jarvis system default model
            "model_server": model_server,
            "max_tokens": 6000,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 20,
            "system_prompt": _get_default_extraction_prompt(),
            "enable_chunking": True,
            "chunk_size": 2000,
            "chunk_overlap": 100,
            "max_context_usage": 0.35,  # Default 35% context usage for extraction
            "confidence_threshold": 0.8  # Default 80% confidence threshold for extraction
        },
        "validation": {
            "model": default_model,  # Use Jarvis system default model
            "model_server": model_server,
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 0.85,
            "system_prompt": _get_default_validation_prompt(),
            "max_context_usage": 0.35,  # Default 35% context usage for validation
            "confidence_threshold": 0.8,
            "enable_structured_output": True
        },
        "comparison": {
            "algorithm": "semantic",  # semantic, structural, or hybrid
            "similarity_threshold": 0.85,
            "ignore_whitespace": True,
            "ignore_case": False,
            "enable_fuzzy_matching": True,
            "fuzzy_threshold": 0.9
        },
        "_fallback_mode": "emergency",
        "_created_at": json.dumps({"timestamp": "emergency_fallback", "reason": "configuration_retrieval_failure"})
    }
    
    # Try to cache the emergency settings for future use
    try:
        redis_client = _get_redis_client()
        if redis_client:
            redis_client.set(f"{IDC_SETTINGS_KEY}_emergency", json.dumps(emergency_settings), ex=get_settings_cache_ttl())
            logger.info("[IDC_SETTINGS] Emergency settings cached for future fallbacks")
    except Exception as e:
        logger.warning(f"[IDC_SETTINGS] Could not cache emergency settings: {e}")
    
    return emergency_settings