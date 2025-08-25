import json
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
NOTEBOOK_LLM_SETTINGS_KEY = 'notebook_llm_settings_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
    return r

def get_notebook_llm_settings():
    """
    Get notebook LLM settings with comprehensive failsafe system to prevent cascading failures.
    
    Failsafe hierarchy:
    1. Redis cache (fastest)
    2. Database reload (reliable)
    3. Emergency fallback settings (prevents total failure)
    """
    # Layer 1: Try Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(NOTEBOOK_LLM_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                # Validate settings contain required keys
                if _validate_notebook_llm_settings(settings):
                    return settings
                else:
                    print(f"[NOTEBOOK_LLM_SETTINGS] Cached settings invalid, falling back to database")
        except Exception as e:
            print(f"[NOTEBOOK_LLM_SETTINGS] Redis error: {e}, falling back to database")
    
    # Layer 2: Try database reload
    try:
        return reload_notebook_llm_settings()
    except Exception as e:
        print(f"[NOTEBOOK_LLM_SETTINGS] Database reload failed: {e}, using emergency fallback")
        
        # Layer 3: Emergency fallback to prevent total system failure
        return _get_emergency_fallback_settings()

def set_notebook_llm_settings(settings_dict):
    redis_client = _get_redis_client()
    if redis_client:
        try:
            from app.core.timeout_settings_cache import get_settings_cache_ttl
            ttl = get_settings_cache_ttl()
            redis_client.setex(NOTEBOOK_LLM_SETTINGS_KEY, ttl, json.dumps(settings_dict))
        except Exception as e:
            print(f"Failed to cache notebook LLM settings in Redis: {e}")

# Call this after updating settings in DB
def reload_notebook_llm_settings():
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'notebook_llm').first()
            if row:
                settings = row.settings
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        from app.core.timeout_settings_cache import get_settings_cache_ttl
                        ttl = get_settings_cache_ttl()
                        redis_client.setex(NOTEBOOK_LLM_SETTINGS_KEY, ttl, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache notebook LLM settings in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, raise error
                raise RuntimeError('No notebook LLM settings found in database')
        finally:
            db.close()
    except Exception as e:
        print(f"[NOTEBOOK_LLM_SETTINGS] Failed to load notebook LLM settings from database: {e}")
        print(f"[NOTEBOOK_LLM_SETTINGS] This is a critical failure - check database connectivity and settings table")
        raise RuntimeError(f"Database configuration retrieval failed: {e}")

def get_notebook_llm_full_config(settings=None, override_mode=None):
    """Construct full notebook_llm configuration by merging base config with mode parameters
    
    Args:
        settings: Notebook LLM settings dictionary
        override_mode: Override mode detected dynamically ('thinking' or 'non-thinking')
    """
    try:
        if settings is None:
            settings = get_notebook_llm_settings()
    except Exception as e:
        print(f"[NOTEBOOK_LLM_CONFIG] Failed to get settings: {e}, using emergency fallback")
        settings = _get_emergency_fallback_settings()
    
    notebook_llm = settings.get('notebook_llm', {})
    configured_mode = notebook_llm.get('mode', 'thinking')
    
    # Use override mode if provided (from dynamic detection)
    mode = override_mode if override_mode else configured_mode
    
    # Get the appropriate mode parameters
    if mode == 'thinking':
        mode_params = settings.get('thinking_mode_params', {})
    else:
        mode_params = settings.get('non_thinking_mode_params', {})
    
    # Merge base config with mode parameters
    full_config = notebook_llm.copy()
    full_config.update(mode_params)
    
    # Store the effective mode used
    full_config['effective_mode'] = mode
    full_config['configured_mode'] = configured_mode
    full_config['mode_overridden'] = override_mode is not None
    
    return full_config

def detect_and_override_mode(model_name: str, sample_response: str = None):
    """Detect model behavior dynamically and return appropriate mode override
    
    Args:
        model_name: Name of the model to analyze
        sample_response: Optional sample response to analyze immediately
        
    Returns:
        str or None: 'thinking' or 'non-thinking' if detected, None if no override needed
    """
    try:
        from app.llm.response_analyzer import get_model_behavior_profile, detect_model_thinking_behavior
        
        # Check if we have a cached profile first
        profile = get_model_behavior_profile(model_name)
        if profile and profile.confidence > 0.7:
            return 'thinking' if profile.is_thinking_model else 'non-thinking'
        
        # If we have a sample response, analyze it immediately
        if sample_response:
            is_thinking, confidence = detect_model_thinking_behavior(sample_response, model_name)
            if confidence > 0.6:
                return 'thinking' if is_thinking else 'non-thinking'
        
        # No confident detection available
        return None
        
    except ImportError:
        # Response analyzer not available, use configured mode
        return None
    except Exception as e:
        print(f"Error in dynamic mode detection: {e}")
        return None

def get_notebook_llm_full_config_with_detection(settings=None, sample_response=None):
    """Get notebook LLM config with dynamic mode detection
    
    This is a convenience function that combines configuration loading with
    automatic behavior detection.
    
    Args:
        settings: Notebook LLM settings dictionary
        sample_response: Sample response to analyze for behavior detection
        
    Returns:
        dict: Full notebook LLM configuration with dynamically detected mode
    """
    if settings is None:
        settings = get_notebook_llm_settings()
    
    notebook_llm = settings.get('notebook_llm', {})
    model_name = notebook_llm.get('model', 'unknown')
    
    # Attempt dynamic detection
    detected_mode = detect_and_override_mode(model_name, sample_response)
    
    # Get config with potential override
    return get_notebook_llm_full_config(settings, detected_mode)

def _validate_notebook_llm_settings(settings):
    """Validate that notebook LLM settings contain required keys to prevent runtime failures"""
    if not isinstance(settings, dict):
        return False
    
    required_keys = ['notebook_llm']
    for key in required_keys:
        if key not in settings or not isinstance(settings[key], dict):
            print(f"[NOTEBOOK_LLM_SETTINGS] Missing or invalid key: {key}")
            return False
        
        # Check that each LLM config has a model
        if 'model' not in settings[key]:
            print(f"[NOTEBOOK_LLM_SETTINGS] Missing model in {key}")
            return False
    
    return True

def _get_emergency_fallback_settings():
    """
    Emergency fallback settings to prevent total system failure.
    Uses safe, known-working configuration for notebook LLM.
    """
    print("[NOTEBOOK_LLM_SETTINGS] EMERGENCY FALLBACK ACTIVATED - Using minimal safe configuration")
    
    # Use the default model for notebook interactions
    working_model = "qwen2.5:72b"
    
    import os
    # Get model server from environment or use empty string (no hardcoded defaults)
    model_server = os.environ.get("OLLAMA_BASE_URL", "")
    
    if not model_server:
        # Log critical error but don't hardcode a fallback
        print("[NOTEBOOK_LLM_SETTINGS] CRITICAL: No model server configured in environment (OLLAMA_BASE_URL)")
        print("[NOTEBOOK_LLM_SETTINGS] System will not function without proper model server configuration")
    
    emergency_settings = {
        "notebook_llm": {
            "mode": "thinking",
            "model": working_model,
            "max_tokens": 8192,
            "model_server": model_server,  # Use environment variable only
            "system_prompt": "You are a helpful assistant for Jupyter notebook interactions. Provide clear, concise responses that are suitable for notebook environments. Focus on code execution, data analysis, and technical explanations.",
            "context_length": 131072,
            "repeat_penalty": 1.1,
            "temperature": 0.7,
            "top_p": 0.9
        },
        # Mode parameters
        "thinking_mode_params": {
            "min_p": 0,
            "top_k": 20,
            "top_p": 0.95,
            "temperature": 0.6
        },
        "non_thinking_mode_params": {
            "min_p": 0,
            "top_k": 20,
            "top_p": 0.8,
            "temperature": 0.7
        },
        # Mark as emergency fallback
        "_fallback_mode": "emergency",
        "_created_at": json.dumps({"timestamp": "emergency_fallback", "reason": "configuration_retrieval_failure"})
    }
    
    # Try to cache the emergency settings for future use
    try:
        redis_client = _get_redis_client()
        if redis_client:
            redis_client.set(f"{NOTEBOOK_LLM_SETTINGS_KEY}_emergency", json.dumps(emergency_settings), ex=3600)  # 1 hour TTL
            print("[NOTEBOOK_LLM_SETTINGS] Emergency settings cached for future fallbacks")
    except Exception as e:
        print(f"[NOTEBOOK_LLM_SETTINGS] Could not cache emergency settings: {e}")
    
    return emergency_settings