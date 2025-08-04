import json
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
LLM_SETTINGS_KEY = 'llm_settings_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
    return r

def get_llm_settings():
    """
    Get LLM settings with comprehensive failsafe system to prevent cascading failures.
    
    Failsafe hierarchy:
    1. Redis cache (fastest)
    2. Database reload (reliable)
    3. Emergency fallback settings (prevents total failure)
    """
    # Layer 1: Try Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(LLM_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                # Validate settings contain required keys
                if _validate_llm_settings(settings):
                    return settings
                else:
                    print(f"[LLM_SETTINGS] Cached settings invalid, falling back to database")
        except Exception as e:
            print(f"[LLM_SETTINGS] Redis error: {e}, falling back to database")
    
    # Layer 2: Try database reload
    try:
        return reload_llm_settings()
    except Exception as e:
        print(f"[LLM_SETTINGS] Database reload failed: {e}, using emergency fallback")
        
        # Layer 3: Emergency fallback to prevent total system failure
        return _get_emergency_fallback_settings()

def set_llm_settings(settings_dict):
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(LLM_SETTINGS_KEY, json.dumps(settings_dict))
        except Exception as e:
            print(f"Failed to cache settings in Redis: {e}")

# Call this after updating settings in DB
def reload_llm_settings():
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row:
                settings = row.settings
                
                # Merge in latest query classifier settings with new LLM fields
                try:
                    from app.core.query_classifier_settings_cache import get_query_classifier_settings
                    latest_query_classifier = get_query_classifier_settings()
                    settings['query_classifier'] = latest_query_classifier
                except Exception as e:
                    print(f"Warning: Failed to merge query classifier settings: {e}")
                
                # Merge in search optimization settings with defaults
                try:
                    search_config = get_search_optimization_config(settings)
                    settings['search_optimization'] = search_config
                    
                    # Save updated settings back to database if new search optimization was added
                    row.settings = settings
                    db.commit()
                    print("Updated LLM settings with search optimization defaults")
                except Exception as e:
                    print(f"Warning: Failed to merge search optimization settings: {e}")
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(LLM_SETTINGS_KEY, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache settings in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, raise error
                raise RuntimeError('No LLM settings found in database')
        finally:
            db.close()
    except Exception as e:
        print(f"[LLM_SETTINGS] Failed to load LLM settings from database: {e}")
        print(f"[LLM_SETTINGS] This is a critical failure - check database connectivity and settings table")
        raise RuntimeError(f"Database configuration retrieval failed: {e}")

def get_main_llm_full_config(settings=None, override_mode=None):
    """Construct full main_llm configuration by merging base config with mode parameters
    
    Args:
        settings: LLM settings dictionary
        override_mode: Override mode detected dynamically ('thinking' or 'non-thinking')
    """
    try:
        if settings is None:
            settings = get_llm_settings()
    except Exception as e:
        print(f"[MAIN_LLM_CONFIG] Failed to get settings: {e}, using emergency fallback")
        settings = _get_emergency_fallback_settings()
    
    main_llm = settings.get('main_llm', {})
    configured_mode = main_llm.get('mode', 'thinking')
    
    # Use override mode if provided (from dynamic detection)
    mode = override_mode if override_mode else configured_mode
    
    # Get the appropriate mode parameters
    if mode == 'thinking':
        mode_params = settings.get('thinking_mode_params', {})
    else:
        mode_params = settings.get('non_thinking_mode_params', {})
    
    # Merge base config with mode parameters
    full_config = main_llm.copy()
    full_config.update(mode_params)
    
    # Store the effective mode used
    full_config['effective_mode'] = mode
    full_config['configured_mode'] = configured_mode
    full_config['mode_overridden'] = override_mode is not None
    
    return full_config

def get_knowledge_graph_full_config(settings=None):
    """Construct full knowledge_graph configuration by merging base config with mode parameters"""
    if settings is None:
        settings = get_llm_settings()
    
    knowledge_graph = settings.get('knowledge_graph', {})
    mode = knowledge_graph.get('mode', 'thinking')
    
    # Get the appropriate mode parameters
    if mode == 'thinking':
        mode_params = settings.get('thinking_mode_params', {})
    else:
        mode_params = settings.get('non_thinking_mode_params', {})
    
    # Merge base config with mode parameters
    full_config = knowledge_graph.copy()
    full_config.update(mode_params)
    
    return full_config

def get_query_classifier_full_config(settings=None, override_mode=None):
    """Construct full query_classifier configuration by merging base config with mode parameters
    
    Args:
        settings: LLM settings dictionary
        override_mode: Override mode detected dynamically ('thinking' or 'non-thinking')
    """
    try:
        if settings is None:
            settings = get_llm_settings()
    except Exception as e:
        print(f"[QUERY_CLASSIFIER_CONFIG] Failed to get settings: {e}, using emergency fallback")
        settings = _get_emergency_fallback_settings()
    
    query_classifier = settings.get('query_classifier', {})
    # Query classifier should default to non-thinking mode for simple classification responses
    configured_mode = query_classifier.get('mode', 'non-thinking')
    
    # Use override mode if provided (from dynamic detection)
    mode = override_mode if override_mode else configured_mode
    
    # Get the appropriate mode parameters
    if mode == 'thinking':
        mode_params = settings.get('thinking_mode_params', {})
    else:
        mode_params = settings.get('non_thinking_mode_params', {})
    
    # Merge base config with mode parameters
    full_config = query_classifier.copy()
    full_config.update(mode_params)
    
    # Store the effective mode used
    full_config['effective_mode'] = mode
    full_config['configured_mode'] = configured_mode
    full_config['mode_overridden'] = override_mode is not None
    
    return full_config

def get_second_llm_full_config(settings=None, override_mode=None):
    """Construct full second_llm configuration by merging base config with mode parameters
    
    Args:
        settings: LLM settings dictionary
        override_mode: Override mode detected dynamically ('thinking' or 'non-thinking')
    """
    try:
        if settings is None:
            settings = get_llm_settings()
    except Exception as e:
        print(f"[SECOND_LLM_CONFIG] Failed to get settings: {e}, using emergency fallback")
        settings = _get_emergency_fallback_settings()
    
    second_llm = settings.get('second_llm', {})
    configured_mode = second_llm.get('mode', 'thinking')
    
    # Use override mode if provided (from dynamic detection)
    mode = override_mode if override_mode else configured_mode
    
    # Get the appropriate mode parameters
    if mode == 'thinking':
        mode_params = settings.get('thinking_mode_params', {})
    else:
        mode_params = settings.get('non_thinking_mode_params', {})
    
    # Merge base config with mode parameters
    full_config = second_llm.copy()
    full_config.update(mode_params)
    
    # Store the effective mode used
    full_config['effective_mode'] = mode
    full_config['configured_mode'] = configured_mode
    full_config['mode_overridden'] = override_mode is not None
    
    return full_config

def get_search_optimization_config(settings=None):
    """Get search query optimization configuration with defaults"""
    if settings is None:
        settings = get_llm_settings()
    
    # Default search optimization settings
    default_config = {
        'enable_search_optimization': True,
        'optimization_timeout': 12,
        'optimization_prompt': '''Transform the user's conversational question into an optimized search query for better results.

Current context: It's 2025, so add year context when asking about recent/current information.

User Question: {query}

## Optimization Guidelines:
1. Remove conversational words (please, can you, I want to know, etc.)
2. Add temporal context when needed (2025, current, latest, recent)
3. Use specific keywords instead of general terms
4. Keep the core intent and meaning intact
5. Make it concise but comprehensive
6. Add relevant context for better search results

## Examples:
- "Can you tell me about the latest AI developments?" → "latest AI developments 2025 current trends"
- "What's the weather like today?" → "weather today current conditions"
- "I want to know about Python vs JavaScript" → "Python vs JavaScript comparison 2025"

Return ONLY the optimized search query, no explanations.'''
    }
    
    # Get user-defined settings or defaults
    search_config = settings.get('search_optimization', default_config)
    
    # Merge with defaults to ensure all fields are present
    merged_config = default_config.copy()
    merged_config.update(search_config)
    
    return merged_config

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

def get_main_llm_full_config_with_detection(settings=None, sample_response=None):
    """Get LLM config with dynamic mode detection
    
    This is a convenience function that combines configuration loading with
    automatic behavior detection.
    
    Args:
        settings: LLM settings dictionary
        sample_response: Sample response to analyze for behavior detection
        
    Returns:
        dict: Full LLM configuration with dynamically detected mode
    """
    if settings is None:
        settings = get_llm_settings()
    
    main_llm = settings.get('main_llm', {})
    model_name = main_llm.get('model', 'unknown')
    
    # Attempt dynamic detection
    detected_mode = detect_and_override_mode(model_name, sample_response)
    
    # Get config with potential override
    return get_main_llm_full_config(settings, detected_mode)

def get_query_classifier_full_config_with_detection(settings=None, sample_response=None):
    """Get query classifier config with dynamic mode detection
    
    This function combines query classifier configuration loading with
    automatic behavior detection for optimal classification performance.
    
    Args:
        settings: LLM settings dictionary
        sample_response: Sample response to analyze for behavior detection
        
    Returns:
        dict: Full query classifier configuration with dynamically detected mode
    """
    if settings is None:
        settings = get_llm_settings()
    
    query_classifier = settings.get('query_classifier', {})
    model_name = query_classifier.get('model', 'unknown')
    
    # Attempt dynamic detection
    detected_mode = detect_and_override_mode(model_name, sample_response)
    
    # Get config with potential override
    return get_query_classifier_full_config(settings, detected_mode)

def get_second_llm_full_config_with_detection(settings=None, sample_response=None):
    """Get second_llm config with dynamic mode detection
    
    This function combines second_llm configuration loading with
    automatic behavior detection for multi-agent and task processing.
    
    Args:
        settings: LLM settings dictionary
        sample_response: Sample response to analyze for behavior detection
        
    Returns:
        dict: Full second_llm configuration with dynamically detected mode
    """
    if settings is None:
        settings = get_llm_settings()
    
    second_llm = settings.get('second_llm', {})
    model_name = second_llm.get('model', 'unknown')
    
    # Attempt dynamic detection
    detected_mode = detect_and_override_mode(model_name, sample_response)
    
    # Get config with potential override
    return get_second_llm_full_config(settings, detected_mode)

def _validate_llm_settings(settings):
    """Validate that LLM settings contain required keys to prevent runtime failures"""
    if not isinstance(settings, dict):
        return False
    
    required_keys = ['main_llm', 'second_llm', 'query_classifier']
    for key in required_keys:
        if key not in settings or not isinstance(settings[key], dict):
            print(f"[LLM_SETTINGS] Missing or invalid key: {key}")
            return False
        
        # Check that each LLM config has a model
        if 'model' not in settings[key]:
            print(f"[LLM_SETTINGS] Missing model in {key}")
            return False
    
    return True

def _get_emergency_fallback_settings():
    """
    Emergency fallback settings to prevent total system failure.
    Uses safe, known-working configuration.
    """
    print("[LLM_SETTINGS] EMERGENCY FALLBACK ACTIVATED - Using minimal safe configuration")
    
    # Use the working model that we know generates proper responses
    working_model = "qwen3:30b-a3b-q4_K_M"
    
    import os
    emergency_settings = {
        "main_llm": {
            "mode": "thinking",
            "model": working_model,  # Use the working model
            "max_tokens": 196608,
            "model_server": os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            "system_prompt": "You are Jarvis, an AI assistant. Always provide detailed, comprehensive responses with thorough explanations, examples, and step-by-step breakdowns when appropriate. Be verbose and informative.",
            "context_length": 262144,
            "repeat_penalty": 1.1
        },
        "second_llm": {
            "mode": "thinking", 
            "model": working_model,  # Use the working model
            "max_tokens": 16384,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "query_classifier": {
            "mode": "thinking",
            "model": working_model,  # Use the working model
            "max_tokens": 2048,
            "temperature": 0.3,
            "top_p": 0.8
        },
        "knowledge_graph": {
            "mode": "thinking",
            "model": working_model,
            "max_tokens": 8192,
            "temperature": 0.6,
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
        # Search optimization with safe defaults
        "search_optimization": {
            "hybrid_search_enabled": True,
            "vector_weight": 0.7,
            "bm25_weight": 0.3,
            "result_fusion_method": "rrf",
            "reranking_enabled": False,
            "max_results": 20
        },
        # Mark as emergency fallback
        "_fallback_mode": "emergency",
        "_created_at": json.dumps({"timestamp": "emergency_fallback", "reason": "configuration_retrieval_failure"})
    }
    
    # Try to cache the emergency settings for future use
    try:
        redis_client = _get_redis_client()
        if redis_client:
            redis_client.set(f"{LLM_SETTINGS_KEY}_emergency", json.dumps(emergency_settings), ex=3600)  # 1 hour TTL
            print("[LLM_SETTINGS] Emergency settings cached for future fallbacks")
    except Exception as e:
        print(f"[LLM_SETTINGS] Could not cache emergency settings: {e}")
    
    return emergency_settings 