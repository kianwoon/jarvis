"""
Timeout Settings Cache
Manages timeout configuration with Redis caching and database persistence
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
CACHE_KEY = "timeout_settings"

# Don't create connection at import time
r = None

# Default timeout settings for all system components
DEFAULT_TIMEOUT_SETTINGS = {
    # API & Network timeouts
    "api_network": {
        "http_request_timeout": 30,
        "http_streaming_timeout": 120,
        "http_upload_timeout": 180,
        "database_connection_timeout": 10,
        "database_query_timeout": 60,
        "redis_operation_timeout": 5,
        "redis_connection_timeout": 10
    },
    
    # LLM & AI processing timeouts
    "llm_ai": {
        "llm_inference_timeout": 60,
        "llm_streaming_timeout": 120,
        "query_classification_timeout": 10,
        "multi_agent_timeout": 120,
        "agent_processing_timeout": 90,
        "agent_coordination_timeout": 60,
        "thinking_mode_timeout": 180
    },
    
    # Document processing & RAG timeouts
    "document_processing": {
        "rag_retrieval_timeout": 30,
        "rag_processing_timeout": 60,
        "vector_search_timeout": 15,
        "embedding_generation_timeout": 60,
        "document_processing_timeout": 120,
        "collection_search_timeout": 45,
        "bm25_processing_timeout": 20
    },
    
    # MCP tools & integrations timeouts
    "mcp_tools": {
        "tool_execution_timeout": 30,
        "tool_initialization_timeout": 15,
        "manifest_fetch_timeout": 10,
        "server_communication_timeout": 15,
        "server_startup_timeout": 30,
        "stdio_bridge_timeout": 25
    },
    
    # Workflow & automation timeouts
    "workflow_automation": {
        "workflow_execution_timeout": 300,
        "workflow_step_timeout": 180,
        "task_timeout": 180,
        "agent_task_timeout": 120,
        "large_generation_timeout": 600,
        "chunk_generation_timeout": 90
    },
    
    # Knowledge graph processing timeouts
    "knowledge_graph": {
        "base_extraction_timeout": 360,  # Doubled from 180 as immediate safety net
        "max_extraction_timeout": 900,
        "pass_timeout_multiplier": 1.5,
        "large_document_threshold": 20000,
        "ultra_large_document_threshold": 50000,
        "content_size_multiplier": 0.01,
        "complexity_multiplier": 1.2,
        "fallback_timeout": 360,
        "api_call_timeout": 360,  # Doubled from 180 as immediate safety net
        "multi_pass_timeout": 600
    },
    
    # Session & cache timeouts (in seconds)
    "session_cache": {
        "redis_ttl_seconds": 3600,
        "conversation_cache_ttl": 86400,
        "result_cache_ttl": 7200,
        "temp_data_ttl": 1800,
        "session_cleanup_interval": 3600,
        "cache_cleanup_interval": 7200
    },
    
    # Redis Cache TTL Settings (in seconds)
    "redis_cache_ttl": {
        "settings_cache_ttl": 3600,          # Settings cache TTL (1 hour)
        "pipeline_cache_ttl": 3600,          # Pipeline state cache TTL (1 hour)
        "list_cache_ttl": 300,               # List cache TTL (5 minutes)
        "agent_response_cache_ttl": 600,     # Agent response cache TTL (10 minutes)
        "collection_registry_ttl": 300,      # Collection registry TTL (5 minutes)
        "conversation_cache_ttl": 86400,     # Conversation cache TTL (24 hours)
        "temporary_data_ttl": 1800,          # Temporary data TTL (30 minutes)
        "idc_cache_ttl": 3600,               # IDC extraction cache TTL (1 hour)
        "validation_cache_ttl": 7200,        # Validation cache TTL (2 hours)
        "knowledge_graph_cache_ttl": 1800,   # Knowledge graph cache TTL (30 minutes)
        "rag_cache_ttl": 3600,               # RAG cache TTL (1 hour)
        "embedding_cache_ttl": 7200,         # Embedding cache TTL (2 hours)
        "vector_search_cache_ttl": 1800,     # Vector search cache TTL (30 minutes)
        "workflow_state_ttl": 3600,          # Workflow state TTL (1 hour)
        "mcp_tool_cache_ttl": 600            # MCP tool cache TTL (10 minutes)
    }
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

def get_timeout_settings() -> Dict:
    """Get timeout settings from cache or database"""
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
    return reload_timeout_settings()

def set_timeout_settings(settings_dict):
    """Cache timeout settings in Redis"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(CACHE_KEY, json.dumps(settings_dict))
        except Exception as e:
            logger.warning(f"Failed to cache timeout settings in Redis: {e}")

def reload_timeout_settings():
    """Reload timeout settings from database and cache in Redis"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'timeout').first()
            if row and row.settings:
                settings = row.settings
                
                # Merge with defaults to ensure all fields are present
                merged_settings = _merge_with_defaults(settings)
                
                # If new fields were added, save back to database
                if merged_settings != settings:
                    row.settings = merged_settings
                    db.commit()
                    logger.info("Updated timeout settings with new default fields")
                    settings = merged_settings
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(settings))
                    except Exception as e:
                        logger.warning(f"Failed to cache timeout settings in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, initialize with defaults and save to DB
                if row:
                    # Update existing row
                    row.settings = DEFAULT_TIMEOUT_SETTINGS
                    db.commit()
                    logger.info("Initialized timeout settings with defaults (updated existing row)")
                else:
                    # Create new row
                    new_row = SettingsModel(
                        category='timeout', 
                        settings=DEFAULT_TIMEOUT_SETTINGS
                    )
                    db.add(new_row)
                    db.commit()
                    logger.info("Created new timeout settings with defaults")
                
                # Cache the defaults
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(DEFAULT_TIMEOUT_SETTINGS))
                    except Exception as e:
                        logger.warning(f"Failed to cache default timeout settings in Redis: {e}")
                
                return DEFAULT_TIMEOUT_SETTINGS
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to load timeout settings from database: {e}")
        # Return default settings to prevent complete failure
        return DEFAULT_TIMEOUT_SETTINGS

def _merge_with_defaults(user_settings: Dict) -> Dict:
    """Merge user settings with defaults to ensure all fields are present"""
    merged = DEFAULT_TIMEOUT_SETTINGS.copy()
    
    for category, category_settings in user_settings.items():
        if category in merged:
            # Update existing category with user values
            merged[category].update(category_settings)
        else:
            # Add completely new category
            merged[category] = category_settings
    
    return merged

def validate_timeout_settings(settings: Dict) -> Dict:
    """Validate and sanitize timeout settings"""
    validated = {}
    
    for category, category_settings in settings.items():
        validated[category] = {}
        
        for setting_key, value in category_settings.items():
            try:
                # Convert to int and validate range
                timeout_value = int(value)
                
                # Set reasonable bounds based on category
                if category == "session_cache" or category == "redis_cache_ttl":
                    # Cache timeouts can be longer (1 second to 7 days)
                    min_val, max_val = 1, 604800
                elif category == "workflow_automation":
                    # Workflow timeouts can be very long (1 second to 30 minutes)
                    min_val, max_val = 1, 1800
                else:
                    # Standard timeouts (1 second to 10 minutes)
                    min_val, max_val = 1, 600
                
                validated[category][setting_key] = max(min_val, min(max_val, timeout_value))
                
                # Log if value was clamped
                if timeout_value != validated[category][setting_key]:
                    logger.warning(f"Timeout value for {category}.{setting_key} was clamped from {timeout_value} to {validated[category][setting_key]}")
                    
            except (ValueError, TypeError) as e:
                # Use default value if invalid
                default_value = DEFAULT_TIMEOUT_SETTINGS.get(category, {}).get(setting_key, 30)
                validated[category][setting_key] = default_value
                logger.warning(f"Invalid timeout value for {category}.{setting_key}: {value}, using default: {default_value}")
    
    return validated

def update_timeout_settings(settings: Dict) -> bool:
    """Update timeout settings in database and cache"""
    try:
        # Validate settings first
        validated_settings = validate_timeout_settings(settings)
        
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'timeout').first()
            if row:
                row.settings = validated_settings
                db.commit()
                
                # Update Redis cache
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(validated_settings))
                    except Exception as e:
                        logger.warning(f"Failed to update Redis cache: {e}")
                
                logger.info("Timeout settings updated successfully")
                return True
            else:
                # Create new row if it doesn't exist
                new_row = SettingsModel(
                    category='timeout',
                    settings=validated_settings
                )
                db.add(new_row)
                db.commit()
                
                # Update Redis cache
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(CACHE_KEY, json.dumps(validated_settings))
                    except Exception as e:
                        logger.warning(f"Failed to update Redis cache: {e}")
                
                logger.info("Timeout settings created successfully")
                return True
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to update timeout settings: {e}")
        return False

def clear_cache():
    """Clear the timeout settings cache"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.delete(CACHE_KEY)
            logger.info("Timeout settings cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear timeout cache: {e}")

def get_category_timeouts(category: str) -> Dict:
    """Get timeout settings for a specific category"""
    all_settings = get_timeout_settings()
    return all_settings.get(category, {})

def get_timeout_value(category: str, setting_key: str, default: int = 30) -> int:
    """Get a specific timeout value with fallback to default"""
    category_settings = get_category_timeouts(category)
    return category_settings.get(setting_key, default)

# Convenience functions for commonly used timeouts
def get_http_timeout() -> int:
    """Get HTTP request timeout"""
    return get_timeout_value("api_network", "http_request_timeout", 30)

def get_llm_timeout() -> int:
    """Get LLM inference timeout"""
    return get_timeout_value("llm_ai", "llm_inference_timeout", 60)

def get_query_classification_timeout() -> int:
    """Get query classification timeout"""
    return get_timeout_value("llm_ai", "query_classification_timeout", 10)

def get_rag_timeout() -> int:
    """Get RAG retrieval timeout"""
    return get_timeout_value("document_processing", "rag_retrieval_timeout", 30)

def get_tool_execution_timeout() -> int:
    """Get MCP tool execution timeout"""
    return get_timeout_value("mcp_tools", "tool_execution_timeout", 30)

def get_agent_timeout() -> int:
    """Get agent processing timeout"""
    return get_timeout_value("llm_ai", "agent_processing_timeout", 90)

def get_redis_timeout() -> int:
    """Get Redis operation timeout"""
    return get_timeout_value("api_network", "redis_operation_timeout", 5)

def get_knowledge_graph_timeout(setting_key: str, default: int = 180) -> int:
    """Get knowledge graph specific timeout"""
    return get_timeout_value("knowledge_graph", setting_key, default)

def get_kg_base_timeout() -> int:
    """Get base knowledge graph extraction timeout"""
    return get_knowledge_graph_timeout("base_extraction_timeout", 180)

def get_kg_max_timeout() -> int:
    """Get maximum knowledge graph extraction timeout"""
    return get_knowledge_graph_timeout("max_extraction_timeout", 900)

def get_kg_fallback_timeout() -> int:
    """Get knowledge graph fallback timeout"""
    return get_knowledge_graph_timeout("fallback_timeout", 360)

# Redis Cache TTL convenience functions
def get_redis_cache_ttl(setting_key: str, default: int = 3600) -> int:
    """Get Redis cache TTL setting"""
    return get_timeout_value("redis_cache_ttl", setting_key, default)

def get_settings_cache_ttl() -> int:
    """Get settings cache TTL"""
    return get_redis_cache_ttl("settings_cache_ttl", 3600)

def get_pipeline_cache_ttl() -> int:
    """Get pipeline cache TTL"""
    return get_redis_cache_ttl("pipeline_cache_ttl", 3600)

def get_list_cache_ttl() -> int:
    """Get list cache TTL"""
    return get_redis_cache_ttl("list_cache_ttl", 300)

def get_agent_response_cache_ttl() -> int:
    """Get agent response cache TTL"""
    return get_redis_cache_ttl("agent_response_cache_ttl", 600)

def get_collection_registry_ttl() -> int:
    """Get collection registry TTL"""
    return get_redis_cache_ttl("collection_registry_ttl", 300)

def get_conversation_cache_ttl() -> int:
    """Get conversation cache TTL"""
    return get_redis_cache_ttl("conversation_cache_ttl", 86400)

def get_temporary_data_ttl() -> int:
    """Get temporary data TTL"""
    return get_redis_cache_ttl("temporary_data_ttl", 1800)

def get_mcp_tool_cache_ttl() -> int:
    """Get MCP tool cache TTL"""
    return get_redis_cache_ttl("mcp_tool_cache_ttl", 600)