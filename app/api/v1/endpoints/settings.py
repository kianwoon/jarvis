from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, Settings as SettingsModel, MCPManifest, MCPTool, Base, get_db
from app.core.llm_settings_cache import reload_llm_settings
from app.core.vector_db_settings_cache import reload_vector_db_settings
from app.core.embedding_settings_cache import reload_embedding_settings
from app.core.iceberg_settings_cache import reload_iceberg_settings
from app.core.mcp_tools_cache import reload_enabled_mcp_tools
from app.core.large_generation_settings_cache import reload_large_generation_settings, validate_large_generation_config, merge_with_defaults
from app.core.rag_settings_cache import reload_rag_settings
from app.core.query_classifier_settings_cache import reload_query_classifier_settings
from app.core.timeout_settings_cache import reload_timeout_settings
from app.core.knowledge_graph_settings_cache import reload_knowledge_graph_settings
from app.core.radiating_settings_cache import reload_radiating_settings
from app.core.meta_task_settings_cache import reload_meta_task_settings
from app.core.synthesis_prompts_cache import reload_synthesis_prompts
from app.services.neo4j_service import test_neo4j_connection
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
import requests
import json
import logging
from sqlalchemy import inspect
from sqlalchemy.exc import ProgrammingError, DatabaseError
from sqlalchemy.sql import text

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class CacheReloadRequest(BaseModel):
    settings: Optional[Dict[str, Any]] = None

def deep_merge_settings(existing_settings: Dict[str, Any], new_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge new settings into existing settings, preserving all existing fields
    and only updating the fields that are explicitly provided in new_settings.
    """
    if not existing_settings:
        return new_settings.copy()
    
    merged = existing_settings.copy()
    
    for key, value in new_settings.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = deep_merge_settings(merged[key], value)
        else:
            # Override with new value
            merged[key] = value
    
    return merged

def _flush_redis_cache():
    """Flush all settings-related Redis cache entries"""
    try:
        from app.core.redis_client import get_redis_client
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis client not available - cache flush skipped")
            return False, "Redis client not available"
        
        # Comprehensive list of cache keys used across the system
        cache_keys = [
            # Core settings caches
            'llm_settings_cache',
            'query_classifier_settings',
            'vector_db_settings_cache',
            'embedding_settings_cache',
            'rag_settings_cache',
            'knowledge_graph_settings_cache',
            'radiating_settings_cache',
            'timeout_settings_cache',
            'overflow_settings_cache',
            'meta_task_settings_cache',
            'iceberg_settings_cache',
            'langfuse_settings_cache',
            'idc_settings_cache',
            
            # MCP related caches
            'mcp_tools_cache',
            'mcp_manifests_cache',
            'mcp_server_cache',
            
            # Prompt and synthesis caches
            'synthesis_prompts_cache',
            'formatting_templates_cache',
            'system_behaviors_cache',
            
            # Automation caches
            'automation_workflows_cache',
            'automation_executions_cache',
            
            # Agent and performance caches
            'langgraph_agents_cache',
            'collection_registry_cache',
            'enterprise_llm_cache',
            
            # Emergency fallback cache
            'llm_settings_cache_emergency'
        ]
        
        # Delete all cache keys
        deleted_keys = []
        for key in cache_keys:
            try:
                result = redis_client.delete(key)
                if result > 0:
                    deleted_keys.append(key)
            except Exception as e:
                logger.warning(f"Failed to delete cache key {key}: {e}")
        
        logger.info(f"Redis cache flush completed - deleted {len(deleted_keys)} keys: {deleted_keys}")
        return True, f"Cleared {len(deleted_keys)} cache entries"
        
    except Exception as e:
        logger.error(f"Failed to flush Redis cache: {str(e)}")
        return False, f"Cache flush failed: {str(e)}"

@router.post("/llm/cache/reload")
def reload_llm_cache(request: Optional[CacheReloadRequest] = Body(None), db: Session = Depends(get_db)):
    """Force reload LLM settings cache from database and clear Redis cache
    
    If settings are provided in the request body, they will be saved to the database first
    before reloading the cache, ensuring the cache contains the latest form data.
    """
    try:
        settings_saved = False
        
        # Debug: Log the received request
        logger.info(f"Reload cache request received: {request}")
        if request:
            logger.info(f"Request settings: {request.settings}")
        
        # If settings are provided, save them to database first
        if request and request.settings:
            logger.info("Saving LLM settings to database before cache reload")
            logger.info(f"Settings to save: {request.settings}")
            
            # Get existing settings for deep merge
            settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            existing_settings = settings_row.settings if settings_row else {}
            
            # Deep merge to preserve existing complex fields
            merged_settings = deep_merge_settings(existing_settings, request.settings)
            logger.info(f"Merged LLM settings keys: {list(merged_settings.keys())}")
            
            # Validate that critical fields are preserved
            critical_fields = ['main_llm', 'second_llm', 'query_classifier', 'search_optimization', 'thinking_mode_params', 'non_thinking_mode_params']
            preserved_fields = [field for field in critical_fields if field in existing_settings and field in merged_settings]
            if preserved_fields:
                logger.info(f"Preserved critical LLM fields: {preserved_fields}")
            
            # Update or create settings row
            if settings_row:
                settings_row.settings = merged_settings
            else:
                logger.info("Creating new LLM settings")
                settings_row = SettingsModel(category='llm', settings=merged_settings)
                db.add(settings_row)
            
            # Commit the changes
            db.commit()
            db.refresh(settings_row)
            logger.info("LLM settings saved with deep merge")
            settings_saved = True
        else:
            if not request:
                logger.info("No request body provided, proceeding with cache reload only")
            elif not request.settings:
                logger.info("Request provided but no settings data, proceeding with cache reload only")
            else:
                logger.info("Unknown case in request handling")
        
        # First, flush Redis cache to ensure fresh data
        cache_flushed, flush_message = _flush_redis_cache()
        
        # Reload LLM settings from database
        settings = reload_llm_settings()
        
        # Also reload query classifier settings to ensure consistency
        reload_query_classifier_settings()
        
        return {
            "success": True, 
            "message": "LLM cache reloaded successfully" + (" (settings saved first)" if settings_saved else ""),
            "model": settings.get("main_llm", {}).get("model", "unknown"),
            "cache_size": len(str(settings)),
            "redis_cache_flushed": cache_flushed,
            "flush_details": flush_message,
            "settings_saved": settings_saved
        }
    except Exception as e:
        logger.error(f"Failed to reload LLM cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload LLM cache: {str(e)}")

@router.post("/query-classifier/cache/reload")
def reload_query_classifier_cache():
    """Force reload Query Classifier settings cache from database and initialize with defaults if needed"""
    try:
        settings = reload_query_classifier_settings()
        return {
            "success": True, 
            "message": "Query Classifier cache reloaded successfully",
            "settings": settings,
            "cache_size": len(str(settings))
        }
    except Exception as e:
        logger.error(f"Failed to reload Query Classifier cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload Query Classifier cache: {str(e)}")

@router.post("/timeout/cache/reload")
def reload_timeout_cache():
    """Force reload timeout settings cache from database and initialize with defaults if needed"""
    try:
        settings = reload_timeout_settings()
        return {
            "success": True, 
            "message": "Timeout settings cache reloaded successfully",
            "settings": settings,
            "cache_size": len(str(settings))
        }
    except Exception as e:
        logger.error(f"Failed to reload timeout cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload timeout cache: {str(e)}")

@router.post("/knowledge-graph/cache/reload")
def reload_knowledge_graph_cache():
    """Force reload knowledge graph settings cache from database"""
    try:
        settings = reload_knowledge_graph_settings()
        return {
            "success": True, 
            "message": "Knowledge graph cache reloaded successfully",
            "settings": settings,
            "cache_size": len(str(settings))
        }
    except Exception as e:
        logger.error(f"Failed to reload knowledge graph cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload knowledge graph cache: {str(e)}")

@router.post("/radiating/cache/reload")
def reload_radiating_cache():
    """Force reload radiating settings cache from database"""
    try:
        settings = reload_radiating_settings()
        return {
            "success": True, 
            "message": "Radiating cache reloaded successfully",
            "settings": settings,
            "cache_size": len(str(settings))
        }
    except Exception as e:
        logger.error(f"Failed to reload radiating cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload radiating cache: {str(e)}")

@router.post("/overflow/cache/reload")
def reload_overflow_cache():
    """Force reload overflow settings cache from database"""
    try:
        from app.core.overflow_settings_cache import reload_overflow_settings
        settings = reload_overflow_settings()
        return {
            "success": True,
            "message": "Overflow cache reloaded successfully",
            "settings": settings,
            "cache_size": len(str(settings))
        }
    except Exception as e:
        logger.error(f"Failed to reload overflow cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload overflow cache: {str(e)}")

@router.post("/meta-task/cache/reload")
def reload_meta_task_cache():
    """Force reload meta-task settings cache from database"""
    try:
        settings = reload_meta_task_settings()
        return {
            "success": True,
            "message": "Meta-task cache reloaded successfully",
            "settings": settings,
            "cache_size": len(str(settings))
        }
    except Exception as e:
        logger.error(f"Failed to reload meta-task cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload meta-task cache: {str(e)}")

@router.post("/synthesis-prompts/cache/reload")
def reload_synthesis_prompts_cache():
    """Force reload synthesis prompts cache from database"""
    try:
        settings = reload_synthesis_prompts()
        return {
            "success": True,
            "message": "Synthesis prompts cache reloaded successfully",
            "settings": settings,
            "cache_size": len(str(settings))
        }
    except Exception as e:
        logger.error(f"Failed to reload synthesis prompts cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload synthesis prompts cache: {str(e)}")

@router.post("/cache/flush")
def flush_redis_cache():
    """Flush all Redis cache entries - comprehensive cache clear"""
    try:
        cache_flushed, flush_message = _flush_redis_cache()
        
        if cache_flushed:
            return {
                "success": True,
                "message": "Redis cache flushed successfully",
                "details": flush_message
            }
        else:
            return {
                "success": False,
                "message": "Failed to flush Redis cache",
                "details": flush_message
            }
    except Exception as e:
        logger.error(f"Failed to flush Redis cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to flush Redis cache: {str(e)}")

@router.post("/cache/reload-all")
def reload_all_caches():
    """Comprehensive cache reload - flush Redis and reload all settings from database"""
    try:
        # First, flush Redis cache
        cache_flushed, flush_message = _flush_redis_cache()
        
        # Then reload all settings caches
        reloaded_caches = []
        errors = []
        
        cache_functions = [
            ("LLM", reload_llm_settings),
            ("Query Classifier", reload_query_classifier_settings),
            ("Vector DB", reload_vector_db_settings),
            ("Embedding", reload_embedding_settings),
            ("RAG", reload_rag_settings),
            ("Knowledge Graph", reload_knowledge_graph_settings),
            ("Radiating", reload_radiating_settings),
            ("Timeout", reload_timeout_settings),
            ("Meta Task", reload_meta_task_settings),
            ("Synthesis Prompts", reload_synthesis_prompts),
        ]
        
        for cache_name, reload_func in cache_functions:
            try:
                reload_func()
                reloaded_caches.append(cache_name)
            except Exception as e:
                error_msg = f"{cache_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to reload {cache_name} cache: {e}")
        
        # Try to reload additional caches that might exist
        try:
            reload_iceberg_settings()
            reloaded_caches.append("Iceberg")
        except:
            pass
            
        try:
            from app.core.overflow_settings_cache import reload_overflow_settings
            reload_overflow_settings()
            reloaded_caches.append("Overflow")
        except:
            pass
            
        try:
            reload_enabled_mcp_tools()
            reloaded_caches.append("MCP Tools")
        except:
            pass
        
        return {
            "success": len(errors) == 0,
            "message": f"Cache reload completed - {len(reloaded_caches)} caches reloaded",
            "redis_cache_flushed": cache_flushed,
            "flush_details": flush_message,
            "reloaded_caches": reloaded_caches,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        logger.error(f"Failed to reload all caches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reload all caches: {str(e)}")


class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]
    persist_to_db: Optional[bool] = False
    reload_cache: Optional[bool] = False

@router.get("/init-mcp-tables")
def initialize_mcp_tables(db: Session = Depends(get_db)):
    """Initialize MCP tables if they don't exist"""
    try:
        inspector = inspect(db.bind)
        
        # Check if tables exist
        mcp_manifests_exists = 'mcp_manifests' in inspector.get_table_names()
        mcp_tools_exists = 'mcp_tools' in inspector.get_table_names()
        
        logger.info(f"MCP tables check: mcp_manifests={mcp_manifests_exists}, mcp_tools={mcp_tools_exists}")
        
        # Create tables if they don't exist
        from sqlalchemy.schema import CreateTable
        
        if not mcp_manifests_exists or not mcp_tools_exists:
            # Create the tables using Base metadata
            from app.core.db import MCPManifest, MCPTool
            from sqlalchemy import MetaData
            
            # Create a new MetaData instance to avoid conflicts
            metadata = MetaData()
            
            # Get the table objects from our models
            manifest_table = MCPManifest.__table__.tometadata(metadata) 
            tool_table = MCPTool.__table__.tometadata(metadata)
            
            # Create tables that don't exist
            with db.bind.begin() as conn:
                if not mcp_manifests_exists:
                    logger.info("Creating mcp_manifests table")
                    conn.execute(CreateTable(manifest_table))
                    logger.info("Successfully created mcp_manifests table with api_key column")
                
                if not mcp_tools_exists:
                    logger.info("Creating mcp_tools table")
                    conn.execute(CreateTable(tool_table))
                    logger.info("Successfully created mcp_tools table")
                
            logger.info("Tables created successfully")
        else:
            # Check if the api_key column exists in mcp_manifests
            mcp_manifest_columns = [col['name'] for col in inspector.get_columns('mcp_manifests')]
            if 'api_key' not in mcp_manifest_columns:
                logger.info("Adding api_key column to existing mcp_manifests table")
                with db.bind.begin() as conn:
                    # SQLite and PostgreSQL have different syntax
                    if 'sqlite' in str(db.bind.url):
                        conn.execute(text("ALTER TABLE mcp_manifests ADD COLUMN api_key TEXT"))
                    else:
                        conn.execute(text("ALTER TABLE mcp_manifests ADD COLUMN api_key VARCHAR(255)"))
                logger.info("Successfully added api_key column to mcp_manifests table")
        
        return {"success": True, "message": "MCP tables initialized successfully"}
    except Exception as e:
        logger.error(f"Error initializing MCP tables: {str(e)}")
        return {"success": False, "error": str(e)}

@router.get("/{category}")
def get_settings(category: str, db: Session = Depends(get_db)):
    settings_row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
    if not settings_row:
        # Return appropriate response for specific categories
        if category == 'rag':
            return {
                "category": category, 
                "settings": {},
                "message": "RAG settings not configured. Please configure via Settings UI.",
                "requires_configuration": True
            }
        elif category == 'large_generation':
            from app.core.large_generation_settings_cache import DEFAULT_LARGE_GENERATION_CONFIG
            # Flatten the nested structure for UI
            flattened_settings = {
                # Detection thresholds
                "strong_number_threshold": DEFAULT_LARGE_GENERATION_CONFIG["detection_thresholds"]["strong_number_threshold"],
                "medium_number_threshold": DEFAULT_LARGE_GENERATION_CONFIG["detection_thresholds"]["medium_number_threshold"],
                "small_number_threshold": DEFAULT_LARGE_GENERATION_CONFIG["detection_thresholds"]["small_number_threshold"],
                "min_items_for_chunking": DEFAULT_LARGE_GENERATION_CONFIG["detection_thresholds"]["min_items_for_chunking"],
                # Scoring parameters
                "numeric_score_weight": DEFAULT_LARGE_GENERATION_CONFIG["scoring_parameters"].get("numeric_score_weight", 0.5),
                "keyword_score_weight": DEFAULT_LARGE_GENERATION_CONFIG["scoring_parameters"].get("keyword_score_weight", 0.3),
                "pattern_score_weight": DEFAULT_LARGE_GENERATION_CONFIG["scoring_parameters"]["pattern_score_weight"],
                "score_multiplier_for_chunks": DEFAULT_LARGE_GENERATION_CONFIG["scoring_parameters"]["score_multiplier"],
                "chunking_bonus_multiplier": DEFAULT_LARGE_GENERATION_CONFIG["scoring_parameters"].get("chunking_bonus_multiplier", 1.5),
                # Processing parameters
                "items_per_chunk": DEFAULT_LARGE_GENERATION_CONFIG["processing_parameters"]["default_chunk_size"],
                "target_chunk_count": DEFAULT_LARGE_GENERATION_CONFIG["processing_parameters"]["max_target_count"],
                "time_per_chunk": DEFAULT_LARGE_GENERATION_CONFIG["processing_parameters"]["estimated_seconds_per_chunk"],
                "base_time": DEFAULT_LARGE_GENERATION_CONFIG["processing_parameters"].get("base_time", 10),
                "confidence_threshold": DEFAULT_LARGE_GENERATION_CONFIG["confidence_calculation"]["max_score_for_confidence"],
                # Memory management
                "redis_ttl": DEFAULT_LARGE_GENERATION_CONFIG["memory_management"]["redis_conversation_ttl"],
                "max_messages": DEFAULT_LARGE_GENERATION_CONFIG["memory_management"]["max_redis_messages"],
                "max_history_display": DEFAULT_LARGE_GENERATION_CONFIG["memory_management"]["conversation_history_display"],
                "enable_memory_optimization": True,
                # Keywords and patterns
                "keywords": DEFAULT_LARGE_GENERATION_CONFIG["keywords_and_patterns"]["large_output_indicators"],
                "regex_patterns": DEFAULT_LARGE_GENERATION_CONFIG["keywords_and_patterns"]["large_patterns"]
            }
            return {"category": category, "settings": flattened_settings}
        elif category == 'langfuse':
            # Return default langfuse settings
            default_langfuse = {
                "enabled": False,
                "host": "https://cloud.langfuse.com",
                "project_id": "",
                "public_key": "",
                "secret_key": "",
                "langfuse_sample_rate": 1.0,
                "debug_mode": False,
                "flush_at": 15,
                "flush_interval": 0.5,
                "timeout": 30,
                "s3_enabled": False,
                "s3_bucket_name": "",
                "s3_endpoint_url": "",
                "s3_access_key_id": "",
                "s3_secret_access_key": "",
                "custom_model_definitions": {}
            }
            return {"category": category, "settings": default_langfuse}
        elif category == 'environment':
            # Read environment variables from .env file
            import os
            from pathlib import Path
            
            env_vars = {}
            # Try to find .env file in project root
            project_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to project root
            env_file = project_root / '.env'
            
            if env_file.exists():
                try:
                    with open(env_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            # Skip empty lines and comments
                            if line and not line.startswith('#') and '=' in line:
                                try:
                                    key, value = line.split('=', 1)
                                    key = key.strip()
                                    value = value.strip()
                                    # Remove quotes if present
                                    if value.startswith('"') and value.endswith('"'):
                                        value = value[1:-1]
                                    elif value.startswith("'") and value.endswith("'"):
                                        value = value[1:-1]
                                    env_vars[key] = value
                                except ValueError:
                                    logger.warning(f"Skipping malformed line {line_num} in .env file: {line}")
                except Exception as e:
                    logger.error(f"Error reading .env file: {e}")
            else:
                logger.warning(f".env file not found at {env_file}")
                
            return {"category": category, "settings": {"environment_variables": env_vars}}
        elif category == 'self_reflection':
            # Load YAML defaults and merge with database settings
            import yaml
            import os
            
            yaml_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'langchain', 'self_reflection_config.yaml')
            try:
                with open(yaml_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                return {"category": category, "settings": yaml_config}
            except Exception as e:
                logger.error(f"Error loading self_reflection YAML config: {e}")
                # Return basic defaults if YAML fails
                default_self_reflection = {
                    "reflection": {
                        "enabled": True,
                        "default_mode": "balanced",
                        "min_query_length": 10,
                        "cache": {"enabled": True, "ttl_seconds": 3600, "max_size": 1000},
                        "timeout_seconds": 30
                    },
                    "quality_evaluation": {
                        "dimension_weights": {
                            "completeness": 0.25,
                            "relevance": 0.25,
                            "accuracy": 0.20,
                            "coherence": 0.10,
                            "specificity": 0.10,
                            "confidence": 0.10
                        },
                        "thresholds": {
                            "min_acceptable_score": 0.7,
                            "refinement_threshold": 0.8
                        }
                    }
                }
                return {"category": category, "settings": default_self_reflection}
        elif category == 'query_patterns':
            # Always load from YAML file for query_patterns (file is source of truth)
            import yaml
            import os
            
            yaml_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'langchain', 'query_patterns_config.yaml')
            try:
                with open(yaml_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                return {"category": category, "settings": yaml_config}
            except Exception as e:
                logger.error(f"Error loading query_patterns YAML config: {e}")
                # Return basic defaults if YAML fails
                default_query_patterns = {
                    "tool_patterns": {},
                    "rag_patterns": {},
                    "code_patterns": {},
                    "multi_agent_patterns": {},
                    "direct_llm_patterns": {},
                    "hybrid_indicators": {},
                    "settings": {
                        "min_confidence_threshold": 0.1,
                        "max_classifications": 3,
                        "enable_hybrid_detection": True,
                        "confidence_decay_factor": 0.8,
                        "pattern_combination_bonus": 0.15
                    }
                }
                return {"category": category, "settings": default_query_patterns}
        elif category == 'timeout':
            # Return default timeout settings
            from app.core.timeout_settings_cache import DEFAULT_TIMEOUT_SETTINGS
            return {"category": category, "settings": DEFAULT_TIMEOUT_SETTINGS}
        elif category == 'overflow':
            # Return default overflow settings
            from app.schemas.overflow import OverflowConfig
            default_config = OverflowConfig()
            return {"category": category, "settings": default_config.dict()}
        elif category == 'radiating':
            # Return default radiating settings
            from app.core.radiating_settings_cache import get_default_radiating_config
            return {"category": category, "settings": get_default_radiating_config()}
        elif category == 'meta_task':
            # Return default meta_task settings
            from app.core.meta_task_settings_cache import get_default_meta_task_settings
            return {"category": category, "settings": get_default_meta_task_settings()}
        elif category in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
            # Return default synthesis prompt settings for specific categories
            from app.core.synthesis_prompts_cache import get_default_synthesis_prompts
            defaults = get_default_synthesis_prompts()
            return {"category": category, "settings": defaults.get(category, {})}
        raise HTTPException(status_code=404, detail="Settings not found")
    
    # Special handling for large_generation to flatten nested structure for UI
    if category == 'large_generation' and settings_row.settings:
        # Check if settings are in nested format
        if "detection_thresholds" in settings_row.settings:
            # Convert nested structure to flattened structure for UI
            flattened_settings = {
                # Detection thresholds
                "strong_number_threshold": settings_row.settings.get("detection_thresholds", {}).get("strong_number_threshold", 30),
                "medium_number_threshold": settings_row.settings.get("detection_thresholds", {}).get("medium_number_threshold", 20),
                "small_number_threshold": settings_row.settings.get("detection_thresholds", {}).get("small_number_threshold", 20),
                "min_items_for_chunking": settings_row.settings.get("detection_thresholds", {}).get("min_items_for_chunking", 20),
                # Scoring parameters
                "numeric_score_weight": settings_row.settings.get("scoring_parameters", {}).get("numeric_score_weight", 0.5),
                "keyword_score_weight": settings_row.settings.get("scoring_parameters", {}).get("keyword_score_weight", 0.3),
                "pattern_score_weight": settings_row.settings.get("scoring_parameters", {}).get("pattern_score_weight", 2),
                "score_multiplier_for_chunks": settings_row.settings.get("scoring_parameters", {}).get("score_multiplier", 15),
                "chunking_bonus_multiplier": settings_row.settings.get("scoring_parameters", {}).get("chunking_bonus_multiplier", 1.5),
                # Processing parameters
                "items_per_chunk": settings_row.settings.get("processing_parameters", {}).get("default_chunk_size", 15),
                "target_chunk_count": settings_row.settings.get("processing_parameters", {}).get("max_target_count", 500),
                "time_per_chunk": settings_row.settings.get("processing_parameters", {}).get("estimated_seconds_per_chunk", 45),
                "base_time": settings_row.settings.get("processing_parameters", {}).get("base_time", 10),
                "confidence_threshold": settings_row.settings.get("confidence_calculation", {}).get("max_score_for_confidence", 5.0),
                # Memory management
                "redis_ttl": settings_row.settings.get("memory_management", {}).get("redis_conversation_ttl", 7 * 24 * 3600),
                "max_messages": settings_row.settings.get("memory_management", {}).get("max_redis_messages", 50),
                "max_history_display": settings_row.settings.get("memory_management", {}).get("conversation_history_display", 10),
                "enable_memory_optimization": True,
                # Keywords and patterns
                "keywords": settings_row.settings.get("keywords_and_patterns", {}).get("large_output_indicators", []),
                "regex_patterns": settings_row.settings.get("keywords_and_patterns", {}).get("large_patterns", [])
            }
            return {"category": category, "settings": flattened_settings}
        else:
            # Already in flattened format
            return {"category": category, "settings": settings_row.settings}
    
    # Special handling for RAG settings - just return as-is from database
    if category == 'rag' and settings_row:
        settings = settings_row.settings.copy() if settings_row.settings else {}
        
        # Remove old collection_selection_rules if it exists (legacy cleanup)
        if 'collection_selection_rules' in settings:
            del settings['collection_selection_rules']
        
        return {"category": category, "settings": settings}
    
    # Special handling for query_patterns - always load from YAML file
    if category == 'query_patterns':
        import yaml
        import os
        
        yaml_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'langchain', 'query_patterns_config.yaml')
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            return {"category": category, "settings": yaml_config}
        except Exception as e:
            logger.error(f"Error loading query_patterns YAML config: {e}")
            # Fall back to database settings if YAML fails
            return {"category": category, "settings": settings_row.settings}
    
    return {"category": category, "settings": settings_row.settings}

@router.put("/{category}")
def update_settings(category: str, update: SettingsUpdate, db: Session = Depends(get_db)):
    logger.info(f"Updating settings for category: {category}")
    logger.info(f"Update payload: persist_to_db={update.persist_to_db}, reload_cache={update.reload_cache}")
    
    # Initialize merged_settings variable to track deep merge results  
    merged_settings = None
    
    # Debug log all settings data for MCP
    if category == 'mcp':
        logger.info(f"MCP settings received - keys: {list(update.settings.keys())}")
        logger.info(f"MCP settings content: {update.settings}")
    
    # Debug log for API key
    if category == 'mcp' and 'api_key' in update.settings:
        has_api_key = bool(update.settings.get('api_key'))
        api_key_length = len(update.settings.get('api_key', ''))
        logger.info(f"MCP API Key present: {has_api_key}, length: {api_key_length}")
    
    # For MCP settings, remove API key before saving to settings table
    # It will be saved to the manifest table in handle_mcp_settings_update
    settings_for_db = update.settings.copy()
    if category == 'mcp' and 'api_key' in settings_for_db:
        logger.info("Removing API key from settings before saving to settings table")
        del settings_for_db['api_key']
    
    settings_row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
    
    # CRITICAL: Handle LLM deep merge BEFORE database save to prevent data loss
    if category == 'llm':
        logger.info("Processing LLM settings with deep merge BEFORE database save")
        
        # Get existing settings for deep merge
        existing_settings = settings_row.settings if settings_row else {}
        logger.info(f"Existing LLM settings keys: {list(existing_settings.keys()) if existing_settings else 'None'}")
        logger.info(f"New LLM settings keys: {list(settings_for_db.keys())}")
        
        # Ensure query_classifier is properly nested if present in top-level
        if 'query_classifier' in update.settings and isinstance(update.settings['query_classifier'], dict):
            # query_classifier is already nested, good
            pass
        else:
            # Check if query_classifier fields are at top level and need to be nested
            query_classifier_fields = [
                'min_confidence_threshold', 'max_classifications', 'classifier_max_tokens',
                'enable_hybrid_detection', 'confidence_decay_factor', 'pattern_combination_bonus',
                'llm_direct_threshold', 'multi_agent_threshold', 'direct_execution_threshold',
                # LLM-based classification fields
                'enable_llm_classification', 'llm_model', 'context_length', 'llm_temperature', 'llm_max_tokens',
                'llm_timeout_seconds', 'system_prompt', 'fallback_to_patterns', 'llm_classification_priority'
            ]
            
            # Extract query_classifier fields if they exist at top level
            query_classifier_data = {}
            for field in query_classifier_fields:
                if field in settings_for_db:
                    query_classifier_data[field] = settings_for_db[field]
                    # Remove from top level
                    del settings_for_db[field]
            
            # Add query_classifier as nested object if we found any fields
            if query_classifier_data:
                settings_for_db['query_classifier'] = query_classifier_data
        
        # Deep merge to preserve existing complex fields
        merged_settings = deep_merge_settings(existing_settings, settings_for_db)
        logger.info(f"Merged LLM settings keys: {list(merged_settings.keys())}")
        
        # Validate that critical fields are preserved
        critical_fields = ['main_llm', 'second_llm', 'query_classifier', 'search_optimization', 'thinking_mode_params', 'non_thinking_mode_params']
        preserved_fields = [field for field in critical_fields if field in existing_settings and field in merged_settings]
        if preserved_fields:
            logger.info(f"Preserved critical LLM fields: {preserved_fields}")
        
        # Use merged settings for database save
        settings_for_db = merged_settings
        logger.info("LLM deep merge completed, will save merged data to database")
    
    if settings_row:
        logger.info(f"Updating existing settings for {category}")
        # Debug for existing settings
        if category == 'mcp':
            current_api_key = settings_row.settings.get('api_key') if settings_row.settings else None
            has_current_api_key = bool(current_api_key)
            current_api_key_length = len(current_api_key or '')
            logger.info(f"Existing MCP API Key: {has_current_api_key}, length: {current_api_key_length}")
        
        settings_row.settings = settings_for_db
    else:
        logger.info(f"Creating new settings for {category}")
        settings_row = SettingsModel(category=category, settings=settings_for_db)
        db.add(settings_row)
    
    try:
        db.commit()
        db.refresh(settings_row)
        logger.info(f"Successfully saved settings to {category} table")
        
        # Verify API key was saved
        if category == 'mcp':
            saved_api_key = settings_row.settings.get('api_key')
            has_saved_api_key = bool(saved_api_key)
            saved_api_key_length = len(saved_api_key or '')
            logger.info(f"Saved MCP API Key: {has_saved_api_key}, length: {saved_api_key_length}")
            
            endpoint_prefix = update.settings.get('endpoint_prefix')
            if endpoint_prefix:
                logger.info(f"Settings contains endpoint_prefix: {endpoint_prefix}")
    except Exception as e:
        logger.error(f"Error saving settings to database: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")
    
    # If updating LLM settings, reload cache (deep merge already handled above)
    if category == 'llm':
        logger.info("LLM deep merge completed, reloading caches")
        reload_llm_settings()
        reload_query_classifier_settings()
    
    # If updating storage settings, reload all related caches
    if category == 'storage':
        reload_vector_db_settings()
        reload_embedding_settings()
        reload_iceberg_settings()
    
    # If updating RAG settings, validate and reload cache
    if category == 'rag':
        from app.core.rag_settings_cache import reload_rag_settings, validate_rag_settings
        
        # Validate settings structure
        is_valid, error_message = validate_rag_settings(settings_for_db)
        if not is_valid:
            return {"success": False, "error": f"Invalid RAG settings: {error_message}"}
        
        # Remove old collection_selection_rules if it exists (legacy cleanup)
        if 'collection_selection_rules' in settings_for_db:
            del settings_for_db['collection_selection_rules']
        
        # Update the database with validated settings
        settings_row.settings = settings_for_db
        db.commit()
        db.refresh(settings_row)
        
        # Reload and validate cache
        try:
            reload_rag_settings()
        except Exception as e:
            return {"success": False, "error": f"Failed to reload RAG settings: {str(e)}"}
    
    # If updating MCP settings, handle special processing
    if category == 'mcp':
        logger.info("Processing MCP settings with special handling")
        handle_mcp_settings_update(update, db)
    
    # If updating large generation settings, validate and reload cache
    if category == 'large_generation':
        logger.info("Processing large generation settings")
        
        # Convert flattened UI structure back to nested structure
        nested_settings = {
            "detection_thresholds": {
                "strong_number_threshold": update.settings.get("strong_number_threshold", 30),
                "medium_number_threshold": update.settings.get("medium_number_threshold", 20),
                "small_number_threshold": update.settings.get("small_number_threshold", 20),
                "min_items_for_chunking": update.settings.get("min_items_for_chunking", 20)
            },
            "scoring_parameters": {
                "numeric_score_weight": update.settings.get("numeric_score_weight", 0.5),
                "keyword_score_weight": update.settings.get("keyword_score_weight", 0.3),
                "pattern_score_weight": update.settings.get("pattern_score_weight", 2),
                "score_multiplier": update.settings.get("score_multiplier_for_chunks", 15),
                "chunking_bonus_multiplier": update.settings.get("chunking_bonus_multiplier", 1.5),
                "min_score_for_keywords": 3,
                "min_score_for_medium_numbers": 2,
                "default_comprehensive_items": 30,
                "min_estimated_items": 10
            },
            "confidence_calculation": {
                "max_score_for_confidence": update.settings.get("confidence_threshold", 5.0),
                "max_number_for_confidence": 100.0
            },
            "processing_parameters": {
                "default_chunk_size": update.settings.get("items_per_chunk", 15),
                "max_target_count": update.settings.get("target_chunk_count", 500),
                "estimated_seconds_per_chunk": update.settings.get("time_per_chunk", 45)
            },
            "memory_management": {
                "redis_conversation_ttl": update.settings.get("redis_ttl", 7 * 24 * 3600),
                "max_redis_messages": update.settings.get("max_messages", 50),
                "max_memory_messages": 20,
                "conversation_history_display": update.settings.get("max_history_display", 10)
            },
            "keywords_and_patterns": {
                "large_output_indicators": update.settings.get("keywords", []),
                "comprehensive_keywords": ["comprehensive", "detailed", "all", "many"],
                "large_patterns": update.settings.get("regex_patterns", [])
            }
        }
        
        # Merge with defaults to ensure all required fields exist
        merged_settings = merge_with_defaults(nested_settings)
        
        # Validate configuration
        is_valid, error_msg = validate_large_generation_config(merged_settings)
        if not is_valid:
            logger.error(f"Invalid large generation configuration: {error_msg}")
            raise HTTPException(status_code=400, detail=f"Invalid configuration: {error_msg}")
        
        # Store the flattened version for UI consistency
        settings_row.settings = update.settings
        db.commit()
        
        # Reload cache with nested structure
        reload_large_generation_settings()
        logger.info("Large generation settings validated and cache reloaded")
    
    # If updating timeout settings, validate and reload cache
    if category == 'timeout':
        from app.core.timeout_settings_cache import validate_timeout_settings, reload_timeout_settings
        
        # Validate timeout settings
        validated_settings = validate_timeout_settings(settings_for_db)
        
        # Update database with validated settings
        settings_row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
        if settings_row:
            settings_row.settings = validated_settings
        else:
            settings_row = SettingsModel(category=category, settings=validated_settings)
            db.add(settings_row)
        
        try:
            db.commit()
            db.refresh(settings_row)
            logger.info("Timeout settings validated and saved to database")
        except Exception as e:
            logger.error(f"Error saving validated timeout settings: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to save timeout settings: {str(e)}")
        
        # Reload cache
        reload_timeout_settings()
        logger.info("Timeout settings validated and cache reloaded")
    
    # If updating overflow settings, reload cache
    if category == 'overflow':
        from app.core.overflow_settings_cache import reload_overflow_settings
        reload_overflow_settings()
        logger.info("Overflow settings validated and cache reloaded")
    
    # If updating self_reflection settings, save as YAML structure and update database settings
    if category == 'self_reflection':
        logger.info("Processing self_reflection settings")
        from app.core.self_reflection_settings_cache import update_reflection_settings
        
        # Save the YAML structure as is to settings table for frontend compatibility
        settings_row.settings = update.settings
        db.commit()
        db.refresh(settings_row)
        
        # Also update model-specific settings in the self_reflection cache if model_settings exist
        if 'model_settings' in update.settings:
            model_settings = update.settings['model_settings']
            for model_name, model_config in model_settings.items():
                if model_name != 'default':  # Skip default, it's handled by YAML
                    try:
                        # Convert model config to database format
                        db_config = {
                            "enabled": model_config.get('enabled', True),
                            "reflection_mode": model_config.get('reflection_mode', 'balanced'),
                            "quality_threshold": model_config.get('quality_threshold', 0.8),
                            "max_iterations": model_config.get('max_refinement_iterations', 3),
                            "min_improvement_threshold": 0.05,
                            "enable_caching": True,
                            "cache_ttl_seconds": 3600,
                            "timeout_seconds": 30
                        }
                        # Note: update_reflection_settings is async, but for now we'll skip this
                        # await update_reflection_settings(model_name, db_config)
                    except Exception as e:
                        logger.warning(f"Could not update model-specific self-reflection settings for {model_name}: {e}")
        
        logger.info("Self-reflection settings saved successfully")
    
    # If updating query_patterns settings, save to YAML file and database
    if category == 'query_patterns':
        logger.info("Processing query_patterns settings")
        import yaml
        import os
        
        # Save to database for API consistency
        settings_row.settings = update.settings
        db.commit()
        db.refresh(settings_row)
        
        # Also save to YAML file to maintain file-based configuration
        yaml_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'langchain', 'query_patterns_config.yaml')
        try:
            with open(yaml_path, 'w') as f:
                yaml.dump(update.settings, f, default_flow_style=False, indent=2)
            logger.info("Query patterns YAML file updated successfully")
            
            # Trigger pattern reload if available
            try:
                from app.langchain.enhanced_query_classifier import reload_patterns
                reload_patterns()
                logger.info("Query patterns cache reloaded")
            except Exception as e:
                logger.warning(f"Could not reload query patterns cache: {e}")
                
        except Exception as e:
            logger.error(f"Error saving query_patterns YAML file: {e}")
            # Continue even if file save fails, database is updated
        
        logger.info("Query patterns settings saved successfully")
    
    # Special handling for environment variables - save to .env file
    if category == 'environment' and update.persist_to_db:
        from pathlib import Path
        
        # Get environment variables from the update
        env_vars = update.settings.get('environment_variables', {})
        
        # Find .env file in project root
        project_root = Path(__file__).parent.parent.parent.parent.parent  # Go up to project root
        env_file = project_root / '.env'
        
        try:
            # Write environment variables to .env file
            with open(env_file, 'w') as f:
                f.write("# Environment Variables\n")
                f.write("# Generated by Jarvis Settings UI\n\n")
                
                for key, value in env_vars.items():
                    # Escape values that contain spaces or special characters
                    if ' ' in str(value) or '"' in str(value) or "'" in str(value):
                        escaped_value = str(value).replace('"', '\\"')
                        value = f'"{escaped_value}"'
                    f.write(f"{key}={value}\n")
                    
            logger.info(f"Successfully updated .env file with {len(env_vars)} variables")
            
        except Exception as e:
            logger.error(f"Error writing to .env file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to update .env file: {str(e)}")
    
    # If updating radiating settings, use deep merge and reload cache
    if category == 'radiating':
        logger.info("Processing radiating settings with deep merge")
        
        # Use deep merge to preserve existing settings like prompts
        existing_settings = settings_row.settings or {}
        logger.info(f"Existing radiating settings keys: {list(existing_settings.keys()) if existing_settings else 'None'}")
        logger.info(f"New radiating settings keys: {list(settings_for_db.keys())}")
        
        # Deep merge to preserve existing complex fields
        merged_settings = deep_merge_settings(existing_settings, settings_for_db)
        logger.info(f"Merged radiating settings keys: {list(merged_settings.keys())}")
        
        # Validate that critical fields are preserved
        critical_fields = ['prompts', 'model_config', 'query_expansion', 'extraction', 'traversal', 'coverage', 'synthesis', 'performance', 'monitoring']
        preserved_fields = [field for field in critical_fields if field in existing_settings and field in merged_settings]
        if preserved_fields:
            logger.info(f"Preserved critical radiating fields: {preserved_fields}")
        
        # Update the database with merged settings
        settings_row.settings = merged_settings
        db.commit()
        db.refresh(settings_row)
        logger.info("Radiating settings saved with deep merge")
        
        # Use merged settings for database save (to ensure it's returned correctly)
        settings_for_db = merged_settings
        
        reload_radiating_settings()
        logger.info("Radiating settings validated and cache reloaded")
    
    # If updating knowledge graph settings, sync model fields and reload cache
    if category == 'knowledge_graph':
        logger.info("Processing knowledge graph settings")
        
        # CRITICAL FIX: Use deep merge instead of complete replacement to preserve all existing fields
        existing_settings = settings_row.settings or {}
        logger.info(f"Existing settings keys: {list(existing_settings.keys()) if existing_settings else 'None'}")
        logger.info(f"New settings keys: {list(settings_for_db.keys())}")
        
        # Synchronize model fields: ensure both main 'model' and 'model_config.model' are consistent
        if 'model_config' in settings_for_db and 'model' in settings_for_db['model_config']:
            # Update the main model field to match model_config.model
            settings_for_db['model'] = settings_for_db['model_config']['model']
            logger.info(f"Synchronized main model field to: {settings_for_db['model']}")
        elif 'model' in settings_for_db and 'model_config' in settings_for_db:
            # Update model_config.model to match main model field
            if 'model' not in settings_for_db['model_config']:
                settings_for_db['model_config']['model'] = settings_for_db['model']
                logger.info(f"Synchronized model_config.model field to: {settings_for_db['model']}")
        
        # Deep merge to preserve existing complex fields (prompts, extraction, learning, discovered_schemas, etc.)
        merged_settings = deep_merge_settings(existing_settings, settings_for_db)
        logger.info(f"Merged settings keys: {list(merged_settings.keys())}")
        
        # Validate that critical fields are preserved
        critical_fields = ['prompts', 'extraction', 'learning', 'discovered_schemas', 'entity_discovery', 'relationship_discovery']
        preserved_fields = [field for field in critical_fields if field in existing_settings and field in merged_settings]
        if preserved_fields:
            logger.info(f"Preserved critical fields: {preserved_fields}")
        
        # Update the database with merged settings
        settings_row.settings = merged_settings
        db.commit()
        db.refresh(settings_row)
        logger.info("Knowledge graph settings synchronized and saved")
        
        # Reload the cache
        reload_knowledge_graph_settings()
        logger.info("Knowledge graph settings saved and cache reloaded")
    
    # If updating meta_task settings, use deep merge and reload cache
    if category == 'meta_task':
        logger.info("Processing meta_task settings with deep merge")
        
        # Get existing settings for deep merge
        existing_settings = settings_row.settings if settings_row else {}
        logger.info(f"Existing meta_task settings keys: {list(existing_settings.keys()) if existing_settings else 'None'}")
        logger.info(f"New meta_task settings keys: {list(settings_for_db.keys())}")
        
        # Deep merge to preserve existing nested model configurations
        merged_settings = deep_merge_settings(existing_settings, settings_for_db)
        logger.info(f"Merged meta_task settings keys: {list(merged_settings.keys())}")
        
        # Validate that critical fields are preserved
        critical_fields = ['analyzer_model', 'reviewer_model', 'assembler_model', 'generator_model', 
                          'execution', 'quality_control', 'output', 'caching']
        preserved_fields = [field for field in critical_fields if field in existing_settings and field in merged_settings]
        if preserved_fields:
            logger.info(f"Preserved critical meta_task fields: {preserved_fields}")
        
        # Log the nested model configurations to verify system_prompts are preserved
        for model_key in ['analyzer_model', 'reviewer_model', 'assembler_model', 'generator_model']:
            if model_key in merged_settings and isinstance(merged_settings[model_key], dict):
                if 'system_prompt' in merged_settings[model_key]:
                    prompt_length = len(merged_settings[model_key]['system_prompt'])
                    logger.info(f"{model_key} has system_prompt with length: {prompt_length}")
        
        # Update the database with merged settings
        settings_row.settings = merged_settings
        db.commit()
        db.refresh(settings_row)
        logger.info("Meta_task settings saved with deep merge")
        
        # Use merged settings for return value
        settings_for_db = merged_settings
        
        # Reload the cache
        reload_meta_task_settings()
        logger.info("Meta_task settings validated and cache reloaded")
    
    # If updating synthesis prompt settings, validate and reload cache
    if category in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        logger.info(f"Processing {category} settings with validation")
        
        from app.core.synthesis_prompts_cache import validate_synthesis_prompts, reload_synthesis_prompts
        
        # For individual category updates, we need to validate in context of all categories
        # Get existing settings for other categories
        all_synthesis_settings = {}
        synthesis_categories = ['synthesis_prompts', 'formatting_templates', 'system_behaviors']
        
        for cat in synthesis_categories:
            if cat == category:
                # Use new settings for current category
                all_synthesis_settings[cat] = settings_for_db
            else:
                # Get existing settings for other categories
                existing_row = db.query(SettingsModel).filter(SettingsModel.category == cat).first()
                if existing_row and existing_row.settings:
                    all_synthesis_settings[cat] = existing_row.settings
                else:
                    # Use defaults if not found
                    from app.core.synthesis_prompts_cache import get_default_synthesis_prompts
                    defaults = get_default_synthesis_prompts()
                    all_synthesis_settings[cat] = defaults.get(cat, {})
        
        # Validate the complete synthesis settings
        is_valid, error_message = validate_synthesis_prompts(all_synthesis_settings)
        if not is_valid:
            logger.error(f"Invalid synthesis prompt settings: {error_message}")
            raise HTTPException(status_code=400, detail=f"Invalid synthesis prompt settings: {error_message}")
        
        # Deep merge with existing settings to preserve template metadata
        existing_settings = settings_row.settings if settings_row else {}
        merged_settings = deep_merge_settings(existing_settings, settings_for_db)
        
        # Update the database with merged settings
        settings_row.settings = merged_settings
        db.commit()
        db.refresh(settings_row)
        logger.info(f"{category} settings validated and saved")
        
        # Use merged settings for return value
        settings_for_db = merged_settings
        
        # Reload synthesis prompts cache
        reload_synthesis_prompts()
        logger.info(f"{category} settings cache reloaded")
    
    # Return merged settings if available (for LLM with deep merge), otherwise use database settings
    final_settings = merged_settings if merged_settings is not None else settings_row.settings
    return {"category": category, "settings": final_settings}

def handle_mcp_settings_update(update: SettingsUpdate, db: Session):
    """Handle MCP settings update with special processing for manifest URL and API key"""
    logger.info(f"Starting handle_mcp_settings_update with persist_to_db={update.persist_to_db}")
    
    if not update.persist_to_db:
        logger.info("persist_to_db is False, skipping special processing")
        return
    
    # Check if the required tables exist
    try:
        inspector = inspect(db.bind)
        
        if 'mcp_manifests' not in inspector.get_table_names():
            logger.error("mcp_manifests table does not exist in the database")
            return
            
        if 'mcp_tools' not in inspector.get_table_names():
            logger.error("mcp_tools table does not exist in the database")
            return
            
        logger.info("Verified that mcp_manifests and mcp_tools tables exist")
    except Exception as e:
        logger.error(f"Error checking for table existence: {str(e)}")
        return
    
    settings = update.settings
    manifest_url = settings.get('manifest_url')
    api_key = settings.get('api_key')
    hostname = settings.get('hostname')
    endpoint_prefix = settings.get('endpoint_prefix', '')
    
    # Debug log for incoming API key
    has_api_key = bool(api_key)
    api_key_length = len(api_key or '')
    logger.info(f"Incoming MCP API Key: present={has_api_key}, length={api_key_length}")
    
    logger.info(f"MCP settings: manifest_url={manifest_url}, api_key={'[REDACTED]' if api_key else 'None'}, hostname={hostname}, endpoint_prefix={endpoint_prefix}")
    
    # Check if this is a tool configuration update (no manifest_url required)
    tool_config_fields = ['max_tool_calls', 'tool_timeout_seconds', 'enable_tool_retries', 'max_tool_retries']
    # Check both the top-level settings and nested settings.settings
    is_tool_config_update = any(field in settings for field in tool_config_fields)
    if not is_tool_config_update and 'settings' in settings:
        # Also check nested settings object for tool configuration fields
        nested_settings = settings['settings']
        if isinstance(nested_settings, dict):
            is_tool_config_update = any(field in nested_settings for field in tool_config_fields)
    
    logger.info(f"MCP settings update check - is_tool_config_update: {is_tool_config_update}, settings keys: {list(settings.keys())}")
    
    if not manifest_url and not is_tool_config_update:
        logger.warning("No manifest URL provided for MCP settings update")
        return
    
    # If this is only a tool configuration update, no need for manifest processing
    if is_tool_config_update and not manifest_url:
        logger.info("Processing tool configuration update (no manifest URL required)")
        # Just save the tool configuration settings - they're already handled in the main settings update
        reload_enabled_mcp_tools()
        return
    
    # Flag to track if we should attempt to reload the cache
    should_reload_cache = update.reload_cache
    
    try:
        # Explicitly start a new transaction for the manifest processing
        # This ensures that even if the manifest fetching fails, the settings are still saved
        logger.info("Starting manifest processing")
        
        # Fetch manifest data from the provided URL
        logger.info(f"Fetching manifest from URL: {manifest_url}")
        
        headers = {}
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"
            logger.info("Adding Authorization header with API key")
        
        try:
            # Substitute hostname for localhost if provided
            fetch_url = manifest_url
            if hostname and 'localhost' in manifest_url:
                fetch_url = manifest_url.replace('localhost', hostname)
                logger.info(f"Using hostname '{hostname}' instead of localhost: {fetch_url}")
            
            # Set a reasonable timeout for the request
            logger.info(f"Sending request to fetch manifest using URL: {fetch_url}")
            response = requests.get(fetch_url, headers=headers, timeout=5)
            
            logger.info(f"Got response with status code: {response.status_code}")
            response.raise_for_status()
            
            manifest_data = response.json()
            logger.info("Successfully parsed manifest JSON")
            
            if not manifest_data or not isinstance(manifest_data.get('tools'), list):
                logger.warning(f"Invalid manifest format from {manifest_url}")
                # Return without error, but don't update manifest or tools
                return
                
            # Extract hostname from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(fetch_url)
            hostname_from_url = parsed_url.netloc
            logger.info(f"Extracted hostname: {hostname_from_url}")
            
            # Upsert manifest in the database
            try:
                manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
                if not manifest:
                    logger.info(f"Creating new manifest for URL: {manifest_url}")
                    manifest = MCPManifest(
                        url=manifest_url,
                        hostname=hostname_from_url,
                        api_key=api_key,  # Store API key in manifest table
                        content=manifest_data
                    )
                    db.add(manifest)
                    try:
                        db.commit()
                        db.refresh(manifest)
                        logger.info(f"Successfully created manifest with ID: {manifest.id}")
                    except Exception as commit_error:
                        db.rollback()
                        logger.error(f"Error committing new manifest: {str(commit_error)}")
                        raise
                else:
                    logger.info(f"Updating existing manifest with ID: {manifest.id}")
                    manifest.content = manifest_data
                    manifest.hostname = hostname_from_url
                    manifest.api_key = api_key  # Update API key in manifest table
                    try:
                        db.commit()
                        logger.info("Successfully updated manifest")
                    except Exception as commit_error:
                        db.rollback()
                        
                        # Check if the error is related to the api_key column not existing
                        if "column mcp_manifests.api_key does not exist" in str(commit_error):
                            logger.error("The api_key column is missing from the mcp_manifests table")
                            raise HTTPException(
                                status_code=500, 
                                detail="The api_key column is missing from the mcp_manifests table. Please run the migration script at jarvis/scripts/add_api_key_column.py to add it."
                            )
                        else:
                            raise
                
                # Store endpoint_prefix in settings, but NOT the API key
                settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
                if settings_row:
                    # Make sure we preserve the existing settings but remove api_key
                    current_settings = settings_row.settings or {}
                    current_settings['endpoint_prefix'] = endpoint_prefix
                    current_settings['manifest_url'] = manifest_url
                    current_settings['hostname'] = hostname
                    # Remove API key from settings as it's now stored in the manifest table
                    if 'api_key' in current_settings:
                        del current_settings['api_key']
                    settings_row.settings = current_settings
                    logger.info(f"Updating MCP settings with endpoint_prefix: {endpoint_prefix}, removing API key from settings")
                else:
                    # Create new settings row without the API key
                    settings_row = SettingsModel(
                        category='mcp',
                        settings={
                            'endpoint_prefix': endpoint_prefix,
                            'manifest_url': manifest_url,
                            'hostname': hostname
                        }
                    )
                    db.add(settings_row)
                    logger.info(f"Creating new MCP settings with endpoint_prefix: {endpoint_prefix}, without API key")
                
                try:
                    db.commit()
                    logger.info("Successfully saved endpoint_prefix to settings table")
                except Exception as settings_error:
                    db.rollback()
                    logger.error(f"Error saving endpoint_prefix to settings: {str(settings_error)}")
                
                # Log the saved manifest
                saved_manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
                if saved_manifest:
                    has_saved_api_key = bool(saved_manifest.api_key)
                    api_key_length = len(saved_manifest.api_key or '')
                    logger.info(f"Verified manifest saved with ID: {saved_manifest.id}, API key present: {has_saved_api_key}, length: {api_key_length}")
                else:
                    logger.error("Failed to find saved manifest after commit")
                
                # Upsert tools from the manifest
                logger.info(f"Processing {len(manifest_data.get('tools', []))} tools")
                
                # First, get all existing tools for this manifest to check for uniqueness
                existing_tools = {}
                if manifest.id:
                    for tool in db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).all():
                        existing_tools[tool.name] = tool
                    logger.info(f"Found {len(existing_tools)} existing tools for this manifest")
                
                for i, tool in enumerate(manifest_data.get('tools', [])):
                    tool_name = tool.get('name')
                    if not tool_name:
                        logger.warning(f"Skipping tool at index {i} without a name")
                        continue
                        
                    logger.info(f"Processing tool: {tool_name}")
                    
                    # Check if this tool exists for this manifest
                    if tool_name in existing_tools:
                        db_tool = existing_tools[tool_name]
                        logger.info(f"Updating existing tool: {tool_name} (ID: {db_tool.id})")
                        db_tool.description = tool.get('description')
                        
                        # Apply endpoint prefix if it exists
                        tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                        
                        # First, try to strip any existing prefix to get the clean endpoint
                        clean_endpoint = tool_endpoint
                        try:
                            # Find the last component of the path which is likely the original endpoint
                            path_parts = tool_endpoint.rstrip('/').split('/')
                            if len(path_parts) > 1:
                                # The last part is likely the original tool name or operation
                                clean_endpoint = f"/{path_parts[-1]}"
                        except Exception as parse_error:
                            logger.warning(f"Error parsing endpoint for prefix cleanup: {str(parse_error)}")
                        
                        # Now apply the new prefix if it exists
                        if endpoint_prefix:
                            # Make sure we have clean slashes
                            prefix_clean = endpoint_prefix.rstrip('/')
                            endpoint_clean = clean_endpoint.lstrip('/')
                            
                            # Put them together
                            tool_endpoint = f"{prefix_clean}/{endpoint_clean}"
                            logger.info(f"Applied endpoint prefix to {tool_name}: {tool_endpoint}")
                        else:
                            # Use the original endpoint or default
                            tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                            logger.info(f"No endpoint prefix to apply for {tool_name}, using: {tool_endpoint}")
                        
                        db_tool.endpoint = tool_endpoint
                        db_tool.method = tool.get('method', 'POST')
                        db_tool.parameters = tool.get('parameters')
                        db_tool.headers = tool.get('headers')
                        # Preserve is_active status
                    else:
                        # Check for name collision with other manifests
                        # This can happen if tool name is not unique across manifests
                        name_collision = db.query(MCPTool).filter(MCPTool.name == tool_name).first()
                        if name_collision:
                            logger.warning(f"Tool name '{tool_name}' already exists with different manifest_id: {name_collision.manifest_id}")
                            # We'll create the tool with a unique name to avoid the collision
                            unique_tool_name = f"{tool_name}_{manifest.id}"
                            logger.info(f"Creating tool with modified name: {unique_tool_name}")
                            
                            # Apply endpoint prefix if it exists
                            tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                            
                            # First, try to strip any existing prefix to get the clean endpoint
                            clean_endpoint = tool_endpoint
                            try:
                                # Find the last component of the path which is likely the original endpoint
                                path_parts = tool_endpoint.rstrip('/').split('/')
                                if len(path_parts) > 1:
                                    # The last part is likely the original tool name or operation
                                    clean_endpoint = f"/{path_parts[-1]}"
                            except Exception as parse_error:
                                logger.warning(f"Error parsing endpoint for prefix cleanup: {str(parse_error)}")
                            
                            # Now apply the new prefix if it exists
                            if endpoint_prefix:
                                # Make sure we have clean slashes
                                prefix_clean = endpoint_prefix.rstrip('/')
                                endpoint_clean = clean_endpoint.lstrip('/')
                                
                                # Put them together
                                tool_endpoint = f"{prefix_clean}/{endpoint_clean}"
                                logger.info(f"Applied endpoint prefix to {unique_tool_name}: {tool_endpoint}")
                            else:
                                # Use the original endpoint or default
                                tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                                logger.info(f"No endpoint prefix to apply for {tool_name}, using: {tool_endpoint}")
                            
                            db_tool = MCPTool(
                                name=unique_tool_name,  # Use a unique name
                                description=tool.get('description'),
                                endpoint=tool_endpoint,
                                method=tool.get('method', 'POST'),
                                parameters=tool.get('parameters'),
                                headers=tool.get('headers'),
                                is_active=True,
                                manifest_id=manifest.id
                            )
                        else:
                            logger.info(f"Creating new tool: {tool_name}")
                            
                            # Apply endpoint prefix if it exists
                            tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                            
                            # First, try to strip any existing prefix to get the clean endpoint
                            clean_endpoint = tool_endpoint
                            try:
                                # Find the last component of the path which is likely the original endpoint
                                path_parts = tool_endpoint.rstrip('/').split('/')
                                if len(path_parts) > 1:
                                    # The last part is likely the original tool name or operation
                                    clean_endpoint = f"/{path_parts[-1]}"
                            except Exception as parse_error:
                                logger.warning(f"Error parsing endpoint for prefix cleanup: {str(parse_error)}")
                            
                            # Now apply the new prefix if it exists
                            if endpoint_prefix:
                                # Make sure we have clean slashes
                                prefix_clean = endpoint_prefix.rstrip('/')
                                endpoint_clean = clean_endpoint.lstrip('/')
                                
                                # Put them together
                                tool_endpoint = f"{prefix_clean}/{endpoint_clean}"
                                logger.info(f"Applied endpoint prefix to {tool_name}: {tool_endpoint}")
                            else:
                                # Use the original endpoint or default
                                tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                                logger.info(f"No endpoint prefix to apply for {tool_name}, using: {tool_endpoint}")
                            
                            db_tool = MCPTool(
                                name=tool_name,
                                description=tool.get('description'),
                                endpoint=tool_endpoint,
                                method=tool.get('method', 'POST'),
                                parameters=tool.get('parameters'),
                                headers=tool.get('headers'),
                                is_active=True,
                                manifest_id=manifest.id
                            )
                        db.add(db_tool)
                
                try:
                    logger.info("Committing tool changes")
                    db.commit()
                    logger.info("Successfully committed tool changes")
                    
                    # Verify tool counts
                    tool_count = db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).count()
                    logger.info(f"Verified {tool_count} tools saved for manifest ID {manifest.id}")
                    
                except Exception as commit_error:
                    db.rollback()
                    logger.error(f"Error committing tool changes: {str(commit_error)}")
                    raise
            except Exception as manifest_error:
                logger.error(f"Error processing manifest: {str(manifest_error)}")
                raise
                  
        except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
            # Handle connection errors, timeouts, or invalid JSON
            logger.warning(f"Could not fetch or process manifest from {manifest_url}: {str(e)}")
            # Don't raise an exception, just log the warning
            # Settings will still be saved, but manifest and tools won't be updated
            
            # Make sure we still save the settings even if manifest fetch fails
            settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
            if settings_row:
                # Update existing settings
                current_settings = settings_row.settings or {}
                current_settings['manifest_url'] = manifest_url
                # Remove API key from settings as it should be in the manifest table
                if 'api_key' in current_settings:
                    del current_settings['api_key']
                current_settings['hostname'] = hostname
                current_settings['endpoint_prefix'] = endpoint_prefix
                settings_row.settings = current_settings
                logger.info("Updating MCP settings despite manifest fetch failure (API key removed from settings)")
            else:
                # Create new settings
                settings_row = SettingsModel(
                    category='mcp',
                    settings={
                        'manifest_url': manifest_url,
                        'hostname': hostname,
                        'endpoint_prefix': endpoint_prefix
                    }
                )
                db.add(settings_row)
                logger.info("Creating new MCP settings despite manifest fetch failure (without API key)")
            
            try:
                db.commit()
                logger.info("Successfully saved MCP settings despite manifest fetch failure")
            except Exception as settings_error:
                logger.error(f"Error saving MCP settings: {str(settings_error)}")
                db.rollback()
            
            # See if we have an existing manifest record we can update with just the URL and API key
            manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
            if manifest:
                logger.info(f"Updating existing manifest URL and preserving content")
                # Update hostname if provided
                if hostname:
                    manifest.hostname = hostname
                    logger.info(f"Updated manifest hostname to user-provided value: {hostname}")
                else:
                    # Just update the hostname from the URL if possible
                    try:
                        from urllib.parse import urlparse
                        parsed_url = urlparse(manifest_url)
                        url_hostname = parsed_url.netloc
                        manifest.hostname = url_hostname
                        logger.info(f"Updated manifest hostname from URL: {url_hostname}")
                    except Exception as parse_error:
                        logger.warning(f"Could not parse URL for hostname: {str(parse_error)}")
                
                # Update the API key in the manifest
                manifest.api_key = api_key
                logger.info(f"Updated API key in manifest record. API key present: {bool(api_key)}, length: {len(api_key or '')}")
                
                try:
                    db.commit()
                    logger.info("Updated manifest record")
                except Exception as commit_error:
                    logger.error(f"Failed to update manifest record: {str(commit_error)}")
                    db.rollback()
            else:
                # Create a new manifest record with the URL and hostname even if fetch failed
                logger.info(f"Creating new manifest record for URL: {manifest_url}")
                
                # Get hostname from URL or use provided hostname
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(manifest_url)
                    url_hostname = hostname if hostname else parsed_url.netloc
                    
                    # Create minimal manifest record
                    new_manifest = MCPManifest(
                        url=manifest_url,
                        hostname=url_hostname,
                        api_key=api_key,  # Store API key in manifest record
                        content={"tools": []}  # Empty tools array as placeholder
                    )
                    db.add(new_manifest)
                    db.commit()
                    logger.info(f"Created new manifest record with hostname: {url_hostname}, API key present: {bool(api_key)}, length: {len(api_key or '')}")
                except Exception as create_error:
                    logger.error(f"Failed to create manifest record: {str(create_error)}")
                    db.rollback()
            
            # We should still reload the cache if requested, using existing data
            should_reload_cache = update.reload_cache
        
        # Reload cache if requested
        if should_reload_cache:
            logger.info("Reloading MCP tools cache")
            try:
                # If endpoint_prefix has changed, we should update all tools in the database
                # to apply the new prefix to their endpoints
                if endpoint_prefix:
                    try:
                        # Find all tools in the database
                        all_tools = db.query(MCPTool).all()
                        update_count = 0
                        
                        for tool in all_tools:
                            # Get the original endpoint without any previous prefix
                            original_endpoint = tool.endpoint
                            
                            # Skip if the endpoint already has the correct prefix
                            if original_endpoint.startswith(endpoint_prefix):
                                continue
                            
                            # Check if there might be an old prefix we need to remove
                            clean_endpoint = original_endpoint
                            try:
                                # Find the last component of the path which is likely the original endpoint
                                path_parts = original_endpoint.rstrip('/').split('/')
                                if len(path_parts) > 1:
                                    # The last part is likely the original tool name
                                    clean_endpoint = f"/{path_parts[-1]}"
                            except Exception as parse_error:
                                logger.warning(f"Error parsing endpoint for prefix removal: {str(parse_error)}")
                            
                            # Apply the new prefix
                            new_endpoint = f"{endpoint_prefix.rstrip('/')}/{clean_endpoint.lstrip('/')}"
                            tool.endpoint = new_endpoint
                            update_count += 1
                            logger.info(f"Updated endpoint for tool {tool.name}: {original_endpoint} -> {new_endpoint}")
                        
                        if update_count > 0:
                            logger.info(f"Updated {update_count} tool endpoints with new prefix: {endpoint_prefix}")
                            # Explicitly commit the changes
                            try:
                                db.commit()
                                logger.info("Successfully committed tool endpoint updates")
                            except Exception as commit_error:
                                logger.error(f"Error committing tool endpoint updates: {str(commit_error)}")
                                db.rollback()
                                raise commit_error
                    
                    except Exception as tool_update_error:
                        logger.error(f"Error updating tool endpoints with new prefix: {str(tool_update_error)}")
                        db.rollback()
                
                reload_enabled_mcp_tools()
                logger.info("Successfully reloaded MCP tools cache")
            except Exception as cache_error:
                logger.error(f"Error reloading MCP tools cache: {str(cache_error)}")
            
    except Exception as e:
        logger.error(f"Error processing MCP settings update: {str(e)}")
        # Check if the error is related to the api_key column not existing
        if "column mcp_manifests.api_key does not exist" in str(e):
            logger.error("The api_key column is missing from the mcp_manifests table")
            raise HTTPException(
                status_code=500, 
                detail="The api_key column is missing from the mcp_manifests table. Please run the migration script at jarvis/scripts/add_api_key_column.py to add it."
            )
        
        # Don't roll back the main transaction - the settings should be saved
        # But we won't continue with manifests/tools
        logger.error("Main settings were saved, but manifest/tools processing failed")
        
        # Need to go back and remove the API key from settings table, as it should only be in manifest
        try:
            settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
            if settings_row and settings_row.settings and 'api_key' in settings_row.settings:
                logger.info("Removing API key from settings table after error")
                current_settings = settings_row.settings
                del current_settings['api_key']
                settings_row.settings = current_settings
                db.commit()
                logger.info("Successfully removed API key from settings table")
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up API key from settings: {str(cleanup_error)}")
            
        # Use proper imports for SQLAlchemy error types
        if isinstance(e, (ProgrammingError, DatabaseError)):
            # If we get a database schema error (like missing column)
            if "column mcp_manifests.api_key does not exist" in str(e):
                raise HTTPException(
                    status_code=500, 
                    detail="The api_key column is missing from the mcp_manifests table. "
                           "Please run database migrations or reinitialize MCP tables."
                )
            else:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    logger.info("Completed MCP settings update")

@router.get("/mock-manifest")
def get_mock_manifest():
    """Return a mock manifest for testing purposes"""
    logger.info("Serving mock manifest")
    return {
        "name": "Mock MCP Manifest",
        "description": "A mock manifest for testing MCP tool integration",
        "tools": [
            {
                "name": "get_datetime",
                "description": "Get the current date and time",
                "endpoint": "/get_datetime",
                "method": "GET",
                "parameters": {}
            },
            {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "endpoint": "/get_weather",
                "method": "POST",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or location name"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform a calculation",
                "endpoint": "/calculate",
                "method": "POST",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
    }

@router.post("/insert-test-mcp-data")
def insert_test_mcp_data(db: Session = Depends(get_db)):
    """Insert test data directly into MCP tables for debugging"""
    try:
        logger.info("Inserting test data into MCP tables")
        
        # Create test manifest
        manifest_url = "http://test-manifest.local/manifest"
        manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
        
        if not manifest:
            manifest = MCPManifest(
                url=manifest_url,
                hostname="test-manifest.local",
                content={
                    "name": "Test Manifest",
                    "description": "Test manifest for debugging",
                    "tools": [
                        {
                            "name": "test_datetime",
                            "description": "Test datetime tool",
                            "endpoint": "/test_datetime",
                            "method": "GET"
                        },
                        {
                            "name": "test_calculator",
                            "description": "Test calculator tool",
                            "endpoint": "/test_calculator",
                            "method": "POST"
                        }
                    ]
                }
            )
            db.add(manifest)
            db.commit()
            db.refresh(manifest)
            logger.info(f"Created test manifest with ID: {manifest.id}")
        else:
            logger.info(f"Using existing test manifest with ID: {manifest.id}")
        
        # Create test tools
        tools_data = [
            {
                "name": "test_datetime",
                "description": "Test datetime tool",
                "endpoint": "/test_datetime",
                "method": "GET",
                "parameters": {},
                "headers": {},
                "is_active": True
            },
            {
                "name": "test_calculator",
                "description": "Test calculator tool",
                "endpoint": "/test_calculator",
                "method": "POST",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string"
                        }
                    }
                },
                "headers": {},
                "is_active": True
            }
        ]
        
        created_tools = []
        for tool_data in tools_data:
            tool = db.query(MCPTool).filter(
                MCPTool.name == tool_data["name"],
                MCPTool.manifest_id == manifest.id
            ).first()
            
            if not tool:
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    endpoint=tool_data["endpoint"],
                    method=tool_data["method"],
                    parameters=tool_data["parameters"],
                    headers=tool_data["headers"],
                    is_active=tool_data["is_active"],
                    manifest_id=manifest.id
                )
                db.add(tool)
                created_tools.append(tool_data["name"])
        
        db.commit()
        logger.info(f"Created {len(created_tools)} test tools: {', '.join(created_tools)}")
        
        # Reload MCP tools cache
        reload_enabled_mcp_tools()
        logger.info("Reloaded MCP tools cache")
        
        # Query database to verify
        manifest_count = db.query(MCPManifest).count()
        tool_count = db.query(MCPTool).count()
        
        return {
            "success": True,
            "manifest_count": manifest_count,
            "tool_count": tool_count,
            "manifest_id": manifest.id,
            "created_tools": created_tools
        }
    except Exception as e:
        logger.error(f"Error inserting test data: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to insert test data: {str(e)}")

@router.post("/knowledge-graph/test-connection")
def test_knowledge_graph_connection():
    """Test Neo4j knowledge graph database connection"""
    try:
        # Force reload knowledge graph settings from database
        reload_knowledge_graph_settings()
        
        # Test the connection using current settings
        result = test_neo4j_connection()
        
        if result['success']:
            return {
                "success": True,
                "message": "Neo4j connection test successful",
                "database_info": result.get('database_info', {}),
                "config": result.get('config', {})
            }
        else:
            return {
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "config": result.get('config', {})
            }
            
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {str(e)}")
        return {
            "success": False,
            "error": f"Connection test failed: {str(e)}",
            "config": {}
        }

# Synthesis Prompt Management Endpoints
class TemplateCreate(BaseModel):
    content: str
    description: str
    variables: List[str]
    active: Optional[bool] = True

class TemplateUpdate(BaseModel):
    content: Optional[str] = None
    description: Optional[str] = None
    variables: Optional[List[str]] = None
    active: Optional[bool] = None

class TemplatePreview(BaseModel):
    sample_variables: Optional[Dict[str, Any]] = None

@router.get("/synthesis-prompts/{category}/templates")
def get_templates(category: str, db: Session = Depends(get_db)):
    """Get all templates in a synthesis prompt category"""
    if category not in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    try:
        from app.core.synthesis_prompts_cache import get_synthesis_prompts
        settings = get_synthesis_prompts()
        templates = settings.get(category, {})
        
        return {
            "category": category,
            "templates": templates,
            "count": len(templates)
        }
    except Exception as e:
        logger.error(f"Failed to get templates for {category}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

@router.get("/synthesis-prompts/{category}/templates/{template_name}")
def get_template(category: str, template_name: str, db: Session = Depends(get_db)):
    """Get a specific template"""
    if category not in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    try:
        from app.core.synthesis_prompts_cache import get_synthesis_setting
        template = get_synthesis_setting(category, template_name)
        
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_name} not found")
        
        return {
            "category": category,
            "template_name": template_name,
            "template": template
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {category}.{template_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")

@router.post("/synthesis-prompts/{category}/templates/{template_name}")
def create_template(category: str, template_name: str, template: TemplateCreate, db: Session = Depends(get_db)):
    """Create a new template"""
    if category not in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    try:
        from app.core.synthesis_prompts_cache import create_template
        
        template_data = {
            "content": template.content,
            "description": template.description,
            "variables": template.variables,
            "active": template.active
        }
        
        success, message = create_template(category, template_name, template_data)
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        return {
            "success": True,
            "message": message,
            "category": category,
            "template_name": template_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create template {category}.{template_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")

@router.put("/synthesis-prompts/{category}/templates/{template_name}")
def update_template(category: str, template_name: str, template: TemplateUpdate, db: Session = Depends(get_db)):
    """Update an existing template"""
    if category not in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    try:
        from app.core.synthesis_prompts_cache import get_synthesis_setting, update_synthesis_setting
        
        # Check if template exists
        existing_template = get_synthesis_setting(category, template_name)
        if not existing_template:
            raise HTTPException(status_code=404, detail=f"Template {template_name} not found")
        
        # Update fields that were provided
        updates = []
        if template.content is not None:
            success = update_synthesis_setting(category, template_name, "content", template.content)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to update content")
            updates.append("content")
        
        if template.description is not None:
            success = update_synthesis_setting(category, template_name, "description", template.description)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to update description")
            updates.append("description")
        
        if template.variables is not None:
            success = update_synthesis_setting(category, template_name, "variables", template.variables)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to update variables")
            updates.append("variables")
        
        if template.active is not None:
            success = update_synthesis_setting(category, template_name, "active", template.active)
            if not success:
                raise HTTPException(status_code=400, detail="Failed to update active status")
            updates.append("active")
        
        return {
            "success": True,
            "message": f"Template updated successfully",
            "category": category,
            "template_name": template_name,
            "updated_fields": updates
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update template {category}.{template_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")

@router.delete("/synthesis-prompts/{category}/templates/{template_name}")
def delete_template(category: str, template_name: str, db: Session = Depends(get_db)):
    """Delete a template"""
    if category not in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    try:
        from app.core.synthesis_prompts_cache import delete_template
        
        success, message = delete_template(category, template_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=message)
        
        return {
            "success": True,
            "message": message,
            "category": category,
            "template_name": template_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete template {category}.{template_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")

@router.post("/synthesis-prompts/{category}/templates/{template_name}/preview")
def preview_template(category: str, template_name: str, preview_data: TemplatePreview, db: Session = Depends(get_db)):
    """Preview a template with sample variables"""
    if category not in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    try:
        from app.core.synthesis_prompts_cache import get_template_preview
        
        preview_result, error = get_template_preview(
            category, 
            template_name, 
            preview_data.sample_variables
        )
        
        if error:
            raise HTTPException(status_code=404, detail=error)
        
        return {
            "success": True,
            "category": category,
            "template_name": template_name,
            "preview": preview_result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to preview template {category}.{template_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to preview template: {str(e)}")

@router.post("/synthesis-prompts/{category}/reset-defaults")
def reset_category_defaults(category: str, db: Session = Depends(get_db)):
    """Reset a category to default templates"""
    if category not in ['synthesis_prompts', 'formatting_templates', 'system_behaviors']:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    try:
        from app.core.synthesis_prompts_cache import reset_to_defaults
        
        success, message = reset_to_defaults(category)
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        return {
            "success": True,
            "message": message,
            "category": category
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset {category} to defaults: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset category: {str(e)}")

@router.post("/synthesis-prompts/reset-all-defaults")
def reset_all_defaults(db: Session = Depends(get_db)):
    """Reset all synthesis prompt categories to defaults"""
    try:
        from app.core.synthesis_prompts_cache import reset_to_defaults
        
        success, message = reset_to_defaults()
        
        if not success:
            raise HTTPException(status_code=500, detail=message)
        
        return {
            "success": True,
            "message": message
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset all synthesis prompts to defaults: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to reset all categories: {str(e)}")

@router.post("/synthesis-prompts/validate")
def validate_synthesis_prompts_endpoint(db: Session = Depends(get_db)):
    """Validate all synthesis prompt settings"""
    try:
        from app.core.synthesis_prompts_cache import get_synthesis_prompts, validate_synthesis_prompts
        
        settings = get_synthesis_prompts()
        is_valid, message = validate_synthesis_prompts(settings)
        
        return {
            "valid": is_valid,
            "message": message,
            "categories_count": len(settings),
            "total_templates": sum(len(cat_settings) for cat_settings in settings.values())
        }
    except Exception as e:
        logger.error(f"Failed to validate synthesis prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to validate: {str(e)}")

@router.get("/synthesis-prompts/status")
def get_synthesis_prompts_status(db: Session = Depends(get_db)):
    """Get status and statistics of synthesis prompt system"""
    try:
        from app.core.synthesis_prompts_cache import get_synthesis_prompts, validate_synthesis_prompts
        
        settings = get_synthesis_prompts()
        is_valid, validation_message = validate_synthesis_prompts(settings)
        
        # Count templates by category
        template_counts = {}
        total_templates = 0
        active_templates = 0
        
        for category, templates in settings.items():
            template_counts[category] = len(templates)
            total_templates += len(templates)
            
            for template in templates.values():
                if template.get('active', True):
                    active_templates += 1
        
        return {
            "status": "healthy" if is_valid else "validation_errors",
            "validation": {
                "valid": is_valid,
                "message": validation_message
            },
            "statistics": {
                "total_categories": len(settings),
                "total_templates": total_templates,
                "active_templates": active_templates,
                "template_counts_by_category": template_counts
            },
            "categories": list(settings.keys())
        }
    except Exception as e:
        logger.error(f"Failed to get synthesis prompts status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")
