"""
Radiating Settings Cache Module

Manages radiating coverage system settings with Redis caching and PostgreSQL persistence.
Follows the established pattern from other settings cache modules in the system.
"""

import json
import logging
import os
from app.core.redis_base import RedisCache
from app.core.timeout_settings_cache import get_settings_cache_ttl
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

RADIATING_SETTINGS_KEY = 'radiating_settings_cache'

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_radiating_settings() -> Dict[str, Any]:
    """Get radiating settings from cache or reload from database"""
    try:
        cached = cache.get(RADIATING_SETTINGS_KEY)
        if cached:
            return cached
        return reload_radiating_settings()
    except Exception as e:
        print(f"[ERROR] Failed to get radiating settings from cache: {str(e)}")
        # Return default settings if cache fails
        return get_default_radiating_settings()

def get_radiating_config() -> Dict[str, Any]:
    """Get radiating configuration with all parameters"""
    try:
        settings = get_radiating_settings()
        
        # Ensure all required fields are present with defaults
        return {
            'enabled': settings.get('enabled', True),
            'default_depth': settings.get('default_depth', 3),
            'max_depth': settings.get('max_depth', 5),
            'relevance_threshold': settings.get('relevance_threshold', 0.7),
            'expansion_strategy': settings.get('expansion_strategy', 'adaptive'),
            'cache_ttl': settings.get('cache_ttl', 3600),
            'traversal_strategy': settings.get('traversal_strategy', 'hybrid'),
            'max_entities_per_hop': settings.get('max_entities_per_hop', 10),
            'relationship_weight_threshold': settings.get('relationship_weight_threshold', 0.5),
            'query_expansion': {
                'enabled': settings.get('query_expansion', {}).get('enabled', True),
                'max_expansions': settings.get('query_expansion', {}).get('max_expansions', 5),
                'confidence_threshold': settings.get('query_expansion', {}).get('confidence_threshold', 0.6),
                'preserve_context': settings.get('query_expansion', {}).get('preserve_context', True),
                'expansion_method': settings.get('query_expansion', {}).get('expansion_method', 'semantic')
            },
            'extraction': {
                'entity_confidence_threshold': settings.get('extraction', {}).get('entity_confidence_threshold', 0.4),
                'relationship_confidence_threshold': settings.get('extraction', {}).get('relationship_confidence_threshold', 0.65),
                'enable_universal_discovery': settings.get('extraction', {}).get('enable_universal_discovery', True),
                'max_entities_per_query': settings.get('extraction', {}).get('max_entities_per_query', 50),
                'max_relationships_per_query': settings.get('extraction', {}).get('max_relationships_per_query', 30)
            },
            'performance': {
                'enable_caching': settings.get('performance', {}).get('enable_caching', True),
                'batch_size': settings.get('performance', {}).get('batch_size', 10),
                'parallel_processing': settings.get('performance', {}).get('parallel_processing', True),
                'max_concurrent_queries': settings.get('performance', {}).get('max_concurrent_queries', 5),
                'timeout_seconds': settings.get('performance', {}).get('timeout_seconds', 30)
            }
        }
    except Exception as e:
        print(f"[ERROR] Failed to get radiating config: {str(e)}")
        return get_default_radiating_config()

def get_default_radiating_config() -> Dict[str, Any]:
    """Get default radiating configuration"""
    return {
        'enabled': True,
        'default_depth': 3,
        'max_depth': 5,
        'relevance_threshold': 0.7,
        'expansion_strategy': 'adaptive',
        'cache_ttl': 3600,
        'traversal_strategy': 'hybrid',
        'max_entities_per_hop': 10,
        'relationship_weight_threshold': 0.5,
        'query_expansion': {
            'enabled': True,
            'max_expansions': 5,
            'confidence_threshold': 0.6,
            'preserve_context': True,
            'expansion_method': 'semantic'
        },
        'extraction': {
            'entity_confidence_threshold': 0.4,
            'relationship_confidence_threshold': 0.65,
            'enable_universal_discovery': True,
            'max_entities_per_query': 50,
            'max_relationships_per_query': 30
        },
        'performance': {
            'enable_caching': True,
            'batch_size': 10,
            'parallel_processing': True,
            'max_concurrent_queries': 5,
            'timeout_seconds': 30
        }
    }

def get_default_radiating_settings() -> Dict[str, Any]:
    """Get default radiating settings with comprehensive configuration"""
    import os
    return {
        'enabled': True,
        'default_depth': 3,
        'max_depth': 5,
        'relevance_threshold': 0.7,
        'expansion_strategy': 'adaptive',  # 'adaptive', 'breadth_first', 'depth_first', 'best_first'
        'cache_ttl': 3600,
        'traversal_strategy': 'hybrid',  # 'hybrid', 'graph_only', 'vector_only'
        'max_entities_per_hop': 10,
        'relationship_weight_threshold': 0.5,
        
        # Query expansion settings
        'query_expansion': {
            'enabled': True,
            'max_expansions': 5,
            'confidence_threshold': 0.6,
            'preserve_context': True,
            'expansion_method': 'semantic',  # 'semantic', 'syntactic', 'hybrid'
            'intent_detection': True,
            'domain_hints': True,
            'synonym_expansion': True,
            'concept_expansion': True,
            'temporal_expansion': False,
            'geographic_expansion': False,
            'hierarchical_expansion': True,
            'cross_domain_expansion': False
        },
        
        # Universal extraction settings
        'extraction': {
            'entity_confidence_threshold': 0.4,
            'relationship_confidence_threshold': 0.65,
            'enable_universal_discovery': True,
            'max_entities_per_query': 50,
            'max_relationships_per_query': 30,
            'enable_pattern_detection': True,
            'enable_semantic_inference': True,
            'enable_context_preservation': True,
            'bidirectional_relationships': True,
            'extract_implicit_relationships': False,
            'extract_temporal_context': True,
            'extract_spatial_context': False,
            'extract_causal_relationships': True,
            'extract_hierarchical_relationships': True,
            'extract_part_whole_relationships': True,
            'extract_comparison_relationships': False
        },
        
        # Traversal settings
        'traversal': {
            'strategy': 'hybrid',  # 'breadth_first', 'depth_first', 'best_first', 'hybrid'
            'prioritization': 'relevance',  # 'relevance', 'recency', 'frequency', 'combined'
            'enable_pruning': True,
            'pruning_threshold': 0.4,
            'enable_cycle_detection': True,
            'max_cycles': 2,
            'enable_path_optimization': True,
            'path_weight_decay': 0.8,
            'enable_dynamic_depth': True,
            'depth_adjustment_factor': 0.9,
            'enable_context_switching': True,
            'context_switch_threshold': 0.5
        },
        
        # Coverage optimization
        'coverage': {
            'enable_gap_detection': True,
            'gap_threshold': 0.6,
            'enable_overlap_detection': True,
            'overlap_threshold': 0.8,
            'enable_completeness_checking': True,
            'completeness_threshold': 0.7,
            'enable_redundancy_elimination': True,
            'redundancy_threshold': 0.9,
            'enable_coverage_metrics': True,
            'metric_calculation_interval': 100  # queries
        },
        
        # Result synthesis settings
        'synthesis': {
            'enable_result_merging': True,
            'merge_strategy': 'weighted',  # 'weighted', 'ranked', 'voting', 'consensus'
            'enable_conflict_resolution': True,
            'conflict_strategy': 'confidence',  # 'confidence', 'recency', 'frequency', 'manual'
            'enable_result_ranking': True,
            'ranking_algorithm': 'combined',  # 'relevance', 'confidence', 'combined'
            'enable_result_filtering': True,
            'filter_duplicates': True,
            'filter_low_confidence': True,
            'enable_result_enrichment': True,
            'enrichment_sources': ['knowledge_graph', 'vector_store', 'llm']
        },
        
        # Performance settings
        'performance': {
            'enable_caching': True,
            'cache_strategy': 'lru',  # 'lru', 'lfu', 'ttl', 'adaptive'
            'cache_size': 1000,
            'batch_size': 10,
            'parallel_processing': True,
            'max_concurrent_queries': 5,
            'timeout_seconds': 30,
            'enable_query_optimization': True,
            'enable_index_optimization': True,
            'enable_memory_optimization': True,
            'memory_limit_mb': 512,
            'enable_progressive_loading': True,
            'progressive_chunk_size': 100
        },
        
        # Model configuration
        'model_config': {
            'model': os.environ.get("RADIATING_MODEL", "qwen3:30b-a3b-q4_K_M"),
            'model_server': '',  # Will be populated from LLM settings
            'temperature': 0.3,
            'max_tokens': 4096,  # Reasonable limit for entity extraction
            'context_length': 32768,
            'repeat_penalty': 1.05,
            'top_p': 0.9,
            'top_k': 40,
            'llm_mode': 'non-thinking'  # Default to non-thinking for faster entity extraction
        },
        
        # Monitoring and debugging
        'monitoring': {
            'enable_metrics': True,
            'enable_tracing': False,
            'enable_profiling': False,
            'log_level': 'INFO',
            'metrics_export_interval': 60,  # seconds
            'enable_query_logging': True,
            'enable_result_logging': False,
            'enable_performance_logging': True
        }
    }

def deep_merge_dicts(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge new dict into existing dict"""
    merged = existing.copy()
    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def set_radiating_settings(settings_dict: Dict[str, Any]):
    """Set radiating settings in cache and persist to database"""
    # Always save to cache first
    cache.set(RADIATING_SETTINGS_KEY, settings_dict, expire=get_settings_cache_ttl())
    
    # Then persist to database
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        from datetime import datetime
        
        db = SessionLocal()
        try:
            # Try to find existing radiating category
            radiating_row = db.query(SettingsModel).filter(SettingsModel.category == 'radiating').first()
            if radiating_row:
                # Deep merge to preserve existing settings
                existing_settings = radiating_row.settings or {}
                merged_settings = deep_merge_dicts(existing_settings, settings_dict)
                radiating_row.settings = merged_settings
                radiating_row.updated_at = datetime.now()
            else:
                # Create new radiating settings row
                radiating_row = SettingsModel(
                    category='radiating', 
                    settings=settings_dict,
                    updated_at=datetime.now()
                )
                db.add(radiating_row)
            
            db.commit()
            print("Radiating settings persisted to database successfully")
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to persist radiating settings to database: {e}")
        # Still return success since cache worked, but log the error

def reload_radiating_settings() -> Dict[str, Any]:
    """Reload radiating settings from database"""
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # Try the radiating category
            radiating_row = db.query(SettingsModel).filter(SettingsModel.category == 'radiating').first()
            if radiating_row and radiating_row.settings:
                settings = radiating_row.settings
                
                # Cache the settings
                cache.set(RADIATING_SETTINGS_KEY, settings, expire=get_settings_cache_ttl())
                return settings
            
            # No existing settings found, use defaults  
            default_settings = get_default_radiating_settings()
            cache.set(RADIATING_SETTINGS_KEY, default_settings, expire=get_settings_cache_ttl())
            print("Using default radiating settings")
            return default_settings
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to load radiating settings from database: {e}")
        print("Using default radiating settings")
        return get_default_radiating_settings()

def get_query_expansion_config() -> Dict[str, Any]:
    """Get query expansion specific configuration"""
    settings = get_radiating_settings()
    return settings.get('query_expansion', {
        'enabled': True,
        'max_expansions': 5,
        'confidence_threshold': 0.6,
        'preserve_context': True,
        'expansion_method': 'semantic'
    })

def get_model_config() -> Dict[str, Any]:
    """Get model configuration for radiating LLM operations"""
    try:
        settings = get_radiating_settings()
        model_config = settings.get('model_config', {})
        
        # Get model server from config or LLM settings as fallback
        model_server = model_config.get('model_server', '')
        if not model_server:
            # Try to get from main LLM settings
            from app.core.llm_settings_cache import get_main_llm_full_config
            main_llm_config = get_main_llm_full_config()
            model_server = main_llm_config.get('model_server', '')
        
        if not model_server:
            # Use environment variable as last resort
            import os
            model_server = os.getenv('OLLAMA_BASE_URL', '')
        
        if not model_server:
            raise ValueError("Model server URL must be configured in radiating or LLM settings")
        
        # Ensure all required fields with defaults
        # Force max_tokens to be reasonable for entity extraction
        max_tokens = model_config.get('max_tokens', 4096)
        # Cap max_tokens at 4096 for entity extraction to prevent timeouts
        if max_tokens > 4096:
            logger.info(f"Capping max_tokens from {max_tokens} to 4096 for entity extraction")
            max_tokens = 4096
            
        return {
            'model': model_config.get('model', 'llama3.1:8b'),
            'max_tokens': max_tokens,
            'temperature': model_config.get('temperature', 0.7),
            'context_length': model_config.get('context_length', 128000),
            'model_server': model_server,
            'system_prompt': model_config.get('system_prompt', ''),
            'llm_mode': model_config.get('llm_mode', 'non-thinking')  # Default to non-thinking for faster extraction
        }
    except Exception as e:
        print(f"[ERROR] Failed to get model config: {str(e)}")
        # Try to get from main LLM settings as fallback
        try:
            from app.core.llm_settings_cache import get_main_llm_full_config
            main_llm_config = get_main_llm_full_config()
            model_server = main_llm_config.get('model_server', '')
            if model_server:
                return {
                    'model': 'llama3.1:8b',
                    'max_tokens': 4096,
                    'temperature': 0.7,
                    'context_length': 128000,
                    'model_server': model_server,
                    'system_prompt': '',
                    'llm_mode': 'non-thinking'
                }
        except:
            pass
        
        # Last resort - use environment variable
        import os
        model_server = os.getenv('OLLAMA_BASE_URL', '')
        if not model_server:
            raise ValueError("Model server URL must be configured")
        
        return {
            'model': 'llama3.1:8b',
            'max_tokens': 4096,
            'temperature': 0.7,
            'context_length': 128000,
            'model_server': model_server,
            'system_prompt': '',
            'llm_mode': 'non-thinking'
        }

def get_extraction_config() -> Dict[str, Any]:
    """Get extraction specific configuration"""
    settings = get_radiating_settings()
    return settings.get('extraction', {
        'entity_confidence_threshold': 0.4,
        'relationship_confidence_threshold': 0.65,
        'enable_universal_discovery': True,
        'max_entities_per_query': 50,
        'max_relationships_per_query': 30
    })

def get_traversal_config() -> Dict[str, Any]:
    """Get traversal specific configuration"""
    settings = get_radiating_settings()
    return settings.get('traversal', {
        'strategy': 'hybrid',
        'prioritization': 'relevance',
        'enable_pruning': True,
        'pruning_threshold': 0.4
    })

def is_radiating_enabled() -> bool:
    """Check if radiating system is enabled"""
    settings = get_radiating_settings()
    return settings.get('enabled', True)

def get_radiating_depth() -> int:
    """Get the default radiating depth"""
    settings = get_radiating_settings()
    return settings.get('default_depth', 3)

def get_max_radiating_depth() -> int:
    """Get the maximum allowed radiating depth"""
    settings = get_radiating_settings()
    return settings.get('max_depth', 5)

def get_radiating_prompts() -> Dict[str, Any]:
    """Get all radiating prompts from settings"""
    settings = get_radiating_settings()
    return settings.get('prompts', get_default_radiating_prompts())

def get_entity_extraction_prompts() -> Dict[str, str]:
    """Get entity extraction prompts"""
    prompts = get_radiating_prompts()
    return prompts.get('entity_extraction', {})

def get_relationship_discovery_prompts() -> Dict[str, str]:
    """Get relationship discovery prompts"""
    prompts = get_radiating_prompts()
    return prompts.get('relationship_discovery', {})

def get_query_analysis_prompts() -> Dict[str, str]:
    """Get query analysis prompts"""
    prompts = get_radiating_prompts()
    return prompts.get('query_analysis', {})

def get_expansion_strategy_prompts() -> Dict[str, str]:
    """Get expansion strategy prompts"""
    prompts = get_radiating_prompts()
    return prompts.get('expansion_strategy', {})

def get_prompt(category: str, prompt_name: str, default: str = "") -> str:
    """
    Get a specific prompt by category and name.
    
    Args:
        category: Prompt category (entity_extraction, relationship_discovery, etc.)
        prompt_name: Name of the specific prompt
        default: Default value if prompt not found
        
    Returns:
        The prompt string or default if not found
    """
    prompts = get_radiating_prompts()
    category_prompts = prompts.get(category, {})
    return category_prompts.get(prompt_name, default)

def update_prompt(category: str, prompt_name: str, prompt_text: str):
    """
    Update a specific prompt in the settings.
    
    Args:
        category: Prompt category
        prompt_name: Name of the specific prompt
        prompt_text: New prompt text
    """
    settings = get_radiating_settings()
    
    # Ensure prompts structure exists
    if 'prompts' not in settings:
        settings['prompts'] = {}
    if category not in settings['prompts']:
        settings['prompts'][category] = {}
    
    # Update the specific prompt
    settings['prompts'][category][prompt_name] = prompt_text
    
    # Save the updated settings
    set_radiating_settings(settings)

def reload_radiating_prompts():
    """Force reload of radiating prompts from database"""
    # Clear cache and reload from database
    cache.delete(RADIATING_SETTINGS_KEY)
    return reload_radiating_settings()

def get_default_radiating_prompts() -> Dict[str, Any]:
    """Get default radiating prompts as fallback"""
    return {
        "entity_extraction": {
            "discovery_comprehensive": "You are analyzing a query about MODERN LLM-ERA technologies...",
            "discovery_regular": "Analyze this text and identify the types of entities present...",
            "extraction_comprehensive": "You are an expert on MODERN LLM-ERA technologies...",
            "extraction_regular": "Extract all important entities from this text..."
        },
        "relationship_discovery": {
            "llm_discovery": "You are an expert in technology, AI, ML, software systems...",
            "relationship_analysis": "Analyze the relationships between these entities...",
            "implicit_relationships": "Analyze these entities and infer implicit relationships..."
        },
        "query_analysis": {
            "entity_extraction": "Extract key entities from the following query...",
            "intent_identification": "Identify the primary intent of this query...",
            "domain_extraction": "Identify the knowledge domains relevant to this query...",
            "temporal_extraction": "Extract temporal context from this query if present..."
        },
        "expansion_strategy": {
            "semantic_expansion": "Find semantically related terms and entities for...",
            "concept_expansion": "Identify related concepts and topics for this query...",
            "hierarchical_expansion": "Find hierarchical relationships for..."
        }
    }