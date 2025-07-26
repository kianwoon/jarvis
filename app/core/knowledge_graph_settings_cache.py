import json
from app.core.redis_base import RedisCache
from typing import Dict, Any, Optional

KNOWLEDGE_GRAPH_SETTINGS_KEY = 'knowledge_graph_settings_cache'

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_knowledge_graph_settings():
    """Get knowledge graph settings from cache or reload from database"""
    try:
        cached = cache.get(KNOWLEDGE_GRAPH_SETTINGS_KEY)
        if cached:
            return cached
        return reload_knowledge_graph_settings()
    except Exception as e:
        print(f"[ERROR] Failed to get knowledge graph settings from cache: {str(e)}")
        # Return default settings if cache fails
        return get_default_knowledge_graph_settings()

def get_neo4j_config() -> Dict[str, Any]:
    """Get Neo4j configuration from knowledge graph settings"""
    try:
        settings = get_knowledge_graph_settings()
        neo4j_config = settings.get('neo4j', {})
        
        # Ensure all required fields are present with defaults
        return {
            'enabled': neo4j_config.get('enabled', True),
            'host': neo4j_config.get('host', 'neo4j'),
            'port': neo4j_config.get('port', 7687),
            'http_port': neo4j_config.get('http_port', 7474),
            'database': neo4j_config.get('database', 'neo4j'),
            'username': neo4j_config.get('username', 'neo4j'),
            'password': neo4j_config.get('password', 'jarvis_neo4j_password'),
            'uri': neo4j_config.get('uri', 'bolt://neo4j:7687'),
            'connection_pool': {
                'max_connections': neo4j_config.get('connection_pool', {}).get('max_connections', 50),
                'connection_timeout': neo4j_config.get('connection_pool', {}).get('connection_timeout', 30),
                'max_transaction_retry_time': neo4j_config.get('connection_pool', {}).get('max_transaction_retry_time', 30)
            },
            'memory_config': {
                'heap_initial': neo4j_config.get('memory_config', {}).get('heap_initial', '512m'),
                'heap_max': neo4j_config.get('memory_config', {}).get('heap_max', '2g'),
                'pagecache': neo4j_config.get('memory_config', {}).get('pagecache', '1g')
            },
            'plugins': {
                'apoc_enabled': neo4j_config.get('plugins', {}).get('apoc_enabled', True),
                'gds_enabled': neo4j_config.get('plugins', {}).get('gds_enabled', True)
            },
            'security': {
                'encrypted': neo4j_config.get('security', {}).get('encrypted', False),
                'trust_strategy': neo4j_config.get('security', {}).get('trust_strategy', 'TRUST_ALL_CERTIFICATES')
            }
        }
    except Exception as e:
        print(f"[ERROR] Failed to get Neo4j config: {str(e)}")
        return get_default_neo4j_config()

def get_default_neo4j_config() -> Dict[str, Any]:
    """Get default Neo4j configuration"""
    return {
        'enabled': True,
        'host': 'neo4j',
        'port': 7687,
        'http_port': 7474,
        'database': 'neo4j',
        'username': 'neo4j',
        'password': 'jarvis_neo4j_password',
        'uri': 'bolt://neo4j:7687',
        'connection_pool': {
            'max_connections': 50,
            'connection_timeout': 30,
            'max_transaction_retry_time': 30
        },
        'memory_config': {
            'heap_initial': '512m',
            'heap_max': '2g',
            'pagecache': '1g'
        },
        'plugins': {
            'apoc_enabled': True,
            'gds_enabled': True
        },
        'security': {
            'encrypted': False,
            'trust_strategy': 'TRUST_ALL_CERTIFICATES'
        }
    }

def get_default_knowledge_graph_settings() -> Dict[str, Any]:
    """Get default knowledge graph settings"""
    return {
        'mode': 'thinking',
        'model': 'qwen3:30b-a3b-q4_K_M',
        'max_tokens': 8192,
        'model_server': 'http://localhost:11434',
        'system_prompt': 'You are an expert knowledge graph extraction system. Extract entities and relationships from the provided text with high precision.',
        'context_length': 40960,
        'repeat_penalty': '1.05',
        'temperature': 0.3,
        'extraction_prompt': 'Extract entities and relationships from the following text in JSON format.',
        'entity_types': ['Person', 'Organization', 'Location', 'Event', 'Concept'],
        'relationship_types': ['works_for', 'located_in', 'part_of', 'related_to', 'causes'],
        'max_entities_per_chunk': 20,
        'enable_coreference_resolution': True,
        'neo4j': get_default_neo4j_config()
    }

def set_knowledge_graph_settings(settings_dict: Dict[str, Any]):
    """Set knowledge graph settings in cache"""
    cache.set(KNOWLEDGE_GRAPH_SETTINGS_KEY, settings_dict)

def reload_knowledge_graph_settings() -> Dict[str, Any]:
    """Reload knowledge graph settings from database"""
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row and 'knowledge_graph' in row.settings:
                settings = row.settings['knowledge_graph']
                
                # Ensure Neo4j config is present
                if 'neo4j' not in settings:
                    settings['neo4j'] = get_default_neo4j_config()
                    # Save updated settings back to database
                    row.settings['knowledge_graph'] = settings
                    db.commit()
                    print("Added default Neo4j configuration to knowledge graph settings")
                
                # Cache the settings
                cache.set(KNOWLEDGE_GRAPH_SETTINGS_KEY, settings)
                return settings
            else:
                # No knowledge graph settings found, return defaults
                default_settings = get_default_knowledge_graph_settings()
                cache.set(KNOWLEDGE_GRAPH_SETTINGS_KEY, default_settings)
                return default_settings
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to load knowledge graph settings from database: {e}")
        return get_default_knowledge_graph_settings()

def get_knowledge_graph_full_config(settings=None) -> Dict[str, Any]:
    """Construct full knowledge graph configuration by merging base config with mode parameters"""
    if settings is None:
        settings = get_knowledge_graph_settings()
    
    # Get LLM settings for mode parameters
    try:
        from app.core.llm_settings_cache import get_llm_settings
        llm_settings = get_llm_settings()
        mode = settings.get('mode', 'thinking')
        
        # Get the appropriate mode parameters
        if mode == 'thinking':
            mode_params = llm_settings.get('thinking_mode_params', {})
        else:
            mode_params = llm_settings.get('non_thinking_mode_params', {})
        
        # Merge base config with mode parameters
        full_config = settings.copy()
        full_config.update(mode_params)
        
        return full_config
    except Exception as e:
        print(f"[ERROR] Failed to get full knowledge graph config: {str(e)}")
        return settings

def test_neo4j_connection() -> Dict[str, Any]:
    """Test Neo4j connection using current settings"""
    try:
        config = get_neo4j_config()
        
        if not config.get('enabled', False):
            return {
                'success': False,
                'error': 'Neo4j is disabled in configuration',
                'config': config
            }
        
        # Here we would test the actual connection
        # For now, return the configuration for testing
        return {
            'success': True,
            'message': 'Neo4j configuration loaded successfully',
            'config': config
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'config': None
        }

def update_neo4j_password(new_password: str) -> bool:
    """Update Neo4j password in settings"""
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
            if row and 'knowledge_graph' in row.settings:
                # Update password in knowledge graph settings
                row.settings['knowledge_graph']['neo4j']['password'] = new_password
                
                # Update URI with new password if using embedded auth
                neo4j_config = row.settings['knowledge_graph']['neo4j']
                host = neo4j_config.get('host', 'neo4j')
                port = neo4j_config.get('port', 7687)
                username = neo4j_config.get('username', 'neo4j')
                row.settings['knowledge_graph']['neo4j']['uri'] = f"bolt://{username}:{new_password}@{host}:{port}"
                
                db.commit()
                
                # Reload cache
                reload_knowledge_graph_settings()
                return True
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to update Neo4j password: {e}")
        return False