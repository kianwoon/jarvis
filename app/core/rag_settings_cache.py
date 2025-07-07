import json
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
RAG_SETTINGS_KEY = 'rag_settings_cache'

# Don't create connection at import time
r = None

def _get_redis_client():
    """Get Redis client using the centralized redis_client"""
    global r
    if r is None:
        from app.core.redis_client import get_redis_client
        r = get_redis_client()
    return r

def get_rag_settings():
    """Get RAG settings from cache or database"""
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(RAG_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                # Validate structure has main categories
                required_categories = ['document_retrieval', 'search_strategy', 'reranking', 'bm25_scoring', 'performance', 'agent_settings']
                if all(cat in settings for cat in required_categories):
                    # Ensure collection_selection exists (migrate from old collection_selection_rules if needed)
                    if 'collection_selection' not in settings:
                        settings['collection_selection'] = get_default_rag_settings()['collection_selection']
                    
                    # Remove old collection_selection_rules if it exists
                    if 'collection_selection_rules' in settings:
                        del settings['collection_selection_rules']
                    
                    return settings
        except Exception as e:
            print(f"Redis error: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_rag_settings()

def set_rag_settings(settings_dict):
    """Set RAG settings in cache"""
    # Ensure collection_selection exists and remove old collection_selection_rules
    if 'collection_selection' not in settings_dict:
        settings_dict['collection_selection'] = get_default_rag_settings()['collection_selection']
    
    # Remove old collection_selection_rules if it exists
    if 'collection_selection_rules' in settings_dict:
        del settings_dict['collection_selection_rules']
    
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.set(RAG_SETTINGS_KEY, json.dumps(settings_dict))
        except Exception as e:
            print(f"Failed to cache RAG settings in Redis: {e}")

def reload_rag_settings():
    """Reload RAG settings from database and cache them"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'rag').first()
            if row:
                settings = row.settings
                
                # Validate required categories exist
                required_categories = ['document_retrieval', 'search_strategy', 'reranking', 'bm25_scoring', 'performance', 'agent_settings']
                if not all(cat in settings for cat in required_categories):
                    print("RAG settings missing required categories, using defaults")
                    settings = get_default_rag_settings()
                
                # Ensure collection_selection exists (migrate from old collection_selection_rules if needed)
                if 'collection_selection' not in settings:
                    settings['collection_selection'] = get_default_rag_settings()['collection_selection']
                
                # Remove old collection_selection_rules if it exists
                if 'collection_selection_rules' in settings:
                    del settings['collection_selection_rules']
                
                # Try to cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        redis_client.set(RAG_SETTINGS_KEY, json.dumps(settings))
                    except Exception as e:
                        print(f"Failed to cache RAG settings in Redis: {e}")
                
                return settings
            else:
                # No user-defined settings, use defaults
                print('No RAG settings found in database, using defaults')
                default_settings = get_default_rag_settings()
                set_rag_settings(default_settings)
                return default_settings
        finally:
            db.close()
    except Exception as e:
        print(f"Failed to load RAG settings from database: {e}")
        # Return default settings to prevent complete failure
        return get_default_rag_settings()

def get_default_rag_settings():
    """Get default RAG settings with current hardcoded values as defaults"""
    return {
        "document_retrieval": {
            "similarity_threshold": 1.5,
            "num_docs_retrieve": 20,
            "max_documents_mcp": 8,
            "cache_max_size": 100,
            "default_collections": ["default_knowledge"],
            "enable_query_expansion": True
        },
        "search_strategy": {
            "default_max_results": 10,
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
            "hybrid_threshold": 0.7,
            "enable_focused_search": True,
            "top_k_vector_search": 50,
            "search_strategy": "auto"  # auto, semantic, keyword, hybrid
        },
        "reranking": {
            "enable_qwen_reranker": True,
            "rerank_weight": 0.7,
            "num_to_rerank": 20,
            "batch_size": 10,
            "enable_advanced_reranking": True,
            "rerank_threshold": 0.5
        },
        "bm25_scoring": {
            "k1": 1.2,
            "b": 0.75,
            "corpus_batch_size": 1000,
            "enable_bm25": True,
            "bm25_weight": 0.3
        },
        "performance": {
            "execution_timeout_ms": 30000,
            "connection_timeout_s": 30,
            "cache_ttl_hours": 2,
            "max_concurrent_searches": 5,
            "enable_caching": True,
            "vector_search_nprobe": 10
        },
        "agent_settings": {
            "confidence_threshold": 0.6,
            "max_results_per_collection": 10,
            "collection_size_threshold": 0,
            "enable_collection_auto_detection": True,
            "default_query_strategy": "auto",
            "min_relevance_score": 0.25,
            "complex_query_threshold": 0.15
        },
        "query_processing": {
            "enable_query_classification": True,
            "max_query_length": 1000,
            "enable_stop_word_removal": True,
            "enable_stemming": False,
            "query_expansion_methods": ["llm", "synonym"],
            "window_size": 50
        },
        "collection_selection": {
            "enable_llm_selection": True,
            "selection_prompt_template": "Given the following query and available collections, determine which collections are most relevant to search:\n\nQuery: {query}\n\nAvailable Collections:\n{collections}\n\nReturn only the collection names that are relevant, separated by commas.",
            "max_collections": 3,
            "confidence_threshold": 0.7,
            "cache_selections": True,
            "fallback_collections": ["default_knowledge"]
        }
    }

# Convenience functions for specific setting categories
def get_document_retrieval_settings():
    """Get document retrieval specific settings"""
    return get_rag_settings().get('document_retrieval', get_default_rag_settings()['document_retrieval'])

def get_search_strategy_settings():
    """Get search strategy specific settings"""
    return get_rag_settings().get('search_strategy', get_default_rag_settings()['search_strategy'])

def get_reranking_settings():
    """Get reranking specific settings"""
    return get_rag_settings().get('reranking', get_default_rag_settings()['reranking'])

def get_bm25_settings():
    """Get BM25 scoring specific settings"""
    return get_rag_settings().get('bm25_scoring', get_default_rag_settings()['bm25_scoring'])

def get_performance_settings():
    """Get performance specific settings"""
    return get_rag_settings().get('performance', get_default_rag_settings()['performance'])

def get_agent_settings():
    """Get agent specific settings"""
    return get_rag_settings().get('agent_settings', get_default_rag_settings()['agent_settings'])

def get_query_processing_settings():
    """Get query processing specific settings"""
    return get_rag_settings().get('query_processing', get_default_rag_settings()['query_processing'])

def get_collection_selection_settings():
    """Get collection selection settings"""
    return get_rag_settings().get('collection_selection', get_default_rag_settings()['collection_selection'])