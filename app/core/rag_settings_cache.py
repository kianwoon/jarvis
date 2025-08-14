import json
import logging
from app.core.config import get_settings

logger = logging.getLogger(__name__)

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
    """Get RAG settings from cache or database - NO HARDCODED FALLBACKS"""
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(RAG_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                logger.debug(f"Retrieved RAG settings from Redis cache with {len(settings)} categories")
                return settings
        except Exception as e:
            logger.warning(f"Redis error retrieving RAG settings: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_rag_settings()

def set_rag_settings(settings_dict):
    """Set RAG settings in cache and database"""
    if not settings_dict:
        logger.error("Cannot set empty RAG settings")
        return False
        
    try:
        # Store in database first
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # Update or create database entry
            row = db.query(SettingsModel).filter(SettingsModel.category == 'rag').first()
            if row:
                row.settings = settings_dict
                logger.info("Updated existing RAG settings in database")
            else:
                new_settings = SettingsModel(category='rag', settings=settings_dict)
                db.add(new_settings)
                logger.info("Created new RAG settings in database")
            
            db.commit()
            
            # Cache in Redis after successful database update
            redis_client = _get_redis_client()
            if redis_client:
                try:
                    from app.core.timeout_settings_cache import get_settings_cache_ttl
                    ttl = get_settings_cache_ttl()
                    redis_client.setex(RAG_SETTINGS_KEY, ttl, json.dumps(settings_dict))
                    logger.debug(f"Cached RAG settings in Redis with TTL {ttl}s")
                except Exception as e:
                    logger.warning(f"Failed to cache RAG settings in Redis: {e}")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to set RAG settings: {e}")
        return False

def reload_rag_settings():
    """Reload RAG settings from database and cache them"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'rag').first()
            if row and row.settings:
                settings = row.settings
                logger.info(f"Loaded RAG settings from database with {len(settings)} categories")
                
                # Cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        from app.core.timeout_settings_cache import get_settings_cache_ttl
                        ttl = get_settings_cache_ttl()
                        redis_client.setex(RAG_SETTINGS_KEY, ttl, json.dumps(settings))
                        logger.debug(f"Cached RAG settings in Redis with TTL {ttl}s")
                    except Exception as e:
                        logger.warning(f"Failed to cache RAG settings in Redis: {e}")
                
                return settings
            else:
                logger.error("No RAG settings found in database and no defaults available")
                logger.error("RAG settings must be configured via the UI before use")
                raise ValueError("RAG settings not configured - please configure via Settings UI")
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to load RAG settings from database: {e}")
        raise

def clear_rag_settings_cache():
    """Clear RAG settings from Redis cache to force reload from database"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.delete(RAG_SETTINGS_KEY)
            logger.info("Cleared RAG settings cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear RAG settings cache: {e}")
            return False
    return False

def validate_rag_settings(settings):
    """Validate RAG settings structure and required fields"""
    required_categories = [
        'document_retrieval', 'search_strategy', 'reranking', 
        'bm25_scoring', 'performance', 'agent_settings',
        'query_processing', 'collection_selection'
    ]
    
    if not isinstance(settings, dict):
        return False, "Settings must be a dictionary"
    
    missing_categories = [cat for cat in required_categories if cat not in settings]
    if missing_categories:
        return False, f"Missing required categories: {missing_categories}"
    
    # Validate critical numeric settings exist and are reasonable
    try:
        doc_retrieval = settings['document_retrieval']
        if 'similarity_threshold' not in doc_retrieval or not isinstance(doc_retrieval['similarity_threshold'], (int, float)):
            return False, "similarity_threshold must be a number"
        if 'num_docs_retrieve' not in doc_retrieval or not isinstance(doc_retrieval['num_docs_retrieve'], int):
            return False, "num_docs_retrieve must be an integer"
            
        agent_settings = settings['agent_settings']
        if 'min_relevance_score' not in agent_settings or not isinstance(agent_settings['min_relevance_score'], (int, float)):
            return False, "min_relevance_score must be a number"
            
    except KeyError as e:
        return False, f"Missing required setting: {e}"
    
    return True, "Settings validation passed"

def get_rag_setting(category, key, default=None):
    """Get a specific RAG setting value"""
    try:
        settings = get_rag_settings()
        return settings.get(category, {}).get(key, default)
    except Exception as e:
        logger.error(f"Failed to get RAG setting {category}.{key}: {e}")
        if default is not None:
            return default
        raise

def update_rag_setting(category, key, value):
    """Update a specific RAG setting value"""
    try:
        settings = get_rag_settings()
        if category not in settings:
            settings[category] = {}
        settings[category][key] = value
        
        is_valid, error = validate_rag_settings(settings)
        if not is_valid:
            logger.error(f"Invalid RAG settings after update: {error}")
            return False
            
        return set_rag_settings(settings)
    except Exception as e:
        logger.error(f"Failed to update RAG setting {category}.{key}: {e}")
        return False

# Convenience functions for specific setting categories
def get_document_retrieval_settings():
    """Get document retrieval specific settings"""
    try:
        return get_rag_settings()['document_retrieval']
    except KeyError:
        logger.error("Document retrieval settings not found")
        raise ValueError("Document retrieval settings not configured")

def get_search_strategy_settings():
    """Get search strategy specific settings"""
    try:
        return get_rag_settings()['search_strategy']
    except KeyError:
        logger.error("Search strategy settings not found")
        raise ValueError("Search strategy settings not configured")

def get_reranking_settings():
    """Get reranking specific settings"""
    try:
        return get_rag_settings()['reranking']
    except KeyError:
        logger.error("Reranking settings not found")
        raise ValueError("Reranking settings not configured")

def get_bm25_settings():
    """Get BM25 scoring specific settings"""
    try:
        return get_rag_settings()['bm25_scoring']
    except KeyError:
        logger.error("BM25 settings not found")
        raise ValueError("BM25 settings not configured")

def get_performance_settings():
    """Get performance specific settings"""
    try:
        return get_rag_settings()['performance']
    except KeyError:
        logger.error("Performance settings not found")
        raise ValueError("Performance settings not configured")

def get_agent_settings():
    """Get agent specific settings"""
    try:
        return get_rag_settings()['agent_settings']
    except KeyError:
        logger.error("Agent settings not found")
        raise ValueError("Agent settings not configured")

def get_query_processing_settings():
    """Get query processing specific settings"""
    try:
        return get_rag_settings()['query_processing']
    except KeyError:
        logger.error("Query processing settings not found")
        raise ValueError("Query processing settings not configured")

def get_collection_selection_settings():
    """Get collection selection settings"""
    try:
        return get_rag_settings()['collection_selection']
    except KeyError:
        logger.error("Collection selection settings not found")
        raise ValueError("Collection selection settings not configured")

def ensure_rag_settings_exist():
    """
    Check if RAG settings exist and are valid.
    Returns True if settings exist, False if they need to be configured.
    """
    try:
        settings = get_rag_settings()
        is_valid, error = validate_rag_settings(settings)
        if not is_valid:
            logger.warning(f"RAG settings exist but are invalid: {error}")
            return False
        return True
    except Exception:
        logger.warning("RAG settings do not exist or are inaccessible")
        return False