"""
Notebook LLM Settings Cache
Manages notebook LLM configuration with Redis caching and database persistence
"""

import json
import logging
from typing import Dict, Optional
from app.core.config import get_settings

logger = logging.getLogger(__name__)

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
    """Get notebook LLM settings from cache or database - NO HARDCODED FALLBACKS"""
    # Try to get from Redis cache first
    redis_client = _get_redis_client()
    if redis_client:
        try:
            cached = redis_client.get(NOTEBOOK_LLM_SETTINGS_KEY)
            if cached:
                settings = json.loads(cached)
                logger.debug(f"Retrieved notebook LLM settings from Redis cache")
                return settings
        except Exception as e:
            logger.warning(f"Redis error retrieving notebook LLM settings: {e}, falling back to database")
    
    # If not cached or Redis failed, load from DB
    return reload_notebook_llm_settings()

def get_notebook_llm_full_config():
    """Get complete notebook LLM configuration"""
    return get_notebook_llm_settings()

def set_notebook_llm_settings(settings_dict):
    """Set notebook LLM settings in cache and database"""
    if not settings_dict:
        logger.error("Cannot set empty notebook LLM settings")
        return False
        
    try:
        # Store in database first
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            # Update or create database entry
            row = db.query(SettingsModel).filter(SettingsModel.category == 'notebook_llm').first()
            if row:
                row.settings = settings_dict
                logger.info("Updated existing notebook LLM settings in database")
            else:
                new_settings = SettingsModel(category='notebook_llm', settings=settings_dict)
                db.add(new_settings)
                logger.info("Created new notebook LLM settings in database")
            
            db.commit()
            
            # Cache in Redis after successful database update
            redis_client = _get_redis_client()
            if redis_client:
                try:
                    from app.core.timeout_settings_cache import get_settings_cache_ttl
                    ttl = get_settings_cache_ttl()
                    redis_client.setex(NOTEBOOK_LLM_SETTINGS_KEY, ttl, json.dumps(settings_dict))
                    logger.debug(f"Cached notebook LLM settings in Redis with TTL {ttl}s")
                except Exception as e:
                    logger.warning(f"Failed to cache notebook LLM settings in Redis: {e}")
            
            return True
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to set notebook LLM settings: {e}")
        return False

def reload_notebook_llm_settings():
    """Reload notebook LLM settings from database and cache them"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, Settings as SettingsModel
        
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'notebook_llm').first()
            if row and row.settings:
                settings = row.settings
                logger.info(f"Loaded notebook LLM settings from database")
                
                # Cache in Redis
                redis_client = _get_redis_client()
                if redis_client:
                    try:
                        from app.core.timeout_settings_cache import get_settings_cache_ttl
                        ttl = get_settings_cache_ttl()
                        redis_client.setex(NOTEBOOK_LLM_SETTINGS_KEY, ttl, json.dumps(settings))
                        logger.debug(f"Cached notebook LLM settings in Redis with TTL {ttl}s")
                    except Exception as e:
                        logger.warning(f"Failed to cache notebook LLM settings in Redis: {e}")
                
                return settings
            else:
                logger.error("No notebook LLM settings found in database and no defaults available")
                logger.error("Notebook LLM settings must be configured via the UI before use")
                raise ValueError("Notebook LLM settings not configured - please configure via Settings UI")
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to load notebook LLM settings from database: {e}")
        raise

def clear_notebook_llm_settings_cache():
    """Clear notebook LLM settings from Redis cache to force reload from database"""
    redis_client = _get_redis_client()
    if redis_client:
        try:
            redis_client.delete(NOTEBOOK_LLM_SETTINGS_KEY)
            logger.info("Cleared notebook LLM settings cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear notebook LLM settings cache: {e}")
            return False
    return False

# Memory deduplication functions preserved for manual execution if needed
def deduplicate_memory_records():
    """Remove duplicate memory records from knowledge_graph_documents table"""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    import os
    
    def get_database_url() -> str:
        return os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/llm_platform')
    
    engine = create_engine(get_database_url())
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        try:
            find_duplicates_query = text("""
                SELECT file_hash, array_agg(document_id ORDER BY created_at) as document_ids,
                       COUNT(*) as count
                FROM knowledge_graph_documents 
                WHERE content_type = 'memory' 
                GROUP BY file_hash 
                HAVING COUNT(*) > 1
            """)
            
            duplicates = db.execute(find_duplicates_query).fetchall()
            
            if not duplicates:
                logger.info("No duplicate memory records found")
                return
            
            logger.info(f"Found {len(duplicates)} sets of duplicate memory records")
            
            for duplicate_set in duplicates:
                file_hash = duplicate_set.file_hash
                document_ids = duplicate_set.document_ids
                count = duplicate_set.count
                
                # Keep the oldest record, delete the rest
                records_to_keep = [document_ids[0]]
                records_to_delete = document_ids[1:]
                
                if records_to_delete:
                    delete_query = text("""
                        DELETE FROM knowledge_graph_documents 
                        WHERE document_id = ANY(:doc_ids) AND content_type = 'memory'
                    """)
                    
                    result = db.execute(delete_query, {'doc_ids': records_to_delete})
                    logger.info(f"Deleted {result.rowcount} duplicate records for hash {file_hash[:12]}...")
                    
            db.commit()
            logger.info("Memory deduplication completed successfully")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error during memory deduplication: {str(e)}")
            raise

def verify_deduplication():
    """Verify that deduplication was successful"""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    import os
    
    def get_database_url() -> str:
        return os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/llm_platform')
    
    engine = create_engine(get_database_url())
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as db:
        find_duplicates_query = text("""
            SELECT file_hash, COUNT(*) as count
            FROM knowledge_graph_documents 
            WHERE content_type = 'memory' 
            GROUP BY file_hash 
            HAVING COUNT(*) > 1
        """)
        
        remaining_duplicates = db.execute(find_duplicates_query).fetchall()
        
        if remaining_duplicates:
            logger.warning(f"Still have {len(remaining_duplicates)} sets of duplicates!")
            for dup in remaining_duplicates:
                logger.warning(f"Hash {dup.file_hash[:12]}... has {dup.count} records")
        else:
            logger.info("No duplicate memory records found - deduplication successful!")