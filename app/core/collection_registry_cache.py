"""
Collection Registry Cache Module

This module provides caching functionality for collection configurations,
including metadata schemas, search configs, and access controls.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from app.core.redis_base import RedisCache
from app.core.db import SessionLocal
from sqlalchemy import text

logger = logging.getLogger(__name__)

class CollectionRegistryCache(RedisCache):
    """Cache for collection configurations and metadata"""
    
    def __init__(self):
        super().__init__(key_prefix="collection_registry:")
        self.cache_ttl = 3600  # 1 hour cache
        
    def _get_cache_key(self, collection_name: str) -> str:
        """Generate cache key for a collection"""
        return collection_name
    
    def _get_list_cache_key(self) -> str:
        """Generate cache key for collection list"""
        return "list"
    
    def get_collection(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection configuration from cache or database"""
        # Try cache first
        cache_key = self._get_cache_key(collection_name)
        cached_data = self.get(cache_key)
        
        if cached_data:
            logger.debug(f"Collection {collection_name} found in cache")
            return cached_data
        
        # Fetch from database
        db = SessionLocal()
        try:
            query = """
                SELECT c.*, 
                       s.document_count, 
                       s.total_chunks, 
                       s.storage_size_mb
                FROM collection_registry c
                LEFT JOIN collection_statistics s ON c.collection_name = s.collection_name
                WHERE c.collection_name = :collection_name
            """
            
            result = db.execute(text(query), {"collection_name": collection_name}).fetchone()
            
            if result:
                collection_data = {
                    "id": result[0],
                    "collection_name": result[1],
                    "collection_type": result[2],
                    "description": result[3],
                    "metadata_schema": result[4] if isinstance(result[4], dict) else (json.loads(result[4]) if result[4] else {}),
                    "search_config": result[5] if isinstance(result[5], dict) else (json.loads(result[5]) if result[5] else {}),
                    "access_config": result[6] if isinstance(result[6], dict) else (json.loads(result[6]) if result[6] else {}),
                    "created_at": result[7].isoformat() if hasattr(result[7], 'isoformat') else str(result[7]),
                    "updated_at": result[8].isoformat() if hasattr(result[8], 'isoformat') else str(result[8]),
                    "statistics": {
                        "document_count": result[9] or 0,
                        "total_chunks": result[10] or 0,
                        "storage_size_mb": result[11] or 0.0
                    }
                }
                
                # Cache the result
                self.set(cache_key, collection_data, self.cache_ttl)
                logger.info(f"Collection {collection_name} cached")
                
                return collection_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching collection {collection_name}: {e}")
            return None
        finally:
            db.close()
    
    def get_all_collections(self) -> List[Dict[str, Any]]:
        """Get all collections from cache or database"""
        # Try cache first
        cache_key = self._get_list_cache_key()
        cached_data = self.get(cache_key)
        
        if cached_data:
            logger.debug("Collection list found in cache")
            return cached_data
        
        # Fetch from database
        db = SessionLocal()
        try:
            query = """
                SELECT c.*, 
                       s.document_count, 
                       s.total_chunks, 
                       s.storage_size_mb
                FROM collection_registry c
                LEFT JOIN collection_statistics s ON c.collection_name = s.collection_name
                ORDER BY c.created_at DESC
            """
            
            results = db.execute(text(query)).fetchall()
            collections = []
            
            for result in results:
                collection_data = {
                    "id": result[0],
                    "collection_name": result[1],
                    "collection_type": result[2],
                    "description": result[3],
                    "metadata_schema": result[4] if isinstance(result[4], dict) else (json.loads(result[4]) if result[4] else {}),
                    "search_config": result[5] if isinstance(result[5], dict) else (json.loads(result[5]) if result[5] else {}),
                    "access_config": result[6] if isinstance(result[6], dict) else (json.loads(result[6]) if result[6] else {}),
                    "created_at": result[7].isoformat() if hasattr(result[7], 'isoformat') else str(result[7]),
                    "updated_at": result[8].isoformat() if hasattr(result[8], 'isoformat') else str(result[8]),
                    "statistics": {
                        "document_count": result[9] or 0,
                        "total_chunks": result[10] or 0,
                        "storage_size_mb": result[11] or 0.0
                    }
                }
                collections.append(collection_data)
                
                # Cache individual collections too
                self.set(
                    self._get_cache_key(collection_data["collection_name"]),
                    collection_data,
                    self.cache_ttl
                )
            
            # Cache the list
            self.set(cache_key, collections, self.cache_ttl)
            logger.info(f"Cached {len(collections)} collections")
            
            return collections
            
        except Exception as e:
            logger.error(f"Error fetching collection list: {e}")
            return []
        finally:
            db.close()
    
    def invalidate_collection(self, collection_name: str):
        """Invalidate cache for a specific collection"""
        try:
            # Delete specific collection cache
            self.delete(self._get_cache_key(collection_name))
            # Delete list cache to force refresh
            self.delete(self._get_list_cache_key())
            logger.info(f"Invalidated cache for collection {collection_name}")
        except Exception as e:
            logger.error(f"Error invalidating cache for collection {collection_name}: {e}")
    
    def invalidate_all(self):
        """Invalidate all collection caches"""
        try:
            # Get all collection names first
            collections = self.get_all_collections()
            
            # Delete all individual collection caches
            for collection in collections:
                self.delete(self._get_cache_key(collection["collection_name"]))
            
            # Delete list cache
            self.delete(self._get_list_cache_key())
            
            logger.info("Invalidated all collection caches")
        except Exception as e:
            logger.error(f"Error invalidating all collection caches: {e}")
    
    def get_collection_by_type(self, collection_type: str) -> List[Dict[str, Any]]:
        """Get all collections of a specific type"""
        all_collections = self.get_all_collections()
        return [c for c in all_collections if c["collection_type"] == collection_type]
    
    def has_access(self, collection_name: str, user_id: str) -> bool:
        """Check if user has access to a collection"""
        collection = self.get_collection(collection_name)
        
        if not collection:
            return False
        
        access_config = collection.get("access_config", {})
        
        # If not restricted, everyone has access
        if not access_config.get("restricted", False):
            return True
        
        # Check if user is in allowed users list
        allowed_users = access_config.get("allowed_users", [])
        return user_id in allowed_users
    
    def get_accessible_collections(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all collections accessible by a user"""
        all_collections = self.get_all_collections()
        accessible = []
        
        for collection in all_collections:
            if self.has_access(collection["collection_name"], user_id):
                accessible.append(collection)
        
        return accessible

# Global instance
_collection_cache = None

def get_collection_cache() -> CollectionRegistryCache:
    """Get or create the global collection cache instance"""
    global _collection_cache
    if _collection_cache is None:
        _collection_cache = CollectionRegistryCache()
    return _collection_cache

# Convenience functions
def get_collection_config(collection_name: str) -> Optional[Dict[str, Any]]:
    """Get collection configuration"""
    cache = get_collection_cache()
    return cache.get_collection(collection_name)

def get_all_collections() -> List[Dict[str, Any]]:
    """Get all collections"""
    cache = get_collection_cache()
    return cache.get_all_collections()

def invalidate_collection_cache(collection_name: str):
    """Invalidate collection cache"""
    cache = get_collection_cache()
    cache.invalidate_collection(collection_name)

def check_collection_access(collection_name: str, user_id: str) -> bool:
    """Check if user has access to collection"""
    cache = get_collection_cache()
    return cache.has_access(collection_name, user_id)