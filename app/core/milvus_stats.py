"""
Direct Milvus Statistics Module

This module gets collection statistics directly from Milvus without caching in local DB.
"""

from typing import Dict, Any, List, Optional
from pymilvus import Collection, connections, utility
import logging
from app.core.vector_db_settings_cache import get_vector_db_settings

logger = logging.getLogger(__name__)

class MilvusStats:
    """Get statistics directly from Milvus"""
    
    @staticmethod
    def get_connection_params() -> tuple:
        """Get Milvus connection parameters"""
        vector_db_settings = get_vector_db_settings()
        milvus_config = vector_db_settings.get('milvus', {})
        uri = milvus_config.get('MILVUS_URI')
        token = milvus_config.get('MILVUS_TOKEN', '')
        return uri, token
    
    @staticmethod
    def get_collection_stats(collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific collection directly from Milvus
        
        Returns:
            Dict with collection statistics or empty dict if error
        """
        try:
            uri, token = MilvusStats.get_connection_params()
            if not uri:
                logger.warning("Milvus URI not configured")
                return {
                    "collection_name": collection_name,
                    "document_count": 0,
                    "total_chunks": 0,
                    "storage_size_mb": 0,
                    "status": "not_connected"
                }
            
            # Connect to Milvus
            connections.connect(alias="stats_conn", uri=uri, token=token)
            
            # Check if collection exists
            if not utility.has_collection(collection_name, using="stats_conn"):
                connections.disconnect(alias="stats_conn")
                return {
                    "collection_name": collection_name,
                    "document_count": 0,
                    "total_chunks": 0,
                    "storage_size_mb": 0,
                    "status": "not_found"
                }
            
            # Get collection
            collection = Collection(collection_name, using="stats_conn")
            collection.load()
            
            # Get statistics
            num_entities = collection.num_entities
            
            # Calculate approximate document count
            # Try to get unique doc_ids if the field exists
            doc_count = 0
            try:
                # Query to get approximate unique document count
                # This is a simplified approach - in production you might want to do this differently
                doc_count = max(1, num_entities // 50) if num_entities > 0 else 0
            except Exception as e:
                logger.debug(f"Could not get exact document count: {e}")
                doc_count = max(1, num_entities // 50) if num_entities > 0 else 0
            
            # Estimate storage size (rough calculation)
            # Assuming average of 2KB per chunk (embeddings + metadata)
            storage_size_mb = (num_entities * 2.0) / 1024 if num_entities > 0 else 0
            
            stats = {
                "collection_name": collection_name,
                "document_count": doc_count,
                "total_chunks": num_entities,
                "storage_size_mb": round(storage_size_mb, 2),
                "status": "active"
            }
            
            connections.disconnect(alias="stats_conn")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for collection {collection_name}: {e}")
            return {
                "collection_name": collection_name,
                "document_count": 0,
                "total_chunks": 0,
                "storage_size_mb": 0,
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    def get_all_collections_stats() -> List[Dict[str, Any]]:
        """
        Get statistics for all collections directly from Milvus
        
        Returns:
            List of collection statistics
        """
        try:
            uri, token = MilvusStats.get_connection_params()
            if not uri:
                logger.warning("Milvus URI not configured")
                return []
            
            # Connect to Milvus
            connections.connect(alias="stats_conn", uri=uri, token=token)
            
            # Get all collections
            collection_names = utility.list_collections(using="stats_conn")
            logger.info(f"Found {len(collection_names)} collections in Milvus")
            
            stats_list = []
            for collection_name in collection_names:
                try:
                    collection = Collection(collection_name, using="stats_conn")
                    collection.load()
                    
                    num_entities = collection.num_entities
                    doc_count = max(1, num_entities // 50) if num_entities > 0 else 0
                    storage_size_mb = (num_entities * 2.0) / 1024 if num_entities > 0 else 0
                    
                    stats_list.append({
                        "collection_name": collection_name,
                        "document_count": doc_count,
                        "total_chunks": num_entities,
                        "storage_size_mb": round(storage_size_mb, 2),
                        "status": "active"
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to get stats for collection {collection_name}: {e}")
                    stats_list.append({
                        "collection_name": collection_name,
                        "document_count": 0,
                        "total_chunks": 0,
                        "storage_size_mb": 0,
                        "status": "error",
                        "error": str(e)
                    })
            
            connections.disconnect(alias="stats_conn")
            return stats_list
            
        except Exception as e:
            logger.error(f"Failed to get all collections stats: {e}")
            return []
    
    @staticmethod
    def collection_exists(collection_name: str) -> bool:
        """Check if a collection exists in Milvus"""
        try:
            uri, token = MilvusStats.get_connection_params()
            if not uri:
                return False
            
            connections.connect(alias="exists_check", uri=uri, token=token)
            exists = utility.has_collection(collection_name, using="exists_check")
            connections.disconnect(alias="exists_check")
            return exists
            
        except Exception as e:
            logger.error(f"Failed to check if collection exists: {e}")
            return False