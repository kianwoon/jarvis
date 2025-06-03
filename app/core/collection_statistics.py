"""
Collection Statistics Management

This module handles updating and retrieving statistics for vector collections.
"""

from typing import Dict, Any, Optional
from app.core.db import SessionLocal, CollectionStatistics
from pymilvus import Collection, connections, utility
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def update_collection_statistics(
    collection_name: str, 
    chunks_added: int = 0,
    uri: str = None,
    token: str = None
) -> Dict[str, Any]:
    """
    Update statistics for a collection after document upload
    
    Args:
        collection_name: Name of the collection
        chunks_added: Number of chunks added in this operation
        uri: Milvus URI
        token: Milvus token
    
    Returns:
        Updated statistics
    """
    db = SessionLocal()
    try:
        # Get or create statistics record
        stats = db.query(CollectionStatistics).filter(
            CollectionStatistics.collection_name == collection_name
        ).first()
        
        if not stats:
            stats = CollectionStatistics(
                collection_name=collection_name,
                document_count=0,
                total_chunks=0,
                storage_size_mb=0.0
            )
            db.add(stats)
        
        # Update chunk count
        stats.total_chunks = (stats.total_chunks or 0) + chunks_added
        
        # Try to get real-time stats from Milvus
        if uri and token:
            try:
                connections.connect(alias="stats_connection", uri=uri, token=token)
                
                if utility.has_collection(collection_name, using="stats_connection"):
                    collection = Collection(collection_name, using="stats_connection")
                    collection.load()
                    
                    # Get actual entity count
                    stats.total_chunks = collection.num_entities
                    
                    # Estimate storage size (rough calculation)
                    # Assuming average chunk size of 1.5KB + overhead
                    stats.storage_size_mb = (collection.num_entities * 2.0) / 1024
                    
                    logger.info(f"Updated stats from Milvus for {collection_name}: {collection.num_entities} entities")
                
                connections.disconnect(alias="stats_connection")
            except Exception as e:
                logger.error(f"Failed to get Milvus stats for {collection_name}: {e}")
        
        # Update timestamp
        stats.last_updated = datetime.utcnow()
        
        # Increment document count (rough estimate based on chunks per doc)
        if chunks_added > 0:
            # Estimate ~50 chunks per document on average
            estimated_docs = max(1, chunks_added // 50)
            stats.document_count = (stats.document_count or 0) + estimated_docs
        
        db.commit()
        
        return {
            "collection_name": stats.collection_name,
            "document_count": stats.document_count,
            "total_chunks": stats.total_chunks,
            "storage_size_mb": stats.storage_size_mb,
            "last_updated": stats.last_updated.isoformat() if stats.last_updated else None
        }
        
    except Exception as e:
        logger.error(f"Failed to update collection statistics: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def get_collection_statistics(collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a specific collection
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Statistics dict or None if not found
    """
    db = SessionLocal()
    try:
        stats = db.query(CollectionStatistics).filter(
            CollectionStatistics.collection_name == collection_name
        ).first()
        
        if stats:
            return {
                "collection_name": stats.collection_name,
                "document_count": stats.document_count or 0,
                "total_chunks": stats.total_chunks or 0,
                "storage_size_mb": stats.storage_size_mb or 0.0,
                "avg_search_latency_ms": stats.avg_search_latency_ms,
                "last_updated": stats.last_updated.isoformat() if stats.last_updated else None
            }
        return None
    finally:
        db.close()

def refresh_all_collection_statistics(uri: str, token: str) -> Dict[str, Dict[str, Any]]:
    """
    Refresh statistics for all collections from Milvus
    
    Args:
        uri: Milvus URI
        token: Milvus token
        
    Returns:
        Dict of collection_name -> statistics
    """
    db = SessionLocal()
    results = {}
    
    try:
        connections.connect(alias="refresh_stats", uri=uri, token=token)
        
        # Get all collections from Milvus
        collection_names = utility.list_collections(using="refresh_stats")
        logger.info(f"Found {len(collection_names)} collections in Milvus")
        
        for collection_name in collection_names:
            try:
                collection = Collection(collection_name, using="refresh_stats")
                collection.load()
                
                # Get or create statistics record
                stats = db.query(CollectionStatistics).filter(
                    CollectionStatistics.collection_name == collection_name
                ).first()
                
                if not stats:
                    stats = CollectionStatistics(collection_name=collection_name)
                    db.add(stats)
                
                # Update with real data
                stats.total_chunks = collection.num_entities
                stats.storage_size_mb = (collection.num_entities * 2.0) / 1024
                
                # Estimate documents (rough calculation)
                stats.document_count = max(1, collection.num_entities // 50)
                stats.last_updated = datetime.utcnow()
                
                results[collection_name] = {
                    "document_count": stats.document_count,
                    "total_chunks": stats.total_chunks,
                    "storage_size_mb": stats.storage_size_mb
                }
                
                logger.info(f"Updated stats for {collection_name}: {stats.total_chunks} chunks")
                
            except Exception as e:
                logger.error(f"Failed to refresh stats for {collection_name}: {e}")
        
        db.commit()
        connections.disconnect(alias="refresh_stats")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to refresh collection statistics: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def update_search_latency(collection_name: str, latency_ms: int):
    """
    Update average search latency for a collection
    
    Args:
        collection_name: Name of the collection
        latency_ms: Search latency in milliseconds
    """
    db = SessionLocal()
    try:
        stats = db.query(CollectionStatistics).filter(
            CollectionStatistics.collection_name == collection_name
        ).first()
        
        if stats:
            # Calculate running average
            if stats.avg_search_latency_ms:
                # Simple moving average with weight towards recent
                stats.avg_search_latency_ms = int(
                    (stats.avg_search_latency_ms * 0.9) + (latency_ms * 0.1)
                )
            else:
                stats.avg_search_latency_ms = latency_ms
            
            db.commit()
    except Exception as e:
        logger.error(f"Failed to update search latency: {e}")
        db.rollback()
    finally:
        db.close()