from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import logging
from sqlalchemy import text
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from app.core.db import get_db, SessionLocal
from app.core.vector_db_settings_cache import get_vector_db_settings
from app.core.config import get_settings
from app.core.collection_registry_cache import (
    get_collection_cache,
    invalidate_collection_cache
)
from app.core.milvus_stats import MilvusStats
from app.core.collection_initializer import (
    initialize_default_collection,
    get_collection_status
)

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

@router.get("/test")
async def test_collections():
    """Test endpoint to verify collections router is working"""
    return {"message": "Collections API is working"}

@router.get("/test-index")
async def test_index_endpoint():
    """Test endpoint to verify index endpoints are working"""
    try:
        get_milvus_connection()
        all_collections = utility.list_collections()
        return {
            "message": "Index endpoint working",
            "total_collections_in_milvus": len(all_collections),
            "collections": all_collections
        }
    except Exception as e:
        return {
            "message": "Index endpoint working but Milvus error",
            "error": str(e)
        }

# Pydantic models
class CollectionCreate(BaseModel):
    collection_name: str
    collection_type: str
    description: str
    metadata_schema: Dict[str, Any]
    search_config: Dict[str, Any]
    access_config: Dict[str, Any]

class CollectionUpdate(BaseModel):
    collection_type: Optional[str] = None
    description: Optional[str] = None
    metadata_schema: Optional[Dict[str, Any]] = None
    search_config: Optional[Dict[str, Any]] = None
    access_config: Optional[Dict[str, Any]] = None

class CollectionResponse(BaseModel):
    id: int
    collection_name: str
    collection_type: str
    description: str
    metadata_schema: Dict[str, Any]
    search_config: Dict[str, Any]
    access_config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    statistics: Optional[Dict[str, Any]] = None

def get_milvus_connection():
    """Get Milvus connection based on settings"""
    vector_db_settings = get_vector_db_settings()
    if vector_db_settings.get('active') != 'milvus':
        raise HTTPException(status_code=400, detail="Milvus is not the active vector database")
    
    milvus_config = vector_db_settings.get('milvus', {})
    try:
        # Check if using cloud Milvus (Zilliz) with URI and token
        milvus_uri = milvus_config.get('MILVUS_URI')
        milvus_token = milvus_config.get('MILVUS_TOKEN')
        
        if milvus_uri and milvus_token:
            # Cloud Milvus connection
            connections.connect(
                alias="default",
                uri=milvus_uri,
                token=milvus_token,
                timeout=10
            )
        else:
            # Local Milvus connection
            connections.connect(
                alias="default",
                host=milvus_config.get('host', 'localhost'),
                port=milvus_config.get('port', 19530),
                timeout=5
            )
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to vector database")

def create_milvus_collection(collection_name: str, metadata_schema: Dict[str, Any]):
    """Create a new Milvus collection with the specified schema"""
    try:
        # Define the schema fields
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=2560),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="uploaded_at", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            # BM25 fields
            FieldSchema(name="bm25_tokens", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="bm25_term_count", dtype=DataType.INT64),
            FieldSchema(name="bm25_unique_terms", dtype=DataType.INT64),
            FieldSchema(name="bm25_top_terms", dtype=DataType.VARCHAR, max_length=1000),
            # Date metadata fields
            FieldSchema(name="creation_date", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="last_modified_date", dtype=DataType.VARCHAR, max_length=100),
        ]
        
        # FIXED: Do NOT add custom fields from metadata_schema to Milvus
        # All collections MUST use the same unified 15-field schema
        # Custom metadata fields stay in PostgreSQL only, not in Milvus
        
        # Create collection schema
        schema = CollectionSchema(
            fields=fields,
            description=f"Collection: {collection_name}"
        )
        
        # Create the collection
        collection = Collection(
            name=collection_name,
            schema=schema,
            consistency_level="Strong"
        )
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256}
        }
        collection.create_index(field_name="vector", index_params=index_params)
        
        # Load collection
        collection.load()
        
        return True
    except Exception as e:
        logger.error(f"Failed to create Milvus collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")

@router.get("/", response_model=List[CollectionResponse])
async def list_collections(db=Depends(get_db), include_stats: bool = False):
    """List all collections with their configurations and statistics"""
    try:
        # Check if tables exist first in PostgreSQL
        try:
            # Check if table exists in PostgreSQL
            result = db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'collection_registry'
                )
            """)).fetchone()
            
            if not result or not result[0]:
                logger.info("Collection tables don't exist yet in PostgreSQL")
                return []
        except Exception as e:
            logger.error(f"Error checking for collection tables: {e}")
            return []
            
        # Get collections from database
        query = """
            SELECT id, collection_name, collection_type, description, 
                   metadata_schema, search_config, access_config, 
                   created_at, updated_at
            FROM collection_registry 
            ORDER BY created_at DESC
        """
        result = db.execute(text(query)).fetchall()
        
        collections = []
        
        # Only fetch stats if requested
        milvus_stats = {}
        if include_stats:
            try:
                milvus_stats = {stat['collection_name']: stat for stat in MilvusStats.get_all_collections_stats()}
            except Exception as e:
                logger.warning(f"Failed to fetch Milvus stats: {e}")
        
        for row in result:
            collection_name = row[1]
            
            collection_data = {
                "id": row[0],
                "collection_name": collection_name,
                "collection_type": row[2],
                "description": row[3],
                "metadata_schema": row[4] if isinstance(row[4], dict) else (json.loads(row[4]) if row[4] else {}),
                "search_config": row[5] if isinstance(row[5], dict) else (json.loads(row[5]) if row[5] else {}),
                "access_config": row[6] if isinstance(row[6], dict) else (json.loads(row[6]) if row[6] else {}),
                "created_at": row[7],
                "updated_at": row[8]
            }
            
            # Add statistics only if requested
            if include_stats:
                stats = milvus_stats.get(collection_name, {
                    "document_count": 0,
                    "total_chunks": 0,
                    "storage_size_mb": 0.0,
                    "status": "not_fetched"
                })
                collection_data["statistics"] = {
                    "document_count": stats.get('document_count', 0),
                    "total_chunks": stats.get('total_chunks', 0),
                    "storage_size_mb": stats.get('storage_size_mb', 0.0),
                    "status": stats.get('status', 'unknown')
                }
            else:
                # Return placeholder stats when not fetching
                collection_data["statistics"] = {
                    "document_count": 0,
                    "total_chunks": 0,
                    "storage_size_mb": 0.0,
                    "status": "not_loaded"
                }
            collections.append(CollectionResponse(**collection_data))
        
        return collections
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")

@router.get("/cache")
async def get_collections_cache():
    """
    Get all collections from Redis cache with database fallback.
    
    This endpoint provides faster access to collection data by using
    Redis cache when available, falling back to PostgreSQL when needed.
    
    Returns:
        List of collections with their configurations and statistics
    """
    try:
        from app.core.collection_registry_cache import get_all_collections
        collections = get_all_collections()
        return collections
    except Exception as e:
        logger.error(f"Failed to get collections from cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collections from cache: {str(e)}")

@router.post("/sync-milvus")
async def sync_collections_to_milvus(db=Depends(get_db)):
    """
    Synchronize all PostgreSQL collections to Milvus.
    
    This endpoint creates missing Milvus collections for all collections
    that exist in PostgreSQL but not in Milvus.
    
    Returns:
        Dict with synchronization results
    """
    try:
        # Get all collections from PostgreSQL
        query = """
            SELECT collection_name, metadata_schema 
            FROM collection_registry 
            ORDER BY created_at ASC
        """
        result = db.execute(text(query)).fetchall()
        
        sync_results = {
            "total_collections": len(result),
            "created": [],
            "already_existed": [],
            "failed": [],
            "milvus_available": False
        }
        
        # Try to connect to Milvus
        try:
            get_milvus_connection()
            sync_results["milvus_available"] = True
        except Exception as e:
            logger.error(f"Milvus not available for sync: {e}")
            raise HTTPException(status_code=500, detail="Milvus vector database not available")
        
        # Create each collection in Milvus
        for row in result:
            collection_name = row[0]
            metadata_schema = row[1] if isinstance(row[1], dict) else (json.loads(row[1]) if row[1] else {})
            
            try:
                # Check if collection already exists in Milvus
                if utility.has_collection(collection_name):
                    sync_results["already_existed"].append(collection_name)
                    logger.info(f"Collection {collection_name} already exists in Milvus")
                    continue
                
                # Create the collection in Milvus
                success = create_milvus_collection(collection_name, metadata_schema)
                if success:
                    sync_results["created"].append(collection_name)
                    logger.info(f"Successfully created Milvus collection: {collection_name}")
                else:
                    sync_results["failed"].append(collection_name)
                    logger.error(f"Failed to create Milvus collection: {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")
                sync_results["failed"].append(collection_name)
        
        # Return results
        return {
            "message": f"Sync completed: {len(sync_results['created'])} created, {len(sync_results['already_existed'])} existed, {len(sync_results['failed'])} failed",
            "results": sync_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to sync collections to Milvus: {e}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.get("/index-summary")
async def get_all_collections_index_summary():
    """Get index type summary for all collections in Milvus"""
    try:
        # Connect to Milvus
        get_milvus_connection()
        
        # Get all collections from Milvus
        all_collections = utility.list_collections()
        
        index_summary = {
            "total_collections": len(all_collections),
            "hnsw_collections": [],
            "ivf_flat_collections": [],
            "other_index_collections": [],
            "no_index_collections": [],
            "error_collections": []
        }
        
        for collection_name in all_collections:
            try:
                collection = Collection(collection_name)
                indexes = collection.indexes
                
                if not indexes:
                    index_summary["no_index_collections"].append(collection_name)
                    continue
                
                # Find vector field index
                vector_index = None
                for index in indexes:
                    if index.field_name == "vector":
                        vector_index = index
                        break
                
                if vector_index:
                    index_type = vector_index.params.get("index_type", "unknown")
                    if index_type == "HNSW":
                        index_summary["hnsw_collections"].append({
                            "collection_name": collection_name,
                            "index_type": index_type,
                            "metric_type": vector_index.params.get("metric_type", "unknown"),
                            "params": vector_index.params.get("params", {})
                        })
                    elif index_type == "IVF_FLAT":
                        index_summary["ivf_flat_collections"].append({
                            "collection_name": collection_name,
                            "index_type": index_type,
                            "metric_type": vector_index.params.get("metric_type", "unknown"),
                            "params": vector_index.params.get("params", {})
                        })
                    else:
                        index_summary["other_index_collections"].append({
                            "collection_name": collection_name,
                            "index_type": index_type,
                            "metric_type": vector_index.params.get("metric_type", "unknown"),
                            "params": vector_index.params.get("params", {})
                        })
                else:
                    index_summary["no_index_collections"].append(collection_name)
                    
            except Exception as e:
                logger.error(f"Error checking index for collection {collection_name}: {e}")
                index_summary["error_collections"].append({
                    "collection_name": collection_name,
                    "error": str(e)
                })
        
        # Add summary counts
        index_summary["summary"] = {
            "hnsw_count": len(index_summary["hnsw_collections"]),
            "ivf_flat_count": len(index_summary["ivf_flat_collections"]),
            "other_index_count": len(index_summary["other_index_collections"]),
            "no_index_count": len(index_summary["no_index_collections"]),
            "error_count": len(index_summary["error_collections"]),
            "hnsw_percentage": (len(index_summary["hnsw_collections"]) / len(all_collections) * 100) if all_collections else 0
        }
        
        return index_summary
        
    except Exception as e:
        logger.error(f"Failed to get index summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get index summary: {str(e)}")

@router.get("/status")
async def get_collections_status():
    """
    Get the current status of the collections system.
    
    This endpoint provides information about:
    - Whether database tables exist
    - Whether the default collection exists
    - Milvus availability
    - Total collection count
    
    Returns:
        Dict with system status information
    """
    try:
        status = get_collection_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get collections status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@router.get("/{collection_name}", response_model=CollectionResponse)
async def get_collection(collection_name: str, db=Depends(get_db), include_stats: bool = False):
    """Get a specific collection by name"""
    try:
        query = """
            SELECT id, collection_name, collection_type, description, 
                   metadata_schema, search_config, access_config, 
                   created_at, updated_at
            FROM collection_registry 
            WHERE collection_name = :collection_name
        """
        result = db.execute(text(query), {"collection_name": collection_name}).fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        collection_data = {
            "id": result[0],
            "collection_name": result[1],
            "collection_type": result[2],
            "description": result[3],
            "metadata_schema": result[4] if isinstance(result[4], dict) else (json.loads(result[4]) if result[4] else {}),
            "search_config": result[5] if isinstance(result[5], dict) else (json.loads(result[5]) if result[5] else {}),
            "access_config": result[6] if isinstance(result[6], dict) else (json.loads(result[6]) if result[6] else {}),
            "created_at": result[7],
            "updated_at": result[8]
        }
        
        # Get stats only if requested
        if include_stats:
            stats = MilvusStats.get_collection_stats(collection_name)
            collection_data["statistics"] = {
                "document_count": stats.get('document_count', 0),
                "total_chunks": stats.get('total_chunks', 0),
                "storage_size_mb": stats.get('storage_size_mb', 0.0),
                "status": stats.get('status', 'unknown')
            }
        else:
            collection_data["statistics"] = {
                "document_count": 0,
                "total_chunks": 0,
                "storage_size_mb": 0.0,
                "status": "not_loaded"
            }
        
        return CollectionResponse(**collection_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collection")

@router.post("/", response_model=CollectionResponse)
async def create_collection(collection: CollectionCreate, db=Depends(get_db)):
    """Create a new collection"""
    try:
        # Check if collection already exists
        existing = db.execute(
            text("SELECT id FROM collection_registry WHERE collection_name = :name"),
            {"name": collection.collection_name}
        ).fetchone()
        
        if existing:
            raise HTTPException(status_code=400, detail="Collection already exists")
        
        # Validate collection name for Milvus
        if not collection.collection_name or not collection.collection_name.replace('_', '').replace('-', '').isalnum():
            raise HTTPException(status_code=400, detail="Collection name must contain only letters, numbers, hyphens, and underscores")
        
        # Try to connect to Milvus and create collection
        try:
            get_milvus_connection()
            
            # Check if Milvus collection exists
            if utility.has_collection(collection.collection_name):
                raise HTTPException(status_code=400, detail="Collection already exists in vector database")
            
            # Create Milvus collection
            create_milvus_collection(collection.collection_name, collection.metadata_schema)
        except HTTPException as e:
            # If it's a connection error, log it but continue
            # The collection registry can still work without Milvus
            if "Failed to connect to vector database" in str(e.detail):
                logger.warning(f"Milvus not available, creating collection registry entry only: {e.detail}")
            else:
                raise
        
        # Insert into database
        query = """
            INSERT INTO collection_registry 
            (collection_name, collection_type, description, metadata_schema, 
             search_config, access_config, created_at, updated_at)
            VALUES (:collection_name, :collection_type, :description, :metadata_schema, 
                    :search_config, :access_config, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        
        db.execute(text(query), {
            "collection_name": collection.collection_name,
            "collection_type": collection.collection_type,
            "description": collection.description,
            "metadata_schema": json.dumps(collection.metadata_schema),
            "search_config": json.dumps(collection.search_config),
            "access_config": json.dumps(collection.access_config)
        })
        db.commit()
        
        # Initialize statistics
        stats_query = """
            INSERT INTO collection_statistics 
            (collection_name, document_count, total_chunks, storage_size_mb, last_updated)
            VALUES (:collection_name, 0, 0, 0.0, CURRENT_TIMESTAMP)
        """
        db.execute(text(stats_query), {"collection_name": collection.collection_name})
        db.commit()
        
        # Invalidate cache
        invalidate_collection_cache(collection.collection_name)
        
        # Return the created collection
        return await get_collection(collection.collection_name, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create collection: {str(e)}")

@router.put("/{collection_name}", response_model=CollectionResponse)
async def update_collection(collection_name: str, update: CollectionUpdate, db=Depends(get_db)):
    """Update an existing collection"""
    try:
        # Check if collection exists
        existing = db.execute(
            text("SELECT id FROM collection_registry WHERE collection_name = :name"),
            {"name": collection_name}
        ).fetchone()
        
        if not existing:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Build update query
        update_fields = []
        update_params = {"collection_name": collection_name}
        
        if update.collection_type is not None:
            update_fields.append("collection_type = :collection_type")
            update_params["collection_type"] = update.collection_type
        
        if update.description is not None:
            update_fields.append("description = :description")
            update_params["description"] = update.description
        
        if update.metadata_schema is not None:
            update_fields.append("metadata_schema = :metadata_schema")
            update_params["metadata_schema"] = json.dumps(update.metadata_schema)
        
        if update.search_config is not None:
            update_fields.append("search_config = :search_config")
            update_params["search_config"] = json.dumps(update.search_config)
        
        if update.access_config is not None:
            update_fields.append("access_config = :access_config")
            update_params["access_config"] = json.dumps(update.access_config)
        
        if update_fields:
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            query = f"""
                UPDATE collection_registry 
                SET {', '.join(update_fields)}
                WHERE collection_name = :collection_name
            """
            
            db.execute(text(query), update_params)
            db.commit()
            
            # Invalidate cache
            invalidate_collection_cache(collection_name)
        
        # Return updated collection
        return await get_collection(collection_name, db)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update collection {collection_name}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update collection")

@router.delete("/{collection_name}")
async def delete_collection(collection_name: str, db=Depends(get_db)):
    """Delete a collection"""
    try:
        # Check if collection exists
        existing = db.execute(
            text("SELECT id FROM collection_registry WHERE collection_name = :name"),
            {"name": collection_name}
        ).fetchone()
        
        if not existing:
            raise HTTPException(status_code=404, detail="Collection not found")
        
        # Try to connect to Milvus and drop collection
        try:
            get_milvus_connection()
            
            # Drop Milvus collection if it exists
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.drop()
        except HTTPException as e:
            # If Milvus is not available, just log and continue
            logger.warning(f"Could not drop Milvus collection: {e.detail}")
        
        # Delete from database
        db.execute(text("DELETE FROM collection_statistics WHERE collection_name = :name"), {"name": collection_name})
        db.execute(text("DELETE FROM user_collection_access WHERE collection_name = :name"), {"name": collection_name})
        db.execute(text("DELETE FROM collection_registry WHERE collection_name = :name"), {"name": collection_name})
        db.commit()
        
        # Invalidate cache
        invalidate_collection_cache(collection_name)
        
        return {"message": f"Collection {collection_name} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection {collection_name}: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete collection")

@router.post("/{collection_name}/analyze")
async def analyze_collection(collection_name: str):
    """Get real-time collection statistics from Milvus"""
    try:
        # Get stats directly from Milvus
        stats = MilvusStats.get_collection_stats(collection_name)
        
        if stats.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail="Collection not found in vector database")
        elif stats.get('status') == 'error':
            raise HTTPException(status_code=500, detail=f"Failed to analyze collection: {stats.get('error', 'Unknown error')}")
        
        return {
            "message": "Collection analyzed successfully",
            "statistics": stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze collection {collection_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze collection")

@router.post("/refresh-statistics")
async def refresh_statistics():
    """Get real-time statistics for all collections from Milvus"""
    try:
        # Get stats directly from Milvus
        stats_list = MilvusStats.get_all_collections_stats()
        
        # Convert list to dict for response
        stats_dict = {stat['collection_name']: stat for stat in stats_list}
        
        return {
            "message": "Statistics retrieved successfully",
            "collections_found": len(stats_list),
            "statistics": stats_dict
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/{collection_name}/statistics")
async def get_collection_stats(collection_name: str):
    """Get real-time statistics for a specific collection from Milvus"""
    stats = MilvusStats.get_collection_stats(collection_name)
    
    if stats.get('status') == 'not_found':
        raise HTTPException(status_code=404, detail="Collection not found in vector database")
    elif stats.get('status') == 'error':
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {stats.get('error', 'Unknown error')}")
    
    return stats

@router.get("/{collection_name}/index-info")
async def get_collection_index_info(collection_name: str):
    """Get detailed index information for a specific collection"""
    try:
        # Connect to Milvus
        get_milvus_connection()
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            raise HTTPException(status_code=404, detail="Collection not found in vector database")
        
        # Get collection object
        collection = Collection(collection_name)
        
        # Get index information
        indexes = collection.indexes
        index_info = []
        
        for index in indexes:
            index_info.append({
                "field_name": index.field_name,
                "index_name": index.index_name,
                "index_type": index.params.get("index_type", "unknown"),
                "metric_type": index.params.get("metric_type", "unknown"),
                "index_params": index.params.get("params", {}),
                "full_params": index.params
            })
        
        return {
            "collection_name": collection_name,
            "indexes": index_info,
            "total_indexes": len(index_info),
            "has_hnsw": any(idx["index_type"] == "HNSW" for idx in index_info),
            "vector_index_type": next((idx["index_type"] for idx in index_info if idx["field_name"] == "vector"), "not_found")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get index info for {collection_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get index information: {str(e)}")

@router.post("/rebuild-with-hnsw")
async def rebuild_collections_with_hnsw(db=Depends(get_db)):
    """
    Rebuild all collections with HNSW index (WARNING: This will delete all vector data)
    
    This endpoint:
    1. Gets all collections from PostgreSQL
    2. Drops existing Milvus collections 
    3. Recreates them with HNSW index
    4. Returns rebuild results
    
    WARNING: All vector data will be lost and need to be re-uploaded!
    """
    try:
        # Get all collections from PostgreSQL
        query = """
            SELECT collection_name, metadata_schema 
            FROM collection_registry 
            ORDER BY created_at ASC
        """
        result = db.execute(text(query)).fetchall()
        
        rebuild_results = {
            "total_collections": len(result),
            "rebuilt": [],
            "failed": [],
            "milvus_available": False,
            "warning": "All vector data has been deleted and needs to be re-uploaded!"
        }
        
        # Try to connect to Milvus
        try:
            get_milvus_connection()
            rebuild_results["milvus_available"] = True
        except Exception as e:
            logger.error(f"Milvus not available for rebuild: {e}")
            raise HTTPException(status_code=500, detail="Milvus vector database not available")
        
        # Rebuild each collection with HNSW
        for row in result:
            collection_name = row[0]
            metadata_schema = row[1] if isinstance(row[1], dict) else (json.loads(row[1]) if row[1] else {})
            
            try:
                # Drop existing collection if it exists
                if utility.has_collection(collection_name):
                    existing_collection = Collection(collection_name)
                    existing_collection.drop()
                    logger.info(f"Dropped existing collection: {collection_name}")
                
                # Create new collection with HNSW
                success = create_milvus_collection(collection_name, metadata_schema)
                if success:
                    rebuild_results["rebuilt"].append(collection_name)
                    logger.info(f"Successfully rebuilt collection with HNSW: {collection_name}")
                else:
                    rebuild_results["failed"].append(collection_name)
                    logger.error(f"Failed to rebuild collection: {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error rebuilding collection {collection_name}: {e}")
                rebuild_results["failed"].append(collection_name)
        
        # Return results
        return {
            "message": f"Rebuild completed: {len(rebuild_results['rebuilt'])} rebuilt, {len(rebuild_results['failed'])} failed",
            "results": rebuild_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rebuild collections with HNSW: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild failed: {str(e)}")

@router.post("/initialize-default")
async def initialize_default_collection_endpoint(db=Depends(get_db)):
    """
    Initialize the default_knowledge collection if it doesn't exist.
    
    This endpoint can be called manually to ensure the default collection
    is properly set up. It's safe to call multiple times.
    
    Returns:
        Dict with initialization results and status
    """
    try:
        result = initialize_default_collection(db)
        
        if result["success"]:
            return {
                "message": result["message"],
                "collection_name": result["collection_name"],
                "already_exists": result["already_exists"],
                "database_created": result["database_created"],
                "milvus_created": result["milvus_created"],
                "milvus_available": result["milvus_available"]
            }
        else:
            raise HTTPException(
                status_code=500 if result["error"] else 409,
                detail=result["message"] or result["error"]
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to initialize default collection: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")