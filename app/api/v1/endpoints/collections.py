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

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

@router.get("/test")
async def test_collections():
    """Test endpoint to verify collections router is working"""
    return {"message": "Collections API is working"}

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
        ]
        
        # Add any custom fields from metadata_schema
        custom_fields = metadata_schema.get('fields', [])
        for field in custom_fields:
            if field['name'] not in [f.name for f in fields]:
                dtype = DataType.VARCHAR if field['type'] == 'string' else DataType.INT64
                max_length = field.get('max_length', 255) if field['type'] == 'string' else None
                field_schema = FieldSchema(
                    name=field['name'],
                    dtype=dtype,
                    max_length=max_length
                )
                fields.append(field_schema)
        
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
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
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