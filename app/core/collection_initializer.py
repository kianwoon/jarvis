"""
Collection initialization utilities

This module provides functions to automatically initialize default collections
for the knowledge management system.
"""

import json
import logging
from typing import Dict, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, get_db
from app.core.vector_db_settings_cache import get_vector_db_settings
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility

logger = logging.getLogger(__name__)

def get_default_collection_config() -> Dict[str, Any]:
    """
    Get the default configuration for the default_knowledge collection.
    
    This matches the 'general' collection type from the frontend with
    appropriate defaults for a default knowledge collection.
    
    Returns:
        Dict containing the collection configuration
    """
    return {
        "collection_name": "default_knowledge",
        "collection_type": "general",
        "description": "Default collection for general documents and knowledge storage. This collection is automatically created and used when no specific collection is specified.",
        "metadata_schema": {
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "fields": []  # Default fields will be added by the API
        },
        "search_config": {
            "strategy": "balanced",
            "similarity_threshold": 0.7,
            "max_results": 10,
            "enable_bm25": True,
            "bm25_weight": 0.3
        },
        "access_config": {
            "restricted": False,
            "allowed_users": []
        }
    }

def check_database_tables_exist(db: Session) -> bool:
    """
    Check if the required collection tables exist in the database.
    
    Args:
        db: Database session
        
    Returns:
        bool: True if tables exist, False otherwise
    """
    try:
        # Check if collection_registry table exists
        result = db.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'collection_registry'
            )
        """)).fetchone()
        
        registry_exists = result and result[0]
        
        # Check if collection_statistics table exists
        result = db.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'collection_statistics'
            )
        """)).fetchone()
        
        stats_exists = result and result[0]
        
        return registry_exists and stats_exists
        
    except Exception as e:
        logger.error(f"Error checking database tables: {e}")
        return False

def check_default_collection_exists(db: Session) -> bool:
    """
    Check if the default_knowledge collection already exists.
    
    Args:
        db: Database session
        
    Returns:
        bool: True if collection exists, False otherwise
    """
    try:
        result = db.execute(
            text("SELECT id FROM collection_registry WHERE collection_name = :name"),
            {"name": "default_knowledge"}
        ).fetchone()
        
        return result is not None
        
    except Exception as e:
        logger.error(f"Error checking for default collection: {e}")
        return False

def get_milvus_connection() -> bool:
    """
    Attempt to connect to Milvus vector database.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        vector_db_settings = get_vector_db_settings()
        if vector_db_settings.get('active') != 'milvus':
            logger.info("Milvus is not the active vector database")
            return False
        
        milvus_config = vector_db_settings.get('milvus', {})
        
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
        logger.warning(f"Failed to connect to Milvus: {e}")
        return False

def create_milvus_collection(collection_name: str, metadata_schema: Dict[str, Any]) -> bool:
    """
    Create a new Milvus collection with the default schema.
    
    Args:
        collection_name: Name of the collection
        metadata_schema: Schema configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if collection already exists
        if utility.has_collection(collection_name):
            logger.info(f"Milvus collection {collection_name} already exists")
            return True
        
        # Define the standard schema fields
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
            # BM25 fields for hybrid search
            FieldSchema(name="bm25_tokens", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="bm25_term_count", dtype=DataType.INT64),
            FieldSchema(name="bm25_unique_terms", dtype=DataType.INT64),
            FieldSchema(name="bm25_top_terms", dtype=DataType.VARCHAR, max_length=1000),
            # Date metadata fields
            FieldSchema(name="creation_date", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="last_modified_date", dtype=DataType.VARCHAR, max_length=100),
        ]
        
        # Create collection schema
        schema = CollectionSchema(
            fields=fields,
            description=f"Default knowledge collection: {collection_name}"
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
        
        logger.info(f"Successfully created Milvus collection: {collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create Milvus collection {collection_name}: {e}")
        return False

def create_default_collection_in_db(db: Session, config: Dict[str, Any]) -> bool:
    """
    Create the default collection in the PostgreSQL database.
    
    Args:
        db: Database session
        config: Collection configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Insert into collection_registry
        query = """
            INSERT INTO collection_registry 
            (collection_name, collection_type, description, metadata_schema, 
             search_config, access_config, created_at, updated_at)
            VALUES (:collection_name, :collection_type, :description, :metadata_schema, 
                    :search_config, :access_config, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """
        
        db.execute(text(query), {
            "collection_name": config["collection_name"],
            "collection_type": config["collection_type"],
            "description": config["description"],
            "metadata_schema": json.dumps(config["metadata_schema"]),
            "search_config": json.dumps(config["search_config"]),
            "access_config": json.dumps(config["access_config"])
        })
        
        # Initialize statistics
        stats_query = """
            INSERT INTO collection_statistics 
            (collection_name, document_count, total_chunks, storage_size_mb, last_updated)
            VALUES (:collection_name, 0, 0, 0.0, CURRENT_TIMESTAMP)
        """
        db.execute(text(stats_query), {"collection_name": config["collection_name"]})
        
        db.commit()
        
        logger.info(f"Successfully created collection registry entry: {config['collection_name']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create collection in database: {e}")
        db.rollback()
        return False

def initialize_default_collection(db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Initialize the default_knowledge collection if it doesn't exist.
    
    This function:
    1. Checks if database tables exist
    2. Checks if default_knowledge collection already exists
    3. Creates the collection in PostgreSQL
    4. Attempts to create the collection in Milvus (optional)
    
    Args:
        db: Optional database session. If not provided, creates a new one.
        
    Returns:
        Dict with initialization results and status information
    """
    result = {
        "success": False,
        "collection_name": "default_knowledge",
        "already_exists": False,
        "database_created": False,
        "milvus_created": False,
        "milvus_available": False,
        "error": None,
        "message": ""
    }
    
    # Create database session if not provided
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    
    try:
        # Check if database tables exist
        if not check_database_tables_exist(db):
            result["error"] = "Required database tables (collection_registry, collection_statistics) do not exist"
            result["message"] = "Database tables need to be created first by running the application"
            return result
        
        # Check if collection already exists
        if check_default_collection_exists(db):
            result["already_exists"] = True
            result["success"] = True
            result["message"] = "Default collection 'default_knowledge' already exists"
            return result
        
        # Get default configuration
        config = get_default_collection_config()
        
        # Create collection in database
        if create_default_collection_in_db(db, config):
            result["database_created"] = True
            
            # Attempt to create in Milvus (optional)
            if get_milvus_connection():
                result["milvus_available"] = True
                if create_milvus_collection(config["collection_name"], config["metadata_schema"]):
                    result["milvus_created"] = True
                    result["message"] = "Successfully created default collection in both PostgreSQL and Milvus"
                else:
                    result["message"] = "Successfully created default collection in PostgreSQL, but Milvus creation failed"
            else:
                result["message"] = "Successfully created default collection in PostgreSQL (Milvus not available)"
            
            result["success"] = True
        else:
            result["error"] = "Failed to create collection in database"
            result["message"] = "Could not create default collection in PostgreSQL"
    
    except Exception as e:
        logger.error(f"Error during default collection initialization: {e}")
        result["error"] = str(e)
        result["message"] = f"Initialization failed: {str(e)}"
    
    finally:
        if close_db:
            db.close()
    
    return result

def initialize_default_collection_sync() -> Dict[str, Any]:
    """
    Synchronous wrapper for initializing the default collection.
    
    This is useful for startup scripts and synchronous contexts.
    
    Returns:
        Dict with initialization results
    """
    return initialize_default_collection()

async def initialize_default_collection_async() -> Dict[str, Any]:
    """
    Asynchronous wrapper for initializing the default collection.
    
    This is useful for FastAPI startup events and async contexts.
    
    Returns:
        Dict with initialization results
    """
    return initialize_default_collection()

def get_collection_status() -> Dict[str, Any]:
    """
    Get the current status of the default collection.
    
    Returns:
        Dict with status information
    """
    status = {
        "default_collection_exists": False,
        "database_tables_exist": False,
        "milvus_available": False,
        "collection_count": 0,
        "error": None
    }
    
    db = SessionLocal()
    try:
        # Check if database tables exist
        status["database_tables_exist"] = check_database_tables_exist(db)
        
        if status["database_tables_exist"]:
            # Check if default collection exists
            status["default_collection_exists"] = check_default_collection_exists(db)
            
            # Get total collection count
            result = db.execute(text("SELECT COUNT(*) FROM collection_registry")).fetchone()
            status["collection_count"] = result[0] if result else 0
        
        # Check Milvus availability
        status["milvus_available"] = get_milvus_connection()
        
    except Exception as e:
        logger.error(f"Error getting collection status: {e}")
        status["error"] = str(e)
    
    finally:
        db.close()
    
    return status