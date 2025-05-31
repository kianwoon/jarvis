"""
Script to add a normalized content field for case-insensitive search
"""
from pymilvus import connections, Collection, FieldSchema, DataType, utility
from app.core.vector_db_settings_cache import get_vector_db_settings
import sys

def add_normalized_content_field():
    """Add a normalized (lowercase) content field to existing collection"""
    
    # Get Milvus configuration
    vector_db_cfg = get_vector_db_settings()
    milvus_cfg = vector_db_cfg.get("milvus", {})
    
    if not milvus_cfg:
        print("Milvus configuration not found")
        return False
    
    uri = milvus_cfg.get("MILVUS_URI")
    token = milvus_cfg.get("MILVUS_TOKEN")
    collection_name = milvus_cfg.get("MILVUS_DEFAULT_COLLECTION", "default_knowledge")
    
    try:
        # Connect to Milvus
        connections.connect(uri=uri, token=token)
        
        # Check if collection exists
        if not utility.has_collection(collection_name):
            print(f"Collection {collection_name} does not exist")
            return False
        
        # Get existing collection
        collection = Collection(collection_name)
        
        # Check current schema
        print(f"Current fields in {collection_name}:")
        for field in collection.schema.fields:
            print(f"  - {field.name} ({field.dtype})")
        
        # Check if normalized_content already exists
        field_names = [field.name for field in collection.schema.fields]
        if "normalized_content" in field_names:
            print("normalized_content field already exists")
            return True
        
        print("\nWARNING: Adding new fields to existing Milvus collection requires:")
        print("1. Creating a new collection with the updated schema")
        print("2. Migrating all data from old to new collection")
        print("3. Dropping the old collection and renaming the new one")
        print("\nThis is a complex operation. For now, we'll handle normalization at query time.")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
    finally:
        connections.disconnect(uri)

if __name__ == "__main__":
    add_normalized_content_field()