"""
Utility to migrate vector database settings from legacy format to new dynamic format.
This ensures backward compatibility while supporting the new flexible structure.
"""

def migrate_vector_db_settings(settings):
    """
    Migrate vector database settings from legacy format to new format.
    
    Legacy format:
    {
        "active": "milvus",
        "milvus": { "status": true, "MILVUS_URI": "...", ... },
        "qdrant": { "status": false, "QDRANT_HOST": "...", ... }
    }
    
    New format:
    {
        "active": "milvus",
        "databases": [
            {
                "id": "milvus",
                "name": "Milvus",
                "enabled": true,
                "config": { "MILVUS_URI": "...", ... }
            },
            ...
        ]
    }
    """
    if not settings or not isinstance(settings, dict):
        return {"active": "milvus", "databases": []}
    
    # If already in new format, return as-is
    if "databases" in settings and isinstance(settings.get("databases"), list):
        return settings
    
    # Convert from legacy format
    migrated = {
        "active": settings.get("active", "milvus"),
        "databases": []
    }
    
    # Known database types to migrate
    known_dbs = {
        "milvus": "Milvus",
        "qdrant": "Qdrant",
        "pinecone": "Pinecone",
        "weaviate": "Weaviate",
        "chroma": "Chroma"
    }
    
    for db_id, db_name in known_dbs.items():
        if db_id in settings and isinstance(settings[db_id], dict):
            db_config = settings[db_id].copy()
            
            # Extract status and remove from config
            # Default to False unless explicitly set to True
            enabled = db_config.pop("status", False)
            
            # Create database entry
            migrated["databases"].append({
                "id": db_id,
                "name": db_name,
                "enabled": enabled,
                "config": db_config
            })
    
    # Ensure at least one database exists
    if not migrated["databases"]:
        migrated["databases"].append({
            "id": "milvus",
            "name": "Milvus",
            "enabled": True,
            "config": {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_TOKEN": "",
                "MILVUS_DEFAULT_COLLECTION": "default_knowledge",
                "dimension": 1536
            }
        })
    
    return migrated


def get_active_vector_db_config(settings):
    """
    Get the configuration for the currently active vector database.
    Works with both legacy and new formats.
    """
    # Migrate to ensure we have the new format
    migrated = migrate_vector_db_settings(settings)
    
    active_id = migrated.get("active", "milvus")
    databases = migrated.get("databases", [])
    
    # Find the active database
    for db in databases:
        if db["id"] == active_id and db.get("enabled", False):
            return {
                "id": db["id"],
                "name": db["name"],
                "config": db["config"]
            }
    
    # If active database not found or disabled, return first enabled one
    for db in databases:
        if db.get("enabled", False):
            return {
                "id": db["id"],
                "name": db["name"],
                "config": db["config"]
            }
    
    # No enabled databases found
    return None


def convert_to_legacy_format(settings):
    """
    Convert new format back to legacy format for backward compatibility.
    This is useful for services that expect the old format.
    """
    if not settings or "databases" not in settings:
        return settings
    
    legacy = {"active": settings.get("active", "milvus")}
    
    for db in settings.get("databases", []):
        db_id = db.get("id")
        if db_id:
            legacy[db_id] = {
                "status": db.get("enabled", False),
                **db.get("config", {})
            }
    
    return legacy