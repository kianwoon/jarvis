"""
Vector Database Service
Provides a unified interface for accessing vector database configurations
and handles both legacy and new formats transparently.
"""

from app.core.vector_db_settings_cache import get_vector_db_settings
from app.utils.vector_db_migration import get_active_vector_db_config, convert_to_legacy_format


def get_active_vector_db():
    """
    Get the currently active vector database configuration.
    Returns the config in a standardized format regardless of storage format.
    
    Returns:
        dict: {
            "id": "milvus",
            "name": "Milvus", 
            "config": { ... }
        }
        or None if no enabled database found
    """
    settings = get_vector_db_settings()
    return get_active_vector_db_config(settings)


def get_vector_db_config_legacy():
    """
    Get vector database settings in legacy format for backward compatibility.
    This is useful for existing code that expects the old structure.
    
    Returns:
        dict: {
            "active": "milvus",
            "milvus": { "status": true, ... },
            "qdrant": { "status": false, ... }
        }
    """
    settings = get_vector_db_settings()
    return convert_to_legacy_format(settings)


def is_vector_db_enabled(db_id: str) -> bool:
    """
    Check if a specific vector database is enabled.
    
    Args:
        db_id: The database identifier (e.g., "milvus", "qdrant")
        
    Returns:
        bool: True if the database is enabled, False otherwise
    """
    settings = get_vector_db_settings()
    
    # Handle new format
    if "databases" in settings:
        for db in settings["databases"]:
            if db["id"] == db_id:
                return db.get("enabled", False)
        return False
    
    # Handle legacy format
    if db_id in settings:
        return settings[db_id].get("status", False)
    
    return False


def get_vector_db_by_id(db_id: str):
    """
    Get configuration for a specific vector database by ID.
    
    Args:
        db_id: The database identifier (e.g., "milvus", "qdrant")
        
    Returns:
        dict: Database configuration or None if not found
    """
    settings = get_vector_db_settings()
    
    # Handle new format
    if "databases" in settings:
        for db in settings["databases"]:
            if db["id"] == db_id:
                return {
                    "id": db["id"],
                    "name": db["name"],
                    "enabled": db.get("enabled", False),
                    "config": db["config"]
                }
        return None
    
    # Handle legacy format
    if db_id in settings:
        return {
            "id": db_id,
            "name": db_id.capitalize(),
            "enabled": settings[db_id].get("status", False),
            "config": {k: v for k, v in settings[db_id].items() if k != "status"}
        }
    
    return None