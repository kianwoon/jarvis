import json
from app.core.redis_base import RedisCache
from app.utils.vector_db_migration import migrate_vector_db_settings

VECTOR_DB_SETTINGS_KEY = 'vector_db_settings_cache'

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_vector_db_settings():
    try:
        cached = cache.get(VECTOR_DB_SETTINGS_KEY)
        if cached:
            # Ensure settings are in the new format
            return migrate_vector_db_settings(cached)
        return reload_vector_db_settings()
    except Exception as e:
        print(f"[ERROR] Failed to get vector DB settings from cache: {str(e)}")
        # Return default settings in new format if cache fails
        return {
            "active": "milvus",
            "databases": [
                {
                    "id": "milvus",
                    "name": "Milvus",
                    "enabled": True,
                    "config": {
                        "MILVUS_URI": "http://milvus:19530",
                        "MILVUS_TOKEN": "",
                        "MILVUS_DEFAULT_COLLECTION": "default_knowledge",
                        "dimension": 1536
                    }
                }
            ]
        }

def set_vector_db_settings(settings_dict):
    cache.set(VECTOR_DB_SETTINGS_KEY, settings_dict)

# Call this after updating settings in DB
def reload_vector_db_settings():
    try:
        from app.core.db import SessionLocal, Settings as SettingsModel
        db = SessionLocal()
        try:
            row = db.query(SettingsModel).filter(SettingsModel.category == 'storage').first()
            if row and 'vector_db' in row.settings:
                # Migrate to new format if needed
                migrated = migrate_vector_db_settings(row.settings['vector_db'])
                cache.set(VECTOR_DB_SETTINGS_KEY, migrated)
                return migrated
            else:
                # Default settings in new format
                default = {
                    "active": "milvus",
                    "databases": [
                        {
                            "id": "milvus",
                            "name": "Milvus",
                            "enabled": True,
                            "config": {
                                "MILVUS_URI": "http://milvus:19530",
                                "MILVUS_TOKEN": "",
                                "MILVUS_DEFAULT_COLLECTION": "default_knowledge",
                                "dimension": 1536
                            }
                        },
                        {
                            "id": "qdrant",
                            "name": "Qdrant",
                            "enabled": False,
                            "config": {
                                "QDRANT_HOST": "localhost",
                                "QDRANT_PORT": 6333,
                                "collection": "default_knowledge",
                                "dimension": 1536
                            }
                        }
                    ]
                }
                cache.set(VECTOR_DB_SETTINGS_KEY, default)
                return default
        finally:
            db.close()
    except Exception as e:
        print(f"[ERROR] Failed to reload vector DB settings from database: {str(e)}")
        # Return default settings in new format if database fails
        default = {
            "active": "milvus",
            "databases": [
                {
                    "id": "milvus",
                    "name": "Milvus",
                    "enabled": True,
                    "config": {
                        "MILVUS_URI": "http://milvus:19530",
                        "MILVUS_TOKEN": "",
                        "MILVUS_DEFAULT_COLLECTION": "default_knowledge",
                        "dimension": 1536
                    }
                }
            ]
        }
        cache.set(VECTOR_DB_SETTINGS_KEY, default)
        return default 