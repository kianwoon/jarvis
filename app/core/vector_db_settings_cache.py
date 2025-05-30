import json
from app.core.db import SessionLocal, Settings as SettingsModel
from app.core.redis_base import RedisCache

VECTOR_DB_SETTINGS_KEY = 'vector_db_settings_cache'

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_vector_db_settings():
    try:
        cached = cache.get(VECTOR_DB_SETTINGS_KEY)
        if cached:
            return cached
        return reload_vector_db_settings()
    except Exception as e:
        print(f"[ERROR] Failed to get vector DB settings from cache: {str(e)}")
        # Return default settings if cache fails
        return {
            "active": "milvus",
            "milvus": {
                "MILVUS_URI": "http://milvus:19530",
                "MILVUS_TOKEN": "",
                "MILVUS_DEFAULT_COLLECTION": "default_knowledge"
            },
            "qdrant": {}
        }

def set_vector_db_settings(settings_dict):
    cache.set(VECTOR_DB_SETTINGS_KEY, settings_dict)

# Call this after updating settings in DB
def reload_vector_db_settings():
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'storage').first()
        if row and 'vector_db' in row.settings:
            cache.set(VECTOR_DB_SETTINGS_KEY, row.settings['vector_db'])
            return row.settings['vector_db']
        else:
            default = {"active": "milvus", "milvus": {}, "qdrant": {}}
            cache.set(VECTOR_DB_SETTINGS_KEY, default)
            return default
    finally:
        db.close() 