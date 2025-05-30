import json
from app.core.db import SessionLocal, Settings as SettingsModel
from app.core.redis_base import RedisCache

EMBEDDING_SETTINGS_KEY = 'embedding_settings_cache'

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_embedding_settings():
    try:
        cached = cache.get(EMBEDDING_SETTINGS_KEY)
        if cached:
            return cached
        return reload_embedding_settings()
    except Exception as e:
        print(f"[ERROR] Failed to get embedding settings from cache: {str(e)}")
        # Return default settings if cache fails
        return {"embedding_model": "BAAI/bge-base-en-v1.5", "embedding_endpoint": ""}

def set_embedding_settings(settings_dict):
    cache.set(EMBEDDING_SETTINGS_KEY, settings_dict)

# Call this after updating settings in DB
def reload_embedding_settings():
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'storage').first()
        if row and 'embedding_model' in row.settings and 'embedding_endpoint' in row.settings:
            embedding = {
                "embedding_model": row.settings['embedding_model'],
                "embedding_endpoint": row.settings['embedding_endpoint']
            }
            cache.set(EMBEDDING_SETTINGS_KEY, embedding)
            return embedding
        else:
            default = {"embedding_model": "", "embedding_endpoint": ""}
            cache.set(EMBEDDING_SETTINGS_KEY, default)
            return default
    finally:
        db.close() 