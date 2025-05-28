import redis
import json
from app.core.db import SessionLocal, Settings as SettingsModel

REDIS_HOST = 'redis'
REDIS_PORT = 6379
EMBEDDING_SETTINGS_KEY = 'embedding_settings_cache'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def get_embedding_settings():
    try:
        cached = r.get(EMBEDDING_SETTINGS_KEY)
        if cached:
            return json.loads(cached)
        return reload_embedding_settings()
    except Exception as e:
        print(f"[ERROR] Failed to get embedding settings from cache: {str(e)}")
        # Return default settings if cache fails
        return {"embedding_model": "BAAI/bge-base-en-v1.5", "embedding_endpoint": ""}

def set_embedding_settings(settings_dict):
    r.set(EMBEDDING_SETTINGS_KEY, json.dumps(settings_dict))

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
            r.set(EMBEDDING_SETTINGS_KEY, json.dumps(embedding))
            return embedding
        else:
            default = {"embedding_model": "", "embedding_endpoint": ""}
            r.set(EMBEDDING_SETTINGS_KEY, json.dumps(default))
            return default
    finally:
        db.close() 