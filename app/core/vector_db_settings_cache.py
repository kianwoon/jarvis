import redis
import json
from app.core.db import SessionLocal, Settings as SettingsModel

REDIS_HOST = 'redis'
REDIS_PORT = 6379
VECTOR_DB_SETTINGS_KEY = 'vector_db_settings_cache'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def get_vector_db_settings():
    try:
        cached = r.get(VECTOR_DB_SETTINGS_KEY)
        if cached:
            return json.loads(cached)
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
    r.set(VECTOR_DB_SETTINGS_KEY, json.dumps(settings_dict))

# Call this after updating settings in DB
def reload_vector_db_settings():
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'storage').first()
        if row and 'vector_db' in row.settings:
            r.set(VECTOR_DB_SETTINGS_KEY, json.dumps(row.settings['vector_db']))
            return row.settings['vector_db']
        else:
            default = {"active": "milvus", "milvus": {}, "qdrant": {}}
            r.set(VECTOR_DB_SETTINGS_KEY, json.dumps(default))
            return default
    finally:
        db.close() 