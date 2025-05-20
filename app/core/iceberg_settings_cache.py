import redis
import json
from app.core.db import SessionLocal, Settings as SettingsModel

REDIS_HOST = 'redis'
REDIS_PORT = 6379
ICEBERG_SETTINGS_KEY = 'iceberg_settings_cache'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def get_iceberg_settings():
    cached = r.get(ICEBERG_SETTINGS_KEY)
    if cached:
        return json.loads(cached)
    return reload_iceberg_settings()

def set_iceberg_settings(settings_dict):
    r.set(ICEBERG_SETTINGS_KEY, json.dumps(settings_dict))

# Call this after updating settings in DB
def reload_iceberg_settings():
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'storage').first()
        if row and 'iceberg' in row.settings:
            iceberg = row.settings['iceberg']
            r.set(ICEBERG_SETTINGS_KEY, json.dumps(iceberg))
            return iceberg
        else:
            default = {}
            r.set(ICEBERG_SETTINGS_KEY, json.dumps(default))
            return default
    finally:
        db.close() 