import json
from app.core.db import SessionLocal, Settings as SettingsModel
from app.core.redis_base import RedisCache

ICEBERG_SETTINGS_KEY = 'iceberg_settings_cache'

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_iceberg_settings():
    cached = cache.get(ICEBERG_SETTINGS_KEY)
    if cached:
        return cached
    return reload_iceberg_settings()

def set_iceberg_settings(settings_dict):
    cache.set(ICEBERG_SETTINGS_KEY, settings_dict)

# Call this after updating settings in DB
def reload_iceberg_settings():
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'storage').first()
        if row and 'iceberg' in row.settings:
            iceberg = row.settings['iceberg']
            cache.set(ICEBERG_SETTINGS_KEY, iceberg)
            return iceberg
        else:
            default = {}
            cache.set(ICEBERG_SETTINGS_KEY, default)
            return default
    finally:
        db.close() 