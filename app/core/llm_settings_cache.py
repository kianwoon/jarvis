import redis
import json
from app.core.db import SessionLocal, Settings as SettingsModel
from app.core.config import get_settings

config = get_settings()
REDIS_HOST = config.REDIS_HOST
REDIS_PORT = config.REDIS_PORT
REDIS_PASSWORD = config.REDIS_PASSWORD
LLM_SETTINGS_KEY = 'llm_settings_cache'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD, decode_responses=True)

def get_llm_settings():
    cached = r.get(LLM_SETTINGS_KEY)
    if cached:
        settings = json.loads(cached)
        # Validate structure
        if 'thinking_mode' not in settings or 'non_thinking_mode' not in settings:
            raise RuntimeError('LLM settings missing thinking_mode or non_thinking_mode')
        return settings
    # If not cached, load from DB and cache it
    return reload_llm_settings()

def set_llm_settings(settings_dict):
    r.set(LLM_SETTINGS_KEY, json.dumps(settings_dict))

# Call this after updating settings in DB
def reload_llm_settings():
    db = SessionLocal()
    try:
        row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
        if row:
            settings = row.settings
            if 'thinking_mode' not in settings or 'non_thinking_mode' not in settings:
                raise RuntimeError('LLM settings missing thinking_mode or non_thinking_mode')
            r.set(LLM_SETTINGS_KEY, json.dumps(settings))
            return settings
        else:
            # No user-defined settings, raise error
            raise RuntimeError('No LLM settings found in database')
    finally:
        db.close() 