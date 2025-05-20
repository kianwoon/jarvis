import redis
import json
from app.core.db import SessionLocal, Settings as SettingsModel

REDIS_HOST = 'redis'
REDIS_PORT = 6379
LLM_SETTINGS_KEY = 'llm_settings_cache'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def get_llm_settings():
    cached = r.get(LLM_SETTINGS_KEY)
    if cached:
        return json.loads(cached)
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
            r.set(LLM_SETTINGS_KEY, json.dumps(row.settings))
            return row.settings
        else:
            # No user-defined settings, use default
            default = {
                "model": "qwen3:30b-a3b",
                "temperature": 0.7,
                "top_p": 1.0,
                "max_tokens": 2048,
                "system_prompt": "You are a helpful assistant."
            }
            r.set(LLM_SETTINGS_KEY, json.dumps(default))
            return default
    finally:
        db.close() 