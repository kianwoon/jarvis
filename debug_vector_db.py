#!/usr/bin/env python3
"""
Debug script to check and reset vector DB settings
"""
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, '/Users/kianwoonwong/Downloads/jarvis')

try:
    from app.core.vector_db_settings_cache import get_vector_db_settings, cache, VECTOR_DB_SETTINGS_KEY
    from app.utils.vector_db_migration import migrate_vector_db_settings
    
    print("🔍 Debugging vector DB settings...")
    
    # Clear the cache first
    print("🗑️ Clearing vector DB cache...")
    cache.delete(VECTOR_DB_SETTINGS_KEY)
    
    # Try to get settings
    print("📦 Getting vector DB settings...")
    try:
        settings = get_vector_db_settings()
        print(f"✅ Success! Settings: {settings}")
        print(f"📋 Settings type: {type(settings)}")
        if isinstance(settings, dict):
            print(f"🔑 Keys: {list(settings.keys())}")
    except Exception as e:
        print(f"❌ Error getting settings: {e}")
        import traceback
        print(f"📍 Traceback: {traceback.format_exc()}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory")