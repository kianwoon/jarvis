#!/usr/bin/env python3
"""
Force upgrade search_optimization configuration to full LLM format.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def force_upgrade():
    """Force upgrade search_optimization to full LLM configuration"""
    print("=== Forcing Search Optimization Upgrade ===")
    
    try:
        from app.core.llm_settings_cache import reload_llm_settings
        from app.core.redis_client import get_redis_client
        
        # Clear any cached settings first
        redis_client = get_redis_client()
        if redis_client:
            redis_client.delete('llm_settings_cache')
            print("Cleared cached LLM settings")
        
        # Force reload from database which should trigger upgrade
        settings = reload_llm_settings()
        
        print("LLM settings reloaded with keys:", list(settings.keys()))
        
        search_optimization = settings.get('search_optimization', {})
        print("Search optimization keys:", list(search_optimization.keys()))
        
        if 'model' in search_optimization:
            print(f"✅ Search optimization has model: {search_optimization.get('model')}")
            print(f"✅ Search optimization has model_server: {search_optimization.get('model_server', 'empty')}")
            print("✅ Upgrade successful!")
            return True
        else:
            print("❌ Search optimization still missing model field")
            return False
            
    except Exception as e:
        print(f"❌ Error during force upgrade: {e}")
        return False

if __name__ == "__main__":
    success = force_upgrade()
    sys.exit(0 if success else 1)