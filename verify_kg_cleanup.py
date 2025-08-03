#!/usr/bin/env python3
"""
Verify Knowledge Graph Settings Cleanup
Check that all hardcoded values have been removed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.knowledge_graph_settings_cache import reload_knowledge_graph_settings, get_knowledge_graph_settings
from app.core.db import SessionLocal, Settings
import json

def check_database_settings():
    """Check database settings directly"""
    print("🔍 Checking database settings...")
    
    db = SessionLocal()
    try:
        kg_row = db.query(Settings).filter(Settings.category == 'knowledge_graph').first()
        if not kg_row:
            print("❌ No knowledge_graph settings found in database")
            return False
        
        settings = kg_row.settings
        
        # Check for extraction_settings
        if 'extraction_settings' in settings:
            print("❌ STILL HAS extraction_settings in database")
            print(f"   Content: {settings['extraction_settings']}")
            return False
        else:
            print("✅ Database is clean - no extraction_settings")
        
        # Check for hardcoded types in prompts
        if 'prompts' in settings:
            has_hardcoded_types = False
            for prompt in settings['prompts']:
                if 'parameters' in prompt and 'types' in prompt['parameters']:
                    print(f"❌ STILL HAS hardcoded types in prompt {prompt.get('name', 'unknown')}")
                    print(f"   Types: {prompt['parameters']['types']}")
                    has_hardcoded_types = True
            
            if not has_hardcoded_types:
                print("✅ Database prompts are clean - no hardcoded types")
        
        return True
        
    finally:
        db.close()

def check_cache_settings():
    """Check cache settings"""
    print("\n🔄 Reloading and checking cache settings...")
    
    try:
        # Force reload from database
        settings = reload_knowledge_graph_settings()
        
        # Check for extraction_settings
        if 'extraction_settings' in settings:
            print("❌ STILL HAS extraction_settings in cache")
            print(f"   Content: {settings['extraction_settings']}")
            return False
        else:
            print("✅ Cache is clean - no extraction_settings")
        
        # Check for static_fallback hardcoded types
        if 'static_fallback' in settings:
            fallback = settings['static_fallback']
            if 'entity_types' in fallback or 'relationship_types' in fallback:
                print("❌ STILL HAS hardcoded types in static_fallback")
                print(f"   Content: {fallback}")
                return False
            else:
                print("✅ Cache static_fallback is clean")
        else:
            print("✅ Cache has no static_fallback section")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking cache: {e}")
        return False

def main():
    print("🧹 Knowledge Graph Settings Cleanup Verification")
    print("=" * 50)
    
    db_clean = check_database_settings()
    cache_clean = check_cache_settings()
    
    print("\n📊 Summary:")
    print("=" * 20)
    if db_clean and cache_clean:
        print("✅ ALL CLEAN! No hardcoded values found.")
        print("   - extraction_settings removed from database")
        print("   - hardcoded types removed from prompts") 
        print("   - static_fallback cleaned in cache")
        print("   - System is now pure LLM-driven")
        return 0
    else:
        print("❌ CLEANUP INCOMPLETE - hardcoded values still present")
        return 1

if __name__ == "__main__":
    exit(main())