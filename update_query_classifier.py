#!/usr/bin/env python3
"""
Script to update query classifier settings with new LLM fields
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.db import SessionLocal, Settings as SettingsModel
from app.core.query_classifier_settings_cache import DEFAULT_QUERY_CLASSIFIER_SETTINGS
import json

def update_query_classifier_settings():
    """Update query classifier settings in database with new LLM fields"""
    db = SessionLocal()
    try:
        # Get LLM settings row
        row = db.query(SettingsModel).filter(SettingsModel.category == 'llm').first()
        if not row:
            print("No LLM settings found in database")
            return False
            
        current_settings = row.settings or {}
        existing_classifier = current_settings.get('query_classifier', {})
        
        print(f"Current query_classifier fields: {list(existing_classifier.keys())}")
        print(f"Default fields: {list(DEFAULT_QUERY_CLASSIFIER_SETTINGS.keys())}")
        
        # Merge existing with new defaults
        merged_classifier = DEFAULT_QUERY_CLASSIFIER_SETTINGS.copy()
        merged_classifier.update(existing_classifier)
        
        # Update in database
        current_settings['query_classifier'] = merged_classifier
        row.settings = current_settings
        db.commit()
        
        print(f"Updated query_classifier with fields: {list(merged_classifier.keys())}")
        print("Database updated successfully!")
        
        # Show the new LLM fields
        llm_fields = {k: v for k, v in merged_classifier.items() if k.startswith('llm_') or k == 'enable_llm_classification' or k == 'fallback_to_patterns' or k == 'llm_classification_priority'}
        print(f"New LLM fields added: {llm_fields}")
        
        return True
        
    except Exception as e:
        print(f"Error updating settings: {e}")
        db.rollback()
        return False
    finally:
        db.close()

if __name__ == "__main__":
    print("Updating query classifier settings with new LLM fields...")
    success = update_query_classifier_settings()
    if success:
        print("\nNow refresh your browser and the new fields should appear!")
        print("You can also call: curl -X POST http://localhost:8000/api/v1/settings/llm/cache/reload")
    else:
        print("Failed to update settings")