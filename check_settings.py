#!/usr/bin/env python
"""
Script to check MCP settings in the database
"""
import os
import sys
import json

# Add the parent directory to the path so we can import app
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from app.core.db import SessionLocal, Settings

def check_mcp_settings():
    db = SessionLocal()
    try:
        print("Checking MCP settings in the database...")
        settings_row = db.query(Settings).filter(Settings.category == 'mcp').first()
        
        if settings_row:
            print(f"Settings found for category 'mcp':")
            print(f"  ID: {settings_row.id}")
            print(f"  Updated at: {settings_row.updated_at}")
            
            # Pretty print the settings
            print("  Settings:")
            for key, value in settings_row.settings.items():
                if key == 'api_key':
                    if value:
                        print(f"    api_key: [REDACTED] (length: {len(value)})")
                    else:
                        print(f"    api_key: None")
                else:
                    print(f"    {key}: {value}")
        else:
            print("No settings found for category 'mcp'")
        
        # Check all settings
        all_settings = db.query(Settings).all()
        print(f"\nAll settings categories ({len(all_settings)}):")
        for s in all_settings:
            print(f"  - {s.category}")
    finally:
        db.close()

if __name__ == "__main__":
    check_mcp_settings() 