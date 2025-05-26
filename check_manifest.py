#!/usr/bin/env python
"""
Script to check if API keys are properly stored in the MCPManifest table
"""
import os
import sys
import json

# Add the parent directory to the path so we can import app
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from app.core.db import SessionLocal, MCPManifest, Settings, Base, engine

def check_manifest_api_keys():
    # First, ensure the tables exist
    try:
        print("Creating database tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        print("Database tables created/verified")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        return

    db = SessionLocal()
    try:
        print("\nChecking MCPManifest table for API keys...")
        manifests = db.query(MCPManifest).all()
        
        if manifests:
            print(f"Found {len(manifests)} manifests:")
            for manifest in manifests:
                print(f"\n  Manifest ID: {manifest.id}")
                print(f"  URL: {manifest.url}")
                print(f"  Hostname: {manifest.hostname}")
                
                # Show API key info (redacted)
                if manifest.api_key:
                    print(f"  API Key: [REDACTED] (length: {len(manifest.api_key)})")
                else:
                    print(f"  API Key: None")
                
                # Count tools for this manifest
                tools_count = len(manifest.tools)
                print(f"  Tools: {tools_count}")
        else:
            print("No manifests found in the database")
        
        print("\nChecking Settings table for comparison...")
        settings_row = db.query(Settings).filter(Settings.category == 'mcp').first()
        
        if settings_row:
            print(f"Found 'mcp' settings:")
            print(f"  ID: {settings_row.id}")
            print(f"  Updated at: {settings_row.updated_at}")
            
            # Check if API key is in settings (it shouldn't be)
            if 'api_key' in settings_row.settings:
                print(f"  WARNING: API key found in settings table! This should be in the manifest table.")
                print(f"  API Key in settings: [REDACTED] (length: {len(settings_row.settings['api_key'])})")
            else:
                print(f"  Good: No API key in settings table (correct)")
                
            # Show other settings
            print("  Other settings:")
            for key, value in settings_row.settings.items():
                print(f"    {key}: {value}")
        else:
            print("No 'mcp' settings found in the database")
            
    finally:
        db.close()

if __name__ == "__main__":
    check_manifest_api_keys() 