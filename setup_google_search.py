#!/usr/bin/env python3
"""
Interactive setup script for Google Search configuration.
Helps users configure real Google Custom Search credentials.
"""

import os
import sys
import json
import getpass
import psycopg2
from psycopg2.extras import Json
from pathlib import Path


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="postgres",
        database="llm_platform"
    )


def validate_api_key(api_key):
    """Basic validation for Google API key format"""
    if not api_key or len(api_key) < 30:
        return False
    if not api_key.startswith("AIza"):
        return False
    return True


def validate_search_engine_id(engine_id):
    """Basic validation for search engine ID"""
    if not engine_id or len(engine_id) < 10:
        return False
    if engine_id == "d77ac8c3d3e124c3c":
        print("❌ This is the TEST search engine that returns fake data!")
        return False
    return True


def main():
    print("=" * 60)
    print("Google Search Setup for Jarvis")
    print("=" * 60)
    print()
    print("This script will help you configure real Google Search.")
    print("You'll need:")
    print("1. A Google Custom Search Engine ID")
    print("2. A Google API Key")
    print()
    print("If you don't have these yet, follow the guide at:")
    print("https://programmablesearchengine.google.com/")
    print()
    
    # Check current configuration
    print("Checking current configuration...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT env FROM mcp_servers WHERE id = 9")
        result = cur.fetchone()
        
        if result and result[0]:
            env = result[0]
            current_api_key = env.get("GOOGLE_SEARCH_API_KEY", "Not set")
            current_engine_id = env.get("GOOGLE_SEARCH_ENGINE_ID", "Not set")
            
            print(f"Current API Key: {current_api_key[:10]}..." if len(current_api_key) > 10 else f"Current API Key: {current_api_key}")
            print(f"Current Engine ID: {current_engine_id}")
            
            if current_engine_id == "d77ac8c3d3e124c3c":
                print("⚠️  WARNING: Currently using TEST engine that returns fake data!")
            print()
    except Exception as e:
        print(f"Could not check current configuration: {e}")
        print()
    
    # Get new credentials
    print("Enter your Google Search credentials:")
    print("(Press Enter to skip and keep current value)")
    print()
    
    # Get API Key
    while True:
        api_key = getpass.getpass("Google API Key (starts with AIza): ").strip()
        if not api_key:
            print("Keeping current API key.")
            api_key = None
            break
        if validate_api_key(api_key):
            print("✅ API key format looks valid")
            break
        else:
            print("❌ Invalid API key format. Should start with 'AIza' and be ~39 characters")
            retry = input("Try again? (y/n): ").lower()
            if retry != 'y':
                api_key = None
                break
    
    # Get Search Engine ID
    while True:
        engine_id = input("Search Engine ID: ").strip()
        if not engine_id:
            print("Keeping current search engine ID.")
            engine_id = None
            break
        if validate_search_engine_id(engine_id):
            print("✅ Search engine ID format looks valid")
            break
        else:
            print("❌ Invalid search engine ID format")
            retry = input("Try again? (y/n): ").lower()
            if retry != 'y':
                engine_id = None
                break
    
    # Update configuration
    if api_key or engine_id:
        print()
        print("Updating configuration...")
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Get current env
            cur.execute("SELECT env FROM mcp_servers WHERE id = 9")
            result = cur.fetchone()
            env = result[0] if result and result[0] else {}
            
            # Update values
            if api_key:
                env["GOOGLE_SEARCH_API_KEY"] = api_key
            if engine_id:
                env["GOOGLE_SEARCH_ENGINE_ID"] = engine_id
            
            # Remove setup instructions if we have valid credentials
            if api_key and engine_id and engine_id != "d77ac8c3d3e124c3c":
                env.pop("GOOGLE_SEARCH_SETUP_INSTRUCTIONS", None)
            
            # Update database
            cur.execute(
                "UPDATE mcp_servers SET env = %s WHERE id = 9",
                (Json(env),)
            )
            conn.commit()
            
            print("✅ Configuration updated successfully!")
            print()
            print("Next steps:")
            print("1. Clear Redis cache: redis-cli FLUSHALL")
            print("2. Restart the API service")
            print("3. Test with a search query in the UI")
            
            # Optionally update .env.local
            print()
            update_env = input("Also update .env.local file? (y/n): ").lower()
            if update_env == 'y':
                env_file = Path(__file__).parent / ".env.local"
                lines = []
                
                # Read existing file
                if env_file.exists():
                    with open(env_file, 'r') as f:
                        lines = f.readlines()
                
                # Update or add lines
                updated = False
                for i, line in enumerate(lines):
                    if api_key and line.startswith("GOOGLE_SEARCH_API_KEY="):
                        lines[i] = f"GOOGLE_SEARCH_API_KEY={api_key}\n"
                        updated = True
                    elif engine_id and line.startswith("GOOGLE_SEARCH_ENGINE_ID="):
                        lines[i] = f"GOOGLE_SEARCH_ENGINE_ID={engine_id}\n"
                        updated = True
                
                # Add if not found
                if not updated:
                    if api_key:
                        lines.append(f"GOOGLE_SEARCH_API_KEY={api_key}\n")
                    if engine_id:
                        lines.append(f"GOOGLE_SEARCH_ENGINE_ID={engine_id}\n")
                
                # Write back
                with open(env_file, 'w') as f:
                    f.writelines(lines)
                
                print("✅ Updated .env.local file")
            
            cur.close()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error updating configuration: {e}")
            return 1
    else:
        print()
        print("No changes made.")
    
    print()
    print("=" * 60)
    print("Setup complete!")
    print("For more information, see: GOOGLE_SEARCH_SETUP_GUIDE.md")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)