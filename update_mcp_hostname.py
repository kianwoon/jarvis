#!/usr/bin/env python3
"""
Update MCP server hostname to use Docker service name
"""
import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.docker')

# Database connection
conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST', 'localhost'),
    port=os.getenv('POSTGRES_PORT', 5432),
    database=os.getenv('POSTGRES_DB', 'llm_platform'),
    user=os.getenv('POSTGRES_USER', 'postgres'),
    password=os.getenv('POSTGRES_PASSWORD', 'postgres')
)

try:
    with conn.cursor() as cur:
        # First, let's see what MCP manifests we have
        cur.execute("SELECT id, url, hostname FROM mcp_manifests")
        manifests = cur.fetchall()
        
        print("Current MCP Manifests:")
        for manifest in manifests:
            print(f"ID: {manifest[0]}, URL: {manifest[1]}, Hostname: {manifest[2]}")
        
        # Update hostname for localhost URLs to use Docker service name
        cur.execute("""
            UPDATE mcp_manifests 
            SET hostname = 'mcp' 
            WHERE url LIKE '%localhost:9000%' 
            AND (hostname IS NULL OR hostname = '' OR hostname = 'localhost')
        """)
        
        updated_count = cur.rowcount
        
        if updated_count > 0:
            conn.commit()
            print(f"\nUpdated {updated_count} manifest(s) to use 'mcp' as hostname")
            
            # Show updated manifests
            cur.execute("SELECT id, url, hostname FROM mcp_manifests WHERE url LIKE '%localhost:9000%'")
            updated = cur.fetchall()
            print("\nUpdated manifests:")
            for manifest in updated:
                print(f"ID: {manifest[0]}, URL: {manifest[1]}, Hostname: {manifest[2]}")
        else:
            print("\nNo manifests needed updating")
            
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    conn.close()