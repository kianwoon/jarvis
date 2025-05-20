#!/usr/bin/env python3
import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database connection parameters (from environment or defaults)
DB_USER = os.environ.get('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'postgres')
DB_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
DB_PORT = os.environ.get('POSTGRES_PORT', '5432')
DB_NAME = os.environ.get('POSTGRES_DB', 'llm_platform')

# List of SQL migration files to execute in order
MIGRATION_FILES = [
    'scripts/create_settings_table.sql',
    'scripts/create_mcp_tools_table.sql',
]

def run_migrations():
    """Execute all pending migrations"""
    print(f"Running migrations for database {DB_NAME}...")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Run each migration file
        for migration_file in MIGRATION_FILES:
            print(f"Applying migration: {migration_file}")
            
            try:
                with open(migration_file, 'r') as f:
                    sql = f.read()
                    cursor.execute(sql)
                print(f"✅ Migration applied: {migration_file}")
            except Exception as e:
                print(f"❌ Error applying migration {migration_file}: {str(e)}")
                
        print("Migrations completed!")
        
    except Exception as e:
        print(f"Failed to connect to database: {str(e)}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    run_migrations() 