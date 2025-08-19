#!/usr/bin/env python3
"""
Script to apply the communication_protocol migration to the database.
This adds the communication_protocol field to the mcp_servers table.
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Get database connection details
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'llm_platform')

# Create database URL
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def apply_migration():
    """Apply the communication_protocol migration."""
    try:
        # Create engine
        engine = create_engine(DATABASE_URL)
        
        # Read migration SQL
        migration_file = 'migrations/add_communication_protocol.sql'
        if not os.path.exists(migration_file):
            print(f"Error: Migration file {migration_file} not found")
            return False
        
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        # Apply migration
        with engine.connect() as conn:
            # Split the migration into individual statements
            statements = [s.strip() for s in migration_sql.split(';') if s.strip() and not s.strip().startswith('--')]
            
            for statement in statements:
                if statement:
                    print(f"Executing: {statement[:100]}...")
                    conn.execute(text(statement))
                    conn.commit()
        
        print("\n✅ Migration applied successfully!")
        
        # Verify the migration
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type, column_default 
                FROM information_schema.columns 
                WHERE table_name = 'mcp_servers' 
                AND column_name = 'communication_protocol'
            """))
            row = result.fetchone()
            
            if row:
                print(f"\n✅ Verified: communication_protocol column exists")
                print(f"   Data type: {row[1]}")
                print(f"   Default value: {row[2]}")
            else:
                print("\n⚠️  Warning: Column not found after migration")
        
        # Show current servers and their protocols
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT name, config_type, communication_protocol 
                FROM mcp_servers 
                ORDER BY id
            """))
            rows = result.fetchall()
            
            if rows:
                print("\n📊 Current MCP Servers:")
                print("-" * 60)
                print(f"{'Name':<30} {'Type':<15} {'Protocol':<15}")
                print("-" * 60)
                for row in rows:
                    print(f"{row[0]:<30} {row[1]:<15} {row[2]:<15}")
            else:
                print("\nNo MCP servers found in database")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error applying migration: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Applying communication_protocol migration...")
    print(f"Database: {POSTGRES_DB} on {POSTGRES_HOST}:{POSTGRES_PORT}")
    print("-" * 60)
    
    success = apply_migration()
    sys.exit(0 if success else 1)