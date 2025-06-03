#!/usr/bin/env python3
"""
PostgreSQL Migration Script for MCP Schema Enhancement
Run this script in your Docker environment to migrate the database schema.
"""

import sys
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from app.core.config import get_settings
    settings = get_settings()
except ImportError:
    # Fallback to environment variables if config module not available
    class Settings:
        POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
        POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
        POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
        POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
        POSTGRES_DB = os.getenv('POSTGRES_DB', 'llm_platform')
    
    settings = Settings()

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the PostgreSQL migration"""
    
    # Database connection parameters
    db_params = {
        'host': settings.POSTGRES_HOST,
        'port': settings.POSTGRES_PORT,
        'user': settings.POSTGRES_USER,
        'password': settings.POSTGRES_PASSWORD,
        'database': settings.POSTGRES_DB
    }
    
    logger.info(f"Connecting to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        logger.info("Connected to PostgreSQL successfully")
        
        # Read the SQL migration file
        migration_file = os.path.join(os.path.dirname(__file__), 'migrate_postgres_schema.sql')
        
        if not os.path.exists(migration_file):
            raise FileNotFoundError(f"Migration file not found: {migration_file}")
        
        with open(migration_file, 'r') as f:
            migration_sql = f.read()
        
        logger.info("Executing migration script...")
        
        # Split the SQL into individual statements (excluding comments and empty lines)
        statements = []
        current_statement = []
        in_function = False
        
        for line in migration_sql.split('\n'):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('--'):
                continue
            
            # Detect function definitions
            if 'CREATE OR REPLACE FUNCTION' in line.upper():
                in_function = True
            elif line.endswith('$$ LANGUAGE plpgsql;'):
                in_function = False
                current_statement.append(line)
                statements.append('\n'.join(current_statement))
                current_statement = []
                continue
            
            current_statement.append(line)
            
            # For non-function statements, split on semicolon
            if not in_function and line.endswith(';'):
                statements.append('\n'.join(current_statement))
                current_statement = []
        
        # Add any remaining statement
        if current_statement:
            statements.append('\n'.join(current_statement))
        
        # Execute each statement
        for i, statement in enumerate(statements):
            if statement.strip():
                try:
                    logger.info(f"Executing statement {i+1}/{len(statements)}")
                    cursor.execute(statement)
                    logger.info(f"✅ Statement {i+1} executed successfully")
                except psycopg2.Error as e:
                    if "already exists" in str(e) or "does not exist" in str(e):
                        logger.warning(f"⚠️ Statement {i+1}: {e}")
                    else:
                        logger.error(f"❌ Error in statement {i+1}: {e}")
                        logger.error(f"Statement: {statement[:100]}...")
                        raise
        
        # Verify migration results
        logger.info("Verifying migration results...")
        
        verification_queries = [
            "SELECT COUNT(*) FROM mcp_servers",
            "SELECT COUNT(*) FROM mcp_tools WHERE server_id IS NOT NULL",
            "SELECT COUNT(*) FROM mcp_manifests WHERE server_id IS NOT NULL",
            """
            SELECT 
                table_name,
                column_name,
                data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND table_name IN ('mcp_servers', 'mcp_tools', 'mcp_manifests')
            AND column_name = 'server_id'
            ORDER BY table_name
            """
        ]
        
        for query in verification_queries:
            cursor.execute(query)
            results = cursor.fetchall()
            logger.info(f"Query result: {results}")
        
        logger.info("🎉 Migration completed successfully!")
        
        # Print summary
        cursor.execute("""
            SELECT 
                'mcp_servers' as table_name, 
                COUNT(*) as total_count,
                COUNT(CASE WHEN config_type = 'manifest' THEN 1 END) as manifest_count,
                COUNT(CASE WHEN config_type = 'command' THEN 1 END) as command_count
            FROM mcp_servers
        """)
        
        server_stats = cursor.fetchone()
        logger.info(f"📊 MCP Servers: {server_stats[1]} total ({server_stats[2]} manifest, {server_stats[3]} command)")
        
        cursor.execute("SELECT COUNT(*) FROM mcp_tools WHERE server_id IS NOT NULL")
        tools_with_server = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mcp_tools")
        total_tools = cursor.fetchone()[0]
        
        logger.info(f"📊 MCP Tools: {tools_with_server}/{total_tools} linked to servers")
        
        cursor.execute("SELECT COUNT(*) FROM mcp_manifests WHERE server_id IS NOT NULL")
        manifests_with_server = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM mcp_manifests")
        total_manifests = cursor.fetchone()[0]
        
        logger.info(f"📊 MCP Manifests: {manifests_with_server}/{total_manifests} linked to servers")
        
    except psycopg2.Error as e:
        logger.error(f"❌ PostgreSQL error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")
    
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting PostgreSQL MCP schema migration...")
    logger.info("This will add support for command-based MCP servers")
    
    success = run_migration()
    
    if success:
        logger.info("✅ Migration completed successfully!")
        logger.info("")
        logger.info("📋 Changes made:")
        logger.info("  • Created mcp_servers table with command and manifest support")
        logger.info("  • Added server_id column to mcp_tools table")
        logger.info("  • Added server_id column to mcp_manifests table")
        logger.info("  • Migrated existing manifests to new server structure")
        logger.info("  • Updated foreign key relationships")
        logger.info("  • Added performance indexes")
        logger.info("")
        logger.info("🚀 Your MCP platform now supports command-based servers!")
    else:
        logger.error("❌ Migration failed!")
        sys.exit(1)