#!/usr/bin/env python3
"""
Database migration script to update MCP schema for command-based servers
This script migrates the existing manifest-based MCP system to support both
manifest and command-based MCP server configurations.
"""

import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import get_settings
from app.core.db import Base, MCPServer, MCPManifest, MCPTool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the database migration"""
    settings = get_settings()
    
    # Try PostgreSQL first, fall back to SQLite
    try:
        DATABASE_URL = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            pass
        logger.info("Using PostgreSQL database")
        is_sqlite = False
    except Exception as e:
        logger.info(f"PostgreSQL connection failed: {e}")
        logger.info("Using SQLite database")
        DB_PATH = os.path.join(os.path.dirname(__file__), "sqlite.db")
        DATABASE_URL = f"sqlite:///{DB_PATH}"
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        is_sqlite = True
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        with engine.begin() as conn:
            logger.info("Starting migration...")
            
            # Step 1: Create new mcp_servers table
            logger.info("Creating mcp_servers table...")
            if is_sqlite:
                create_servers_sql = """
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name VARCHAR(255) NOT NULL,
                    config_type VARCHAR(20) NOT NULL,
                    manifest_url VARCHAR(255),
                    hostname VARCHAR(255),
                    api_key VARCHAR(255),
                    command VARCHAR(500),
                    args JSON,
                    env JSON,
                    working_directory VARCHAR(500),
                    process_id INTEGER,
                    is_running BOOLEAN DEFAULT 0,
                    restart_policy VARCHAR(20) DEFAULT 'on-failure',
                    max_restarts INTEGER DEFAULT 3,
                    restart_count INTEGER DEFAULT 0,
                    is_active BOOLEAN NOT NULL DEFAULT 1,
                    last_health_check TIMESTAMP,
                    health_status VARCHAR(20) DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            else:
                create_servers_sql = """
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    config_type VARCHAR(20) NOT NULL,
                    manifest_url VARCHAR(255),
                    hostname VARCHAR(255),
                    api_key VARCHAR(255),
                    command VARCHAR(500),
                    args JSONB,
                    env JSONB,
                    working_directory VARCHAR(500),
                    process_id INTEGER,
                    is_running BOOLEAN DEFAULT false,
                    restart_policy VARCHAR(20) DEFAULT 'on-failure',
                    max_restarts INTEGER DEFAULT 3,
                    restart_count INTEGER DEFAULT 0,
                    is_active BOOLEAN NOT NULL DEFAULT true,
                    last_health_check TIMESTAMP WITH TIME ZONE,
                    health_status VARCHAR(20) DEFAULT 'unknown',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
                );
                """
            
            conn.execute(text(create_servers_sql))
            logger.info("mcp_servers table created successfully")
            
            # Step 2: Migrate existing manifests to new server structure
            logger.info("Migrating existing manifests...")
            
            # Check if old manifest table exists
            if is_sqlite:
                check_manifests_sql = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='mcp_manifests';
                """
            else:
                check_manifests_sql = """
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'mcp_manifests';
                """
            
            result = conn.execute(text(check_manifests_sql))
            table_exists = result.fetchone() is not None
            
            if table_exists:
                # Add server_id column to mcp_manifests first
                logger.info("Adding server_id column to mcp_manifests...")
                if is_sqlite:
                    check_column_sql = """
                    PRAGMA table_info(mcp_manifests);
                    """
                    columns = conn.execute(text(check_column_sql)).fetchall()
                    has_server_id = any(col[1] == 'server_id' for col in columns)
                    
                    if not has_server_id:
                        alter_manifests_sql = """
                        ALTER TABLE mcp_manifests ADD COLUMN server_id INTEGER;
                        """
                        conn.execute(text(alter_manifests_sql))
                        logger.info("Added server_id column to mcp_manifests")
                else:
                    alter_manifests_sql = """
                    ALTER TABLE mcp_manifests 
                    ADD COLUMN IF NOT EXISTS server_id INTEGER REFERENCES mcp_servers(id);
                    """
                    conn.execute(text(alter_manifests_sql))
                    logger.info("Added server_id column to mcp_manifests")
                
                # Get existing manifests
                get_manifests_sql = "SELECT * FROM mcp_manifests;"
                manifests = conn.execute(text(get_manifests_sql)).fetchall()
                
                logger.info(f"Found {len(manifests)} existing manifests to migrate")
                
                for manifest in manifests:
                    # Extract server name from manifest content or use hostname
                    try:
                        import json
                        content = json.loads(manifest.content) if isinstance(manifest.content, str) else manifest.content
                        server_name = content.get("name", manifest.hostname or f"Server {manifest.id}")
                    except:
                        server_name = manifest.hostname or f"Server {manifest.id}"
                    
                    # Insert into new servers table
                    insert_server_sql = """
                    INSERT INTO mcp_servers (
                        name, config_type, manifest_url, hostname, api_key,
                        is_active, health_status, created_at, updated_at
                    ) VALUES (
                        :name, 'manifest', :manifest_url, :hostname, :api_key,
                        :is_active, 'unknown', :created_at, :updated_at
                    );
                    """
                    
                    server_result = conn.execute(text(insert_server_sql), {
                        "name": server_name,
                        "manifest_url": manifest.url,
                        "hostname": manifest.hostname,
                        "api_key": manifest.api_key,
                        "is_active": True,
                        "created_at": manifest.created_at,
                        "updated_at": manifest.updated_at
                    })
                    
                    # Get the new server ID
                    if is_sqlite:
                        new_server_id = server_result.lastrowid
                    else:
                        new_server_id = conn.execute(text("SELECT lastval();")).scalar()
                    
                    # Update manifest to reference new server
                    update_manifest_sql = """
                    UPDATE mcp_manifests SET server_id = :server_id WHERE id = :manifest_id;
                    """
                    conn.execute(text(update_manifest_sql), {
                        "server_id": new_server_id,
                        "manifest_id": manifest.id
                    })
                    
                    logger.info(f"Migrated manifest {manifest.id} to server {new_server_id}")
                
                logger.info("Manifest migration completed")
            else:
                logger.info("No existing manifests found to migrate")
            
            # Step 3: Update mcp_tools table to reference servers instead of manifests
            logger.info("Updating mcp_tools table structure...")
            
            if is_sqlite:
                # Check if server_id column exists
                check_tools_column_sql = """
                PRAGMA table_info(mcp_tools);
                """
                tool_columns = conn.execute(text(check_tools_column_sql)).fetchall()
                has_server_id = any(col[1] == 'server_id' for col in tool_columns)
                
                if not has_server_id:
                    alter_tools_sql = """
                    ALTER TABLE mcp_tools ADD COLUMN server_id INTEGER;
                    """
                    conn.execute(text(alter_tools_sql))
                    logger.info("Added server_id column to mcp_tools")
                    
                    # Update existing tools to reference servers
                    update_tools_sql = """
                    UPDATE mcp_tools 
                    SET server_id = (
                        SELECT server_id 
                        FROM mcp_manifests 
                        WHERE mcp_manifests.id = mcp_tools.manifest_id
                    )
                    WHERE manifest_id IS NOT NULL;
                    """
                    conn.execute(text(update_tools_sql))
                    logger.info("Updated existing tools to reference servers")
            else:
                alter_tools_sql = """
                ALTER TABLE mcp_tools 
                ADD COLUMN IF NOT EXISTS server_id INTEGER REFERENCES mcp_servers(id);
                """
                conn.execute(text(alter_tools_sql))
                
                # Update existing tools to reference servers
                update_tools_sql = """
                UPDATE mcp_tools 
                SET server_id = (
                    SELECT server_id 
                    FROM mcp_manifests 
                    WHERE mcp_manifests.id = mcp_tools.manifest_id
                )
                WHERE manifest_id IS NOT NULL AND server_id IS NULL;
                """
                conn.execute(text(update_tools_sql))
                logger.info("Updated mcp_tools table structure and references")
            
            logger.info("Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    
    # Step 5: Create all tables using SQLAlchemy (for any missing tables)
    logger.info("Creating any missing tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema is up to date!")

if __name__ == "__main__":
    try:
        run_migration()
        logger.info("✅ Migration completed successfully!")
        logger.info("\n📋 Summary of changes:")
        logger.info("   • Created new mcp_servers table with support for both manifest and command configurations")
        logger.info("   • Migrated existing manifest records to the new server structure")
        logger.info("   • Updated mcp_tools and mcp_manifests to reference servers")
        logger.info("   • Added process management fields for command-based servers")
        logger.info("\n🚀 Your MCP platform now supports both manifest and command-based server configurations!")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        sys.exit(1)