#!/usr/bin/env python3
"""
Database migration script to add remote_config column to mcp_servers table
"""

import sys
import os
sys.path.append('/app')

from sqlalchemy import create_engine, text
from app.core.config import get_settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_database():
    """Add remote_config column to mcp_servers table"""
    
    # Get database settings
    settings = get_settings()
    DATABASE_URL = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )
    
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.connect() as conn:
            # Check if remote_config column already exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'mcp_servers' 
                AND column_name = 'remote_config'
            """))
            
            if result.fetchone():
                logger.info("remote_config column already exists. No migration needed.")
                return True
            
            # Add the remote_config column
            logger.info("Adding remote_config column to mcp_servers table...")
            conn.execute(text("""
                ALTER TABLE mcp_servers 
                ADD COLUMN remote_config JSON NULL
            """))
            
            # Commit the transaction
            conn.commit()
            logger.info("Successfully added remote_config column to mcp_servers table")
            return True
            
    except Exception as e:
        logger.error(f"Failed to add remote_config column: {str(e)}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    if success:
        print("Database migration completed successfully")
        sys.exit(0)
    else:
        print("Database migration failed")
        sys.exit(1)