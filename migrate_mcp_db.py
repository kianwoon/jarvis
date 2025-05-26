#!/usr/bin/env python3
"""
Script to migrate the MCP database schema, adding the api_key column to mcp_manifests table
and migrating existing api_keys from settings to mcp_manifests.
"""

import sys
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the migration script.
    """
    logger.info("Starting MCP database migration")
    
    # Step 1: Initialize MCP tables if they don't exist and add the api_key column
    try:
        logger.info("Step 1: Initializing MCP tables")
        response = requests.get("http://localhost:8000/api/v1/settings/init-mcp-tables")
        if response.status_code != 200:
            logger.error(f"Failed to initialize MCP tables: {response.status_code}")
            logger.error(response.text)
            return False
        
        result = response.json()
        logger.info(f"MCP tables initialization result: {result}")
        
        if not result.get("success"):
            logger.error("MCP tables initialization was not successful")
            return False
        
        logger.info("Successfully initialized MCP tables")
    except Exception as e:
        logger.error(f"Error initializing MCP tables: {str(e)}")
        return False
    
    # Step 2: Run the migration script to migrate data
    try:
        logger.info("Step 2: Running migration script")
        from scripts.add_api_key_column import add_api_key_column
        
        success = add_api_key_column()
        if not success:
            logger.error("Migration script was not successful")
            return False
        
        logger.info("Successfully ran migration script")
    except Exception as e:
        logger.error(f"Error running migration script: {str(e)}")
        return False
    
    logger.info("MCP database migration completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("Migration failed")
        sys.exit(1) 