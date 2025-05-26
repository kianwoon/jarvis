#!/usr/bin/env python3
"""
This script adds the api_key column to the mcp_manifests table.
Run this if you see errors about "column mcp_manifests.api_key does not exist".
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Add the project root directory to the Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Import the migration function
        from scripts.add_api_key_column import add_api_key_column
        
        # Run the migration
        logger.info("Starting API key column migration")
        success = add_api_key_column()
        
        if success:
            logger.info("Successfully added api_key column to mcp_manifests table")
            logger.info("You can now save MCP settings with API keys properly")
            return 0
        else:
            logger.error("Failed to add api_key column to mcp_manifests table")
            logger.error("Try running the initialize_mcp_tables endpoint first: GET /api/v1/settings/init-mcp-tables")
            return 1
    except Exception as e:
        logger.error(f"Error running migration: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 