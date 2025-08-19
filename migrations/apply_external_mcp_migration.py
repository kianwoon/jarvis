#!/usr/bin/env python3
"""
Apply migration to add External MCP Server as a configurable command server.
This script can be run manually or integrated into the startup process.
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.db import get_db_session, MCPServer
from app.core.config import get_settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_migration():
    """Apply the External MCP Server migration."""
    try:
        with get_db_session() as db:
            # Check if External MCP Server already exists
            existing_server = db.query(MCPServer).filter(
                MCPServer.name == "External MCP Server"
            ).first()
            
            if not existing_server:
                # Create new External MCP Server entry
                external_server = MCPServer(
                    name="External MCP Server",
                    config_type="command",
                    command="npm",
                    args=["start"],
                    working_directory="/Users/kianwoonwong/Downloads/MCP",
                    env={
                        "MCP_MODE": "http",
                        "MCP_PORT": "3001"
                    },
                    restart_policy="on-failure",
                    max_restarts=3,
                    is_active=True,
                    enhanced_error_handling_config={
                        "enabled": True,
                        "max_tool_retries": 3,
                        "retry_base_delay": 1.0,
                        "retry_max_delay": 60.0,
                        "retry_backoff_multiplier": 2.0,
                        "timeout_seconds": 30,
                        "enable_circuit_breaker": True,
                        "circuit_failure_threshold": 5,
                        "circuit_recovery_timeout": 60
                    },
                    auth_refresh_config={
                        "enabled": False,
                        "server_type": "custom",
                        "auth_type": "oauth2",
                        "refresh_method": "POST",
                        "token_expiry_buffer_minutes": 5
                    }
                )
                
                db.add(external_server)
                db.commit()
                logger.info("✅ External MCP Server added successfully as a command-based server")
                return True
                
            elif existing_server.config_type != "command":
                # Update existing server to command type
                existing_server.config_type = "command"
                existing_server.command = "npm"
                existing_server.args = ["start"]
                existing_server.working_directory = "/Users/kianwoonwong/Downloads/MCP"
                existing_server.env = {
                    "MCP_MODE": "http",
                    "MCP_PORT": "3001"
                }
                existing_server.restart_policy = "on-failure"
                existing_server.max_restarts = 3
                existing_server.updated_at = datetime.utcnow()
                
                db.commit()
                logger.info("✅ External MCP Server updated to command-based configuration")
                return True
                
            else:
                logger.info("ℹ️ External MCP Server already exists with correct configuration")
                
                # Optional: Update working directory if it's different
                if existing_server.working_directory != "/Users/kianwoonwong/Downloads/MCP":
                    existing_server.working_directory = "/Users/kianwoonwong/Downloads/MCP"
                    db.commit()
                    logger.info("✅ Updated working directory for External MCP Server")
                    
                return False
                
    except Exception as e:
        logger.error(f"❌ Error applying migration: {e}")
        raise

def verify_migration():
    """Verify the migration was applied correctly."""
    try:
        with get_db_session() as db:
            server = db.query(MCPServer).filter(
                MCPServer.name == "External MCP Server"
            ).first()
            
            if server:
                logger.info("\n📋 External MCP Server Configuration:")
                logger.info(f"  ID: {server.id}")
                logger.info(f"  Name: {server.name}")
                logger.info(f"  Type: {server.config_type}")
                logger.info(f"  Command: {server.command}")
                logger.info(f"  Args: {server.args}")
                logger.info(f"  Working Dir: {server.working_directory}")
                logger.info(f"  Environment: {json.dumps(server.env, indent=2)}")
                logger.info(f"  Active: {server.is_active}")
                logger.info(f"  Restart Policy: {server.restart_policy}")
                return True
            else:
                logger.error("❌ External MCP Server not found in database")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error verifying migration: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting External MCP Server migration...")
    
    # Apply the migration
    applied = apply_migration()
    
    # Verify the migration
    if verify_migration():
        logger.info("\n✅ Migration completed successfully!")
        if applied:
            logger.info("💡 You can now manage the External MCP Server from the Settings UI")
            logger.info("💡 Use the Actions menu to Start/Stop/Restart the server")
    else:
        logger.error("\n❌ Migration verification failed!")
        sys.exit(1)