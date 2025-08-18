#!/usr/bin/env python3
"""
MCP Bridge Server Startup Script

This script properly initializes the asyncio event loop before importing modules
that create async tasks at import time, preventing the "no running event loop" error.

The issue occurs because unified_mcp_service.py creates an MCPSubprocessPool instance
at import time, which tries to start an async cleanup task, but there's no event
loop running yet.

Usage:
    python start_mcp_bridge.py
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_event_loop():
    """
    Set up the asyncio event loop before importing modules that need it.
    
    This prevents the "RuntimeError: no running event loop" error that occurs
    when unified_mcp_service.py tries to create async tasks at import time.
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            logger.info("Found existing event loop")
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info("Created new event loop")
        
        return loop
    except Exception as e:
        logger.error(f"Failed to setup event loop: {e}")
        raise


def main():
    """
    Main entry point that properly initializes the event loop and starts the server.
    """
    try:
        logger.info("Starting MCP HTTP Bridge Server...")
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(".env.local")
            logger.info("Loaded environment variables from .env.local")
        except ImportError:
            logger.warning("python-dotenv not available, skipping .env.local loading")
        except Exception as e:
            logger.warning(f"Failed to load .env.local: {e}")
        
        # Setup event loop BEFORE importing modules that need it
        setup_event_loop()
        logger.info("Event loop initialized successfully")
        
        # Now it's safe to import modules that create async tasks at import time
        import uvicorn
        
        # Import the FastAPI app (this will now work without the asyncio error)
        from mcp_http_bridge_server import app
        logger.info("Successfully imported MCP bridge server app")
        
        # Start the server
        logger.info("Starting uvicorn server on 0.0.0.0:3001")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=3001,
            log_level="info",
            access_log=True,
            # Use the existing event loop
            loop="asyncio"
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start MCP HTTP Bridge Server: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()