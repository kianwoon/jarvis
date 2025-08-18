#!/usr/bin/env python3
"""
MCP HTTP Bridge Server

This server provides an HTTP interface to MCP tools, allowing external services
to call MCP tools via HTTP requests. It integrates with the existing Jarvis
MCP infrastructure.

The server:
- Listens on port 3001
- Provides a REST API for MCP tool execution
- Uses the existing unified MCP service for tool execution
- Supports the google_search tool with server_id=9

Usage:
    python mcp_http_bridge_server.py
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Import the existing MCP infrastructure
from app.core.db import MCPTool, MCPServer
from app.core.config import get_settings
from app.core.unified_mcp_service import unified_mcp_service, call_mcp_tool_unified
from app.core.mcp_tools_cache import get_enabled_mcp_tools, get_tool_info

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MCP HTTP Bridge Server",
    description="HTTP interface for MCP tool execution",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
settings = get_settings()
DATABASE_URL = (
    f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pydantic models for request/response
class ToolExecutionRequest(BaseModel):
    """Request model for tool execution"""
    tool_name: str = Field(..., description="Name of the MCP tool to execute")
    parameters: Dict[str, Any] = Field(default={}, description="Parameters for the tool")
    server_id: Optional[int] = Field(None, description="Optional server ID to use specific MCP server")

class ToolExecutionResponse(BaseModel):
    """Response model for tool execution"""
    success: bool = Field(..., description="Whether the execution was successful")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    tool_name: str = Field(..., description="Name of the executed tool")
    server_id: Optional[int] = Field(None, description="ID of the MCP server used")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Server health status")
    timestamp: str = Field(..., description="Current server time")
    available_tools: int = Field(..., description="Number of available MCP tools")
    database_connected: bool = Field(..., description="Database connection status")

# Helper functions
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

async def get_tool_config(tool_name: str, server_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Get tool configuration from database or cache
    
    Args:
        tool_name: Name of the tool
        server_id: Optional specific server ID to use
        
    Returns:
        Tool configuration dict or None if not found
    """
    try:
        # Try to get from cache first
        tool_info = get_tool_info(tool_name)
        if tool_info:
            logger.info(f"Found tool {tool_name} in cache")
            return tool_info
        
        # If not in cache or specific server_id requested, query database
        db = get_db()
        try:
            query = db.query(MCPTool).filter(
                MCPTool.name == tool_name,
                MCPTool.is_active == True
            )
            
            if server_id:
                query = query.filter(MCPTool.server_id == server_id)
            
            tool = query.first()
            
            if not tool:
                logger.warning(f"Tool {tool_name} not found in database")
                return None
            
            # Build tool info dict
            tool_config = {
                "name": tool.name,
                "description": tool.description,
                "endpoint": tool.endpoint,
                "method": tool.method,
                "parameters": tool.parameters,
                "headers": tool.headers,
                "server_id": tool.server_id
            }
            
            # Get server info if available
            if tool.server_id:
                server = db.query(MCPServer).filter(MCPServer.id == tool.server_id).first()
                if server:
                    tool_config["server_name"] = server.name
                    tool_config["server_config"] = {
                        "command": server.command,
                        "args": server.args,
                        "env": server.env,
                        "working_directory": server.working_directory
                    }
            
            logger.info(f"Found tool {tool_name} in database with server_id={tool.server_id}")
            return tool_config
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting tool config for {tool_name}: {e}")
        return None

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns server status and basic statistics
    """
    try:
        # Check database connection
        db_connected = False
        available_tools = 0
        
        try:
            db = get_db()
            available_tools = db.query(MCPTool).filter(MCPTool.is_active == True).count()
            db_connected = True
            db.close()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            available_tools=available_tools,
            database_connected=db_connected
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools():
    """
    List all available MCP tools
    
    Returns a list of all active MCP tools with their configurations
    """
    try:
        # Get tools from cache
        cached_tools = get_enabled_mcp_tools()
        
        # Also get from database for complete list
        db = get_db()
        try:
            db_tools = db.query(MCPTool).filter(MCPTool.is_active == True).all()
            
            tools_list = []
            for tool in db_tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "endpoint": tool.endpoint,
                    "method": tool.method,
                    "server_id": tool.server_id,
                    "in_cache": tool.name in cached_tools
                }
                
                # Add server name if available
                if tool.server_id:
                    server = db.query(MCPServer).filter(MCPServer.id == tool.server_id).first()
                    if server:
                        tool_info["server_name"] = server.name
                
                tools_list.append(tool_info)
            
            return {
                "total": len(tools_list),
                "tools": tools_list
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools/{tool_name}")
async def get_tool_info_endpoint(tool_name: str):
    """
    Get detailed information about a specific tool
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Detailed tool configuration
    """
    try:
        tool_config = await get_tool_config(tool_name)
        
        if not tool_config:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        return tool_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool info for {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/{tool_name}", response_model=ToolExecutionResponse)
async def execute_tool(tool_name: str, request: ToolExecutionRequest):
    """
    Execute an MCP tool
    
    This endpoint executes the specified MCP tool with the provided parameters.
    It uses the existing unified MCP service infrastructure.
    
    Args:
        tool_name: Name of the tool to execute
        request: Execution request with parameters
        
    Returns:
        Tool execution result
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Executing tool: {tool_name} with params: {request.parameters}")
        
        # Validate tool name matches
        if request.tool_name != tool_name:
            raise HTTPException(
                status_code=400, 
                detail=f"Tool name mismatch: URL says '{tool_name}', body says '{request.tool_name}'"
            )
        
        # Get tool configuration
        tool_config = await get_tool_config(tool_name, request.server_id)
        
        if not tool_config:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        # Special handling for google_search
        if tool_name == "google_search":
            # Direct call to the emergency bypass in unified_mcp_service
            result = await unified_mcp_service._direct_google_search(request.parameters)
        else:
            # Use the unified MCP service to execute the tool
            result = await call_mcp_tool_unified(
                tool_info=tool_config,
                tool_name=tool_name,
                parameters=request.parameters
            )
        
        # Check for errors in result
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Tool execution error: {result['error']}")
            return ToolExecutionResponse(
                success=False,
                error=result["error"],
                execution_time=time.time() - start_time,
                tool_name=tool_name,
                server_id=tool_config.get("server_id")
            )
        
        # Successful execution
        logger.info(f"Tool {tool_name} executed successfully")
        return ToolExecutionResponse(
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            tool_name=tool_name,
            server_id=tool_config.get("server_id")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return ToolExecutionResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            tool_name=tool_name,
            server_id=request.server_id
        )

@app.post("/tools/google_search", response_model=ToolExecutionResponse)
async def execute_google_search(request: Dict[str, Any]):
    """
    Direct endpoint for Google Search tool
    
    This endpoint specifically handles the google_search tool,
    which is configured with server_id=9 in the database.
    
    Args:
        request: Search parameters (query, num_results, etc.)
        
    Returns:
        Search results
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Google Search request: {request}")
        
        # Extract parameters
        if "parameters" in request:
            parameters = request["parameters"]
        else:
            parameters = request
        
        # Use the direct Google search method
        result = await unified_mcp_service._direct_google_search(parameters)
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Google Search error: {result['error']}")
            return ToolExecutionResponse(
                success=False,
                error=result["error"],
                execution_time=time.time() - start_time,
                tool_name="google_search",
                server_id=9
            )
        
        # Successful search
        logger.info("Google Search completed successfully")
        return ToolExecutionResponse(
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            tool_name="google_search",
            server_id=9
        )
        
    except Exception as e:
        logger.error(f"Google Search failed: {e}")
        return ToolExecutionResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            tool_name="google_search",
            server_id=9
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to catch any unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "detail": "An unexpected error occurred"
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler
    """
    logger.info("MCP HTTP Bridge Server starting up...")
    
    # Test database connection
    try:
        db = get_db()
        tool_count = db.query(MCPTool).filter(MCPTool.is_active == True).count()
        logger.info(f"Database connected. Found {tool_count} active MCP tools")
        db.close()
    except Exception as e:
        logger.error(f"Database connection failed during startup: {e}")
    
    logger.info("MCP HTTP Bridge Server started successfully on port 3001")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    """
    logger.info("MCP HTTP Bridge Server shutting down...")
    
    # Clean up unified MCP service resources
    try:
        await unified_mcp_service.close()
        logger.info("Unified MCP service resources cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up unified MCP service: {e}")
    
    logger.info("MCP HTTP Bridge Server shutdown complete")

# Main entry point
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(".env.local")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_level="info",
        access_log=True
    )