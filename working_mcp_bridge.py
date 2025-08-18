#!/usr/bin/env python3
"""
Working MCP HTTP Bridge Server

A fully functional MCP HTTP bridge server that:
1. Actually executes MCP tools (not mock responses)
2. Works with the fixed unified_mcp_service.py
3. Correctly imports only available functions
4. Provides real results, not mock data

The server:
- Listens on port 3001
- Has /health endpoint
- Has /tools/google_search endpoint that ACTUALLY calls the google_search MCP tool
- Uses the correct imports and functions available
- Returns REAL results
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# Import the existing MCP infrastructure
from app.core.config import get_settings
from app.core.unified_mcp_service import unified_mcp_service, call_mcp_tool_unified
from app.core.mcp_tools_cache import get_enabled_mcp_tools, reload_enabled_mcp_tools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Working MCP HTTP Bridge Server",
    description="Fully functional HTTP interface for MCP tool execution",
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

# Pydantic models for request/response
class ToolExecutionRequest(BaseModel):
    """Request model for tool execution"""
    query: Optional[str] = Field(None, description="Search query for google_search")
    num_results: Optional[int] = Field(5, description="Number of results to return")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Generic parameters for tools")

class ToolExecutionResponse(BaseModel):
    """Response model for tool execution"""
    success: bool = Field(..., description="Whether the execution was successful")
    result: Optional[Any] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    tool_name: str = Field(..., description="Name of the executed tool")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Server health status")
    timestamp: str = Field(..., description="Current server time")
    available_tools: int = Field(..., description="Number of available MCP tools")
    message: str = Field(..., description="Health check message")

# Helper function to get tool info from cache
def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get tool information from the MCP tools cache
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool configuration dict or None if not found
    """
    try:
        # Get all enabled tools from cache
        enabled_tools = get_enabled_mcp_tools()
        
        # Return the tool info if found
        if tool_name in enabled_tools:
            return enabled_tools[tool_name]
        
        # Try reloading cache if not found
        logger.info(f"Tool {tool_name} not in cache, reloading...")
        enabled_tools = reload_enabled_mcp_tools()
        
        if tool_name in enabled_tools:
            return enabled_tools[tool_name]
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting tool info for {tool_name}: {e}")
        return None

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns server status and basic statistics
    """
    try:
        # Get available tools from cache
        enabled_tools = get_enabled_mcp_tools()
        tool_count = len(enabled_tools) if enabled_tools else 0
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            available_tools=tool_count,
            message=f"Working MCP Bridge Server is running. {tool_count} tools available."
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def list_tools():
    """
    List all available MCP tools
    
    Returns a list of all cached MCP tools with their configurations
    """
    try:
        # Get tools from cache
        enabled_tools = get_enabled_mcp_tools()
        
        if not enabled_tools:
            # Try reloading
            enabled_tools = reload_enabled_mcp_tools()
        
        tools_list = []
        for tool_name, tool_info in enabled_tools.items():
            tools_list.append({
                "name": tool_name,
                "description": tool_info.get("description", ""),
                "endpoint": tool_info.get("endpoint", ""),
                "method": tool_info.get("method", ""),
                "server_id": tool_info.get("server_id"),
                "has_api_key": bool(tool_info.get("api_key"))
            })
        
        return {
            "total": len(tools_list),
            "tools": tools_list
        }
        
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/google_search")
async def execute_google_search(request: ToolExecutionRequest):
    """
    Direct endpoint for Google Search tool
    
    This endpoint specifically handles the google_search tool using the
    direct implementation in unified_mcp_service.
    
    Args:
        request: Search parameters (query, num_results, etc.)
        
    Returns:
        Search results from Google
    """
    start_time = time.time()
    
    try:
        logger.info(f"Google Search request: query='{request.query}', num_results={request.num_results}")
        
        # Prepare parameters
        parameters = {
            "query": request.query or "",
            "num_results": request.num_results or 5
        }
        
        # Merge with any additional parameters
        if request.parameters:
            parameters.update(request.parameters)
        
        # Validate query
        if not parameters.get("query"):
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        # Get tool configuration from cache for google_search
        tool_info = get_tool_info("google_search")
        
        if not tool_info:
            raise HTTPException(status_code=404, detail="google_search tool not found")
        
        # Use the unified MCP service to execute the tool
        logger.info(f"Calling google_search via unified MCP service with params: {parameters}")
        result = await call_mcp_tool_unified(
            tool_info=tool_info,
            tool_name="google_search",
            parameters=parameters
        )
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Google Search error: {result['error']}")
            return ToolExecutionResponse(
                success=False,
                error=result["error"],
                execution_time=time.time() - start_time,
                tool_name="google_search"
            )
        
        # Extract the text content if present
        if isinstance(result, dict) and "content" in result:
            # The result is in MCP format with content array
            content_text = ""
            for content_item in result.get("content", []):
                if content_item.get("type") == "text":
                    content_text = content_item.get("text", "")
                    break
            
            # Return the formatted result
            return ToolExecutionResponse(
                success=True,
                result={"content": content_text} if content_text else result,
                execution_time=time.time() - start_time,
                tool_name="google_search"
            )
        
        # Return raw result if not in expected format
        logger.info("Google Search completed successfully")
        return ToolExecutionResponse(
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            tool_name="google_search"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google Search failed: {e}", exc_info=True)
        return ToolExecutionResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            tool_name="google_search"
        )

@app.post("/tools/{tool_name}")
async def execute_tool(tool_name: str, request: Dict[str, Any]):
    """
    Execute any MCP tool by name
    
    This endpoint executes the specified MCP tool with the provided parameters
    using the unified MCP service infrastructure.
    
    Args:
        tool_name: Name of the tool to execute
        request: Parameters for the tool
        
    Returns:
        Tool execution result
    """
    start_time = time.time()
    
    try:
        logger.info(f"Executing tool: {tool_name} with params: {request}")
        
        # Get tool configuration from cache
        tool_info = get_tool_info(tool_name)
        
        if not tool_info:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
        
        # Extract parameters
        if "parameters" in request:
            parameters = request["parameters"]
        else:
            parameters = request
        
        # Use the unified MCP service to execute the tool
        result = await call_mcp_tool_unified(
            tool_info=tool_info,
            tool_name=tool_name,
            parameters=parameters
        )
        
        # Check for errors in result
        if isinstance(result, dict) and "error" in result:
            logger.error(f"Tool execution error: {result['error']}")
            return ToolExecutionResponse(
                success=False,
                error=result["error"],
                execution_time=time.time() - start_time,
                tool_name=tool_name
            )
        
        # Successful execution
        logger.info(f"Tool {tool_name} executed successfully")
        return ToolExecutionResponse(
            success=True,
            result=result,
            execution_time=time.time() - start_time,
            tool_name=tool_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
        return ToolExecutionResponse(
            success=False,
            error=str(e),
            execution_time=time.time() - start_time,
            tool_name=tool_name
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler to catch any unhandled errors
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
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
    logger.info("Working MCP HTTP Bridge Server starting up...")
    
    # Load/reload MCP tools cache
    try:
        enabled_tools = reload_enabled_mcp_tools()
        logger.info(f"Loaded {len(enabled_tools)} MCP tools into cache")
        
        # Log some of the available tools
        tool_names = list(enabled_tools.keys())[:10]  # First 10 tools
        logger.info(f"Sample tools available: {tool_names}")
        
        # Check if google_search is available
        if "google_search" in enabled_tools:
            logger.info("✓ google_search tool is available")
        else:
            logger.warning("⚠ google_search tool not found in cache")
            
    except Exception as e:
        logger.error(f"Failed to load MCP tools cache: {e}")
    
    logger.info("Working MCP HTTP Bridge Server started successfully on port 3001")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    """
    logger.info("Working MCP HTTP Bridge Server shutting down...")
    
    # Clean up unified MCP service resources
    try:
        await unified_mcp_service.close()
        logger.info("Unified MCP service resources cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up unified MCP service: {e}")
    
    logger.info("Working MCP HTTP Bridge Server shutdown complete")

# Main entry point
if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    
    # Try multiple env file locations
    env_files = [".env.local", ".env"]
    for env_file in env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
            break
    
    # MCP tools are now self-describing through their database configuration
    # No need for hardcoded environment variable setup
    
    # Run the server
    logger.info("Starting Working MCP HTTP Bridge Server on port 3001...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_level="info",
        access_log=True
    )