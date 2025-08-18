#!/usr/bin/env python3
"""
Simple MCP HTTP Bridge Server

This is a minimal, standalone HTTP bridge server that provides basic MCP tool endpoints
without importing the complex unified_mcp_service (which has event loop issues).

Features:
- Runs on port 3001
- Health check endpoint: GET /health
- Google Search endpoint: POST /tools/google_search (returns mock response for now)
- No complex dependencies or imports that cause event loop conflicts

This serves as a temporary solution while the event loop issues in unified_mcp_service
are being resolved.

Usage:
    python simple_mcp_bridge.py
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Simple MCP HTTP Bridge Server",
    description="Minimal HTTP interface for MCP tools (mock responses)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Server health status")
    timestamp: str = Field(..., description="Current server time")
    message: str = Field(..., description="Status message")

class GoogleSearchRequest(BaseModel):
    """Request model for Google Search"""
    query: str = Field(..., description="Search query")
    num_results: int = Field(default=10, description="Number of results to return")

class GoogleSearchResponse(BaseModel):
    """Response model for Google Search"""
    success: bool = Field(..., description="Whether the request was successful")
    result: Dict[str, Any] = Field(..., description="Search result data")
    execution_time: float = Field(..., description="Execution time in seconds")

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint - returns basic server status
    """
    try:
        return HealthResponse(
            status="ok",
            timestamp=datetime.utcnow().isoformat(),
            message="Simple MCP bridge is running"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tools/google_search", response_model=GoogleSearchResponse)
async def google_search(request: GoogleSearchRequest):
    """
    Google Search endpoint
    
    Currently returns a mock response since we can't use unified_mcp_service
    due to event loop issues. The actual tool execution will be implemented
    once the event loop issue is resolved.
    
    Args:
        request: Search request with query and optional num_results
        
    Returns:
        Mock search response with placeholder data
    """
    start_time = time.time()
    
    try:
        logger.info(f"Google Search request: query='{request.query}', num_results={request.num_results}")
        
        # Create mock response to show the bridge is working
        mock_result = {
            "message": "MCP bridge is running but needs unified_mcp_service fix to actually execute tools",
            "query": request.query,
            "note": "This is a placeholder response. The actual tool execution requires fixing the event loop issue in unified_mcp_service.py",
            "requested_results": request.num_results,
            "mock_data": {
                "search_performed": True,
                "timestamp": datetime.utcnow().isoformat(),
                "bridge_status": "operational",
                "next_steps": [
                    "Fix event loop issue in unified_mcp_service.py",
                    "Replace this mock response with actual Google Search API call",
                    "Test with real MCP tool execution"
                ]
            }
        }
        
        execution_time = time.time() - start_time
        logger.info(f"Mock Google Search completed in {execution_time:.3f}s")
        
        return GoogleSearchResponse(
            success=True,
            result=mock_result,
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Google Search error: {e}")
        
        return GoogleSearchResponse(
            success=False,
            result={
                "error": str(e),
                "query": request.query,
                "message": "Error in simple MCP bridge"
            },
            execution_time=execution_time
        )

@app.post("/tools/google_search")
async def google_search_flexible(request: Dict[str, Any]):
    """
    Alternative Google Search endpoint that accepts flexible JSON input
    
    This endpoint handles various request formats for compatibility
    with different frontend implementations.
    """
    start_time = time.time()
    
    try:
        # Extract query from various possible request formats
        query = request.get("query", "")
        if not query:
            # Try alternative field names
            query = request.get("q", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        num_results = request.get("num_results", 10)
        
        logger.info(f"Flexible Google Search request: query='{query}', num_results={num_results}")
        
        # Create mock response
        mock_result = {
            "message": "MCP bridge is running but needs unified_mcp_service fix to actually execute tools",
            "query": query,
            "note": "This is a placeholder response. The actual tool execution requires fixing the event loop issue in unified_mcp_service.py",
            "requested_results": num_results,
            "request_format": "flexible",
            "mock_data": {
                "search_performed": True,
                "timestamp": datetime.utcnow().isoformat(),
                "bridge_status": "operational"
            }
        }
        
        execution_time = time.time() - start_time
        logger.info(f"Flexible Mock Google Search completed in {execution_time:.3f}s")
        
        return {
            "success": True,
            "result": mock_result,
            "execution_time": execution_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Flexible Google Search error: {e}")
        
        return {
            "success": False,
            "result": {
                "error": str(e),
                "query": request.get("query", "unknown"),
                "message": "Error in simple MCP bridge (flexible endpoint)"
            },
            "execution_time": execution_time
        }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Global exception handler
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "message": "Simple MCP bridge encountered an error"
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler
    """
    logger.info("=" * 60)
    logger.info("Simple MCP HTTP Bridge Server starting up...")
    logger.info("Port: 3001")
    logger.info("Endpoints:")
    logger.info("  GET  /health                - Health check")
    logger.info("  POST /tools/google_search   - Google Search (mock)")
    logger.info("=" * 60)
    logger.info("Note: This is a temporary bridge with mock responses.")
    logger.info("Actual tool execution requires fixing unified_mcp_service.py")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    """
    logger.info("Simple MCP HTTP Bridge Server shutting down...")

# Main entry point
if __name__ == "__main__":
    print("Starting Simple MCP HTTP Bridge Server...")
    print("This server provides mock responses while the event loop issue is fixed.")
    print("Listening on: http://localhost:3001")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_level="info",
        access_log=True
    )