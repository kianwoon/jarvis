#!/usr/bin/env python
"""
Script to add a test manifest with an API key
"""
import os
import sys
import json

# Add the parent directory to the path so we can import app
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from app.core.db import SessionLocal, MCPManifest, MCPTool, Base, engine

def add_test_manifest():
    # First, ensure the tables exist
    try:
        print("Creating database tables if they don't exist...")
        Base.metadata.create_all(bind=engine)
        print("Database tables created/verified")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        return

    db = SessionLocal()
    try:
        # Create a test manifest with API key
        manifest_url = "http://localhost:9000/manifest"
        api_key = "test-api-key-12345"
        hostname = "mcp-jira-outlook:9000"
        
        print(f"\nAdding test manifest with URL: {manifest_url}")
        print(f"API Key: {api_key}")
        print(f"Hostname: {hostname}")
        
        # Check if manifest already exists
        existing_manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
        if existing_manifest:
            print(f"Manifest with URL {manifest_url} already exists. Updating it.")
            existing_manifest.api_key = api_key
            existing_manifest.hostname = hostname
            manifest = existing_manifest
        else:
            # Create sample manifest content
            manifest_content = {
                "name": "MCP Server",
                "description": "MCP Server for Jira and Outlook integration",
                "version": "1.0.0",
                "tools": [
                    {
                        "name": "get_datetime",
                        "description": "Get the current date and time",
                        "parameters": {
                            "type": "object",
                            "required": [],
                            "properties": {}
                        }
                    }
                ]
            }
            
            # Create new manifest
            manifest = MCPManifest(
                url=manifest_url,
                api_key=api_key,
                hostname=hostname,
                content=manifest_content
            )
            db.add(manifest)
        
        # Commit changes
        db.commit()
        db.refresh(manifest)
        print(f"Successfully added/updated manifest with ID: {manifest.id}")
        
        # Add a test tool
        tool_name = "get_datetime"
        existing_tool = db.query(MCPTool).filter(
            MCPTool.name == tool_name,
            MCPTool.manifest_id == manifest.id
        ).first()
        
        if existing_tool:
            print(f"Tool {tool_name} already exists. Updating it.")
            tool = existing_tool
        else:
            tool = MCPTool(
                name=tool_name,
                description="Get the current date and time",
                endpoint="/get_datetime",
                method="GET",
                parameters={},
                headers={},
                is_active=True,
                manifest_id=manifest.id
            )
            db.add(tool)
        
        # Commit changes
        db.commit()
        print(f"Successfully added/updated tool: {tool_name}")
        
        print("\nTest data added successfully!")
            
    finally:
        db.close()

if __name__ == "__main__":
    add_test_manifest() 