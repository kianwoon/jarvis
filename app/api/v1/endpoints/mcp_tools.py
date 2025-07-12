from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, MCPTool, MCPManifest
from app.core.mcp_tools_cache import reload_enabled_mcp_tools
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import requests
from urllib.parse import unquote
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models for API
class MCPToolBase(BaseModel):
    name: str
    description: Optional[str] = None
    endpoint: str
    method: str = "POST"
    parameters: Optional[dict] = None
    headers: Optional[dict] = None
    is_active: bool = True

class MCPToolCreate(MCPToolBase):
    pass

class MCPToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    parameters: Optional[dict] = None
    headers: Optional[dict] = None
    is_active: Optional[bool] = None

class MCPToolResponse(MCPToolBase):
    id: int
    created_at: datetime
    updated_at: datetime
    server_id: Optional[int] = None
    server_name: Optional[str] = None
    is_manual: bool = False
    manifest_id: Optional[int] = None

    class Config:
        from_attributes = True

class MCPEnabledToolsUpdate(BaseModel):
    url: str
    enabled_tools: dict

class ManifestCreate(BaseModel):
    url: str
    hostname: str

# Place this above the dynamic route
@router.get("/manifests/all")
def list_manifests_with_tools(db: Session = Depends(get_db)):
    manifests = db.query(MCPManifest).all()
    return [
        {
            "url": manifest.url,
            "hostname": manifest.hostname,
            "tools": manifest.content.get("tools", [])
        }
        for manifest in manifests
    ]

@router.get("/manifests/{manifest_url:path}")
def get_manifest_by_url(manifest_url: str, db: Session = Depends(get_db)):
    decoded_url = unquote(manifest_url)
    logger.debug(f"Looking for manifest: '{decoded_url}' (len={len(decoded_url)})")
    all_manifests = db.query(MCPManifest).all()
    for m in all_manifests:
        logger.debug(f"DB manifest: '{m.url}' (len={len(m.url)})")
    manifest = db.query(MCPManifest).filter(MCPManifest.url == decoded_url).first()
    if not manifest:
        logger.error(f"Manifest not found for: '{decoded_url}'")
        raise HTTPException(status_code=404, detail="Manifest not found")
    return {"url": manifest.url, "hostname": manifest.hostname, "tools": manifest.content.get("tools", [])}

# Manifest routes
@router.post("/manifests")
def add_manifest(data: ManifestCreate, db: Session = Depends(get_db)):
    logger.debug(f"Attempting to add manifest from URL: {data.url} with hostname: {data.hostname}")
    # Fetch manifest from URL
    try:
        # Replace localhost with host.docker.internal for Docker container access
        manifest_url = data.url.replace('localhost', 'host.docker.internal')
        logger.debug(f"Making request to fetch manifest from: {manifest_url}")
        resp = requests.get(manifest_url, timeout=5)  # Add timeout
        resp.raise_for_status()
        manifest_json = resp.json()
        logger.debug(f"Successfully fetched manifest: {manifest_json}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error when fetching manifest from {manifest_url}")
        raise HTTPException(status_code=400, detail=f"Could not connect to MCP server at {data.url}. Please ensure the server is running and accessible.")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout when fetching manifest from {manifest_url}")
        raise HTTPException(status_code=400, detail=f"Connection to MCP server at {data.url} timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error when fetching manifest from {manifest_url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch manifest from {data.url}: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response from {manifest_url}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON response from {data.url}: {str(e)}")
    
    if "tools" not in manifest_json or not isinstance(manifest_json["tools"], list):
        logger.error(f"Invalid manifest format: missing 'tools' list in {manifest_json}")
        raise HTTPException(status_code=400, detail="Manifest must contain a 'tools' list")

    # Upsert manifest
    logger.debug("Checking for existing manifest...")
    manifest = db.query(MCPManifest).filter(MCPManifest.url == data.url).first()
    if not manifest:
        logger.debug("Creating new manifest...")
        manifest = MCPManifest(url=data.url, hostname=data.hostname, content=manifest_json)
        db.add(manifest)
        db.commit()
        db.refresh(manifest)
        logger.debug(f"Created new manifest with ID: {manifest.id}")
    else:
        logger.debug(f"Updating existing manifest with ID: {manifest.id}")
        manifest.content = manifest_json
        manifest.hostname = data.hostname
        db.commit()

    # Upsert tools
    logger.debug("Upserting tools...")
    for tool in manifest_json["tools"]:
        logger.debug(f"Processing tool: {tool['name']}")
        db_tool = db.query(MCPTool).filter(
            MCPTool.name == tool["name"],
            MCPTool.manifest_id == manifest.id
        ).first()
        if not db_tool:
            logger.debug(f"Creating new tool: {tool['name']}")
            db_tool = MCPTool(
                name=tool["name"],
                description=tool.get("description"),
                endpoint=tool.get("endpoint", f"/{tool['name']}"),
                method=tool.get("method", "POST"),
                parameters=tool.get("parameters"),
                headers=tool.get("headers"),
                is_active=True,
                manifest_id=manifest.id
            )
            db.add(db_tool)
        else:
            logger.debug(f"Updating existing tool: {tool['name']}")
            db_tool.description = tool.get("description")
            db_tool.endpoint = tool.get("endpoint", f"/{tool['name']}")
            db_tool.method = tool.get("method", "POST")
            db_tool.parameters = tool.get("parameters")
            db_tool.headers = tool.get("headers")
        # Do not change is_active on upsert
    db.commit()
    logger.debug("Successfully completed manifest and tools registration/update")
    return {"detail": "Manifest and tools registered/updated."}

@router.delete("/manifests/{manifest_url:path}")
def delete_manifest(manifest_url: str, db: Session = Depends(get_db)):
    """Delete a manifest and its associated tools."""
    decoded_url = unquote(manifest_url)
    logger.debug(f"Attempting to delete manifest: '{decoded_url}'")
    
    manifest = db.query(MCPManifest).filter(MCPManifest.url == decoded_url).first()
    if not manifest:
        logger.error(f"Manifest not found for deletion: '{decoded_url}'")
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    try:
        # First delete all associated tools
        db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).delete()
        # Then delete the manifest
        db.delete(manifest)
        db.commit()
        logger.info(f"Successfully deleted manifest and associated tools for: '{decoded_url}'")
        return {"detail": "Manifest and associated tools deleted"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting manifest '{decoded_url}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete manifest: {str(e)}")

# Tool routes
@router.post("/", response_model=MCPToolResponse)
def create_mcp_tool(tool: MCPToolCreate, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.name == tool.name).first()
    if db_tool:
        raise HTTPException(status_code=400, detail="Tool with this name already exists")
    
    db_tool = MCPTool(
        name=tool.name,
        description=tool.description,
        endpoint=tool.endpoint,
        method=tool.method,
        parameters=tool.parameters,
        headers=tool.headers,
        is_active=tool.is_active
    )
    db.add(db_tool)
    db.commit()
    db.refresh(db_tool)
    return db_tool

@router.get("/")
def get_mcp_tools(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    from sqlalchemy.orm import joinedload
    
    # Query tools with their associated servers
    tools = db.query(MCPTool).options(joinedload(MCPTool.server)).order_by(MCPTool.name).offset(skip).limit(limit).all()
    
    # Build response with server names
    response = []
    for tool in tools:
        tool_dict = {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "endpoint": tool.endpoint,
            "method": tool.method,
            "parameters": tool.parameters,
            "headers": tool.headers,
            "is_active": tool.is_active,
            "is_manual": tool.is_manual,
            "created_at": tool.created_at,
            "updated_at": tool.updated_at,
            "server_id": tool.server_id,
            "manifest_id": tool.manifest_id,
            "server_name": tool.server.name if tool.server else None
        }
        response.append(tool_dict)
    
    return response

@router.put("/enabled")
def update_enabled_tools(data: MCPEnabledToolsUpdate, db: Session = Depends(get_db)):
    manifest = db.query(MCPManifest).filter(MCPManifest.url == data.url).first()
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    tools = db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).all()
    updated = 0
    for tool in tools:
        if tool.name in data.enabled_tools:
            tool.is_active = data.enabled_tools[tool.name]
            updated += 1
    db.commit()
    # Reload enabled tools to Redis
    reload_enabled_mcp_tools()
    return {"updated": updated}

# IMPORTANT: Place these routes BEFORE the /{tool_id} route to avoid conflicts
@router.get("/debug/endpoints")
def debug_tool_endpoints(db: Session = Depends(get_db)):
    """Debug endpoint to check all tool endpoints and settings"""
    # Get MCP settings
    from app.core.db import Settings as SettingsModel
    settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
    settings = settings_row.settings if settings_row else {}
    
    # Get all tools
    tools = db.query(MCPTool).all()
    
    # Get all manifests
    manifests = {m.id: m for m in db.query(MCPManifest).all()}
    
    # Format the response
    result = {
        "settings": {
            "endpoint_prefix": settings.get("endpoint_prefix", ""),
            "hostname": settings.get("hostname", ""),
            "manifest_url": settings.get("manifest_url", "")
        },
        "tools": [
            {
                "id": tool.id,
                "name": tool.name,
                "endpoint": tool.endpoint,
                "is_active": tool.is_active,
                "manifest_id": tool.manifest_id,
                "manifest_hostname": manifests.get(tool.manifest_id).hostname if tool.manifest_id in manifests else None
            }
            for tool in tools
        ]
    }
    
    return result 

@router.post("/cache/reload")
async def reload_mcp_tools_cache():
    """Force reload MCP tools cache from database"""
    try:
        # Reload MCP tools cache 
        from app.core.mcp_tools_cache import reload_enabled_mcp_tools
        enabled_tools = reload_enabled_mcp_tools()
        
        # Also reload the langgraph agents cache to ensure consistency
        try:
            from app.core.langgraph_agents_cache import reload_cache_from_db
            reload_cache_from_db()
        except Exception as e:
            logger.warning(f"Failed to reload langgraph agents cache: {e}")
        
        return {
            "status": "success",
            "message": f"MCP tools cache reloaded with {len(enabled_tools)} tools",
            "tools_count": len(enabled_tools),
            "tools": list(enabled_tools.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload MCP tools cache: {str(e)}")

@router.get("/from-db")
def get_mcp_tools_from_db(db: Session = Depends(get_db)):
    """Get all MCP tools from the database with their manifest info."""
    try:
        # Get all manifests first for efficient lookup
        manifests = {m.id: m for m in db.query(MCPManifest).all()}
        
        # Get all tools ordered by name
        tools = db.query(MCPTool).order_by(MCPTool.name).all()
        
        # Format the response
        formatted_tools = []
        for tool in tools:
            manifest = manifests.get(tool.manifest_id)
            
            # Get the raw tool definition from the manifest content
            raw_tool = None
            if manifest and manifest.content and "tools" in manifest.content:
                for t in manifest.content["tools"]:
                    if t.get("name") == tool.name:
                        raw_tool = t
                        break
            
            formatted_tool = {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "endpoint": tool.endpoint,
                "method": tool.method,
                "parameters": tool.parameters,
                "headers": tool.headers,
                "is_active": tool.is_active,
                "manifest_id": tool.manifest_id,
                "manifest_url": manifest.url if manifest else None,
                "manifest_hostname": manifest.hostname if manifest else None,
                "raw_tool": raw_tool
            }
            formatted_tools.append(formatted_tool)
        
        return {
            "tools": formatted_tools,
            "manifests": [
                {
                    "id": m.id,
                    "url": m.url,
                    "hostname": m.hostname,
                    "has_api_key": bool(m.api_key),
                    "tools_count": len([t for t in tools if t.manifest_id == m.id])
                }
                for m in manifests.values()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting MCP tools from database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get MCP tools: {str(e)}")

# Tool-specific routes with ID parameter
@router.get("/{tool_id}", response_model=MCPToolResponse)
def get_mcp_tool(tool_id: str, db: Session = Depends(get_db)):
    # Try to convert tool_id to integer, return 404 if it's not a valid integer
    try:
        tool_id_int = int(tool_id)
    except ValueError:
        logger.error(f"Invalid tool ID format: {tool_id}")
        raise HTTPException(status_code=404, detail="Tool not found")
        
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id_int).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return db_tool

@router.put("/{tool_id}", response_model=MCPToolResponse)
def update_mcp_tool(tool_id: str, tool: MCPToolUpdate, db: Session = Depends(get_db)):
    # Try to convert tool_id to integer, return 404 if it's not a valid integer
    try:
        tool_id_int = int(tool_id)
    except ValueError:
        logger.error(f"Invalid tool ID format: {tool_id}")
        raise HTTPException(status_code=404, detail="Tool not found")
        
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id_int).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Update attributes that are provided
    update_data = tool.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_tool, key, value)
    
    db.commit()
    db.refresh(db_tool)
    return db_tool

@router.delete("/{tool_id}", response_model=MCPToolResponse)
def delete_mcp_tool(tool_id: str, db: Session = Depends(get_db)):
    # Try to convert tool_id to integer, return 404 if it's not a valid integer
    try:
        tool_id_int = int(tool_id)
    except ValueError:
        logger.error(f"Invalid tool ID format: {tool_id}")
        raise HTTPException(status_code=404, detail="Tool not found")
        
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id_int).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    db.delete(db_tool)
    db.commit()
    return db_tool 