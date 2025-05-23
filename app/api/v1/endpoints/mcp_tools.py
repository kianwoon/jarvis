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

    class Config:
        orm_mode = True

class MCPEnabledToolsUpdate(BaseModel):
    url: str
    enabled_tools: dict

class ManifestCreate(BaseModel):
    url: str

# Place this above the dynamic route
@router.get("/manifests/all")
def list_manifests_with_tools(db: Session = Depends(get_db)):
    manifests = db.query(MCPManifest).all()
    return [
        {
            "url": manifest.url,
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
    return {"tools": manifest.content.get("tools", [])}

# Manifest routes
@router.post("/manifests")
def add_manifest(data: ManifestCreate, db: Session = Depends(get_db)):
    logger.debug(f"Attempting to add manifest from URL: {data.url}")
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
        manifest = MCPManifest(url=data.url, content=manifest_json)
        db.add(manifest)
        db.commit()
        db.refresh(manifest)
        logger.debug(f"Created new manifest with ID: {manifest.id}")
    else:
        logger.debug(f"Updating existing manifest with ID: {manifest.id}")
        manifest.content = manifest_json
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

@router.get("/", response_model=List[MCPToolResponse])
def get_mcp_tools(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    tools = db.query(MCPTool).offset(skip).limit(limit).all()
    return tools

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

@router.get("/{tool_id}", response_model=MCPToolResponse)
def get_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return db_tool

@router.put("/{tool_id}", response_model=MCPToolResponse)
def update_mcp_tool(tool_id: int, tool: MCPToolUpdate, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Update attributes that are provided
    update_data = tool.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_tool, key, value)
    
    db.commit()
    db.refresh(db_tool)
    return db_tool

@router.delete("/{tool_id}", response_model=MCPToolResponse)
def delete_mcp_tool(tool_id: int, db: Session = Depends(get_db)):
    db_tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if db_tool is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    db.delete(db_tool)
    db.commit()
    return db_tool 