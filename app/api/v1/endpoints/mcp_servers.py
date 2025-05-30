from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, MCPTool, MCPManifest
from app.core.mcp_tools_cache import reload_enabled_mcp_tools
from app.core.mcp_manifest_cache import get_mcp_manifest_by_id, get_mcp_manifests, reload_mcp_manifests, invalidate_manifest_cache
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import requests
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
class MCPServerBase(BaseModel):
    name: str
    manifest_url: str
    hostname: Optional[str] = None
    endpoint_prefix: Optional[str] = None
    api_key: Optional[str] = None
    is_active: bool = True

class MCPServerCreate(MCPServerBase):
    pass

class MCPServerUpdate(BaseModel):
    name: Optional[str] = None
    manifest_url: Optional[str] = None
    hostname: Optional[str] = None
    endpoint_prefix: Optional[str] = None
    api_key: Optional[str] = None
    is_active: Optional[bool] = None

class MCPServerResponse(BaseModel):
    id: int
    name: str
    manifest_url: str
    hostname: Optional[str] = None
    endpoint_prefix: Optional[str] = None
    api_key: Optional[str] = None
    is_active: bool
    status: Optional[str] = 'disconnected'
    last_check: Optional[str] = None
    tool_count: Optional[int] = 0

class MCPToolToggle(BaseModel):
    is_active: bool

class BulkToolToggle(BaseModel):
    tool_ids: List[int]
    is_active: bool

@router.get("/", response_model=List[MCPServerResponse])
def list_servers(db: Session = Depends(get_db)):
    """List all MCP servers (manifests)."""
    try:
        # Try to get manifests from cache first
        manifest_dict = get_mcp_manifests()
        servers = []
        
        for manifest_id, manifest_data in manifest_dict.items():
            # Count active tools (still need DB for this)
            tool_count = db.query(MCPTool).filter(MCPTool.manifest_id == int(manifest_id)).count()
            
            # Extract server name from manifest content or use hostname
            server_name = manifest_data["content"].get("name", manifest_data["hostname"] or f"Server {manifest_id}")
            
            server = MCPServerResponse(
                id=manifest_data["id"],
                name=server_name,
                manifest_url=manifest_data["url"],
                hostname=manifest_data["hostname"],
                endpoint_prefix="/invoke",  # Default endpoint prefix
                api_key="•••••••••••••••" if manifest_data["api_key"] else None,
                is_active=True,  # All manifests are considered active
                status='disconnected',  # Default to disconnected until health check is performed
                last_check=manifest_data["updated_at"],
                tool_count=tool_count
            )
            servers.append(server)
        return servers
    except Exception as e:
        logger.error(f"Error listing servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list servers: {str(e)}")

@router.post("/", response_model=MCPServerResponse)
def create_server(server: MCPServerCreate, db: Session = Depends(get_db)):
    """Create a new MCP server (manifest)."""
    try:
        # Check if manifest already exists
        existing = db.query(MCPManifest).filter(MCPManifest.url == server.manifest_url).first()
        if existing:
            raise HTTPException(status_code=400, detail="A server with this manifest URL already exists")
        
        # Fetch manifest from URL
        try:
            manifest_url = server.manifest_url.replace('localhost', 'host.docker.internal')
            resp = requests.get(manifest_url, timeout=5)
            resp.raise_for_status()
            manifest_json = resp.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch manifest: {str(e)}")
        
        # Create new manifest
        new_manifest = MCPManifest(
            url=server.manifest_url,
            hostname=server.hostname,
            api_key=server.api_key,
            content=manifest_json
        )
        db.add(new_manifest)
        db.commit()
        db.refresh(new_manifest)
        
        # Create tools from manifest
        # Extract port from manifest URL if available
        import re
        port_match = re.search(r':(\d+)', server.manifest_url)
        port = port_match.group(1) if port_match else '9000'
        
        for tool in manifest_json.get("tools", []):
            db_tool = MCPTool(
                name=tool["name"],
                description=tool.get("description"),
                endpoint=f"http://{server.hostname or 'localhost'}:{port}/invoke/{tool['name']}",
                method=tool.get("method", "POST"),
                parameters=tool.get("parameters"),
                headers=tool.get("headers"),
                is_active=True,
                manifest_id=new_manifest.id
            )
            db.add(db_tool)
        db.commit()
        
        # Reload caches after adding new manifest and tools
        reload_mcp_manifests()
        reload_enabled_mcp_tools()
        
        # Count tools
        tool_count = db.query(MCPTool).filter(MCPTool.manifest_id == new_manifest.id).count()
        
        return MCPServerResponse(
            id=new_manifest.id,
            name=server.name,
            manifest_url=new_manifest.url,
            hostname=new_manifest.hostname,
            endpoint_prefix="/invoke",
            api_key="•••••••••••••••" if new_manifest.api_key else None,
            is_active=True,
            status='connected',
            last_check=new_manifest.updated_at.isoformat() if new_manifest.updated_at else None,
            tool_count=tool_count
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create server: {str(e)}")

@router.get("/{server_id}", response_model=MCPServerResponse)
def get_server(server_id: int, db: Session = Depends(get_db)):
    """Get a specific MCP server."""
    manifest = db.query(MCPManifest).filter(MCPManifest.id == server_id).first()
    if not manifest:
        raise HTTPException(status_code=404, detail="Server not found")
    
    tool_count = db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).count()
    server_name = manifest.content.get("name", manifest.hostname or f"Server {manifest.id}")
    
    return MCPServerResponse(
        id=manifest.id,
        name=server_name,
        manifest_url=manifest.url,
        hostname=manifest.hostname,
        endpoint_prefix="/invoke",
        api_key="•••••••••••••••" if manifest.api_key else None,
        is_active=True,
        status='connected',
        last_check=manifest.updated_at.isoformat() if manifest.updated_at else None,
        tool_count=tool_count
    )

@router.put("/{server_id}", response_model=MCPServerResponse)
def update_server(server_id: int, server: MCPServerUpdate, db: Session = Depends(get_db)):
    """Update an MCP server."""
    manifest = db.query(MCPManifest).filter(MCPManifest.id == server_id).first()
    if not manifest:
        raise HTTPException(status_code=404, detail="Server not found")
    
    try:
        # Update fields
        if server.manifest_url and server.manifest_url != manifest.url:
            # Fetch new manifest
            manifest_url = server.manifest_url.replace('localhost', 'host.docker.internal')
            resp = requests.get(manifest_url, timeout=5)
            resp.raise_for_status()
            manifest_json = resp.json()
            manifest.url = server.manifest_url
            manifest.content = manifest_json
        
        if server.hostname is not None:
            manifest.hostname = server.hostname
        
        if server.api_key is not None:
            manifest.api_key = server.api_key if server.api_key != "•••••••••••••••" else manifest.api_key
        
        db.commit()
        db.refresh(manifest)
        
        tool_count = db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).count()
        server_name = server.name or manifest.content.get("name", manifest.hostname or f"Server {manifest.id}")
        
        return MCPServerResponse(
            id=manifest.id,
            name=server_name,
            manifest_url=manifest.url,
            hostname=manifest.hostname,
            endpoint_prefix="/invoke",
            api_key="•••••••••••••••" if manifest.api_key else None,
            is_active=True,
            status='connected',
            last_check=manifest.updated_at.isoformat() if manifest.updated_at else None,
            tool_count=tool_count
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update server: {str(e)}")

@router.delete("/{server_id}")
def delete_server(server_id: int, db: Session = Depends(get_db)):
    """Delete an MCP server and its tools."""
    manifest = db.query(MCPManifest).filter(MCPManifest.id == server_id).first()
    if not manifest:
        raise HTTPException(status_code=404, detail="Server not found")
    
    try:
        # Delete associated tools first
        db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).delete()
        # Delete manifest
        db.delete(manifest)
        db.commit()
        return {"detail": "Server and associated tools deleted"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete server: {str(e)}")

@router.post("/{server_id}/health")
def check_server_health(server_id: int, db: Session = Depends(get_db)):
    """Check the health of an MCP server."""
    # First try to get from cache
    manifest_data = get_mcp_manifest_by_id(server_id)
    if not manifest_data:
        # Not in cache, check database
        manifest = db.query(MCPManifest).filter(MCPManifest.id == server_id).first()
        if not manifest:
            raise HTTPException(status_code=404, detail="Server not found")
        # Reload cache since we found it in DB
        reload_mcp_manifests()
        manifest_data = {
            "url": manifest.url,
            "hostname": manifest.hostname
        }
    
    try:
        # Use the configured hostname if available, otherwise fall back to host.docker.internal
        manifest_url = manifest_data["url"]
        original_url = manifest_url
        
        # Check if we're running in Docker
        import os
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        if in_docker:
            if manifest_data.get("hostname") and "localhost" in manifest_url:
                manifest_url = manifest_url.replace('localhost', manifest_data["hostname"])
                logger.info(f"Docker environment detected. Using configured hostname: {manifest_data['hostname']}")
            elif "localhost" in manifest_url:
                manifest_url = manifest_url.replace('localhost', 'host.docker.internal')
                logger.info("Docker environment detected. Using host.docker.internal for Docker networking")
        else:
            logger.info("Not in Docker, using original URL")
        
        logger.info(f"Health check - Original URL: {original_url}, Final URL: {manifest_url}, In Docker: {in_docker}")
        
        # Add authentication if API key is available
        headers = {}
        if manifest_data.get("api_key"):
            headers["Authorization"] = f"Bearer {manifest_data['api_key']}"
            logger.info("Added API key authentication for health check")
        
        resp = requests.get(manifest_url, headers=headers, timeout=3)
        resp.raise_for_status()
        status = 'connected'
        logger.info(f"Health check successful for server {server_id}")
    except Exception as e:
        logger.warning(f"Health check failed for server {server_id}: {str(e)}")
        logger.warning(f"Failed URL: {manifest_url}")
        status = 'disconnected'
    
    # Update timestamp in database (still need DB write)
    manifest = db.query(MCPManifest).filter(MCPManifest.id == server_id).first()
    if manifest:
        manifest.updated_at = datetime.utcnow()
        db.commit()
        # Invalidate cache after DB update
        invalidate_manifest_cache()
    
    return {"status": status}

@router.get("/{server_id}/tools")
def list_server_tools(server_id: int, db: Session = Depends(get_db)):
    """List all tools for a specific server."""
    manifest = db.query(MCPManifest).filter(MCPManifest.id == server_id).first()
    if not manifest:
        raise HTTPException(status_code=404, detail="Server not found")
    
    tools = db.query(MCPTool).filter(MCPTool.manifest_id == server_id).all()
    
    return [{
        "id": tool.id,
        "server_id": server_id,
        "server_name": manifest.content.get("name", manifest.hostname or f"Server {server_id}"),
        "name": tool.name,
        "description": tool.description,
        "endpoint": tool.endpoint,
        "method": tool.method,
        "parameters": tool.parameters,
        "headers": tool.headers,
        "is_active": tool.is_active
    } for tool in tools]

@router.put("/tools/{tool_id}/toggle")
def toggle_tool(tool_id: int, toggle: MCPToolToggle, db: Session = Depends(get_db)):
    """Toggle a tool's active status."""
    tool = db.query(MCPTool).filter(MCPTool.id == tool_id).first()
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool.is_active = toggle.is_active
    db.commit()
    
    # Reload enabled tools
    reload_enabled_mcp_tools()
    
    return {"detail": "Tool status updated"}

@router.put("/tools/bulk-toggle")
def bulk_toggle_tools(toggle: BulkToolToggle, db: Session = Depends(get_db)):
    """Toggle multiple tools' active status."""
    tools = db.query(MCPTool).filter(MCPTool.id.in_(toggle.tool_ids)).all()
    
    for tool in tools:
        tool.is_active = toggle.is_active
    
    db.commit()
    
    # Reload enabled tools
    reload_enabled_mcp_tools()
    
    return {"detail": f"{len(tools)} tools updated"}