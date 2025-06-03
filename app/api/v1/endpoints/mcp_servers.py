from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.core.db import SessionLocal, MCPTool, MCPManifest, MCPServer
from app.core.mcp_tools_cache import reload_enabled_mcp_tools
from app.core.mcp_manifest_cache import get_mcp_manifest_by_id, get_mcp_manifests, reload_mcp_manifests, invalidate_manifest_cache
try:
    from app.core.mcp_process_manager import mcp_process_manager
except ImportError:
    logger.warning("MCP process manager not available")
    mcp_process_manager = None
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
    config_type: str  # 'manifest' or 'command'
    
    # Manifest-based configuration
    manifest_url: Optional[str] = None
    hostname: Optional[str] = None
    api_key: Optional[str] = None
    
    # Command-based configuration
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[dict] = None
    working_directory: Optional[str] = None
    restart_policy: Optional[str] = "on-failure"
    max_restarts: Optional[int] = 3
    
    is_active: bool = True

class MCPServerCreate(MCPServerBase):
    pass

class MCPServerUpdate(BaseModel):
    name: Optional[str] = None
    config_type: Optional[str] = None
    
    # Manifest-based configuration
    manifest_url: Optional[str] = None
    hostname: Optional[str] = None
    api_key: Optional[str] = None
    
    # Command-based configuration
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[dict] = None
    working_directory: Optional[str] = None
    restart_policy: Optional[str] = None
    max_restarts: Optional[int] = None
    
    is_active: Optional[bool] = None

class MCPServerResponse(BaseModel):
    id: int
    name: str
    config_type: str
    
    # Manifest-based configuration
    manifest_url: Optional[str] = None
    hostname: Optional[str] = None
    api_key: Optional[str] = None
    
    # Command-based configuration
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[dict] = None
    working_directory: Optional[str] = None
    restart_policy: Optional[str] = None
    max_restarts: Optional[int] = None
    
    # Process management
    process_id: Optional[int] = None
    is_running: Optional[bool] = False
    restart_count: Optional[int] = 0
    health_status: Optional[str] = "unknown"
    
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
    """List all MCP servers."""
    try:
        servers = db.query(MCPServer).all()
        response_servers = []
        
        for server in servers:
            # Count active tools
            tool_count = db.query(MCPTool).filter(MCPTool.server_id == server.id).count()
            
            response_server = MCPServerResponse(
                id=server.id,
                name=server.name,
                config_type=server.config_type,
                manifest_url=server.manifest_url,
                hostname=server.hostname,
                api_key="•••••••••••••••" if server.api_key else None,
                command=server.command,
                args=server.args,
                env=server.env,
                working_directory=server.working_directory,
                restart_policy=server.restart_policy,
                max_restarts=server.max_restarts,
                process_id=server.process_id,
                is_running=server.is_running,
                restart_count=server.restart_count,
                health_status=server.health_status,
                is_active=server.is_active,
                status=server.health_status,
                last_check=server.last_health_check.isoformat() if server.last_health_check else None,
                tool_count=tool_count
            )
            response_servers.append(response_server)
        
        return response_servers
    except Exception as e:
        logger.error(f"Error listing servers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list servers: {str(e)}")

@router.post("/", response_model=MCPServerResponse)
async def create_server(server: MCPServerCreate, db: Session = Depends(get_db)):
    """Create a new MCP server."""
    try:
        # Validate configuration type
        if server.config_type not in ["manifest", "command"]:
            raise HTTPException(status_code=400, detail="config_type must be 'manifest' or 'command'")
        
        # Validate required fields based on config type
        if server.config_type == "manifest" and not server.manifest_url:
            raise HTTPException(status_code=400, detail="manifest_url is required for manifest-based servers")
        
        if server.config_type == "command" and not server.command:
            raise HTTPException(status_code=400, detail="command is required for command-based servers")
        
        # Create new server
        new_server = MCPServer(
            name=server.name,
            config_type=server.config_type,
            manifest_url=server.manifest_url,
            hostname=server.hostname,
            api_key=server.api_key,
            command=server.command,
            args=server.args,
            env=server.env,
            working_directory=server.working_directory,
            restart_policy=server.restart_policy or "on-failure",
            max_restarts=server.max_restarts or 3,
            is_active=server.is_active
        )
        db.add(new_server)
        db.commit()
        db.refresh(new_server)
        
        # Handle manifest-based server
        if server.config_type == "manifest":
            try:
                manifest_url = server.manifest_url.replace('localhost', 'host.docker.internal')
                resp = requests.get(manifest_url, timeout=5)
                resp.raise_for_status()
                manifest_json = resp.json()
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=400, detail=f"Failed to fetch manifest: {str(e)}")
            
            # Create manifest record
            manifest = MCPManifest(
                url=server.manifest_url,
                hostname=server.hostname,
                api_key=server.api_key,
                content=manifest_json,
                server_id=new_server.id
            )
            db.add(manifest)
            
            # Create tools from manifest
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
                    server_id=new_server.id
                )
                db.add(db_tool)
        
        # Handle command-based server
        elif server.config_type == "command":
            # Start the process if process manager is available
            if mcp_process_manager is not None:
                success, message = await mcp_process_manager.start_server(new_server.id, db)
                if not success:
                    logger.warning(f"Failed to start command-based server: {message}")
                    # Don't fail the creation, just log the warning
            else:
                logger.warning("Process manager not available, command-based server created but not started")
        
        db.commit()
        
        # Reload caches
        reload_mcp_manifests()
        reload_enabled_mcp_tools()
        
        # Count tools
        tool_count = db.query(MCPTool).filter(MCPTool.server_id == new_server.id).count()
        
        return MCPServerResponse(
            id=new_server.id,
            name=new_server.name,
            config_type=new_server.config_type,
            manifest_url=new_server.manifest_url,
            hostname=new_server.hostname,
            api_key="•••••••••••••••" if new_server.api_key else None,
            command=new_server.command,
            args=new_server.args,
            env=new_server.env,
            working_directory=new_server.working_directory,
            restart_policy=new_server.restart_policy,
            max_restarts=new_server.max_restarts,
            process_id=new_server.process_id,
            is_running=new_server.is_running,
            restart_count=new_server.restart_count,
            health_status=new_server.health_status,
            is_active=new_server.is_active,
            status='connected' if new_server.config_type == 'command' and new_server.is_running else 'disconnected',
            last_check=new_server.last_health_check.isoformat() if new_server.last_health_check else None,
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
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    tool_count = db.query(MCPTool).filter(MCPTool.server_id == server.id).count()
    
    return MCPServerResponse(
        id=server.id,
        name=server.name,
        config_type=server.config_type,
        manifest_url=server.manifest_url,
        hostname=server.hostname,
        api_key="•••••••••••••••" if server.api_key else None,
        command=server.command,
        args=server.args,
        env=server.env,
        working_directory=server.working_directory,
        restart_policy=server.restart_policy,
        max_restarts=server.max_restarts,
        process_id=server.process_id,
        is_running=server.is_running,
        restart_count=server.restart_count,
        health_status=server.health_status,
        is_active=server.is_active,
        status=server.health_status,
        last_check=server.last_health_check.isoformat() if server.last_health_check else None,
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
        
        tool_count = db.query(MCPTool).filter(MCPTool.server_id == server_id).count()
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
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    try:
        # Delete associated tools first
        db.query(MCPTool).filter(MCPTool.server_id == server_id).delete()
        
        # Delete associated manifest if it exists
        if server.config_type == "manifest":
            manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server_id).first()
            if manifest:
                db.delete(manifest)
        
        # Delete the server
        db.delete(server)
        db.commit()
        return {"detail": "Server and associated tools deleted"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete server: {str(e)}")


@router.post("/{server_id}/refresh")
def refresh_server_tools(server_id: int, db: Session = Depends(get_db)):
    """Refresh tools from the MCP server manifest."""
    try:
        # Get server from database
        server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
        if not server:
            raise HTTPException(status_code=404, detail="Server not found")
        
        # Only manifest-based servers can be refreshed
        if server.config_type != "manifest":
            raise HTTPException(status_code=400, detail="Only manifest-based servers can be refreshed")
        
        # Get the associated manifest
        manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server_id).first()
        if not manifest:
            raise HTTPException(status_code=404, detail="Manifest not found for server")
        
        # Fetch fresh manifest from URL
        manifest_url = manifest.url
        
        # Handle Docker networking
        import os
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
        
        if in_docker and "localhost" in manifest_url:
            # If we have a configured hostname, use it (e.g., 'mcp' for docker service name)
            if manifest.hostname and manifest.hostname != 'localhost':
                manifest_url = manifest_url.replace('localhost', manifest.hostname)
                logger.info(f"Docker environment: Using configured hostname '{manifest.hostname}'")
            # If both services are in the same Docker Compose network, localhost should work
            else:
                logger.info("Docker environment: Keeping localhost (same network)")
                # manifest_url remains unchanged
        
        # Add authentication if API key is available
        headers = {}
        if manifest.api_key:
            headers["Authorization"] = f"Bearer {manifest.api_key}"
            logger.info("Added API key authentication for manifest refresh")
        
        try:
            resp = requests.get(manifest_url, headers=headers, timeout=10)
            resp.raise_for_status()
            manifest_json = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch manifest for refresh: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to fetch manifest: {str(e)}")
        
        # Update manifest content in database
        manifest.content = manifest_json
        manifest.updated_at = datetime.utcnow()
        
        # Get existing tools
        existing_tools = db.query(MCPTool).filter(MCPTool.server_id == server_id).all()
        existing_tool_names = {tool.name: tool for tool in existing_tools}
        
        # Extract port from manifest URL if available
        import re
        port_match = re.search(r':(\d+)', manifest.url)
        port = port_match.group(1) if port_match else '9000'
        
        # Process tools from manifest
        manifest_tool_names = set()
        tools_added = 0
        tools_updated = 0
        
        for tool_data in manifest_json.get("tools", []):
            tool_name = tool_data["name"]
            manifest_tool_names.add(tool_name)
            
            if tool_name in existing_tool_names:
                # Update existing tool
                existing_tool = existing_tool_names[tool_name]
                existing_tool.description = tool_data.get("description")
                existing_tool.parameters = tool_data.get("parameters")
                existing_tool.headers = tool_data.get("headers")
                existing_tool.method = tool_data.get("method", "POST")
                tools_updated += 1
                logger.info(f"Updated existing tool: {tool_name}")
            else:
                # Add new tool
                db_tool = MCPTool(
                    name=tool_name,
                    description=tool_data.get("description"),
                    endpoint=f"http://{server.hostname or 'localhost'}:{port}/invoke/{tool_name}",
                    method=tool_data.get("method", "POST"),
                    parameters=tool_data.get("parameters"),
                    headers=tool_data.get("headers"),
                    is_active=True,
                    server_id=server_id
                )
                db.add(db_tool)
                tools_added += 1
                logger.info(f"Added new tool: {tool_name}")
        
        # Remove tools that are no longer in the manifest
        tools_removed = 0
        for tool_name, tool in existing_tool_names.items():
            if tool_name not in manifest_tool_names:
                db.delete(tool)
                tools_removed += 1
                logger.info(f"Removed tool no longer in manifest: {tool_name}")
        
        # Commit all changes
        db.commit()
        
        # Reload caches after updating tools
        reload_mcp_manifests()
        reload_enabled_mcp_tools()
        
        logger.info(f"Manifest refresh completed for server {server_id}. Added: {tools_added}, Updated: {tools_updated}, Removed: {tools_removed}")
        
        return {
            "status": "success",
            "message": f"Tools refreshed successfully. Added: {tools_added}, Updated: {tools_updated}, Removed: {tools_removed}",
            "tools_added": tools_added,
            "tools_updated": tools_updated,
            "tools_removed": tools_removed,
            "total_tools": len(manifest_tool_names)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error refreshing server tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh server tools: {str(e)}")

@router.get("/{server_id}/tools")
def list_server_tools(server_id: int, db: Session = Depends(get_db)):
    """List all tools for a specific server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    tools = db.query(MCPTool).filter(MCPTool.server_id == server_id).all()
    
    return [{
        "id": tool.id,
        "server_id": server_id,
        "server_name": server.name,
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

# Process management endpoints for command-based servers

@router.post("/{server_id}/start")
async def start_server_process(server_id: int, db: Session = Depends(get_db)):
    """Start a command-based MCP server process."""
    if mcp_process_manager is None:
        raise HTTPException(status_code=503, detail="Process management not available")
        
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    if server.config_type != "command":
        raise HTTPException(status_code=400, detail="Only command-based servers can be started")
    
    try:
        success, message = await mcp_process_manager.start_server(server_id, db)
        if success:
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        logger.error(f"Error starting server {server_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start server: {str(e)}")

@router.post("/{server_id}/stop")
async def stop_server_process(server_id: int, force: bool = False, db: Session = Depends(get_db)):
    """Stop a command-based MCP server process."""
    if mcp_process_manager is None:
        raise HTTPException(status_code=503, detail="Process management not available")
        
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    if server.config_type != "command":
        raise HTTPException(status_code=400, detail="Only command-based servers can be stopped")
    
    try:
        success, message = await mcp_process_manager.stop_server(server_id, db, force=force)
        if success:
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        logger.error(f"Error stopping server {server_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop server: {str(e)}")

@router.post("/{server_id}/restart")
async def restart_server_process(server_id: int, db: Session = Depends(get_db)):
    """Restart a command-based MCP server process."""
    if mcp_process_manager is None:
        raise HTTPException(status_code=503, detail="Process management not available")
        
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    if server.config_type != "command":
        raise HTTPException(status_code=400, detail="Only command-based servers can be restarted")
    
    try:
        success, message = await mcp_process_manager.restart_server(server_id, db)
        if success:
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        logger.error(f"Error restarting server {server_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restart server: {str(e)}")

@router.post("/{server_id}/health")
async def check_server_health(server_id: int, db: Session = Depends(get_db)):
    """Perform health check on an MCP server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    try:
        if server.config_type == "command":
            if mcp_process_manager is None:
                status = "unknown"
                message = "Process management not available"
            else:
                success, message = await mcp_process_manager.health_check_server(server_id, db)
                status = "healthy" if success else "unhealthy"
        else:
            # Handle manifest-based health check (existing logic)
            manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server_id).first()
            if not manifest:
                raise HTTPException(status_code=404, detail="Manifest not found for server")
            
            try:
                manifest_url = manifest.url
                
                # Handle Docker networking consistently
                import os
                in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
                
                if in_docker and "localhost" in manifest_url:
                    # If we have a configured hostname, use it (e.g., 'mcp' for docker service name)
                    if manifest.hostname and manifest.hostname != 'localhost':
                        manifest_url = manifest_url.replace('localhost', manifest.hostname)
                        logger.info(f"Health check: Using configured hostname '{manifest.hostname}'")
                    # If both services are in the same Docker Compose network, localhost should work
                    else:
                        logger.info("Health check: Keeping localhost (same network)")
                
                headers = {}
                if manifest.api_key:
                    headers["Authorization"] = f"Bearer {manifest.api_key}"
                
                logger.info(f"Health check URL: {manifest_url}")
                resp = requests.get(manifest_url, headers=headers, timeout=3)
                resp.raise_for_status()
                status = 'healthy'
                message = "Manifest server is accessible"
            except Exception as e:
                status = 'unhealthy'
                message = f"Health check failed: {str(e)}"
                logger.warning(f"Health check failed for URL {manifest_url}: {e}")
        
        # Update health status
        server.health_status = status
        server.last_health_check = func.now() if hasattr(func, 'now') else None
        db.commit()
        
        return {"status": status, "message": message}
        
    except Exception as e:
        logger.error(f"Health check failed for server {server_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")