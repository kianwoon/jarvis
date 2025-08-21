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
from datetime import datetime, timedelta, timezone
import requests
import logging
import subprocess
import os
import asyncio
import aiohttp
try:
    import psutil
except ImportError:
    psutil = None

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Warning if psutil is not available
if psutil is None:
    logger.warning("psutil not installed - some features will be limited")

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
    config_type: str  # 'manifest', 'command', 'remote_http', or 'http'
    communication_protocol: Optional[str] = 'stdio'  # 'stdio', 'http', 'sse', 'websocket'
    
    # Manifest-based configuration
    manifest_url: Optional[str] = None
    hostname: Optional[str] = None
    api_key: Optional[str] = None
    oauth_credentials: Optional[dict] = None
    
    # Command-based configuration
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[dict] = None
    working_directory: Optional[str] = None
    restart_policy: Optional[str] = "on-failure"
    max_restarts: Optional[int] = 3
    
    # Remote HTTP/SSE MCP Server Configuration
    remote_config: Optional[dict] = None
    
    # Enhanced Error Handling Configuration
    enhanced_error_handling_config: Optional[dict] = None
    auth_refresh_config: Optional[dict] = None
    
    is_active: bool = True

class MCPServerCreate(MCPServerBase):
    pass

class MCPServerUpdate(BaseModel):
    name: Optional[str] = None
    config_type: Optional[str] = None
    communication_protocol: Optional[str] = None
    
    # Manifest-based configuration
    manifest_url: Optional[str] = None
    hostname: Optional[str] = None
    api_key: Optional[str] = None
    oauth_credentials: Optional[dict] = None
    
    # Command-based configuration
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[dict] = None
    working_directory: Optional[str] = None
    restart_policy: Optional[str] = None
    max_restarts: Optional[int] = None
    
    # Remote HTTP/SSE MCP Server Configuration
    remote_config: Optional[dict] = None
    
    # Enhanced Error Handling Configuration
    enhanced_error_handling_config: Optional[dict] = None
    auth_refresh_config: Optional[dict] = None
    
    is_active: Optional[bool] = None

class MCPServerResponse(BaseModel):
    id: int
    name: str
    config_type: str
    communication_protocol: Optional[str] = 'stdio'
    
    # Manifest-based configuration
    manifest_url: Optional[str] = None
    hostname: Optional[str] = None
    api_key: Optional[str] = None
    oauth_credentials: Optional[dict] = None
    
    # Command-based configuration
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[dict] = None
    working_directory: Optional[str] = None
    restart_policy: Optional[str] = None
    max_restarts: Optional[int] = None
    
    # Remote HTTP/SSE MCP Server Configuration
    remote_config: Optional[dict] = None
    
    # Process management
    process_id: Optional[int] = None
    is_running: Optional[bool] = False
    restart_count: Optional[int] = 0
    health_status: Optional[str] = "unknown"
    
    # Enhanced Error Handling Configuration
    enhanced_error_handling_config: Optional[dict] = None
    auth_refresh_config: Optional[dict] = None
    
    is_active: bool
    status: Optional[str] = 'disconnected'
    last_check: Optional[str] = None
    tool_count: Optional[int] = 0

class MCPToolToggle(BaseModel):
    is_active: bool

class BulkToolToggle(BaseModel):
    tool_ids: List[int]
    is_active: bool

class MCPToolCreate(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None
    headers: Optional[dict] = None
    method: Optional[str] = "POST"
    endpoint: Optional[str] = None
    is_active: bool = True

class MCPToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[dict] = None
    headers: Optional[dict] = None
    method: Optional[str] = None
    endpoint: Optional[str] = None
    is_active: Optional[bool] = None

class MCPToolsImport(BaseModel):
    tools: List[MCPToolCreate]
    replace_existing: bool = False






def mask_sensitive_env_vars(env_dict: dict, show_sensitive: bool = False) -> dict:
    """Mask sensitive environment variables unless explicitly requested."""
    if not env_dict or show_sensitive:
        return env_dict
    
    # List of sensitive keywords that should be masked
    sensitive_keywords = [
        'TOKEN', 'KEY', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH',
        'API_KEY', 'ACCESS_TOKEN', 'REFRESH_TOKEN', 'CLIENT_SECRET',
        'PRIVATE', 'CERT', 'CERTIFICATE', 'JIRA_TOKEN', 'MS_GRAPH_TOKEN'
    ]
    
    masked_env = {}
    for key, value in env_dict.items():
        # Check if the key contains any sensitive keyword
        is_sensitive = any(keyword in key.upper() for keyword in sensitive_keywords)
        
        if is_sensitive and value:
            # Show first 4 characters for debugging, rest as dots
            if len(str(value)) > 8:
                masked_env[key] = str(value)[:4] + "•" * 12
            else:
                masked_env[key] = "•" * 12
        else:
            masked_env[key] = value
    
    return masked_env

@router.get("/", response_model=List[MCPServerResponse])
def list_servers(show_sensitive: bool = False, db: Session = Depends(get_db)):
    """List all MCP servers."""
    try:
        servers = db.query(MCPServer).all()
        response_servers = []
        
        for server in servers:
            # Count active tools
            tool_count = db.query(MCPTool).filter(MCPTool.server_id == server.id).count()
            
            # Mask OAuth credentials
            masked_oauth = None
            if server.oauth_credentials:
                if show_sensitive:
                    masked_oauth = server.oauth_credentials
                else:
                    masked_oauth = {
                        "configured": True,
                        "client_id": server.oauth_credentials.get("client_id", "")[:10] + "..." 
                            if server.oauth_credentials.get("client_id") else None,
                        "has_tokens": bool(server.oauth_credentials.get("access_token") or 
                                         server.oauth_credentials.get("refresh_token"))
                    }
            
            # Mask environment variables
            masked_env = mask_sensitive_env_vars(server.env, show_sensitive)
            
            response_server = MCPServerResponse(
                id=server.id,
                name=server.name,
                config_type=server.config_type,
                communication_protocol=server.communication_protocol,
                manifest_url=server.manifest_url,
                hostname=server.hostname,
                api_key=server.api_key if show_sensitive else ("•••••••••••••••" if server.api_key else None),
                oauth_credentials=masked_oauth,
                command=server.command,
                args=server.args,
                env=masked_env,
                working_directory=server.working_directory,
                restart_policy=server.restart_policy,
                max_restarts=server.max_restarts,
                remote_config=server.remote_config,
                enhanced_error_handling_config=server.enhanced_error_handling_config,
                auth_refresh_config=server.auth_refresh_config,
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
        if server.config_type not in ["manifest", "command", "remote_http", "http"]:
            raise HTTPException(status_code=400, detail="config_type must be 'manifest', 'command', 'remote_http', or 'http'")
        
        # Validate required fields based on config type
        if server.config_type == "manifest" and not server.manifest_url:
            raise HTTPException(status_code=400, detail="manifest_url is required for manifest-based servers")
        
        if server.config_type == "command" and not server.command:
            raise HTTPException(status_code=400, detail="command is required for command-based servers")
            
        if server.config_type == "remote_http":
            if not server.remote_config or not server.remote_config.get("server_url"):
                raise HTTPException(status_code=400, detail="remote_config.server_url is required for remote HTTP servers")
                
        if server.config_type == "http":
            if not server.hostname:
                raise HTTPException(status_code=400, detail="hostname is required for HTTP servers")
        
        # Set default enhanced error handling configuration if not provided
        default_error_config = {
            "enabled": True,
            "max_tool_retries": 3,
            "retry_base_delay": 1.0,
            "retry_max_delay": 60.0,
            "retry_backoff_multiplier": 2.0,
            "timeout_seconds": 30,
            "enable_circuit_breaker": True,
            "circuit_failure_threshold": 5,
            "circuit_recovery_timeout": 60
        }
        
        default_auth_config = {
            "enabled": False,
            "server_type": "custom",
            "auth_type": "oauth2",
            "refresh_endpoint": "",
            "refresh_method": "POST",
            "refresh_headers": {},
            "refresh_data_template": {},
            "token_expiry_buffer_minutes": 5
        }
        
        # Set default communication protocol based on config type if not provided
        if not server.communication_protocol:
            if server.config_type == "command":
                server.communication_protocol = "stdio"  # Default for command servers
            elif server.config_type in ["http", "manifest"]:
                server.communication_protocol = "http"  # Default for HTTP servers
            elif server.config_type == "remote_http":
                # Check if SSE is specified in remote_config
                if server.remote_config and server.remote_config.get("transport_type") == "sse":
                    server.communication_protocol = "sse"
                else:
                    server.communication_protocol = "http"
        
        # Create new server
        new_server = MCPServer(
            name=server.name,
            config_type=server.config_type,
            communication_protocol=server.communication_protocol,
            manifest_url=server.manifest_url,
            hostname=server.hostname,
            api_key=server.api_key,
            oauth_credentials=server.oauth_credentials,
            command=server.command,
            args=server.args,
            env=server.env,
            working_directory=server.working_directory,
            restart_policy=server.restart_policy or "on-failure",
            max_restarts=server.max_restarts or 3,
            remote_config=server.remote_config,
            enhanced_error_handling_config=server.enhanced_error_handling_config or default_error_config,
            auth_refresh_config=server.auth_refresh_config or default_auth_config,
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
        
        # Handle HTTP server
        elif server.config_type == "http":
            # For HTTP servers, we can try to discover tools if there's a discovery endpoint
            # For now, just mark as healthy if we can connect
            try:
                if server.hostname:
                    # Try a simple health check
                    hostname = server.hostname
                    if not hostname.startswith(('http://', 'https://')):
                        health_url = f"http://{hostname}/health"
                    else:
                        health_url = hostname.rstrip('/') + '/health'
                    
                    headers = {}
                    if server.api_key:
                        headers["Authorization"] = f"Bearer {server.api_key}"
                    
                    # Try to connect
                    try:
                        resp = requests.get(health_url, headers=headers, timeout=5)
                        if resp.status_code < 500:
                            new_server.health_status = "healthy"
                            new_server.is_running = True
                    except:
                        # Server might not have a health endpoint, that's ok
                        new_server.health_status = "unknown"
                        
            except Exception as e:
                logger.warning(f"Could not perform initial health check for HTTP server: {e}")
                new_server.health_status = "unknown"
        
        # Handle remote HTTP/SSE server
        elif server.config_type == "remote_http":
            # Discover tools from remote MCP server
            try:
                from app.core.remote_mcp_client import remote_mcp_manager
                
                server_config = {
                    "id": new_server.id,
                    "name": new_server.name,
                    "remote_config": new_server.remote_config
                }
                
                logger.info(f"Discovering tools from remote MCP server: {new_server.name}")
                tools = await remote_mcp_manager.discover_tools(server_config)
                
                # Create tools in database
                tools_added = 0
                for tool_def in tools:
                    tool_name = tool_def.get("name")
                    if not tool_name:
                        continue
                        
                    db_tool = MCPTool(
                        name=tool_name,
                        description=tool_def.get("description", ""),
                        endpoint=f"remote://{new_server.name}/{tool_name}",
                        method="POST",
                        parameters=tool_def.get("inputSchema", {}),
                        is_active=True,
                        server_id=new_server.id
                    )
                    db.add(db_tool)
                    tools_added += 1
                
                logger.info(f"Added {tools_added} tools from remote MCP server")
                
            except Exception as e:
                logger.error(f"Failed to discover tools from remote MCP server: {str(e)}")
                # Don't fail the server creation, just log the warning
                logger.warning("Remote MCP server created but tool discovery failed")
        
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
            communication_protocol=new_server.communication_protocol,
            manifest_url=new_server.manifest_url,
            hostname=new_server.hostname,
            api_key="•••••••••••••••" if new_server.api_key else None,
            command=new_server.command,
            args=new_server.args,
            env=new_server.env,
            working_directory=new_server.working_directory,
            restart_policy=new_server.restart_policy,
            max_restarts=new_server.max_restarts,
            remote_config=new_server.remote_config,
            enhanced_error_handling_config=new_server.enhanced_error_handling_config,
            auth_refresh_config=new_server.auth_refresh_config,
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
def get_server(server_id: int, show_sensitive: bool = False, db: Session = Depends(get_db)):
    """Get a specific MCP server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    tool_count = db.query(MCPTool).filter(MCPTool.server_id == server.id).count()
    
    # Handle OAuth credentials based on show_sensitive flag
    oauth_creds = None
    if server.oauth_credentials:
        if show_sensitive:
            # Return full credentials
            oauth_creds = server.oauth_credentials
        else:
            # Mask sensitive data
            oauth_creds = {
                "configured": True,
                "client_id": server.oauth_credentials.get("client_id", "")[:10] + "..." 
                    if server.oauth_credentials.get("client_id") else None,
                "has_tokens": bool(server.oauth_credentials.get("access_token") or 
                                 server.oauth_credentials.get("refresh_token"))
            }
    
    # Mask environment variables
    masked_env = mask_sensitive_env_vars(server.env, show_sensitive)
    
    return MCPServerResponse(
        id=server.id,
        name=server.name,
        config_type=server.config_type,
        communication_protocol=server.communication_protocol,
        manifest_url=server.manifest_url,
        hostname=server.hostname,
        api_key=server.api_key if show_sensitive else ("•••••••••••••••" if server.api_key else None),
        oauth_credentials=oauth_creds,
        command=server.command,
        args=server.args,
        env=masked_env,
        working_directory=server.working_directory,
        restart_policy=server.restart_policy,
        max_restarts=server.max_restarts,
        remote_config=server.remote_config,
        enhanced_error_handling_config=server.enhanced_error_handling_config,
        auth_refresh_config=server.auth_refresh_config,
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
def update_server(server_id: int, server_update: MCPServerUpdate, db: Session = Depends(get_db)):
    """Update an MCP server."""
    db_server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not db_server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    try:
        # Update basic fields
        if server_update.name is not None:
            db_server.name = server_update.name
        if server_update.config_type is not None:
            db_server.config_type = server_update.config_type
        if server_update.communication_protocol is not None:
            db_server.communication_protocol = server_update.communication_protocol
        if server_update.hostname is not None:
            db_server.hostname = server_update.hostname
        if server_update.api_key is not None and server_update.api_key != "•••••••••••••••":
            db_server.api_key = server_update.api_key
        if server_update.oauth_credentials is not None:
            db_server.oauth_credentials = server_update.oauth_credentials
        if server_update.enhanced_error_handling_config is not None:
            db_server.enhanced_error_handling_config = server_update.enhanced_error_handling_config
        if server_update.auth_refresh_config is not None:
            db_server.auth_refresh_config = server_update.auth_refresh_config
        if server_update.remote_config is not None:
            db_server.remote_config = server_update.remote_config
        if server_update.is_active is not None:
            db_server.is_active = server_update.is_active
        
        # Update config-specific fields
        if db_server.config_type == "manifest":
            if server_update.manifest_url is not None:
                db_server.manifest_url = server_update.manifest_url
                # Update associated manifest
                manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server_id).first()
                if manifest:
                    manifest.url = server_update.manifest_url
                    if server_update.hostname is not None:
                        manifest.hostname = server_update.hostname
                    if server_update.api_key is not None and server_update.api_key != "•••••••••••••••":
                        manifest.api_key = server_update.api_key
        
        elif db_server.config_type == "command":
            if server_update.command is not None:
                db_server.command = server_update.command
            if server_update.args is not None:
                db_server.args = server_update.args
            if server_update.env is not None:
                db_server.env = server_update.env
            if server_update.working_directory is not None:
                db_server.working_directory = server_update.working_directory
            if server_update.restart_policy is not None:
                db_server.restart_policy = server_update.restart_policy
            if server_update.max_restarts is not None:
                db_server.max_restarts = server_update.max_restarts
        
        elif db_server.config_type == "remote_http":
            # Remote HTTP servers primarily use remote_config, which is handled above
            pass
            
        elif db_server.config_type == "http":
            # HTTP servers use hostname and api_key, which are handled above
            # Also handle env for local HTTP servers (like the Local MCP on port 3001)
            if server_update.env is not None:
                db_server.env = server_update.env
        
        db.commit()
        db.refresh(db_server)
        
        # Reload caches to ensure UI shows updated data
        reload_mcp_manifests()
        reload_enabled_mcp_tools()
        
        # Prepare response
        tool_count = db.query(MCPTool).filter(MCPTool.server_id == server_id).count()
        
        # Mask OAuth credentials
        masked_oauth = None
        if db_server.oauth_credentials:
            masked_oauth = {
                "configured": True,
                "client_id": db_server.oauth_credentials.get("client_id", "")[:10] + "..." 
                    if db_server.oauth_credentials.get("client_id") else None,
                "has_tokens": bool(db_server.oauth_credentials.get("access_token") or 
                                 db_server.oauth_credentials.get("refresh_token"))
            }
        
        return MCPServerResponse(
            id=db_server.id,
            name=db_server.name,
            config_type=db_server.config_type,
            communication_protocol=db_server.communication_protocol,
            manifest_url=db_server.manifest_url,
            hostname=db_server.hostname,
            api_key="•••••••••••••••" if db_server.api_key else None,
            oauth_credentials=masked_oauth,
            command=db_server.command,
            args=db_server.args,
            env=db_server.env,
            working_directory=db_server.working_directory,
            restart_policy=db_server.restart_policy,
            max_restarts=db_server.max_restarts,
            remote_config=db_server.remote_config,
            enhanced_error_handling_config=db_server.enhanced_error_handling_config,
            auth_refresh_config=db_server.auth_refresh_config,
            process_id=db_server.process_id,
            is_running=db_server.is_running,
            restart_count=db_server.restart_count,
            health_status=db_server.health_status,
            is_active=db_server.is_active,
            status=db_server.health_status,
            last_check=db_server.last_health_check.isoformat() if db_server.last_health_check else None,
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
        # Delete associated OAuth credentials first (if table exists)
        try:
            from sqlalchemy import text
            delete_oauth_query = text("DELETE FROM oauth_credentials WHERE mcp_server_id = :server_id")
            db.execute(delete_oauth_query, {"server_id": server_id})
        except Exception as oauth_error:
            # OAuth credentials table might not exist, which is fine
            logger.debug(f"OAuth credentials deletion skipped: {str(oauth_error)}")
        
        # Delete associated tools
        db.query(MCPTool).filter(MCPTool.server_id == server_id).delete()
        
        # Delete associated manifest if it exists
        if server.config_type == "manifest":
            manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server_id).first()
            if manifest:
                db.delete(manifest)
        
        # Delete the server
        db.delete(server)
        db.commit()
        
        # Reload caches to ensure UI reflects deletion
        reload_mcp_manifests()
        reload_enabled_mcp_tools()
        
        return {"detail": "Server and associated data deleted successfully"}
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
    
    tools = db.query(MCPTool).filter(MCPTool.server_id == server_id).order_by(MCPTool.name).all()
    
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
        "is_active": tool.is_active,
        "is_manual": getattr(tool, 'is_manual', False)  # Show if manually added
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
            # Special handling for Docker-based command servers
            if server.command == "docker" and server.args and "exec" in server.args:
                # For Docker exec commands running from within Docker, assume healthy
                # since we can't check Docker from inside a container
                import os
                in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
                
                if in_docker:
                    # When running inside Docker, we can't check other containers
                    # Assume healthy if the server has been used recently
                    if server.last_health_check:
                        last_check = server.last_health_check
                        if isinstance(last_check, str):
                            last_check = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
                        
                        # Ensure both datetimes are timezone-aware
                        now = datetime.now(timezone.utc)
                        if last_check.tzinfo is None:
                            # Make last_check timezone-aware if it isn't
                            last_check = last_check.replace(tzinfo=timezone.utc)
                        
                        # Consider healthy if checked within last 5 minutes
                        if now - last_check < timedelta(minutes=5):
                            status = "healthy"
                            message = "Command-based server (Docker exec mode)"
                        else:
                            status = "unknown"
                            message = "Command-based server (status unknown)"
                    else:
                        status = "unknown"
                        message = "Command-based server (not yet checked)"
                else:
                    # Only try Docker commands if not running inside Docker
                    try:
                        import subprocess
                        container_idx = server.args.index("-i") + 1
                        if container_idx < len(server.args):
                            container_name = server.args[container_idx]
                            # Check if container is running
                            result = subprocess.run(
                                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if result.returncode == 0 and result.stdout.strip() == "true":
                                status = "healthy"
                                message = f"Docker container {container_name} is running"
                            else:
                                status = "unhealthy"
                                message = f"Docker container {container_name} is not running"
                        else:
                            status = "unknown"
                            message = "Invalid Docker command format"
                    except Exception as e:
                        status = "unknown"
                        message = f"Could not check Docker container: {str(e)}"
            elif server.command in ['npx', 'node', 'python', 'python3']:
                # For stdio-based command servers (npx, node, etc.), check if recently used
                if server.last_health_check:
                    last_check = server.last_health_check
                    if isinstance(last_check, str):
                        last_check = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
                    
                    # Ensure both datetimes are timezone-aware
                    now = datetime.now(timezone.utc)
                    if last_check.tzinfo is None:
                        last_check = last_check.replace(tzinfo=timezone.utc)
                    
                    # Consider healthy if used within last 10 minutes
                    if now - last_check < timedelta(minutes=10):
                        status = "healthy"
                        message = f"Command-based server ({server.command})"
                    else:
                        status = "unknown"
                        message = f"Command-based server (not recently used)"
                else:
                    # Server hasn't been used yet, but tools were discovered successfully
                    tool_count = db.query(MCPTool).filter(MCPTool.server_id == server_id).count()
                    if tool_count > 0:
                        status = "healthy"
                        message = f"Command-based server ({server.command}) - {tool_count} tools available"
                    else:
                        status = "unknown"
                        message = f"Command-based server (no tools discovered)"
            elif mcp_process_manager is None:
                status = "unknown"
                message = "Process management not available"
            else:
                success, message = await mcp_process_manager.health_check_server(server_id, db)
                status = "healthy" if success else "unhealthy"
        elif server.config_type == "manifest":
            # Handle manifest-based health check
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
        
        elif server.config_type == "remote_http":
            # Handle remote HTTP/SSE MCP server health check
            try:
                if not server.remote_config:
                    status = 'unhealthy'
                    message = "No remote configuration found"
                else:
                    from app.core.remote_mcp_client import RemoteMCPClient
                    
                    remote_config = server.remote_config
                    server_url = remote_config.get("server_url")
                    
                    if not server_url:
                        status = 'unhealthy'
                        message = "No server URL in remote configuration"
                    else:
                        # Create a temporary client for health check
                        client = RemoteMCPClient(
                            server_url=server_url,
                            transport_type=remote_config.get("transport_type", "http"),
                            auth_headers=remote_config.get("auth_headers", {}),
                            client_info=remote_config.get("client_info", {}),
                            connection_timeout=10  # Shorter timeout for health checks
                        )
                        
                        logger.info(f"Health check: Testing remote MCP server {server_url}")
                        
                        # Try to connect and initialize
                        async with client:
                            server_info = await client.get_server_info()
                            if server_info.get("initialized", False):
                                status = 'healthy'
                                message = f"Remote MCP server is accessible and initialized"
                            else:
                                status = 'unhealthy'
                                message = "Remote MCP server connection failed"
                        
            except Exception as e:
                status = 'unhealthy'
                message = f"Remote health check failed: {str(e)}"
                logger.warning(f"Remote health check failed for server {server_id}: {e}")
        
        elif server.config_type == "http":
            # Handle HTTP server health check
            try:
                if not server.hostname:
                    status = 'unhealthy'
                    message = "No hostname configured for HTTP server"
                else:
                    # Build health check URL
                    hostname = server.hostname
                    # Add http:// if not present
                    if not hostname.startswith(('http://', 'https://')):
                        health_url = f"http://{hostname}/health"
                    else:
                        # Parse and add /health endpoint
                        health_url = hostname.rstrip('/') + '/health'
                    
                    # Add authentication if configured
                    headers = {}
                    if server.api_key:
                        headers["Authorization"] = f"Bearer {server.api_key}"
                    
                    logger.info(f"Health check for HTTP server: {health_url}")
                    
                    # Perform health check
                    import requests
                    resp = requests.get(health_url, headers=headers, timeout=5)
                    
                    # Check response status
                    if resp.status_code == 200:
                        try:
                            data = resp.json()
                            if data.get('status') == 'healthy' or data.get('ok') or data.get('healthy'):
                                status = 'healthy'
                                server_name = data.get('server', data.get('name', 'Unknown'))
                                version = data.get('version', 'Unknown')
                                message = f"HTTP server is healthy: {server_name} v{version}"
                            else:
                                status = 'healthy'  # Server responded, consider it healthy
                                message = f"HTTP server responded with status {resp.status_code}"
                        except:
                            # Not JSON response, but server is reachable
                            status = 'healthy'
                            message = f"HTTP server is reachable (status {resp.status_code})"
                    elif resp.status_code == 404:
                        # No health endpoint, but server is reachable
                        status = 'healthy'
                        message = "HTTP server is reachable (no health endpoint)"
                    else:
                        status = 'unhealthy'
                        message = f"HTTP server returned status {resp.status_code}"
                        
            except requests.RequestException as e:
                status = 'unhealthy'
                message = f"HTTP health check failed: {str(e)}"
                logger.warning(f"HTTP health check failed for server {server_id}: {e}")
            except Exception as e:
                status = 'unhealthy'
                message = f"Unexpected error during HTTP health check: {str(e)}"
                logger.error(f"Unexpected error during HTTP health check for server {server_id}: {e}")
        
        else:
            # Unknown server type
            status = 'unknown'
            message = f"Unknown server type: {server.config_type}"
        
        # Update health status
        server.health_status = status
        server.last_health_check = datetime.now(timezone.utc)
        db.commit()
        
        return {"status": status, "message": message}
        
    except Exception as e:
        logger.error(f"Health check failed for server {server_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")

@router.get("/{server_id}/oauth")
def get_server_oauth(server_id: int, db: Session = Depends(get_db)):
    """Get OAuth credentials for a server (unmasked for editing)."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    return {
        "oauth_credentials": server.oauth_credentials or {}
    }

@router.get("/{server_id}/env")
def get_server_env(server_id: int, db: Session = Depends(get_db)):
    """Get environment variables for a server (unmasked for editing)."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    return {
        "env": server.env or {},
        "warning": "These environment variables contain sensitive information. Handle with care."
    }

@router.post("/{server_id}/discover-tools")
async def discover_server_tools(server_id: int, db: Session = Depends(get_db)):
    """Discover tools from an MCP server (command-based or remote HTTP)."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    if server.config_type not in ["command", "remote_http", "http"]:
        raise HTTPException(status_code=400, detail="Tool discovery is only for command-based, remote HTTP, or HTTP servers")
    
    try:
        # Use communication_protocol to determine how to discover tools
        comm_protocol = getattr(server, 'communication_protocol', 'stdio')
        
        if comm_protocol == "stdio" and server.config_type == "command":
            # Use the stdio bridge for stdio communication
            from app.core.mcp_stdio_bridge import MCPDockerBridge, MCPStdioBridge
            
            # Check if we're running inside Docker
            in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
            
            # Determine the type of bridge to use
            if server.command == "docker" and server.args and "exec" in server.args:
                container_idx = server.args.index("-i") + 1
                if container_idx < len(server.args):
                    container_name = server.args[container_idx]
                    mcp_command = server.args[container_idx + 1:]
                    
                    if in_docker:
                        # Use network bridge when running inside Docker
                        from app.core.mcp_network_bridge import MCPNetworkBridge
                        bridge = MCPNetworkBridge(container_name, mcp_command)
                    else:
                        # Use Docker bridge when running on host
                        bridge = MCPDockerBridge(container_name, mcp_command, server.env or {})
                else:
                    raise ValueError("Invalid Docker exec command format")
            else:
                # Generic stdio server
                bridge = MCPStdioBridge(server.command, server.args or [], server.env or {}, server.working_directory)
            
            # Start the bridge and discover tools
            await bridge.start()
            try:
                tools = await bridge.list_tools()
                endpoint_prefix = f"stdio://{server.name}"
            finally:
                await bridge.stop()
                
        elif comm_protocol in ["http", "sse"] and server.config_type == "command":
            # Command that starts an HTTP/SSE server (like Local MCP)
            # The command should start the server, then we connect via HTTP
            
            # First, ensure the server is running
            if mcp_process_manager:
                if not server.is_running:
                    # Start the server process
                    logger.info(f"Starting HTTP command server: {server.name} with command: {server.command} {server.args}")
                    success, message = await mcp_process_manager.start_server(server.id, db)
                    if not success:
                        raise HTTPException(status_code=500, detail=f"Failed to start server: {message}")
                    
                    # Wait for HTTP server to be ready with non-blocking polling
                    logger.info(f"Waiting for HTTP server {server.name} to be ready...")
                    max_attempts = 20  # 10 seconds total (20 * 0.5s)
                    server_ready = False
                    
                    for attempt in range(max_attempts):
                        try:
                            # Try to connect to the server
                            test_url = f"http://{server.hostname}/health" if server.hostname else f"http://localhost:{server.env.get('MCP_PORT', 3001)}/health"
                            async with aiohttp.ClientSession() as session:
                                async with session.get(test_url, timeout=aiohttp.ClientTimeout(total=1)) as resp:
                                    if resp.status in [200, 404]:  # 404 is ok - means server is up but no health endpoint
                                        server_ready = True
                                        logger.info(f"HTTP server {server.name} is ready after {(attempt + 1) * 0.5}s")
                                        break
                        except:
                            pass  # Server not ready yet
                        
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(0.5)  # Short non-blocking sleep
                    
                    if not server_ready:
                        logger.warning(f"HTTP server {server.name} did not become ready after 10s, proceeding anyway")
                    
                    # Refresh server state from database
                    db.refresh(server)
                else:
                    logger.info(f"Server {server.name} is already running")
            else:
                logger.warning("MCP process manager not available, assuming server is externally managed")
            
            # Now try to discover tools via HTTP
            tools = []
            
            # Use the hostname field which should be properly configured
            # For Local MCP, this should be "host.docker.internal:3001" or "localhost:3001"
            hostname = server.hostname
            
            # Fallback to defaults if hostname not set
            if not hostname:
                # Check environment for port
                if server.env and 'MCP_PORT' in server.env:
                    hostname = f"localhost:{server.env['MCP_PORT']}"
                elif server.name == "Local MCP":
                    # Special case for Local MCP which runs on port 3001
                    hostname = "localhost:3001"
                else:
                    # Generic default
                    hostname = "localhost:3000"
                logger.warning(f"No hostname configured for server {server.name}, using: {hostname}")
            
            # Replace host.docker.internal with localhost if running outside Docker
            in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER')
            if not in_docker and 'host.docker.internal' in hostname:
                hostname = hostname.replace('host.docker.internal', 'localhost')
                logger.info(f"Replaced host.docker.internal with localhost: {hostname}")
            
            # Set endpoint prefix using the properly determined hostname
            endpoint_prefix = f"http://{hostname}"
            
            try:
                # Build discovery URL
                if not hostname.startswith(('http://', 'https://')):
                    discover_url = f"http://{hostname}/tools"
                else:
                    discover_url = hostname.rstrip('/') + '/tools'
                
                # Add authentication if configured
                headers = {}
                if server.api_key:
                    headers["Authorization"] = f"Bearer {server.api_key}"
                
                # First check if server is healthy
                health_url = discover_url.replace('/tools', '/health')
                logger.info(f"Checking server health at: {health_url}")
                
                # Wait for server to be ready with health checks
                max_health_checks = 10
                for health_attempt in range(max_health_checks):
                    try:
                        health_resp = requests.get(health_url, timeout=5)
                        if health_resp.status_code == 200:
                            logger.info(f"Server is healthy, proceeding to discover tools")
                            break
                    except:
                        pass
                    
                    if health_attempt < max_health_checks - 1:
                        logger.info(f"Health check {health_attempt + 1}/{max_health_checks} failed, waiting 2 seconds...")
                        await asyncio.sleep(2)
                    else:
                        logger.warning("Server health check failed after all attempts, trying tool discovery anyway")
                
                logger.info(f"Discovering tools from HTTP server: {discover_url}")
                
                # Try to discover tools with retries
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        logger.info(f"Tool discovery attempt {attempt + 1}/{max_retries}")
                        resp = requests.get(discover_url, headers=headers, timeout=10)
                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, list):
                                tools = data
                            elif isinstance(data, dict) and 'tools' in data:
                                tools = data['tools']
                            logger.info(f"Successfully discovered {len(tools)} tools from HTTP server")
                            break
                        else:
                            logger.warning(f"Got status code {resp.status_code} from {discover_url}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2)
                    except (requests.ConnectionError, requests.Timeout) as e:
                        if attempt < max_retries - 1:
                            logger.info(f"Connection attempt {attempt + 1} failed ({str(e)}), retrying in 3 seconds...")
                            await asyncio.sleep(3)
                        else:
                            raise e
            except Exception as e:
                logger.error(f"Failed to discover tools from HTTP server at {discover_url}: {e}")
                # Try to provide more helpful error message
                if "Connection refused" in str(e):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Could not connect to HTTP server at {hostname}. Please ensure the server is running and accessible."
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to discover tools: {str(e)}"
                    )
                
        elif comm_protocol == "sse" and server.config_type == "remote_http":
            # SSE protocol for remote servers
            from app.core.remote_mcp_client import remote_mcp_manager
            
            server_config = {
                "id": server.id,
                "name": server.name,
                "remote_config": server.remote_config
            }
            
            tools = await remote_mcp_manager.discover_tools(server_config)
            endpoint_prefix = f"sse://{server.name}"
            
        elif server.config_type == "remote_http":
            # Standard remote HTTP
            from app.core.remote_mcp_client import remote_mcp_manager
            
            server_config = {
                "id": server.id,
                "name": server.name,
                "remote_config": server.remote_config
            }
            
            tools = await remote_mcp_manager.discover_tools(server_config)
            endpoint_prefix = f"remote://{server.name}"
            
        elif server.config_type == "http" or comm_protocol == "http":
            # For HTTP servers, we need to implement tool discovery
            # This would typically involve calling a /tools or /discover endpoint
            # For now, return empty tools list or attempt discovery
            tools = []
            endpoint_prefix = f"http://{server.hostname}"
            
            if server.hostname:
                try:
                    # Build discovery URL
                    hostname = server.hostname
                    if not hostname.startswith(('http://', 'https://')):
                        discover_url = f"http://{hostname}/tools"
                    else:
                        discover_url = hostname.rstrip('/') + '/tools'
                    
                    # Add authentication if configured
                    headers = {}
                    if server.api_key:
                        headers["Authorization"] = f"Bearer {server.api_key}"
                    
                    logger.info(f"Discovering tools from HTTP server: {discover_url}")
                    
                    # Try to discover tools
                    resp = requests.get(discover_url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, list):
                            tools = data
                        elif isinstance(data, dict) and 'tools' in data:
                            tools = data['tools']
                        logger.info(f"Discovered {len(tools)} tools from HTTP server")
                except Exception as e:
                    logger.warning(f"Could not discover tools from HTTP server: {e}")
                    # Continue without tools
        
        # Update server status
        server.health_status = "healthy"
        if server.config_type == "command":
            server.is_running = True
        server.last_health_check = datetime.utcnow()
        
        # Store discovered tools in database
        tools_added = 0
        tools_updated = 0
        
        for tool_def in tools:
            tool_name = tool_def.get("name")
            if not tool_name:
                continue
                
            # Check if tool already exists
            existing_tool = db.query(MCPTool).filter(
                MCPTool.name == tool_name,
                MCPTool.server_id == server_id
            ).first()
            
            if existing_tool:
                # Update existing tool
                existing_tool.description = tool_def.get("description", "")
                existing_tool.parameters = tool_def.get("inputSchema", {})
                existing_tool.endpoint = f"{endpoint_prefix}/tools/{tool_name}"
                existing_tool.updated_at = datetime.utcnow()
                tools_updated += 1
            else:
                # Create new tool
                new_tool = MCPTool(
                    name=tool_name,
                    description=tool_def.get("description", ""),
                    endpoint=f"{endpoint_prefix}/tools/{tool_name}",
                    method="POST",
                    parameters=tool_def.get("inputSchema", {}),
                    server_id=server_id,
                    is_active=True
                )
                db.add(new_tool)
                tools_added += 1
        
        db.commit()
        
        # Reload MCP tools cache
        from app.core.mcp_tools_cache import reload_enabled_mcp_tools
        reload_enabled_mcp_tools()
        
        return {
            "status": "success",
            "tools_discovered": len(tools),
            "tools_added": tools_added,
            "tools_updated": tools_updated,
            "tools": [{"name": t.get("name"), "description": t.get("description", "")} for t in tools if t.get("name")]
        }
            
    except Exception as e:
        # Update server status on error
        server.health_status = "unhealthy"
        server.is_running = False
        db.commit()
        
        logger.error(f"Failed to discover tools for server {server_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Tool discovery failed: {str(e)}")

@router.post("/{server_id}/tools")
async def add_tool_manually(server_id: int, tool: MCPToolCreate, db: Session = Depends(get_db)):
    """Manually add a tool to an MCP server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # Check if tool with same name already exists for this server
    existing_tool = db.query(MCPTool).filter(
        MCPTool.name == tool.name,
        MCPTool.server_id == server_id
    ).first()
    
    if existing_tool:
        raise HTTPException(status_code=400, detail=f"Tool '{tool.name}' already exists for this server")
    
    try:
        # Generate endpoint if not provided
        if not tool.endpoint:
            if server.config_type == "command":
                tool.endpoint = f"stdio://{server.name}/{tool.name}"
            else:
                # For manifest servers, construct from hostname/port
                import re
                port = "9000"  # default
                if server.manifest_url:
                    port_match = re.search(r':(\d+)', server.manifest_url)
                    if port_match:
                        port = port_match.group(1)
                tool.endpoint = f"http://{server.hostname or 'localhost'}:{port}/invoke/{tool.name}"
        
        # Create new tool
        new_tool = MCPTool(
            name=tool.name,
            description=tool.description,
            endpoint=tool.endpoint,
            method=tool.method,
            parameters=tool.parameters,
            headers=tool.headers,
            is_active=tool.is_active,
            is_manual=True,  # Mark as manually added
            server_id=server_id
        )
        
        db.add(new_tool)
        db.commit()
        db.refresh(new_tool)
        
        # Reload cache
        reload_enabled_mcp_tools()
        
        return {
            "id": new_tool.id,
            "name": new_tool.name,
            "description": new_tool.description,
            "endpoint": new_tool.endpoint,
            "method": new_tool.method,
            "parameters": new_tool.parameters,
            "headers": new_tool.headers,
            "is_active": new_tool.is_active,
            "server_id": new_tool.server_id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error adding tool manually: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add tool: {str(e)}")

@router.put("/{server_id}/tools/{tool_id}")
async def update_tool_manually(server_id: int, tool_id: int, tool_update: MCPToolUpdate, db: Session = Depends(get_db)):
    """Update a tool's configuration manually."""
    tool = db.query(MCPTool).filter(
        MCPTool.id == tool_id,
        MCPTool.server_id == server_id
    ).first()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    try:
        # Update fields if provided
        if tool_update.name is not None:
            # Check if name conflicts with another tool
            existing = db.query(MCPTool).filter(
                MCPTool.name == tool_update.name,
                MCPTool.server_id == server_id,
                MCPTool.id != tool_id
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail=f"Tool name '{tool_update.name}' already exists")
            tool.name = tool_update.name
            
        if tool_update.description is not None:
            tool.description = tool_update.description
        if tool_update.parameters is not None:
            tool.parameters = tool_update.parameters
        if tool_update.headers is not None:
            tool.headers = tool_update.headers
        if tool_update.method is not None:
            tool.method = tool_update.method
        if tool_update.endpoint is not None:
            tool.endpoint = tool_update.endpoint
        if tool_update.is_active is not None:
            tool.is_active = tool_update.is_active
            
        tool.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(tool)
        
        # Reload cache
        reload_enabled_mcp_tools()
        
        return {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "endpoint": tool.endpoint,
            "method": tool.method,
            "parameters": tool.parameters,
            "headers": tool.headers,
            "is_active": tool.is_active,
            "server_id": tool.server_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update tool: {str(e)}")

@router.delete("/{server_id}/tools/{tool_id}")
async def delete_tool_manually(server_id: int, tool_id: int, db: Session = Depends(get_db)):
    """Delete a tool from an MCP server."""
    tool = db.query(MCPTool).filter(
        MCPTool.id == tool_id,
        MCPTool.server_id == server_id
    ).first()
    
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    try:
        db.delete(tool)
        db.commit()
        
        # Reload cache
        reload_enabled_mcp_tools()
        
        return {"detail": f"Tool '{tool.name}' deleted successfully"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting tool: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete tool: {str(e)}")

@router.post("/{server_id}/tools/import")
async def import_tools(server_id: int, import_data: MCPToolsImport, db: Session = Depends(get_db)):
    """Import multiple tools at once for an MCP server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    try:
        imported_count = 0
        updated_count = 0
        errors = []
        
        for tool_data in import_data.tools:
            try:
                # Check if tool exists
                existing_tool = db.query(MCPTool).filter(
                    MCPTool.name == tool_data.name,
                    MCPTool.server_id == server_id
                ).first()
                
                if existing_tool:
                    if import_data.replace_existing:
                        # Update existing tool
                        existing_tool.description = tool_data.description
                        existing_tool.parameters = tool_data.parameters
                        existing_tool.headers = tool_data.headers
                        existing_tool.method = tool_data.method
                        if tool_data.endpoint:
                            existing_tool.endpoint = tool_data.endpoint
                        existing_tool.is_active = tool_data.is_active
                        existing_tool.updated_at = datetime.utcnow()
                        updated_count += 1
                    else:
                        errors.append(f"Tool '{tool_data.name}' already exists (skipped)")
                        continue
                else:
                    # Generate endpoint if not provided
                    endpoint = tool_data.endpoint
                    if not endpoint:
                        if server.config_type == "command":
                            endpoint = f"stdio://{server.name}/{tool_data.name}"
                        else:
                            import re
                            port = "9000"
                            if server.manifest_url:
                                port_match = re.search(r':(\d+)', server.manifest_url)
                                if port_match:
                                    port = port_match.group(1)
                            endpoint = f"http://{server.hostname or 'localhost'}:{port}/invoke/{tool_data.name}"
                    
                    # Create new tool
                    new_tool = MCPTool(
                        name=tool_data.name,
                        description=tool_data.description,
                        endpoint=endpoint,
                        method=tool_data.method,
                        parameters=tool_data.parameters,
                        headers=tool_data.headers,
                        is_active=tool_data.is_active,
                        is_manual=True,  # Mark as manually imported
                        server_id=server_id
                    )
                    db.add(new_tool)
                    imported_count += 1
                    
            except Exception as e:
                errors.append(f"Error importing '{tool_data.name}': {str(e)}")
        
        db.commit()
        
        # Reload cache
        reload_enabled_mcp_tools()
        
        return {
            "status": "success",
            "imported": imported_count,
            "updated": updated_count,
            "total": len(import_data.tools),
            "errors": errors
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error importing tools: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to import tools: {str(e)}")

@router.get("/{server_id}/logs")
async def get_server_logs(server_id: int, lines: int = 50, db: Session = Depends(get_db)):
    """Get logs for a specific MCP server."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    # For command-based servers, check if there's a log file
    if server.config_type == "command" and server.working_directory:
        log_file = os.path.join(server.working_directory, "mcp_server.log")
        
        if os.path.exists(log_file):
            try:
                # Read last N lines using tail
                import subprocess
                result = subprocess.run(
                    ["tail", "-n", str(lines), log_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                return {
                    "logs": result.stdout,
                    "server_name": server.name,
                    "log_file": log_file,
                    "lines": lines
                }
            except Exception as e:
                logger.error(f"Error reading logs for server {server_id}: {e}")
                return {
                    "logs": f"Error reading log file: {str(e)}",
                    "server_name": server.name,
                    "lines": lines
                }
        else:
            # Try to get logs from process manager if available
            if mcp_process_manager:
                logs = await mcp_process_manager.get_server_logs(server_id, lines)
                if logs:
                    return {
                        "logs": logs,
                        "server_name": server.name,
                        "lines": lines
                    }
    
    # Default message for servers without log support
    return {
        "logs": f"No logs available for {server.name} ({server.config_type} type).\nCheck if the server is running and has a log file.",
        "server_name": server.name,
        "lines": lines
    }

@router.get("/{server_id}/tools/export")
async def export_tools(server_id: int, db: Session = Depends(get_db)):
    """Export all tools for an MCP server in a format suitable for import."""
    server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    tools = db.query(MCPTool).filter(MCPTool.server_id == server_id).order_by(MCPTool.name).all()
    
    export_data = {
        "server_name": server.name,
        "server_type": server.config_type,
        "exported_at": datetime.utcnow().isoformat(),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "headers": tool.headers,
                "method": tool.method,
                "endpoint": tool.endpoint,
                "is_active": tool.is_active
            }
            for tool in tools
        ]
    }
    
    return export_data

@router.post("/batch/start-all")
async def start_all_servers(db: Session = Depends(get_db)):
    """Start all active command-based MCP servers."""
    servers = db.query(MCPServer).filter(
        MCPServer.config_type == "command",
        MCPServer.is_active == True
    ).all()
    
    results = []
    for server in servers:
        try:
            if mcp_process_manager:
                success, message = await mcp_process_manager.start_server(server.id, db)
                results.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "success": success,
                    "message": message
                })
            else:
                results.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "success": False,
                    "message": "Process manager not available"
                })
        except Exception as e:
            results.append({
                "server_id": server.id,
                "server_name": server.name,
                "success": False,
                "message": str(e)
            })
    
    # Also try to start external MCP server
    try:
        external_result = await start_external_mcp_server()
        results.append({
            "server_id": "external",
            "server_name": "External MCP Server (port 3001)",
            "success": external_result.get("status") in ["success", "already_running"],
            "message": external_result.get("message", "")
        })
    except Exception as e:
        results.append({
            "server_id": "external",
            "server_name": "External MCP Server (port 3001)",
            "success": False,
            "message": str(e)
        })
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    }

@router.post("/batch/stop-all")
async def stop_all_servers(db: Session = Depends(get_db)):
    """Stop all running MCP servers."""
    servers = db.query(MCPServer).filter(
        MCPServer.config_type == "command",
        MCPServer.is_running == True
    ).all()
    
    results = []
    for server in servers:
        try:
            if mcp_process_manager:
                success, message = await mcp_process_manager.stop_server(server.id, db)
                results.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "success": success,
                    "message": message
                })
            else:
                results.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "success": False,
                    "message": "Process manager not available"
                })
        except Exception as e:
            results.append({
                "server_id": server.id,
                "server_name": server.name,
                "success": False,
                "message": str(e)
            })
    
    # Also try to stop external MCP server
    try:
        external_result = await stop_external_mcp_server()
        results.append({
            "server_id": "external",
            "server_name": "External MCP Server (port 3001)",
            "success": external_result.get("status") == "success",
            "message": external_result.get("message", "")
        })
    except Exception as e:
        results.append({
            "server_id": "external",
            "server_name": "External MCP Server (port 3001)",
            "success": False,
            "message": str(e)
        })
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    }

async def check_server_health_concurrent(server_id: int):
    """Helper for concurrent health checks with independent DB session."""
    from app.core.db import SessionLocal
    db = SessionLocal()
    try:
        return await check_server_health(server_id, db)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        db.close()

@router.post("/batch/refresh-health")
async def refresh_health_all_servers(db: Session = Depends(get_db)):
    """Check health status of all configured MCP servers concurrently."""
    servers = db.query(MCPServer).filter(MCPServer.is_active == True).all()
    
    # Create concurrent tasks
    tasks = [check_server_health_concurrent(server.id) for server in servers]
    
    # Execute concurrently
    results_raw = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    results = []
    for server, result in zip(servers, results_raw):
        if isinstance(result, Exception):
            results.append({
                "server_id": server.id,
                "server_name": server.name,
                "server_type": server.config_type,
                "success": False,
                "status": "error",
                "message": str(result)
            })
        else:
            results.append({
                "server_id": server.id,
                "server_name": server.name,
                "server_type": server.config_type,
                "success": True,
                "status": result.get("status", "unknown"),
                "message": result.get("message", "Health check completed")
            })
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    }

@router.post("/batch/discover-tools")
async def discover_tools_all_servers(db: Session = Depends(get_db)):
    """Discover or refresh tools from all active MCP servers."""
    servers = db.query(MCPServer).filter(MCPServer.is_active == True).all()
    
    results = []
    total_tools_discovered = 0
    
    for server in servers:
        try:
            if server.config_type == "manifest":
                # For manifest servers, refresh from manifest
                refresh_response = refresh_server_tools(server.id, db)
                results.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "server_type": server.config_type,
                    "success": True,
                    "tools_discovered": refresh_response.get("total_tools", 0),
                    "message": refresh_response.get("message", "Tools refreshed")
                })
                total_tools_discovered += refresh_response.get("total_tools", 0)
            elif server.config_type in ["command", "remote_http", "http"]:
                # For other server types, count existing tools or attempt rediscovery
                tool_count = db.query(MCPTool).filter(MCPTool.server_id == server.id).count()
                
                if server.config_type == "command" and mcp_process_manager:
                    # For command servers, we can potentially restart to rediscover tools
                    try:
                        # Just count existing tools for now
                        success = tool_count > 0
                        message = f"Found {tool_count} existing tools" if success else "No tools discovered yet"
                    except Exception as e:
                        success = False
                        message = str(e)
                else:
                    # For HTTP/remote servers, tools are discovered at connection time
                    success = tool_count > 0
                    message = f"Found {tool_count} tools" if success else "No tools available"
                
                results.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "server_type": server.config_type,
                    "success": success,
                    "tools_discovered": tool_count,
                    "message": message
                })
                if success:
                    total_tools_discovered += tool_count
            else:
                results.append({
                    "server_id": server.id,
                    "server_name": server.name,
                    "server_type": server.config_type,
                    "success": False,
                    "tools_discovered": 0,
                    "message": f"Tool discovery not supported for {server.config_type} servers"
                })
        except Exception as e:
            results.append({
                "server_id": server.id,
                "server_name": server.name,
                "server_type": server.config_type,
                "success": False,
                "tools_discovered": 0,
                "message": str(e)
            })
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "total_tools": total_tools_discovered,
        "results": results
    }

@router.post("/batch/test-connections")
async def test_connections_all_servers(db: Session = Depends(get_db)):
    """Test connectivity to all configured MCP servers."""
    servers = db.query(MCPServer).filter(MCPServer.is_active == True).all()
    
    results = []
    for server in servers:
        try:
            # Helper function to check if a server is local
            def is_local_server(server) -> bool:
                """Check if a server is running locally based on its configuration."""
                if server.config_type == "command":
                    return True  # Command servers are always local
                elif server.config_type == "http":
                    # Check if hostname is localhost or Docker internal
                    hostname = server.hostname or ""
                    if any(local in hostname.lower() for local in ["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"]):
                        return True
                    # Check if it's a Docker service name (no dots in hostname)
                    if "." not in hostname and not hostname.startswith("http"):
                        return True
                elif server.config_type == "manifest":
                    # Check manifest URL for local addresses
                    url = server.manifest_url or ""
                    if any(local in url.lower() for local in ["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"]):
                        return True
                    # Check configured hostname
                    hostname = server.hostname or ""
                    if any(local in hostname.lower() for local in ["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"]):
                        return True
                    # Docker service names
                    if hostname and "." not in hostname and not hostname.startswith("http"):
                        return True
                return False  # remote_http is always remote, others default to remote
            
            is_local = is_local_server(server)
            
            connection_test = {
                "server_id": server.id,
                "server_name": server.name,
                "server_type": server.config_type,
                "can_control": server.config_type == "command" or (is_local and server.config_type in ["http", "manifest"]),  # Local servers can potentially be controlled
                "is_cloud_managed": server.config_type == "remote_http" or (not is_local and server.config_type in ["http", "manifest"]),
                "is_local": is_local,  # Add explicit local flag
                "success": False,
                "latency_ms": None,
                "message": ""
            }
            
            import time
            start_time = time.time()
            
            if server.config_type == "manifest":
                # Test manifest URL accessibility
                manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server.id).first()
                if manifest:
                    try:
                        headers = {}
                        if manifest.api_key:
                            headers["Authorization"] = f"Bearer {manifest.api_key}"
                        resp = requests.get(manifest.url, headers=headers, timeout=5)
                        resp.raise_for_status()
                        connection_test["success"] = True
                        connection_test["message"] = "Manifest accessible"
                    except Exception as e:
                        connection_test["message"] = f"Manifest unreachable: {str(e)}"
                else:
                    connection_test["message"] = "No manifest configured"
                    
            elif server.config_type == "remote_http":
                # Test remote HTTP server connectivity
                if server.remote_config and server.remote_config.get("server_url"):
                    try:
                        resp = requests.get(
                            server.remote_config["server_url"], 
                            headers=server.remote_config.get("auth_headers", {}),
                            timeout=5
                        )
                        connection_test["success"] = resp.status_code < 500
                        connection_test["message"] = f"Remote server responded with {resp.status_code}"
                    except Exception as e:
                        connection_test["message"] = f"Connection failed: {str(e)}"
                else:
                    connection_test["message"] = "No remote URL configured"
                    
            elif server.config_type == "http":
                # Test HTTP server connectivity
                if server.hostname:
                    try:
                        url = server.hostname if server.hostname.startswith("http") else f"http://{server.hostname}"
                        resp = requests.get(url, timeout=5)
                        connection_test["success"] = resp.status_code < 500
                        connection_test["message"] = f"HTTP server responded with {resp.status_code}"
                    except Exception as e:
                        connection_test["message"] = f"Connection failed: {str(e)}"
                else:
                    connection_test["message"] = "No hostname configured"
                    
            elif server.config_type == "command":
                # For command servers, check if process is running
                if server.is_running:
                    connection_test["success"] = True
                    connection_test["message"] = f"Process running (PID: {server.process_id})"
                else:
                    connection_test["message"] = "Process not running"
            
            # Calculate latency
            end_time = time.time()
            connection_test["latency_ms"] = round((end_time - start_time) * 1000, 2)
            
            results.append(connection_test)
            
        except Exception as e:
            # Use the same helper function for consistency
            is_local = is_local_server(server) if 'is_local_server' in locals() else False
            results.append({
                "server_id": server.id,
                "server_name": server.name,
                "server_type": server.config_type,
                "can_control": server.config_type == "command" or (is_local and server.config_type in ["http", "manifest"]),
                "is_cloud_managed": server.config_type == "remote_http" or (not is_local and server.config_type in ["http", "manifest"]),
                "is_local": is_local,
                "success": False,
                "latency_ms": None,
                "message": str(e)
            })
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "results": results
    }