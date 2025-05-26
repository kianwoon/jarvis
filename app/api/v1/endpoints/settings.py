from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.core.db import SessionLocal, Settings as SettingsModel, MCPManifest, MCPTool, Base
from app.core.llm_settings_cache import reload_llm_settings
from app.core.vector_db_settings_cache import reload_vector_db_settings
from app.core.embedding_settings_cache import reload_embedding_settings
from app.core.iceberg_settings_cache import reload_iceberg_settings
from app.core.mcp_tools_cache import reload_enabled_mcp_tools
from typing import Any, Dict, Optional
from pydantic import BaseModel
import requests
import json
import logging
from sqlalchemy import inspect
from sqlalchemy.exc import ProgrammingError, DatabaseError
from sqlalchemy.sql import text

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class SettingsUpdate(BaseModel):
    settings: Dict[str, Any]
    persist_to_db: Optional[bool] = False
    reload_cache: Optional[bool] = False

@router.get("/init-mcp-tables")
def initialize_mcp_tables(db: Session = Depends(get_db)):
    """Initialize MCP tables if they don't exist"""
    try:
        inspector = inspect(db.bind)
        
        # Check if tables exist
        mcp_manifests_exists = 'mcp_manifests' in inspector.get_table_names()
        mcp_tools_exists = 'mcp_tools' in inspector.get_table_names()
        
        logger.info(f"MCP tables check: mcp_manifests={mcp_manifests_exists}, mcp_tools={mcp_tools_exists}")
        
        # Create tables if they don't exist
        from sqlalchemy.schema import CreateTable
        
        if not mcp_manifests_exists or not mcp_tools_exists:
            # Create the tables using Base metadata
            from app.core.db import MCPManifest, MCPTool
            from sqlalchemy import MetaData
            
            # Create a new MetaData instance to avoid conflicts
            metadata = MetaData()
            
            # Get the table objects from our models
            manifest_table = MCPManifest.__table__.tometadata(metadata) 
            tool_table = MCPTool.__table__.tometadata(metadata)
            
            # Create tables that don't exist
            with db.bind.begin() as conn:
                if not mcp_manifests_exists:
                    logger.info("Creating mcp_manifests table")
                    conn.execute(CreateTable(manifest_table))
                    logger.info("Successfully created mcp_manifests table with api_key column")
                
                if not mcp_tools_exists:
                    logger.info("Creating mcp_tools table")
                    conn.execute(CreateTable(tool_table))
                    logger.info("Successfully created mcp_tools table")
                
            logger.info("Tables created successfully")
        else:
            # Check if the api_key column exists in mcp_manifests
            mcp_manifest_columns = [col['name'] for col in inspector.get_columns('mcp_manifests')]
            if 'api_key' not in mcp_manifest_columns:
                logger.info("Adding api_key column to existing mcp_manifests table")
                with db.bind.begin() as conn:
                    # SQLite and PostgreSQL have different syntax
                    if 'sqlite' in str(db.bind.url):
                        conn.execute(text("ALTER TABLE mcp_manifests ADD COLUMN api_key TEXT"))
                    else:
                        conn.execute(text("ALTER TABLE mcp_manifests ADD COLUMN api_key VARCHAR(255)"))
                logger.info("Successfully added api_key column to mcp_manifests table")
        
        return {"success": True, "message": "MCP tables initialized successfully"}
    except Exception as e:
        logger.error(f"Error initializing MCP tables: {str(e)}")
        return {"success": False, "error": str(e)}

@router.get("/{category}")
def get_settings(category: str, db: Session = Depends(get_db)):
    settings_row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
    if not settings_row:
        raise HTTPException(status_code=404, detail="Settings not found")
    return {"category": category, "settings": settings_row.settings}

@router.put("/{category}")
def update_settings(category: str, update: SettingsUpdate, db: Session = Depends(get_db)):
    logger.info(f"Updating settings for category: {category}")
    logger.info(f"Update payload: persist_to_db={update.persist_to_db}, reload_cache={update.reload_cache}")
    
    # Debug log for API key
    if category == 'mcp' and 'api_key' in update.settings:
        has_api_key = bool(update.settings.get('api_key'))
        api_key_length = len(update.settings.get('api_key', ''))
        logger.info(f"MCP API Key present: {has_api_key}, length: {api_key_length}")
    
    # For MCP settings, remove API key before saving to settings table
    # It will be saved to the manifest table in handle_mcp_settings_update
    settings_for_db = update.settings.copy()
    if category == 'mcp' and 'api_key' in settings_for_db:
        logger.info("Removing API key from settings before saving to settings table")
        del settings_for_db['api_key']
    
    settings_row = db.query(SettingsModel).filter(SettingsModel.category == category).first()
    if settings_row:
        logger.info(f"Updating existing settings for {category}")
        # Debug for existing settings
        if category == 'mcp':
            current_api_key = settings_row.settings.get('api_key') if settings_row.settings else None
            has_current_api_key = bool(current_api_key)
            current_api_key_length = len(current_api_key or '')
            logger.info(f"Existing MCP API Key: {has_current_api_key}, length: {current_api_key_length}")
        
        settings_row.settings = settings_for_db
    else:
        logger.info(f"Creating new settings for {category}")
        settings_row = SettingsModel(category=category, settings=settings_for_db)
        db.add(settings_row)
    
    try:
        db.commit()
        db.refresh(settings_row)
        logger.info(f"Successfully saved settings to {category} table")
        
        # Verify API key was saved
        if category == 'mcp':
            saved_api_key = settings_row.settings.get('api_key')
            has_saved_api_key = bool(saved_api_key)
            saved_api_key_length = len(saved_api_key or '')
            logger.info(f"Saved MCP API Key: {has_saved_api_key}, length: {saved_api_key_length}")
            
            endpoint_prefix = update.settings.get('endpoint_prefix')
            if endpoint_prefix:
                logger.info(f"Settings contains endpoint_prefix: {endpoint_prefix}")
    except Exception as e:
        logger.error(f"Error saving settings to database: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")
    
    # If updating LLM settings, reload cache
    if category == 'llm':
        reload_llm_settings()
    
    # If updating storage settings, reload all related caches
    if category == 'storage':
        reload_vector_db_settings()
        reload_embedding_settings()
        reload_iceberg_settings()
    
    # If updating MCP settings, handle special processing
    if category == 'mcp':
        logger.info("Processing MCP settings with special handling")
        handle_mcp_settings_update(update, db)
    
    return {"category": category, "settings": settings_row.settings}

def handle_mcp_settings_update(update: SettingsUpdate, db: Session):
    """Handle MCP settings update with special processing for manifest URL and API key"""
    logger.info(f"Starting handle_mcp_settings_update with persist_to_db={update.persist_to_db}")
    
    if not update.persist_to_db:
        logger.info("persist_to_db is False, skipping special processing")
        return
    
    # Check if the required tables exist
    try:
        inspector = inspect(db.bind)
        
        if 'mcp_manifests' not in inspector.get_table_names():
            logger.error("mcp_manifests table does not exist in the database")
            return
            
        if 'mcp_tools' not in inspector.get_table_names():
            logger.error("mcp_tools table does not exist in the database")
            return
            
        logger.info("Verified that mcp_manifests and mcp_tools tables exist")
    except Exception as e:
        logger.error(f"Error checking for table existence: {str(e)}")
        return
    
    settings = update.settings
    manifest_url = settings.get('manifest_url')
    api_key = settings.get('api_key')
    hostname = settings.get('hostname')
    endpoint_prefix = settings.get('endpoint_prefix', '')
    
    # Debug log for incoming API key
    has_api_key = bool(api_key)
    api_key_length = len(api_key or '')
    logger.info(f"Incoming MCP API Key: present={has_api_key}, length={api_key_length}")
    
    logger.info(f"MCP settings: manifest_url={manifest_url}, api_key={'[REDACTED]' if api_key else 'None'}, hostname={hostname}, endpoint_prefix={endpoint_prefix}")
    
    if not manifest_url:
        logger.warning("No manifest URL provided for MCP settings update")
        return
    
    # Flag to track if we should attempt to reload the cache
    should_reload_cache = update.reload_cache
    
    try:
        # Explicitly start a new transaction for the manifest processing
        # This ensures that even if the manifest fetching fails, the settings are still saved
        logger.info("Starting manifest processing")
        
        # Fetch manifest data from the provided URL
        logger.info(f"Fetching manifest from URL: {manifest_url}")
        
        headers = {}
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"
            logger.info("Adding Authorization header with API key")
        
        try:
            # Substitute hostname for localhost if provided
            fetch_url = manifest_url
            if hostname and 'localhost' in manifest_url:
                fetch_url = manifest_url.replace('localhost', hostname)
                logger.info(f"Using hostname '{hostname}' instead of localhost: {fetch_url}")
            
            # Set a reasonable timeout for the request
            logger.info(f"Sending request to fetch manifest using URL: {fetch_url}")
            response = requests.get(fetch_url, headers=headers, timeout=5)
            
            logger.info(f"Got response with status code: {response.status_code}")
            response.raise_for_status()
            
            manifest_data = response.json()
            logger.info("Successfully parsed manifest JSON")
            
            if not manifest_data or not isinstance(manifest_data.get('tools'), list):
                logger.warning(f"Invalid manifest format from {manifest_url}")
                # Return without error, but don't update manifest or tools
                return
                
            # Extract hostname from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(fetch_url)
            hostname_from_url = parsed_url.netloc
            logger.info(f"Extracted hostname: {hostname_from_url}")
            
            # Upsert manifest in the database
            try:
                manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
                if not manifest:
                    logger.info(f"Creating new manifest for URL: {manifest_url}")
                    manifest = MCPManifest(
                        url=manifest_url,
                        hostname=hostname_from_url,
                        api_key=api_key,  # Store API key in manifest table
                        content=manifest_data
                    )
                    db.add(manifest)
                    try:
                        db.commit()
                        db.refresh(manifest)
                        logger.info(f"Successfully created manifest with ID: {manifest.id}")
                    except Exception as commit_error:
                        db.rollback()
                        logger.error(f"Error committing new manifest: {str(commit_error)}")
                        raise
                else:
                    logger.info(f"Updating existing manifest with ID: {manifest.id}")
                    manifest.content = manifest_data
                    manifest.hostname = hostname_from_url
                    manifest.api_key = api_key  # Update API key in manifest table
                    try:
                        db.commit()
                        logger.info("Successfully updated manifest")
                    except Exception as commit_error:
                        db.rollback()
                        
                        # Check if the error is related to the api_key column not existing
                        if "column mcp_manifests.api_key does not exist" in str(commit_error):
                            logger.error("The api_key column is missing from the mcp_manifests table")
                            raise HTTPException(
                                status_code=500, 
                                detail="The api_key column is missing from the mcp_manifests table. Please run the migration script at jarvis/scripts/add_api_key_column.py to add it."
                            )
                        else:
                            raise
                
                # Store endpoint_prefix in settings, but NOT the API key
                settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
                if settings_row:
                    # Make sure we preserve the existing settings but remove api_key
                    current_settings = settings_row.settings or {}
                    current_settings['endpoint_prefix'] = endpoint_prefix
                    current_settings['manifest_url'] = manifest_url
                    current_settings['hostname'] = hostname
                    # Remove API key from settings as it's now stored in the manifest table
                    if 'api_key' in current_settings:
                        del current_settings['api_key']
                    settings_row.settings = current_settings
                    logger.info(f"Updating MCP settings with endpoint_prefix: {endpoint_prefix}, removing API key from settings")
                else:
                    # Create new settings row without the API key
                    settings_row = SettingsModel(
                        category='mcp',
                        settings={
                            'endpoint_prefix': endpoint_prefix,
                            'manifest_url': manifest_url,
                            'hostname': hostname
                        }
                    )
                    db.add(settings_row)
                    logger.info(f"Creating new MCP settings with endpoint_prefix: {endpoint_prefix}, without API key")
                
                try:
                    db.commit()
                    logger.info("Successfully saved endpoint_prefix to settings table")
                except Exception as settings_error:
                    db.rollback()
                    logger.error(f"Error saving endpoint_prefix to settings: {str(settings_error)}")
                
                # Log the saved manifest
                saved_manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
                if saved_manifest:
                    has_saved_api_key = bool(saved_manifest.api_key)
                    api_key_length = len(saved_manifest.api_key or '')
                    logger.info(f"Verified manifest saved with ID: {saved_manifest.id}, API key present: {has_saved_api_key}, length: {api_key_length}")
                else:
                    logger.error("Failed to find saved manifest after commit")
                
                # Upsert tools from the manifest
                logger.info(f"Processing {len(manifest_data.get('tools', []))} tools")
                
                # First, get all existing tools for this manifest to check for uniqueness
                existing_tools = {}
                if manifest.id:
                    for tool in db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).all():
                        existing_tools[tool.name] = tool
                    logger.info(f"Found {len(existing_tools)} existing tools for this manifest")
                
                for i, tool in enumerate(manifest_data.get('tools', [])):
                    tool_name = tool.get('name')
                    if not tool_name:
                        logger.warning(f"Skipping tool at index {i} without a name")
                        continue
                        
                    logger.info(f"Processing tool: {tool_name}")
                    
                    # Check if this tool exists for this manifest
                    if tool_name in existing_tools:
                        db_tool = existing_tools[tool_name]
                        logger.info(f"Updating existing tool: {tool_name} (ID: {db_tool.id})")
                        db_tool.description = tool.get('description')
                        
                        # Apply endpoint prefix if it exists
                        tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                        
                        # First, try to strip any existing prefix to get the clean endpoint
                        clean_endpoint = tool_endpoint
                        try:
                            # Find the last component of the path which is likely the original endpoint
                            path_parts = tool_endpoint.rstrip('/').split('/')
                            if len(path_parts) > 1:
                                # The last part is likely the original tool name or operation
                                clean_endpoint = f"/{path_parts[-1]}"
                        except Exception as parse_error:
                            logger.warning(f"Error parsing endpoint for prefix cleanup: {str(parse_error)}")
                        
                        # Now apply the new prefix if it exists
                        if endpoint_prefix:
                            # Make sure we have clean slashes
                            prefix_clean = endpoint_prefix.rstrip('/')
                            endpoint_clean = clean_endpoint.lstrip('/')
                            
                            # Put them together
                            tool_endpoint = f"{prefix_clean}/{endpoint_clean}"
                            logger.info(f"Applied endpoint prefix to {tool_name}: {tool_endpoint}")
                        else:
                            # Use the original endpoint or default
                            tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                            logger.info(f"No endpoint prefix to apply for {tool_name}, using: {tool_endpoint}")
                        
                        db_tool.endpoint = tool_endpoint
                        db_tool.method = tool.get('method', 'POST')
                        db_tool.parameters = tool.get('parameters')
                        db_tool.headers = tool.get('headers')
                        # Preserve is_active status
                    else:
                        # Check for name collision with other manifests
                        # This can happen if tool name is not unique across manifests
                        name_collision = db.query(MCPTool).filter(MCPTool.name == tool_name).first()
                        if name_collision:
                            logger.warning(f"Tool name '{tool_name}' already exists with different manifest_id: {name_collision.manifest_id}")
                            # We'll create the tool with a unique name to avoid the collision
                            unique_tool_name = f"{tool_name}_{manifest.id}"
                            logger.info(f"Creating tool with modified name: {unique_tool_name}")
                            
                            # Apply endpoint prefix if it exists
                            tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                            
                            # First, try to strip any existing prefix to get the clean endpoint
                            clean_endpoint = tool_endpoint
                            try:
                                # Find the last component of the path which is likely the original endpoint
                                path_parts = tool_endpoint.rstrip('/').split('/')
                                if len(path_parts) > 1:
                                    # The last part is likely the original tool name or operation
                                    clean_endpoint = f"/{path_parts[-1]}"
                            except Exception as parse_error:
                                logger.warning(f"Error parsing endpoint for prefix cleanup: {str(parse_error)}")
                            
                            # Now apply the new prefix if it exists
                            if endpoint_prefix:
                                # Make sure we have clean slashes
                                prefix_clean = endpoint_prefix.rstrip('/')
                                endpoint_clean = clean_endpoint.lstrip('/')
                                
                                # Put them together
                                tool_endpoint = f"{prefix_clean}/{endpoint_clean}"
                                logger.info(f"Applied endpoint prefix to {unique_tool_name}: {tool_endpoint}")
                            else:
                                # Use the original endpoint or default
                                tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                                logger.info(f"No endpoint prefix to apply for {tool_name}, using: {tool_endpoint}")
                            
                            db_tool = MCPTool(
                                name=unique_tool_name,  # Use a unique name
                                description=tool.get('description'),
                                endpoint=tool_endpoint,
                                method=tool.get('method', 'POST'),
                                parameters=tool.get('parameters'),
                                headers=tool.get('headers'),
                                is_active=True,
                                manifest_id=manifest.id
                            )
                        else:
                            logger.info(f"Creating new tool: {tool_name}")
                            
                            # Apply endpoint prefix if it exists
                            tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                            
                            # First, try to strip any existing prefix to get the clean endpoint
                            clean_endpoint = tool_endpoint
                            try:
                                # Find the last component of the path which is likely the original endpoint
                                path_parts = tool_endpoint.rstrip('/').split('/')
                                if len(path_parts) > 1:
                                    # The last part is likely the original tool name or operation
                                    clean_endpoint = f"/{path_parts[-1]}"
                            except Exception as parse_error:
                                logger.warning(f"Error parsing endpoint for prefix cleanup: {str(parse_error)}")
                            
                            # Now apply the new prefix if it exists
                            if endpoint_prefix:
                                # Make sure we have clean slashes
                                prefix_clean = endpoint_prefix.rstrip('/')
                                endpoint_clean = clean_endpoint.lstrip('/')
                                
                                # Put them together
                                tool_endpoint = f"{prefix_clean}/{endpoint_clean}"
                                logger.info(f"Applied endpoint prefix to {tool_name}: {tool_endpoint}")
                            else:
                                # Use the original endpoint or default
                                tool_endpoint = tool.get('endpoint', f"/{tool_name}")
                                logger.info(f"No endpoint prefix to apply for {tool_name}, using: {tool_endpoint}")
                            
                            db_tool = MCPTool(
                                name=tool_name,
                                description=tool.get('description'),
                                endpoint=tool_endpoint,
                                method=tool.get('method', 'POST'),
                                parameters=tool.get('parameters'),
                                headers=tool.get('headers'),
                                is_active=True,
                                manifest_id=manifest.id
                            )
                        db.add(db_tool)
                
                try:
                    logger.info("Committing tool changes")
                    db.commit()
                    logger.info("Successfully committed tool changes")
                    
                    # Verify tool counts
                    tool_count = db.query(MCPTool).filter(MCPTool.manifest_id == manifest.id).count()
                    logger.info(f"Verified {tool_count} tools saved for manifest ID {manifest.id}")
                    
                except Exception as commit_error:
                    db.rollback()
                    logger.error(f"Error committing tool changes: {str(commit_error)}")
                    raise
            except Exception as manifest_error:
                logger.error(f"Error processing manifest: {str(manifest_error)}")
                raise
                  
        except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
            # Handle connection errors, timeouts, or invalid JSON
            logger.warning(f"Could not fetch or process manifest from {manifest_url}: {str(e)}")
            # Don't raise an exception, just log the warning
            # Settings will still be saved, but manifest and tools won't be updated
            
            # Make sure we still save the settings even if manifest fetch fails
            settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
            if settings_row:
                # Update existing settings
                current_settings = settings_row.settings or {}
                current_settings['manifest_url'] = manifest_url
                # Remove API key from settings as it should be in the manifest table
                if 'api_key' in current_settings:
                    del current_settings['api_key']
                current_settings['hostname'] = hostname
                current_settings['endpoint_prefix'] = endpoint_prefix
                settings_row.settings = current_settings
                logger.info("Updating MCP settings despite manifest fetch failure (API key removed from settings)")
            else:
                # Create new settings
                settings_row = SettingsModel(
                    category='mcp',
                    settings={
                        'manifest_url': manifest_url,
                        'hostname': hostname,
                        'endpoint_prefix': endpoint_prefix
                    }
                )
                db.add(settings_row)
                logger.info("Creating new MCP settings despite manifest fetch failure (without API key)")
            
            try:
                db.commit()
                logger.info("Successfully saved MCP settings despite manifest fetch failure")
            except Exception as settings_error:
                logger.error(f"Error saving MCP settings: {str(settings_error)}")
                db.rollback()
            
            # See if we have an existing manifest record we can update with just the URL and API key
            manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
            if manifest:
                logger.info(f"Updating existing manifest URL and preserving content")
                # Update hostname if provided
                if hostname:
                    manifest.hostname = hostname
                    logger.info(f"Updated manifest hostname to user-provided value: {hostname}")
                else:
                    # Just update the hostname from the URL if possible
                    try:
                        from urllib.parse import urlparse
                        parsed_url = urlparse(manifest_url)
                        url_hostname = parsed_url.netloc
                        manifest.hostname = url_hostname
                        logger.info(f"Updated manifest hostname from URL: {url_hostname}")
                    except Exception as parse_error:
                        logger.warning(f"Could not parse URL for hostname: {str(parse_error)}")
                
                # Update the API key in the manifest
                manifest.api_key = api_key
                logger.info(f"Updated API key in manifest record. API key present: {bool(api_key)}, length: {len(api_key or '')}")
                
                try:
                    db.commit()
                    logger.info("Updated manifest record")
                except Exception as commit_error:
                    logger.error(f"Failed to update manifest record: {str(commit_error)}")
                    db.rollback()
            else:
                # Create a new manifest record with the URL and hostname even if fetch failed
                logger.info(f"Creating new manifest record for URL: {manifest_url}")
                
                # Get hostname from URL or use provided hostname
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(manifest_url)
                    url_hostname = hostname if hostname else parsed_url.netloc
                    
                    # Create minimal manifest record
                    new_manifest = MCPManifest(
                        url=manifest_url,
                        hostname=url_hostname,
                        api_key=api_key,  # Store API key in manifest record
                        content={"tools": []}  # Empty tools array as placeholder
                    )
                    db.add(new_manifest)
                    db.commit()
                    logger.info(f"Created new manifest record with hostname: {url_hostname}, API key present: {bool(api_key)}, length: {len(api_key or '')}")
                except Exception as create_error:
                    logger.error(f"Failed to create manifest record: {str(create_error)}")
                    db.rollback()
            
            # We should still reload the cache if requested, using existing data
            should_reload_cache = update.reload_cache
        
        # Reload cache if requested
        if should_reload_cache:
            logger.info("Reloading MCP tools cache")
            try:
                # If endpoint_prefix has changed, we should update all tools in the database
                # to apply the new prefix to their endpoints
                if endpoint_prefix:
                    try:
                        # Find all tools in the database
                        all_tools = db.query(MCPTool).all()
                        update_count = 0
                        
                        for tool in all_tools:
                            # Get the original endpoint without any previous prefix
                            original_endpoint = tool.endpoint
                            
                            # Skip if the endpoint already has the correct prefix
                            if original_endpoint.startswith(endpoint_prefix):
                                continue
                            
                            # Check if there might be an old prefix we need to remove
                            clean_endpoint = original_endpoint
                            try:
                                # Find the last component of the path which is likely the original endpoint
                                path_parts = original_endpoint.rstrip('/').split('/')
                                if len(path_parts) > 1:
                                    # The last part is likely the original tool name
                                    clean_endpoint = f"/{path_parts[-1]}"
                            except Exception as parse_error:
                                logger.warning(f"Error parsing endpoint for prefix removal: {str(parse_error)}")
                            
                            # Apply the new prefix
                            new_endpoint = f"{endpoint_prefix.rstrip('/')}/{clean_endpoint.lstrip('/')}"
                            tool.endpoint = new_endpoint
                            update_count += 1
                            logger.info(f"Updated endpoint for tool {tool.name}: {original_endpoint} -> {new_endpoint}")
                        
                        if update_count > 0:
                            logger.info(f"Updated {update_count} tool endpoints with new prefix: {endpoint_prefix}")
                            # Explicitly commit the changes
                            try:
                                db.commit()
                                logger.info("Successfully committed tool endpoint updates")
                            except Exception as commit_error:
                                logger.error(f"Error committing tool endpoint updates: {str(commit_error)}")
                                db.rollback()
                                raise commit_error
                    
                    except Exception as tool_update_error:
                        logger.error(f"Error updating tool endpoints with new prefix: {str(tool_update_error)}")
                        db.rollback()
                
                reload_enabled_mcp_tools()
                logger.info("Successfully reloaded MCP tools cache")
            except Exception as cache_error:
                logger.error(f"Error reloading MCP tools cache: {str(cache_error)}")
            
    except Exception as e:
        logger.error(f"Error processing MCP settings update: {str(e)}")
        # Check if the error is related to the api_key column not existing
        if "column mcp_manifests.api_key does not exist" in str(e):
            logger.error("The api_key column is missing from the mcp_manifests table")
            raise HTTPException(
                status_code=500, 
                detail="The api_key column is missing from the mcp_manifests table. Please run the migration script at jarvis/scripts/add_api_key_column.py to add it."
            )
        
        # Don't roll back the main transaction - the settings should be saved
        # But we won't continue with manifests/tools
        logger.error("Main settings were saved, but manifest/tools processing failed")
        
        # Need to go back and remove the API key from settings table, as it should only be in manifest
        try:
            settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
            if settings_row and settings_row.settings and 'api_key' in settings_row.settings:
                logger.info("Removing API key from settings table after error")
                current_settings = settings_row.settings
                del current_settings['api_key']
                settings_row.settings = current_settings
                db.commit()
                logger.info("Successfully removed API key from settings table")
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up API key from settings: {str(cleanup_error)}")
            
        # Use proper imports for SQLAlchemy error types
        if isinstance(e, (ProgrammingError, DatabaseError)):
            # If we get a database schema error (like missing column)
            if "column mcp_manifests.api_key does not exist" in str(e):
                raise HTTPException(
                    status_code=500, 
                    detail="The api_key column is missing from the mcp_manifests table. "
                           "Please run database migrations or reinitialize MCP tables."
                )
            else:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    logger.info("Completed MCP settings update")

@router.get("/mock-manifest")
def get_mock_manifest():
    """Return a mock manifest for testing purposes"""
    logger.info("Serving mock manifest")
    return {
        "name": "Mock MCP Manifest",
        "description": "A mock manifest for testing MCP tool integration",
        "tools": [
            {
                "name": "get_datetime",
                "description": "Get the current date and time",
                "endpoint": "/get_datetime",
                "method": "GET",
                "parameters": {}
            },
            {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "endpoint": "/get_weather",
                "method": "POST",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or location name"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform a calculation",
                "endpoint": "/calculate",
                "method": "POST",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        ]
    }

@router.post("/insert-test-mcp-data")
def insert_test_mcp_data(db: Session = Depends(get_db)):
    """Insert test data directly into MCP tables for debugging"""
    try:
        logger.info("Inserting test data into MCP tables")
        
        # Create test manifest
        manifest_url = "http://test-manifest.local/manifest"
        manifest = db.query(MCPManifest).filter(MCPManifest.url == manifest_url).first()
        
        if not manifest:
            manifest = MCPManifest(
                url=manifest_url,
                hostname="test-manifest.local",
                content={
                    "name": "Test Manifest",
                    "description": "Test manifest for debugging",
                    "tools": [
                        {
                            "name": "test_datetime",
                            "description": "Test datetime tool",
                            "endpoint": "/test_datetime",
                            "method": "GET"
                        },
                        {
                            "name": "test_calculator",
                            "description": "Test calculator tool",
                            "endpoint": "/test_calculator",
                            "method": "POST"
                        }
                    ]
                }
            )
            db.add(manifest)
            db.commit()
            db.refresh(manifest)
            logger.info(f"Created test manifest with ID: {manifest.id}")
        else:
            logger.info(f"Using existing test manifest with ID: {manifest.id}")
        
        # Create test tools
        tools_data = [
            {
                "name": "test_datetime",
                "description": "Test datetime tool",
                "endpoint": "/test_datetime",
                "method": "GET",
                "parameters": {},
                "headers": {},
                "is_active": True
            },
            {
                "name": "test_calculator",
                "description": "Test calculator tool",
                "endpoint": "/test_calculator",
                "method": "POST",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string"
                        }
                    }
                },
                "headers": {},
                "is_active": True
            }
        ]
        
        created_tools = []
        for tool_data in tools_data:
            tool = db.query(MCPTool).filter(
                MCPTool.name == tool_data["name"],
                MCPTool.manifest_id == manifest.id
            ).first()
            
            if not tool:
                tool = MCPTool(
                    name=tool_data["name"],
                    description=tool_data["description"],
                    endpoint=tool_data["endpoint"],
                    method=tool_data["method"],
                    parameters=tool_data["parameters"],
                    headers=tool_data["headers"],
                    is_active=tool_data["is_active"],
                    manifest_id=manifest.id
                )
                db.add(tool)
                created_tools.append(tool_data["name"])
        
        db.commit()
        logger.info(f"Created {len(created_tools)} test tools: {', '.join(created_tools)}")
        
        # Reload MCP tools cache
        reload_enabled_mcp_tools()
        logger.info("Reloaded MCP tools cache")
        
        # Query database to verify
        manifest_count = db.query(MCPManifest).count()
        tool_count = db.query(MCPTool).count()
        
        return {
            "success": True,
            "manifest_count": manifest_count,
            "tool_count": tool_count,
            "manifest_id": manifest.id,
            "created_tools": created_tools
        }
    except Exception as e:
        logger.error(f"Error inserting test data: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to insert test data: {str(e)}") 