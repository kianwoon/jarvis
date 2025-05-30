import json
import logging
from app.core.db import SessionLocal, MCPTool, Settings as SettingsModel
from app.core.redis_base import RedisCache

MCP_TOOLS_KEY = 'mcp_tools_cache'

# Setup logging
logger = logging.getLogger(__name__)

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_enabled_mcp_tools():
    cached = cache.get(MCP_TOOLS_KEY)
    if cached:
        return cached
    return reload_enabled_mcp_tools()

def reload_enabled_mcp_tools():
    db = SessionLocal()
    try:
        # Get MCP settings to see if there's an endpoint prefix
        settings_row = db.query(SettingsModel).filter(SettingsModel.category == 'mcp').first()
        endpoint_prefix = ''
        
        if settings_row and settings_row.settings:
            endpoint_prefix = settings_row.settings.get('endpoint_prefix', '')
            if endpoint_prefix:
                logger.info(f"Found endpoint prefix in settings: {endpoint_prefix}")
        else:
            logger.warning("No MCP settings found in database")
        
        tools = db.query(MCPTool).filter(MCPTool.is_active == True).all()
        enabled_tools = {}
        for tool in tools:
            manifest = tool.manifest
            
            # Get API key from manifest rather than settings
            api_key = None
            if manifest:
                api_key = manifest.api_key
                if api_key:
                    has_api_key = bool(api_key)
                    api_key_length = len(api_key or '')
                    logger.info(f"Retrieved API key from manifest (ID: {manifest.id}): present={has_api_key}, length={api_key_length}")
            
            # Make sure endpoint has the correct prefix if needed
            tool_endpoint = tool.endpoint
            
            # For logging/debugging only
            original_endpoint = tool_endpoint
            
            enabled_tools[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                "endpoint": tool_endpoint,
                "method": tool.method,
                "parameters": tool.parameters,
                "headers": tool.headers,
                "manifest_id": tool.manifest_id,
                "manifest_hostname": manifest.hostname if manifest else None,
                "endpoint_prefix": endpoint_prefix if endpoint_prefix else None,
                "api_key": api_key  # Include API key from manifest in cached data
            }
            
            # Log if we're using a hostname and/or prefix
            if manifest and manifest.hostname:
                logger.info(f"Tool {tool.name} will use hostname: {manifest.hostname}")
            if endpoint_prefix:
                logger.info(f"Tool {tool.name} endpoint: {original_endpoint} (with prefix: {endpoint_prefix})")
        
        # Debug info before caching
        api_key_count = sum(1 for tool_info in enabled_tools.values() if tool_info.get('api_key'))
        logger.info(f"Caching {len(enabled_tools)} tools, {api_key_count} with API keys")
            
        cache.set(MCP_TOOLS_KEY, enabled_tools)
        logger.info(f"Reloaded {len(enabled_tools)} enabled MCP tools to cache")
        return enabled_tools
    finally:
        db.close() 