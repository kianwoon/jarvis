"""
Enhanced MCP Server Cache Management
Supports both manifest-based and command-based MCP servers with Redis caching
"""
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from app.core.db import get_db, MCPServer, MCPTool, MCPManifest
from app.core.redis_base import RedisCache
import json
import logging

logger = logging.getLogger(__name__)

class MCPServerCache:
    """Enhanced cache manager for MCP servers"""
    
    def __init__(self):
        self.cache = RedisCache(key_prefix="mcp_server:")
        self.cache_ttl = 3600  # 1 hour
    
    def get_all_servers(self) -> Dict[str, Any]:
        """Get all MCP servers from cache"""
        try:
            cached_data = self.cache.get("all_servers")
            if cached_data:
                return cached_data
            
            # If not in cache, load from database
            return self._reload_all_servers()
        except Exception as e:
            logger.error(f"Error getting servers from cache: {e}")
            return self._reload_all_servers()
    
    def get_server_by_id(self, server_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific server from cache"""
        try:
            cached_data = self.cache.get(f"server_{server_id}")
            if cached_data:
                return cached_data
            
            # If not in cache, load from database
            return self._load_server_from_db(server_id)
        except Exception as e:
            logger.error(f"Error getting server {server_id} from cache: {e}")
            return self._load_server_from_db(server_id)
    
    def get_active_tools(self) -> List[Dict[str, Any]]:
        """Get all active MCP tools from cache"""
        try:
            cached_tools = self.cache.get("active_tools")
            if cached_tools:
                return cached_tools
            
            return self._reload_active_tools()
        except Exception as e:
            logger.error(f"Error getting active tools from cache: {e}")
            return self._reload_active_tools()
    
    def get_tools_by_server(self, server_id: int) -> List[Dict[str, Any]]:
        """Get tools for a specific server from cache"""
        try:
            cached_tools = self.cache.get(f"server_{server_id}_tools")
            if cached_tools:
                return cached_tools
            
            return self._load_server_tools_from_db(server_id)
        except Exception as e:
            logger.error(f"Error getting tools for server {server_id} from cache: {e}")
            return self._load_server_tools_from_db(server_id)
    
    def invalidate_server_cache(self, server_id: int = None):
        """Invalidate cache for a specific server or all servers"""
        try:
            if server_id:
                self.cache.delete(f"server_{server_id}")
                self.cache.delete(f"server_{server_id}_tools")
                logger.info(f"Invalidated cache for server {server_id}")
            else:
                self.cache.delete("all_servers")
                self.cache.delete("active_tools")
                logger.info("Invalidated all server cache")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    def reload_all_caches(self):
        """Reload all caches from database"""
        try:
            self._reload_all_servers()
            self._reload_active_tools()
            logger.info("Reloaded all MCP caches")
        except Exception as e:
            logger.error(f"Error reloading caches: {e}")
    
    def _reload_all_servers(self) -> Dict[str, Any]:
        """Reload all servers from database"""
        try:
            with next(get_db()) as db:
                servers = db.query(MCPServer).all()
                server_data = {}
                
                for server in servers:
                    server_dict = {
                        "id": server.id,
                        "name": server.name,
                        "config_type": server.config_type,
                        "manifest_url": server.manifest_url,
                        "hostname": server.hostname,
                        "api_key": server.api_key,
                        "command": server.command,
                        "args": server.args,
                        "env": server.env,
                        "working_directory": server.working_directory,
                        "process_id": server.process_id,
                        "is_running": server.is_running,
                        "restart_policy": server.restart_policy,
                        "max_restarts": server.max_restarts,
                        "restart_count": server.restart_count,
                        "is_active": server.is_active,
                        "health_status": server.health_status,
                        "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None,
                        "created_at": server.created_at.isoformat() if server.created_at else None,
                        "updated_at": server.updated_at.isoformat() if server.updated_at else None
                    }
                    
                    # Add manifest content if available
                    if server.config_type == "manifest":
                        manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server.id).first()
                        if manifest:
                            server_dict["manifest_content"] = manifest.content
                    
                    server_data[str(server.id)] = server_dict
                    
                    # Cache individual server
                    self.cache.set(f"server_{server.id}", server_dict, expire=self.cache_ttl)
                
                # Cache all servers
                self.cache.set("all_servers", server_data, expire=self.cache_ttl)
                return server_data
                
        except Exception as e:
            logger.error(f"Error reloading servers from database: {e}")
            return {}
    
    def _reload_active_tools(self) -> List[Dict[str, Any]]:
        """Reload active tools from database"""
        try:
            with next(get_db()) as db:
                tools = db.query(MCPTool).filter(MCPTool.is_active == True).all()
                tool_data = []
                
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
                        "server_id": tool.server_id,
                        "created_at": tool.created_at.isoformat() if tool.created_at else None
                    }
                    tool_data.append(tool_dict)
                
                # Cache active tools
                self.cache.set("active_tools", tool_data, expire=self.cache_ttl)
                return tool_data
                
        except Exception as e:
            logger.error(f"Error reloading active tools from database: {e}")
            return []
    
    def _load_server_from_db(self, server_id: int) -> Optional[Dict[str, Any]]:
        """Load a specific server from database"""
        try:
            with next(get_db()) as db:
                server = db.query(MCPServer).filter(MCPServer.id == server_id).first()
                if not server:
                    return None
                
                server_dict = {
                    "id": server.id,
                    "name": server.name,
                    "config_type": server.config_type,
                    "manifest_url": server.manifest_url,
                    "hostname": server.hostname,
                    "api_key": server.api_key,
                    "command": server.command,
                    "args": server.args,
                    "env": server.env,
                    "working_directory": server.working_directory,
                    "process_id": server.process_id,
                    "is_running": server.is_running,
                    "restart_policy": server.restart_policy,
                    "max_restarts": server.max_restarts,
                    "restart_count": server.restart_count,
                    "is_active": server.is_active,
                    "health_status": server.health_status,
                    "last_health_check": server.last_health_check.isoformat() if server.last_health_check else None,
                    "created_at": server.created_at.isoformat() if server.created_at else None,
                    "updated_at": server.updated_at.isoformat() if server.updated_at else None
                }
                
                # Add manifest content if available
                if server.config_type == "manifest":
                    manifest = db.query(MCPManifest).filter(MCPManifest.server_id == server.id).first()
                    if manifest:
                        server_dict["manifest_content"] = manifest.content
                
                # Cache the server
                self.cache.set(f"server_{server_id}", server_dict, expire=self.cache_ttl)
                return server_dict
                
        except Exception as e:
            logger.error(f"Error loading server {server_id} from database: {e}")
            return None
    
    def _load_server_tools_from_db(self, server_id: int) -> List[Dict[str, Any]]:
        """Load tools for a specific server from database"""
        try:
            with next(get_db()) as db:
                tools = db.query(MCPTool).filter(MCPTool.server_id == server_id).all()
                tool_data = []
                
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
                        "server_id": tool.server_id,
                        "created_at": tool.created_at.isoformat() if tool.created_at else None
                    }
                    tool_data.append(tool_dict)
                
                # Cache server tools
                self.cache.set(f"server_{server_id}_tools", tool_data, expire=self.cache_ttl)
                return tool_data
                
        except Exception as e:
            logger.error(f"Error loading tools for server {server_id} from database: {e}")
            return []

# Global cache instance
mcp_server_cache = MCPServerCache()

# Convenience functions for backwards compatibility
def get_mcp_servers():
    """Get all MCP servers"""
    return mcp_server_cache.get_all_servers()

def get_mcp_server_by_id(server_id: int):
    """Get MCP server by ID"""
    return mcp_server_cache.get_server_by_id(server_id)

def get_enabled_mcp_tools():
    """Get all enabled MCP tools"""
    return mcp_server_cache.get_active_tools()

def reload_mcp_server_cache():
    """Reload all MCP server caches"""
    mcp_server_cache.reload_all_caches()

def invalidate_mcp_server_cache(server_id: int = None):
    """Invalidate MCP server cache"""
    mcp_server_cache.invalidate_server_cache(server_id)