import logging
from app.core.db import SessionLocal, MCPManifest
from app.core.redis_base import RedisCache

MCP_MANIFESTS_KEY = 'mcp_manifests_cache'

# Setup logging
logger = logging.getLogger(__name__)

# Initialize cache with lazy Redis connection
cache = RedisCache(key_prefix="")

def get_mcp_manifests():
    """Get all MCP manifests from cache or reload from DB"""
    cached = cache.get(MCP_MANIFESTS_KEY)
    if cached:
        return cached
    return reload_mcp_manifests()

def get_mcp_manifest_by_id(manifest_id: int):
    """Get a specific manifest by ID from cache"""
    manifests = get_mcp_manifests()
    return manifests.get(str(manifest_id))

def reload_mcp_manifests():
    """Reload all MCP manifests from database to cache"""
    db = SessionLocal()
    try:
        manifests = db.query(MCPManifest).all()
        manifest_dict = {}
        
        for manifest in manifests:
            manifest_dict[str(manifest.id)] = {
                "id": manifest.id,
                "url": manifest.url,
                "hostname": manifest.hostname,
                "api_key": manifest.api_key,
                "content": manifest.content,
                "created_at": manifest.created_at.isoformat() if manifest.created_at else None,
                "updated_at": manifest.updated_at.isoformat() if manifest.updated_at else None
            }
            
        cache.set(MCP_MANIFESTS_KEY, manifest_dict)
        logger.info(f"Reloaded {len(manifest_dict)} MCP manifests to cache")
        return manifest_dict
    finally:
        db.close()

def invalidate_manifest_cache():
    """Invalidate the manifest cache to force reload on next access"""
    cache.delete(MCP_MANIFESTS_KEY)
    logger.info("Invalidated MCP manifests cache")