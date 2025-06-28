"""
Automation Cache - Redis-based caching for workflows
Following existing Redis patterns from the codebase
"""
import json
import logging
from typing import Dict, List, Any, Optional
from app.core.redis_base import RedisCache
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Cache keys
AUTOMATION_WORKFLOWS_KEY = 'automation_workflows_cache'
AUTOMATION_EXECUTIONS_KEY = 'automation_executions_cache'

# Initialize cache with lazy Redis connection (following existing pattern)
cache = RedisCache(key_prefix="automation_")

def get_automation_workflows() -> Dict[str, Any]:
    """Get all automation workflows from cache"""
    cached = cache.get(AUTOMATION_WORKFLOWS_KEY)
    if cached:
        logger.info(f"[AUTOMATION CACHE] Retrieved {len(cached)} workflows from cache")
        return cached
    return reload_automation_workflows()

def reload_automation_workflows() -> Dict[str, Any]:
    """Reload automation workflows from database and cache them"""
    try:
        # Lazy import to avoid database connection at startup
        from app.core.db import SessionLocal, AutomationWorkflow
        
        db = SessionLocal()
        try:
            logger.info("[AUTOMATION CACHE] Starting workflow cache reload...")
            workflows = db.query(AutomationWorkflow).filter(AutomationWorkflow.is_active == True).all()
            
            workflows_dict = {}
            for workflow in workflows:
                workflows_dict[str(workflow.id)] = {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "langflow_config": workflow.langflow_config,
                    "trigger_config": workflow.trigger_config,
                    "is_active": workflow.is_active,
                    "created_by": workflow.created_by or "system",
                    "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
                    "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None
                }
            
            cache.set(AUTOMATION_WORKFLOWS_KEY, workflows_dict)
            logger.info(f"[AUTOMATION CACHE] Successfully cached {len(workflows_dict)} workflows")
            return workflows_dict
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Failed to reload automation workflows: {e}")
        return {}

def get_workflow_by_id(workflow_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific workflow by ID"""
    workflows = get_automation_workflows()
    return workflows.get(str(workflow_id))

def cache_workflow_execution(execution_id: str, execution_data: Dict[str, Any], ttl: int = 3600):
    """Cache workflow execution data with TTL"""
    execution_key = f"{AUTOMATION_EXECUTIONS_KEY}:{execution_id}"
    cache.set(execution_key, execution_data, expire=ttl)
    logger.info(f"[AUTOMATION CACHE] Cached execution {execution_id}")

def get_workflow_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow execution data from cache"""
    execution_key = f"{AUTOMATION_EXECUTIONS_KEY}:{execution_id}"
    return cache.get(execution_key)

def invalidate_workflow_cache():
    """Invalidate the workflow cache (for use after database updates)"""
    cache.delete(AUTOMATION_WORKFLOWS_KEY)
    logger.info("[AUTOMATION CACHE] Workflow cache invalidated")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    workflows = get_automation_workflows()
    return {
        "workflows_cached": len(workflows),
        "cache_prefix": cache.key_prefix,
        "redis_available": cache._get_client() is not None
    }