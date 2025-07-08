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
                # Validate langflow_config has nodes before caching
                langflow_config = workflow.langflow_config
                if langflow_config and isinstance(langflow_config, dict):
                    nodes = langflow_config.get('nodes', [])
                    if not nodes:
                        logger.warning(f"[AUTOMATION CACHE] Workflow {workflow.id} has empty nodes array in langflow_config")
                else:
                    logger.warning(f"[AUTOMATION CACHE] Workflow {workflow.id} has invalid langflow_config: {type(langflow_config)}")
                
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
    # First try to get from cache
    workflows = get_automation_workflows()
    workflow = workflows.get(str(workflow_id))
    
    # If not found or has invalid data, try direct database query
    if not workflow or not workflow.get('langflow_config', {}).get('nodes'):
        logger.warning(f"[AUTOMATION CACHE] Workflow {workflow_id} not found in cache or has no nodes, querying database directly")
        
        try:
            from app.core.db import SessionLocal, AutomationWorkflow
            db = SessionLocal()
            try:
                db_workflow = db.query(AutomationWorkflow).filter(AutomationWorkflow.id == workflow_id).first()
                if db_workflow:
                    workflow = {
                        "id": db_workflow.id,
                        "name": db_workflow.name,
                        "description": db_workflow.description,
                        "langflow_config": db_workflow.langflow_config,
                        "trigger_config": db_workflow.trigger_config,
                        "is_active": db_workflow.is_active,
                        "created_by": db_workflow.created_by or "system",
                        "created_at": db_workflow.created_at.isoformat() if db_workflow.created_at else None,
                        "updated_at": db_workflow.updated_at.isoformat() if db_workflow.updated_at else None
                    }
                    logger.info(f"[AUTOMATION CACHE] Loaded workflow {workflow_id} directly from database")
                    
                    # Update cache with fresh data
                    workflows[str(workflow_id)] = workflow
                    cache.set(AUTOMATION_WORKFLOWS_KEY, workflows)
            finally:
                db.close()
        except Exception as e:
            logger.error(f"[AUTOMATION CACHE] Failed to load workflow {workflow_id} from database: {e}")
    
    return workflow

def cache_workflow_execution(execution_id: str, execution_data: Dict[str, Any], ttl: int = 3600):
    """Cache workflow execution data with TTL"""
    execution_key = f"{AUTOMATION_EXECUTIONS_KEY}:{execution_id}"
    cache.set(execution_key, execution_data, expire=ttl)
    logger.info(f"[AUTOMATION CACHE] Cached execution {execution_id}")

def get_workflow_execution(execution_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow execution data from cache"""
    execution_key = f"{AUTOMATION_EXECUTIONS_KEY}:{execution_id}"
    return cache.get(execution_key)

def invalidate_workflow_cache(workflow_id: Optional[int] = None):
    """Invalidate the workflow cache and related caches (for use after database updates)"""
    # Add a small delay to ensure database transaction is committed
    import time
    time.sleep(0.1)
    
    # Always clear the main workflow cache
    cache.delete(AUTOMATION_WORKFLOWS_KEY)
    logger.info("[AUTOMATION CACHE] Workflow cache invalidated")
    
    # If specific workflow, immediately reload it to prevent stale data
    if workflow_id:
        logger.info(f"[AUTOMATION CACHE] Preloading workflow {workflow_id} after invalidation")
        # Force reload from database
        reload_automation_workflows()
    
    # Clear execution cache entries
    try:
        # Get all execution cache keys and delete them
        redis_client = cache._get_client()
        if redis_client:
            # Clear execution caches
            execution_keys = redis_client.keys(f"{cache.key_prefix}{AUTOMATION_EXECUTIONS_KEY}:*")
            if execution_keys:
                redis_client.delete(*execution_keys)
                logger.info(f"[AUTOMATION CACHE] Cleared {len(execution_keys)} execution cache entries")
            
            # Clear workflow state caches if workflow_id is provided
            if workflow_id:
                state_keys = redis_client.keys(f"{cache.key_prefix}workflow_state:{workflow_id}:*")
                if state_keys:
                    redis_client.delete(*state_keys)
                    logger.info(f"[AUTOMATION CACHE] Cleared {len(state_keys)} workflow state entries for workflow {workflow_id}")
                
                # Clear node-level caches for this workflow
                node_cache_keys = redis_client.keys(f"{cache.key_prefix}cache_{workflow_id}_*")
                if node_cache_keys:
                    redis_client.delete(*node_cache_keys)
                    logger.info(f"[AUTOMATION CACHE] Cleared {len(node_cache_keys)} node cache entries for workflow {workflow_id}")
            else:
                # Clear all workflow state caches
                all_state_keys = redis_client.keys(f"{cache.key_prefix}workflow_state:*")
                if all_state_keys:
                    redis_client.delete(*all_state_keys)
                    logger.info(f"[AUTOMATION CACHE] Cleared {len(all_state_keys)} workflow state entries")
                
                # Clear all node-level caches
                all_node_keys = redis_client.keys(f"{cache.key_prefix}cache_*")
                if all_node_keys:
                    redis_client.delete(*all_node_keys)
                    logger.info(f"[AUTOMATION CACHE] Cleared {len(all_node_keys)} node cache entries")
                    
    except Exception as e:
        logger.error(f"[AUTOMATION CACHE] Error clearing additional caches: {e}")
    
    logger.info("[AUTOMATION CACHE] Complete cache invalidation finished")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics"""
    workflows = get_automation_workflows()
    return {
        "workflows_cached": len(workflows),
        "cache_prefix": cache.key_prefix,
        "redis_available": cache._get_client() is not None
    }