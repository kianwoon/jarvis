"""
Cache Management API Endpoints
Provides cache inspection, validation, and management capabilities
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime, timezone

from app.automation.integrations.redis_bridge import workflow_redis
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/cache/status/{node_id}")
async def get_cache_status(
    node_id: str,
    workflow_id: Optional[int] = Query(None),
    cache_key: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Get cache status for a specific cache node"""
    try:
        # If cache_key provided, use it directly
        if cache_key:
            actual_cache_key = cache_key
        else:
            # Search for cache keys matching this node
            search_patterns = [
                f"*{node_id}*",  # General pattern for node_id
                f"default:auto:w{workflow_id}:n{node_id}*" if workflow_id else f"default:auto:*:n{node_id}*",  # New cache format
                f"cache_{workflow_id}_{node_id}" if workflow_id else f"cache_{node_id}",  # Old format fallback
            ]
            
            actual_cache_key = None
            for pattern in search_patterns:
                cache_keys = workflow_redis.get_cache_keys_pattern(pattern)
                if cache_keys:
                    # Log all found keys for debugging
                    logger.info(f"[CACHE API] Found {len(cache_keys)} cache keys matching pattern '{pattern}': {cache_keys}")
                    # Sort by key length (longer keys are more specific) and take the most recent
                    cache_keys.sort(key=len, reverse=True)
                    actual_cache_key = cache_keys[0]
                    logger.info(f"[CACHE API] Selected cache key for node {node_id}: {actual_cache_key}")
                    break
            
            if not actual_cache_key:
                # Fallback to simple pattern
                actual_cache_key = f"cache_{workflow_id}_{node_id}" if workflow_id else f"cache_{node_id}"
        
        # Check if cache exists
        cache_exists = workflow_redis.exists(actual_cache_key)
        
        if not cache_exists:
            return {
                "node_id": node_id,
                "cache_key": actual_cache_key,
                "exists": False,
                "status": "empty",
                "message": "No cached data found"
            }
        
        # Get cache metadata
        cache_data = workflow_redis.get_value(actual_cache_key)
        
        if not cache_data:
            return {
                "node_id": node_id,
                "cache_key": actual_cache_key,
                "exists": False,
                "status": "expired",
                "message": "Cache data expired or corrupted"
            }
        
        # Calculate cache metadata
        cache_metadata = _get_cache_metadata(actual_cache_key, cache_data)
        
        return {
            "node_id": node_id,
            "cache_key": actual_cache_key,
            "exists": True,
            "status": "valid",
            "metadata": cache_metadata,
            "preview": _get_cache_preview(cache_data),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[CACHE API] Error getting cache status for {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

@router.get("/cache/preview/{cache_key}")
async def get_cache_preview(cache_key: str, max_length: int = Query(500)) -> Dict[str, Any]:
    """Get a preview of cached content"""
    try:
        # First try the cache_key directly
        actual_cache_key = cache_key
        if not workflow_redis.exists(cache_key):
            # Search for cache keys matching this pattern
            search_patterns = [
                f"*{cache_key}*",  # General pattern
                f"*{cache_key.replace('cache_', '')}*",  # Remove cache_ prefix and search
            ]
            
            found_key = None
            for pattern in search_patterns:
                cache_keys = workflow_redis.get_cache_keys_pattern(pattern)
                if cache_keys:
                    found_key = cache_keys[0]
                    logger.info(f"[CACHE API] Found actual cache key for preview {cache_key}: {found_key}")
                    break
            
            if not found_key:
                raise HTTPException(status_code=404, detail="Cache key not found")
            
            actual_cache_key = found_key
        
        # Get cached data
        cache_data = workflow_redis.get_value(actual_cache_key)
        
        if not cache_data:
            raise HTTPException(status_code=404, detail="Cache data expired or corrupted")
        
        # Generate preview
        preview = _get_detailed_cache_preview(cache_data, max_length)
        metadata = _get_cache_metadata(actual_cache_key, cache_data)
        
        return {
            "cache_key": actual_cache_key,
            "preview": preview,
            "metadata": metadata,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CACHE API] Error getting cache preview for {cache_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache preview: {str(e)}")

@router.post("/cache/validate")
async def validate_cache_existence(request: Dict[str, Any]) -> Dict[str, Any]:
    """Check if cache exists for given workflow configuration"""
    try:
        workflow_id = request.get("workflow_id")
        node_id = request.get("node_id")
        cache_config = request.get("cache_config", {})
        input_data = request.get("input_data", {})
        
        if not node_id:
            raise HTTPException(status_code=400, detail="node_id is required")
        
        # Generate cache key based on configuration
        cache_key = _generate_cache_key(
            workflow_id=workflow_id,
            node_id=node_id,
            cache_config=cache_config,
            input_data=input_data
        )
        
        # Check cache existence
        cache_exists = workflow_redis.exists(cache_key)
        
        result = {
            "cache_key": cache_key,
            "exists": cache_exists,
            "node_id": node_id,
            "workflow_id": workflow_id
        }
        
        if cache_exists:
            cache_data = workflow_redis.get_value(cache_key)
            if cache_data:
                result["metadata"] = _get_cache_metadata(cache_key, cache_data)
                result["preview"] = _get_cache_preview(cache_data)
            else:
                result["exists"] = False
                result["status"] = "expired"
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CACHE API] Error validating cache existence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate cache: {str(e)}")

@router.get("/cache/stats")
async def get_cache_statistics(workflow_id: Optional[int] = Query(None)) -> Dict[str, Any]:
    """Get cache statistics and analytics"""
    try:
        # Get all cache keys
        pattern = f"cache_{workflow_id}_*" if workflow_id else "cache_*"
        cache_keys = workflow_redis.get_cache_keys_pattern(pattern)
        
        if not cache_keys:
            return {
                "total_caches": 0,
                "workflow_id": workflow_id,
                "statistics": {},
                "message": "No cache data found"
            }
        
        # Analyze cache data
        total_size = 0
        valid_caches = 0
        expired_caches = 0
        cache_ages = []
        
        for cache_key in cache_keys:
            try:
                cache_data = workflow_redis.get_value(cache_key)
                if cache_data:
                    valid_caches += 1
                    size = len(str(cache_data))
                    total_size += size
                    
                    # Try to get cache timestamp if available
                    if isinstance(cache_data, dict) and "timestamp" in cache_data:
                        cache_time = datetime.fromisoformat(cache_data["timestamp"].replace('Z', '+00:00'))
                        age_hours = (datetime.now(timezone.utc) - cache_time).total_seconds() / 3600
                        cache_ages.append(age_hours)
                else:
                    expired_caches += 1
            except Exception as e:
                logger.warning(f"[CACHE STATS] Error analyzing cache {cache_key}: {e}")
                expired_caches += 1
        
        # Calculate statistics
        avg_age = sum(cache_ages) / len(cache_ages) if cache_ages else 0
        
        return {
            "workflow_id": workflow_id,
            "total_caches": len(cache_keys),
            "valid_caches": valid_caches,
            "expired_caches": expired_caches,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "average_age_hours": round(avg_age, 2) if avg_age else 0,
            "statistics": {
                "cache_hit_potential": f"{(valid_caches / len(cache_keys) * 100):.1f}%" if cache_keys else "0%",
                "storage_efficiency": "good" if total_size < 50 * 1024 * 1024 else "high",  # 50MB threshold
                "freshness": "fresh" if avg_age < 24 else "aging" if avg_age < 168 else "stale"  # 1 day / 1 week thresholds
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[CACHE API] Error getting cache statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")

@router.delete("/cache/clear/{cache_key}")
async def clear_cache(cache_key: str) -> Dict[str, Any]:
    """Clear a specific cache entry"""
    try:
        # First try the cache_key directly
        actual_cache_key = cache_key
        if not workflow_redis.exists(cache_key):
            # Search for cache keys matching this pattern
            search_patterns = [
                f"*{cache_key}*",  # General pattern
                f"*{cache_key.replace('cache_', '')}*",  # Remove cache_ prefix and search
            ]
            
            found_key = None
            for pattern in search_patterns:
                cache_keys = workflow_redis.get_cache_keys_pattern(pattern)
                if cache_keys:
                    found_key = cache_keys[0]
                    logger.info(f"[CACHE API] Found actual cache key for clearing {cache_key}: {found_key}")
                    break
            
            if not found_key:
                raise HTTPException(status_code=404, detail="Cache key not found")
            
            actual_cache_key = found_key
        
        success = workflow_redis.delete_value(actual_cache_key)
        
        if success:
            return {
                "cache_key": actual_cache_key,
                "cleared": True,
                "message": "Cache cleared successfully",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CACHE API] Error clearing cache {cache_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.delete("/cache/clear-node/{node_id}")
async def clear_node_cache(
    node_id: str,
    workflow_id: Optional[int] = Query(None)
) -> Dict[str, Any]:
    """Clear ALL cache entries for a specific node"""
    try:
        # Search for all cache keys matching this node
        search_patterns = [
            f"*{node_id}*",  # General pattern for node_id
            f"cache_{workflow_id}_{node_id}" if workflow_id else f"cache_{node_id}",  # Fallback pattern
        ]
        
        all_cache_keys = []
        for pattern in search_patterns:
            cache_keys = workflow_redis.get_cache_keys_pattern(pattern)
            if cache_keys:
                all_cache_keys.extend(cache_keys)
        
        # Remove duplicates
        all_cache_keys = list(set(all_cache_keys))
        
        if not all_cache_keys:
            return {
                "node_id": node_id,
                "cleared_count": 0,
                "message": "No cache entries found for this node"
            }
        
        # Clear each cache key
        cleared_count = 0
        cleared_keys = []
        for cache_key in all_cache_keys:
            try:
                if workflow_redis.delete_value(cache_key):
                    cleared_count += 1
                    cleared_keys.append(cache_key)
                    logger.info(f"[CACHE API] Cleared cache key: {cache_key}")
            except Exception as e:
                logger.warning(f"[CACHE CLEAR] Failed to clear {cache_key}: {e}")
        
        return {
            "node_id": node_id,
            "cleared_count": cleared_count,
            "total_found": len(all_cache_keys),
            "cleared_keys": cleared_keys,
            "success_rate": f"{(cleared_count / len(all_cache_keys) * 100):.1f}%",
            "message": f"Cleared {cleared_count} out of {len(all_cache_keys)} cache entries for node {node_id}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[CACHE API] Error clearing cache for node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear node cache: {str(e)}")

@router.delete("/cache/clear-all")
async def clear_all_cache(workflow_id: Optional[int] = Query(None)) -> Dict[str, Any]:
    """Clear all cache entries or all for a specific workflow"""
    try:
        # Get cache keys to clear
        pattern = f"cache_{workflow_id}_*" if workflow_id else "cache_*"
        cache_keys = workflow_redis.get_cache_keys_pattern(pattern)
        
        if not cache_keys:
            return {
                "cleared_count": 0,
                "workflow_id": workflow_id,
                "message": "No cache entries found to clear"
            }
        
        # Clear each cache key
        cleared_count = 0
        for cache_key in cache_keys:
            try:
                if workflow_redis.delete_value(cache_key):
                    cleared_count += 1
            except Exception as e:
                logger.warning(f"[CACHE CLEAR] Failed to clear {cache_key}: {e}")
        
        return {
            "cleared_count": cleared_count,
            "total_found": len(cache_keys),
            "workflow_id": workflow_id,
            "success_rate": f"{(cleared_count / len(cache_keys) * 100):.1f}%",
            "message": f"Cleared {cleared_count} out of {len(cache_keys)} cache entries",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"[CACHE API] Error clearing all cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# Helper functions

def _get_cache_metadata(cache_key: str, cache_data: Any) -> Dict[str, Any]:
    """Extract metadata from cached data"""
    try:
        # Calculate size
        data_size = len(str(cache_data))
        
        # Try to extract timestamp if available
        timestamp = None
        if isinstance(cache_data, dict):
            timestamp = cache_data.get("timestamp") or cache_data.get("created_at")
        
        # Calculate age if timestamp available
        age_info = {}
        if timestamp:
            try:
                cache_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                age_seconds = (datetime.now(timezone.utc) - cache_time).total_seconds()
                age_info = {
                    "age_seconds": int(age_seconds),
                    "age_minutes": round(age_seconds / 60, 1),
                    "age_hours": round(age_seconds / 3600, 2),
                    "created_at": timestamp
                }
            except Exception:
                pass
        
        # Determine data type
        data_type = type(cache_data).__name__
        if isinstance(cache_data, dict):
            data_type = "object"
        elif isinstance(cache_data, list):
            data_type = "array"
        elif isinstance(cache_data, str):
            data_type = "string"
        
        return {
            "size_bytes": data_size,
            "size_kb": round(data_size / 1024, 2),
            "data_type": data_type,
            "cache_key": cache_key,
            **age_info
        }
        
    except Exception as e:
        logger.warning(f"[CACHE METADATA] Error extracting metadata: {e}")
        return {
            "size_bytes": 0,
            "data_type": "unknown",
            "error": str(e)
        }

def _get_cache_preview(cache_data: Any, max_length: int = 100) -> Dict[str, Any]:
    """Generate a preview of cached data"""
    try:
        if isinstance(cache_data, dict):
            # For objects, show key structure
            preview_text = json.dumps(cache_data, indent=2)[:max_length]
            return {
                "type": "object",
                "keys": list(cache_data.keys()) if len(cache_data.keys()) < 10 else list(cache_data.keys())[:10] + ["..."],
                "preview_text": preview_text + "..." if len(preview_text) >= max_length else preview_text
            }
        elif isinstance(cache_data, list):
            preview_text = str(cache_data)[:max_length]
            return {
                "type": "array",
                "length": len(cache_data),
                "preview_text": preview_text + "..." if len(preview_text) >= max_length else preview_text
            }
        else:
            # For strings and other types
            preview_text = str(cache_data)[:max_length]
            return {
                "type": "string",
                "length": len(str(cache_data)),
                "preview_text": preview_text + "..." if len(preview_text) >= max_length else preview_text
            }
            
    except Exception as e:
        return {
            "type": "error",
            "preview_text": f"Error generating preview: {str(e)}"
        }

def _get_detailed_cache_preview(cache_data: Any, max_length: int = 500) -> Dict[str, Any]:
    """Generate a detailed preview of cached data"""
    basic_preview = _get_cache_preview(cache_data, max_length)
    
    try:
        # Add more detailed information
        if isinstance(cache_data, dict):
            basic_preview["key_count"] = len(cache_data.keys())
            if "output" in cache_data:
                basic_preview["has_output"] = True
                basic_preview["output_preview"] = str(cache_data["output"])[:200]
        elif isinstance(cache_data, str):
            basic_preview["word_count"] = len(cache_data.split())
            basic_preview["line_count"] = len(cache_data.split('\n'))
        
        return basic_preview
        
    except Exception:
        return basic_preview

def _generate_cache_key(workflow_id: Optional[int], node_id: str, cache_config: Dict[str, Any], input_data: Dict[str, Any]) -> str:
    """Generate cache key based on configuration"""
    try:
        cache_key_pattern = cache_config.get("cache_key_pattern", "auto")
        custom_key = cache_config.get("cache_key", "")
        
        if cache_key_pattern == "custom" and custom_key:
            return custom_key
        elif cache_key_pattern == "node_only":
            return f"cache_{node_id}"
        elif cache_key_pattern == "input_hash":
            import hashlib
            input_str = json.dumps(input_data, sort_keys=True)
            input_hash = hashlib.md5(input_str.encode()).hexdigest()[:8]
            return f"cache_{node_id}_{input_hash}"
        else:  # auto
            if workflow_id:
                return f"cache_{workflow_id}_{node_id}"
            else:
                return f"cache_{node_id}"
                
    except Exception as e:
        logger.warning(f"[CACHE KEY] Error generating cache key: {e}")
        return f"cache_{node_id}"