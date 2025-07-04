"""
Universal Node Executor with Cache Support
Wraps all node executions with consistent cache checking
"""
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from app.automation.core.cache_manager import get_cache_manager, CacheCheckResult

logger = logging.getLogger(__name__)


class UniversalNodeExecutor:
    """Executes any node type with universal cache support"""
    
    def __init__(self):
        self.cache_manager = get_cache_manager()
        
    async def execute_with_cache(
        self,
        node: Dict[str, Any],
        input_data: str,
        workflow_id: int,
        execution_id: str,
        workflow_config: Dict[str, Any],
        execute_fn: Any,  # The actual node execution function
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Universal node execution wrapper with cache support
        
        Args:
            node: Node configuration
            input_data: Input data for the node
            workflow_id: Workflow ID
            execution_id: Execution ID
            workflow_config: Full workflow configuration
            execute_fn: The specific execution function for this node type
            **kwargs: Additional arguments for the execution function
        """
        node_id = node.get("node_id") or node.get("id")
        node_type = node.get("type") or node.get("data", {}).get("type", "unknown")
        
        # Analyze workflow cache relationships if not already done
        if not self.cache_manager.cache_map:
            nodes = workflow_config.get("nodes", [])
            edges = workflow_config.get("edges", [])
            cache_analysis = self.cache_manager.analyze_workflow(nodes, edges)
            logger.info(f"[NODE EXECUTOR] Cache analysis: {cache_analysis}")
        
        # Step 1: Check all relevant caches before execution
        cache_result = await self.cache_manager.check_all_caches(
            node_id,
            input_data,
            workflow_id,
            check_upstream=True
        )
        
        if cache_result and cache_result.cache_hit:
            # Emit cache hit event
            yield {
                "type": "cache_hit",
                "node_id": node_id,
                "node_type": node_type,
                "cache_key": cache_result.cache_key,
                "cache_node_id": cache_result.cache_node_id,
                "from_cache": True,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Emit node completion with cached data
            yield {
                "type": "node_complete",
                "node_id": node_id,
                "node_type": node_type,
                "output": cache_result.cached_data,
                "from_cache": True,
                "cache_metadata": cache_result.metadata,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Return early - no need to execute
            logger.info(f"[NODE EXECUTOR] Using cached result for {node_type} node {node_id}")
            return
        
        # Step 2: No cache hit - execute the node
        logger.info(f"[NODE EXECUTOR] Executing {node_type} node {node_id} (no cache hit)")
        
        # Collect all outputs from the execution
        outputs = []
        final_output = None
        
        try:
            # Execute the node-specific function
            async for event in execute_fn(node, input_data, workflow_id, execution_id, **kwargs):
                # Pass through all events
                yield event
                
                # Capture output for caching
                if event.get("type") == "node_complete":
                    final_output = event.get("output")
                elif event.get("type") == "agent_response":
                    outputs.append(event.get("content", ""))
            
            # Determine what to cache
            output_to_cache = final_output or "\n".join(outputs) if outputs else None
            
            # Step 3: Store result in all connected caches
            if output_to_cache:
                stored_keys = await self.cache_manager.store_in_all_caches(
                    node_id,
                    output_to_cache,
                    workflow_id,
                    input_data
                )
                
                if stored_keys:
                    # Emit cache storage event
                    yield {
                        "type": "cache_stored",
                        "node_id": node_id,
                        "node_type": node_type,
                        "cache_keys": stored_keys,
                        "cache_count": len(stored_keys),
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"[NODE EXECUTOR] Error executing {node_type} node {node_id}: {e}")
            yield {
                "type": "node_error",
                "node_id": node_id,
                "node_type": node_type,
                "error": str(e),
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            raise


# Global instance
_node_executor = None

def get_node_executor() -> UniversalNodeExecutor:
    """Get or create the global node executor instance"""
    global _node_executor
    if _node_executor is None:
        _node_executor = UniversalNodeExecutor()
    return _node_executor