"""
Example integration of Universal Cache System with workflow execution
This shows how to use the new cache manager for all node types
"""
import logging
from typing import Dict, Any, AsyncGenerator

from app.automation.core.node_executor import get_node_executor
from app.automation.core.cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


class WorkflowExecutorV2:
    """
    Enhanced workflow executor with universal cache support
    This is an example of how to integrate the cache system
    """
    
    def __init__(self):
        self.node_executor = get_node_executor()
        self.cache_manager = get_cache_manager()
    
    async def execute_workflow(
        self,
        workflow_config: Dict[str, Any],
        input_data: str,
        workflow_id: int,
        execution_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute workflow with universal cache support"""
        
        # Analyze cache relationships once at the start
        nodes = workflow_config.get("nodes", [])
        edges = workflow_config.get("edges", [])
        cache_analysis = self.cache_manager.analyze_workflow(nodes, edges)
        
        logger.info(f"[WORKFLOW V2] Starting with cache analysis: {cache_analysis}")
        
        # Process nodes based on execution sequence
        execution_sequence = workflow_config.get("execution_sequence", [])
        
        for node_id in execution_sequence:
            # Find node configuration
            node = self._find_node(nodes, node_id)
            if not node:
                continue
            
            node_type = node.get("type") or node.get("data", {}).get("type", "")
            
            # Route to appropriate handler with cache support
            if node_type in ["agentnode", "AgentNode"]:
                async for event in self._execute_agent_node(
                    node, input_data, workflow_id, execution_id, workflow_config
                ):
                    yield event
                    
            elif node_type in ["parallelnode", "ParallelNode"]:
                async for event in self._execute_parallel_node(
                    node, input_data, workflow_id, execution_id, workflow_config
                ):
                    yield event
                    
            # Add more node types as needed
    
    async def _execute_agent_node(
        self,
        node: Dict[str, Any],
        input_data: str,
        workflow_id: int,
        execution_id: str,
        workflow_config: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agent node with cache support"""
        
        # Define the actual execution function
        async def agent_execution_fn(node, input_data, workflow_id, execution_id, **kwargs):
            # Your existing agent execution logic here
            # This is just a placeholder
            yield {
                "type": "agent_start",
                "node_id": node.get("id"),
                "agent_name": node.get("data", {}).get("node", {}).get("agent_name", "")
            }
            
            # Simulate agent execution
            output = f"Agent output for: {input_data}"
            
            yield {
                "type": "agent_response",
                "content": output
            }
            
            yield {
                "type": "node_complete",
                "node_id": node.get("id"),
                "output": output
            }
        
        # Execute with cache support
        async for event in self.node_executor.execute_with_cache(
            node=node,
            input_data=input_data,
            workflow_id=workflow_id,
            execution_id=execution_id,
            workflow_config=workflow_config,
            execute_fn=agent_execution_fn
        ):
            yield event
    
    async def _execute_parallel_node(
        self,
        node: Dict[str, Any],
        input_data: str,
        workflow_id: int,
        execution_id: str,
        workflow_config: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute parallel node with cache support for each sub-agent"""
        
        # Get all downstream agents
        edges = workflow_config.get("edges", [])
        nodes = workflow_config.get("nodes", [])
        parallel_node_id = node.get("id")
        
        downstream_agents = []
        for edge in edges:
            if edge.get("source") == parallel_node_id:
                target_id = edge.get("target")
                target_node = self._find_node(nodes, target_id)
                if target_node and "agent" in target_node.get("type", "").lower():
                    downstream_agents.append(target_node)
        
        # Execute each agent with cache support
        for agent_node in downstream_agents:
            async for event in self._execute_agent_node(
                agent_node, input_data, workflow_id, execution_id, workflow_config
            ):
                yield event
    
    def _find_node(self, nodes: List[Dict], node_id: str) -> Optional[Dict]:
        """Find node by ID"""
        for node in nodes:
            if node.get("id") == node_id:
                return node
        return None


# Example usage
async def example_usage():
    """Example of using the new workflow executor"""
    executor = WorkflowExecutorV2()
    
    workflow_config = {
        "nodes": [...],  # Your nodes
        "edges": [...],  # Your edges
        "execution_sequence": ["start", "agent1", "cache1", "end"]
    }
    
    async for event in executor.execute_workflow(
        workflow_config=workflow_config,
        input_data="Test query",
        workflow_id=1,
        execution_id="test-123"
    ):
        print(f"Event: {event['type']} - {event.get('node_id', '')}")