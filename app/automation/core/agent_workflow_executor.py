"""
Agent Workflow Executor - Bridges visual workflows with the multi-agent system
Leverages the proven LangGraph multi-agent infrastructure for execution
"""
import logging
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem, agent_instance_pool
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_name
from app.automation.integrations.redis_bridge import workflow_redis
from app.automation.integrations.postgres_bridge import postgres_bridge
from app.core.langfuse_integration import get_tracer
from app.automation.core.workflow_state import WorkflowState, workflow_state_manager
from app.automation.core.resource_monitor import get_resource_monitor

logger = logging.getLogger(__name__)

class AgentWorkflowExecutor:
    """Executes visual workflows using agent-based approach with LangGraph"""
    
    def __init__(self):
        self.tracer = get_tracer()
        self.dynamic_agent_system = None  # Will use instance pool instead
        self.resource_monitor = get_resource_monitor()
    
    async def execute_agent_workflow(
        self,
        workflow_id: int,
        execution_id: str,
        workflow_config: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        trace=None  # Accept trace parameter from API layer like standard chat mode
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute agent-based workflow with real-time streaming
        
        Args:
            workflow_id: Workflow identifier
            execution_id: Unique execution identifier
            workflow_config: Visual workflow configuration (nodes and edges)
            input_data: Optional input data for workflow
            message: Optional message to trigger workflow
            trace: Langfuse trace from API layer (follows standard chat mode pattern)
            
        Yields:
            Real-time execution updates with agent progress
        """
        # Use provided trace from API layer (standard chat mode pattern)
        execution_trace = trace
        
        # Start resource monitoring
        resource_usage = self.resource_monitor.start_workflow_monitoring(workflow_id, execution_id)
        
        try:
            logger.info(f"[AGENT WORKFLOW] Starting workflow {workflow_id}, execution {execution_id}")
            
            # OPTIMIZED: Use agent instance pool instead of creating fresh instances
            # This prevents resource waste while maintaining isolation
            self.dynamic_agent_system = await agent_instance_pool.get_or_create_instance(trace=execution_trace)
            logger.info(f"[AGENT WORKFLOW] Retrieved DynamicMultiAgentSystem from pool for workflow {workflow_id}")
            
            # Use the trace provided by API layer instead of creating our own
            # This follows the exact same pattern as standard chat mode
            
            # Initialize workflow state
            workflow_state_obj = workflow_state_manager.create_state(workflow_id, execution_id)
            
            # Store initial input data and message in workflow state
            if input_data:
                workflow_state_obj.set_state("input_data", input_data, "initial_input")
            if message:
                workflow_state_obj.set_state("user_message", message, "initial_message")
            
            # Initialize execution tracking
            workflow_state = {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "input_data": input_data,
                "message": message,
                "agent_states": {},
                "execution_log": [],
                "workflow_state": workflow_state_obj.to_dict()
            }
            
            # Cache initial state
            workflow_redis.set_workflow_state(workflow_id, execution_id, workflow_state)
            
            # Yield initial status
            yield {
                "type": "workflow_start",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "running"
            }
            
            # Convert visual workflow to agent execution plan with enhanced 4-way connectivity
            agent_plan = self._convert_workflow_to_agent_plan(workflow_config, input_data, message)
            
            # Process enhanced workflow metadata if available
            if workflow_config.get('connectivity_type') == '4-way':
                logger.info(f"[AGENT WORKFLOW] Processing 4-way connectivity workflow v{workflow_config.get('version', '1.0')}")
                
                # Use execution sequence if provided
                if 'execution_sequence' in workflow_config:
                    agent_plan['execution_sequence'] = workflow_config['execution_sequence']
                    logger.info(f"[AGENT WORKFLOW] Using predefined execution sequence: {len(agent_plan['execution_sequence'])} steps")
                
                # Process node relationships for enhanced execution
                if 'node_relationships' in workflow_config:
                    agent_plan['node_relationships'] = workflow_config['node_relationships']
                    logger.info(f"[AGENT WORKFLOW] Loaded node relationships for {len(agent_plan['node_relationships'])} nodes")
                
                # Process enhanced edge information
                if workflow_config.get('edges'):
                    enhanced_edges = [edge for edge in workflow_config['edges'] if 'connectivity_info' in edge]
                    if enhanced_edges:
                        agent_plan['enhanced_edges'] = enhanced_edges
                        logger.info(f"[AGENT WORKFLOW] Processing {len(enhanced_edges)} enhanced edges with 4-way connectivity info")
            
            # Create workflow planning span with enhanced debugging
            workflow_planning_span = None
            if execution_trace:
                try:
                    # SIMPLIFIED: Create simple workflow planning span
                    workflow_planning_span = self.tracer.create_span(
                        execution_trace,
                        name="workflow-planning",
                        metadata={
                            "operation": "workflow_planning",
                            "pattern": agent_plan["pattern"],
                            "agent_count": len(agent_plan["agents"]),
                            "state_enabled_agents": len([a for a in agent_plan["agents"] if a.get("state_enabled", False)])
                        }
                    )
                    logger.debug(f"Workflow planning span created: {workflow_planning_span is not None}")
                    
                    if workflow_planning_span:
                        # Fix: Use correct Langfuse API - spans use .end() method
                        workflow_planning_span.end(
                            output={
                                "agents": [a["agent_name"] for a in agent_plan["agents"]],
                                "pattern": agent_plan["pattern"],
                                "enhanced_features": {
                                    "state_enabled_agents": len([a for a in agent_plan["agents"] if a.get("state_enabled", False)]),
                                    "direct_chaining": any(a.get("state_enabled", False) for a in agent_plan["agents"]),
                                    "mixed_workflow": len(agent_plan.get("state_nodes", [])) > 0 and any(a.get("state_enabled", False) for a in agent_plan["agents"])
                                },
                                "planning_completed": True
                            }
                        )
                        logger.debug("Workflow planning span completed successfully")
                except Exception as e:
                    logger.warning(f"Failed to create workflow planning span: {e}")
                    logger.debug(f"Workflow planning span failed: {e}")
            
            # Yield enhanced agent plan with execution sequence
            yield {
                "type": "agent_plan",
                "agents": agent_plan["agents"],
                "execution_pattern": agent_plan["pattern"],
                "execution_sequence": agent_plan.get("execution_sequence", []),
                "connectivity_type": workflow_config.get('connectivity_type', 'legacy'),
                "workflow_version": workflow_config.get('version', '1.0')
            }
            
            # Execute agents using multi-agent system
            async for update in self._execute_agent_plan(
                agent_plan, 
                workflow_id, 
                execution_id, 
                execution_trace,
                workflow_state_obj
            ):
                yield update
            
            # Final status with workflow state
            final_state = {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat(),
                "workflow_state": workflow_state_obj.to_dict() if workflow_state_obj else None
            }
            workflow_redis.update_workflow_state(workflow_id, execution_id, final_state)
            
            # Update execution status in database
            postgres_bridge.update_execution(execution_id, {
                "status": "completed",
                "completed_at": datetime.utcnow()
            })
            logger.info(f"[AGENT WORKFLOW] Updated execution {execution_id} status to completed in database")
            
            # End execution trace with success and enhanced metadata
            if execution_trace:
                try:
                    # Calculate comprehensive metadata for trace completion
                    enhanced_metadata = {
                        "workflow_completed": True,
                        "final_state": final_state,
                        "state_checkpoints": len(workflow_state_obj.get_checkpoints()) if workflow_state_obj else 0,
                        "enhanced_features_used": bool(workflow_state_obj and any(
                            key.startswith("agent_chain_") for key in workflow_state_obj.get_all_state().keys()
                        )) if workflow_state_obj else False,
                        "execution_duration_info": {
                            "started_at": workflow_state.get("started_at"),
                            "completed_at": final_state["completed_at"]
                        },
                        "workflow_statistics": {
                            "total_agents_planned": len(agent_plan.get("agents", [])),
                            "state_enabled_agents": len([a for a in agent_plan.get("agents", []) if a.get("state_enabled", False)]),
                            "traditional_state_nodes": len(agent_plan.get("state_nodes", []))
                        }
                    }
                    
                    # Fix: Use correct Langfuse API - traces use update() method, not end_span_with_result()
                    execution_trace.update(
                        output=enhanced_metadata,
                        metadata={"success": True}
                    )
                    logger.info(f"[LANGFUSE] Completed automation execution trace for workflow {workflow_id}")
                    
                    # Flush traces to ensure they're sent
                    try:
                        self.tracer.flush()
                        logger.debug("Langfuse traces flushed successfully")
                    except Exception as flush_error:
                        logger.warning(f"Failed to flush traces: {flush_error}")
                        
                except Exception as e:
                    logger.warning(f"Failed to end execution trace: {e}")
            
            yield {
                "type": "workflow_complete",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "completed"
            }
            
            # Clean up workflow state from memory after completion
            if workflow_state_obj:
                workflow_state_manager.remove_state(workflow_id, execution_id)
            
            # OPTIMIZED: Return agent system to pool instead of destroying it
            if self.dynamic_agent_system:
                await agent_instance_pool.release_instance(self.dynamic_agent_system)
                self.dynamic_agent_system = None
                logger.info(f"[AGENT WORKFLOW] Returned DynamicMultiAgentSystem to pool after workflow completion")
            
            # Complete resource monitoring
            final_usage = self.resource_monitor.complete_workflow_monitoring(workflow_id, execution_id)
            if final_usage and final_usage.limits_exceeded:
                logger.warning(f"[AGENT WORKFLOW] Workflow {workflow_id} exceeded resource limits: {final_usage.limits_exceeded}")
            
            # CRITICAL: Comprehensive cleanup after successful completion
            try:
                logger.info(f"[AGENT WORKFLOW] Starting comprehensive cleanup after workflow completion")
                
                # 1. Clean up MCP subprocesses
                from app.core.unified_mcp_service import cleanup_mcp_subprocesses, mcp_subprocess_pool
                await cleanup_mcp_subprocesses()
                
                # 2. Emergency pool reset if pool is corrupted
                pool_stats = mcp_subprocess_pool.get_pool_stats()
                if pool_stats.get("active_processes", 0) > 15:  # Emergency threshold
                    logger.warning(f"[AGENT WORKFLOW] Emergency MCP pool cleanup - {pool_stats['active_processes']} active processes")
                    await mcp_subprocess_pool.cleanup_all()
                
                logger.info(f"[AGENT WORKFLOW] MCP subprocess cleanup completed after successful workflow")
            except Exception as mcp_error:
                logger.warning(f"[AGENT WORKFLOW] MCP subprocess cleanup failed after completion: {mcp_error}")
            
            # Force garbage collection to prevent memory accumulation
            try:
                import gc
                gc.collect()
                logger.info(f"[AGENT WORKFLOW] Garbage collection completed after successful workflow")
            except Exception as gc_error:
                logger.warning(f"[AGENT WORKFLOW] Garbage collection failed: {gc_error}")
            
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Execution failed: {e}")
            
            # End execution trace with error
            if execution_trace:
                try:
                    # Fix: Use correct Langfuse API - traces use update() method, not end_span_with_result()
                    execution_trace.update(
                        output=f"Error: {str(e)}",
                        metadata={
                            "success": False,
                            "error": str(e),
                            "workflow_id": workflow_id,
                            "execution_id": execution_id
                        }
                    )
                    logger.info(f"[LANGFUSE] Ended automation execution trace with error for workflow {workflow_id}")
                    logger.debug("Automation workflow trace ended with error")
                except Exception as trace_error:
                    logger.warning(f"Failed to end execution trace with error: {trace_error}")
            
            # Update error state
            error_state = {
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": str(e)
            }
            workflow_redis.update_workflow_state(workflow_id, execution_id, error_state)
            
            # Update execution status in database
            postgres_bridge.update_execution(execution_id, {
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.utcnow()
            })
            logger.info(f"[AGENT WORKFLOW] Updated execution {execution_id} status to failed in database")
            
            # Update execution trace with error (traces use update, not end_span_with_result)
            if execution_trace:
                try:
                    execution_trace.update(
                        output=f"Error: {str(e)}",
                        metadata={
                            "success": False,
                            "error": str(e),
                            "workflow_id": workflow_id,
                            "execution_id": execution_id
                        }
                    )
                except Exception:
                    pass
            
            yield {
                "type": "workflow_error",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "error": str(e)
            }
            
            # Clean up workflow state on error
            try:
                workflow_state_manager.remove_state(workflow_id, execution_id)
            except:
                pass
            
            # OPTIMIZED: Return agent system to pool even on error
            if self.dynamic_agent_system:
                try:
                    await agent_instance_pool.release_instance(self.dynamic_agent_system)
                    self.dynamic_agent_system = None
                    logger.info(f"[AGENT WORKFLOW] Returned DynamicMultiAgentSystem to pool after workflow error")
                except Exception as cleanup_error:
                    logger.warning(f"[AGENT WORKFLOW] Failed to return agent system to pool: {cleanup_error}")
                    self.dynamic_agent_system = None
            
            # Complete resource monitoring with error status
            try:
                final_usage = self.resource_monitor.complete_workflow_monitoring(workflow_id, execution_id)
                if final_usage and final_usage.limits_exceeded:
                    logger.warning(f"[AGENT WORKFLOW] Failed workflow {workflow_id} exceeded resource limits: {final_usage.limits_exceeded}")
            except Exception as monitor_error:
                logger.warning(f"[AGENT WORKFLOW] Failed to complete resource monitoring: {monitor_error}")
            
            # CRITICAL: Comprehensive cleanup after workflow error
            try:
                logger.info(f"[AGENT WORKFLOW] Starting comprehensive cleanup after workflow error")
                
                # 1. Clean up MCP subprocesses
                from app.core.unified_mcp_service import cleanup_mcp_subprocesses, mcp_subprocess_pool
                await cleanup_mcp_subprocesses()
                
                # 2. Emergency pool reset if pool is corrupted (more aggressive on error)
                pool_stats = mcp_subprocess_pool.get_pool_stats()
                if pool_stats.get("active_processes", 0) > 10:  # Lower threshold on error
                    logger.warning(f"[AGENT WORKFLOW] Emergency MCP pool cleanup after error - {pool_stats['active_processes']} active processes")
                    await mcp_subprocess_pool.cleanup_all()
                
                logger.info(f"[AGENT WORKFLOW] MCP subprocess cleanup completed after error")
            except Exception as mcp_error:
                logger.warning(f"[AGENT WORKFLOW] MCP subprocess cleanup failed after error: {mcp_error}")
            
            # Force garbage collection to prevent memory accumulation
            try:
                import gc
                gc.collect()
                logger.info(f"[AGENT WORKFLOW] Garbage collection completed after workflow error")
            except Exception as gc_error:
                logger.warning(f"[AGENT WORKFLOW] Garbage collection failed: {gc_error}")
    
    def _convert_workflow_to_agent_plan(
        self, 
        workflow_config: Dict[str, Any], 
        input_data: Optional[Dict[str, Any]], 
        message: Optional[str]
    ) -> Dict[str, Any]:
        """Convert visual workflow configuration to agent execution plan"""
        
        nodes = workflow_config.get("nodes", [])
        edges = workflow_config.get("edges", [])
        
        # Extract agent nodes, state nodes, and output nodes from workflow
        agent_nodes = []
        state_nodes = []
        output_node = None
        
        for node in nodes:
            node_data = node.get("data", {})
            node_type = node_data.get("type", "")
            
            # Check for state nodes
            if node_type == "StateNode" or node.get("type") == "statenode":
                state_nodes.append({
                    "node_id": node.get("id"),
                    "node_data": node_data,
                    "position": node.get("position", {}),
                    "state_operation": node_data.get("stateOperation", "merge"),
                    "state_keys": node_data.get("stateKeys", []),
                    "state_values": node_data.get("stateValues", {}),
                    "persistence": node_data.get("persistence", True),
                    "checkpoint_name": node_data.get("checkpointName", "")
                })
                logger.debug(f"[WORKFLOW CONVERSION] Found StateNode: {node.get('id')}")
            
            # Check for output nodes
            elif node_type == "OutputNode" or node.get("type") == "outputnode":
                # Extract OutputNode configuration
                output_config = node_data.get("node", {}) or node_data
                
                output_node = {
                    "node_id": node.get("id"),
                    "output_format": (
                        output_config.get("output_format") or
                        node_data.get("output_format") or
                        "text"
                    ),
                    "include_metadata": (
                        output_config.get("include_metadata") or
                        node_data.get("include_metadata") or
                        False
                    ),
                    "include_tool_calls": (
                        output_config.get("include_tool_calls") or
                        node_data.get("include_tool_calls") or
                        False
                    ),
                    "auto_display": (
                        output_config.get("auto_display") or
                        node_data.get("auto_display") or
                        True
                    ),
                    "auto_save": (
                        output_config.get("auto_save") or
                        node_data.get("auto_save") or
                        False
                    )
                }
                logger.debug(f"[WORKFLOW CONVERSION] Found OutputNode: {node.get('id')} with format: {output_node['output_format']}, auto_display: {output_node['auto_display']}, auto_save: {output_node['auto_save']}")
            
            # Check for both legacy and agent-based node types
            elif node_type == "AgentNode" or node.get("type") == "agentnode":
                # Try multiple ways to extract agent configuration
                agent_config = node_data.get("node", {})
                
                # Get agent name from various possible locations
                agent_name = (
                    agent_config.get("agent_name") or
                    node_data.get("agentName") or  # Frontend format
                    node_data.get("agent_name") or
                    ""
                )
                
                logger.debug(f"[WORKFLOW CONVERSION] Node: {node.get('id')}, Type: {node_type}, Agent: {agent_name}")
                
                if agent_name:
                    # Get agent from cache
                    agent_info = get_agent_by_name(agent_name)
                    if agent_info:
                        # FIXED: Only extract user-configured custom prompt, don't pad with empty strings
                        custom_prompt = ""
                        if agent_config.get("custom_prompt"):
                            custom_prompt = agent_config.get("custom_prompt")
                        elif node_data.get("customPrompt"):
                            custom_prompt = node_data.get("customPrompt")
                        # Do NOT use agent template prompts - only use what user explicitly configured
                        
                        # FIXED: Only use user-configured tools, don't default to empty arrays
                        tools = []
                        if agent_config.get("tools"):
                            tools = agent_config.get("tools")
                        elif node_data.get("tools"):
                            tools = node_data.get("tools")
                        # Do NOT merge agent template tools - only use what user explicitly configured
                        
                        # FIXED: Only use user-configured context, never inject agent template defaults
                        # This prevents hardcoded model specifications and other template configs from corrupting user workflows
                        context = {}
                        if agent_config.get("context"):
                            context = agent_config.get("context")
                        elif node_data.get("context"):
                            context = node_data.get("context")
                        # Do NOT merge agent template context - only use what user explicitly configured
                        
                        # Extract timeout configuration from workflow node
                        # Debug logging to understand timeout extraction
                        context_timeout = context.get("timeout")
                        agent_config_timeout = agent_config.get("timeout")
                        node_data_timeout = node_data.get("timeout")
                        
                        logger.debug(f"[TIMEOUT DEBUG] Agent: {agent_name}")
                        logger.debug(f"[TIMEOUT DEBUG] context timeout: {context_timeout}")
                        logger.debug(f"[TIMEOUT DEBUG] agent_config timeout: {agent_config_timeout}")
                        logger.debug(f"[TIMEOUT DEBUG] node_data timeout: {node_data_timeout}")
                        logger.debug(f"[TIMEOUT DEBUG] agent_config keys: {list(agent_config.keys()) if agent_config else 'None'}")
                        logger.debug(f"[TIMEOUT DEBUG] node_data keys: {list(node_data.keys()) if node_data else 'None'}")
                        
                        configured_timeout = (
                            context_timeout or
                            agent_config_timeout or
                            node_data_timeout or
                            60  # fallback default
                        )
                        
                        logger.info(f"[TIMEOUT EXTRACTION] Agent {agent_name}: extracted timeout = {configured_timeout}s from workflow config")
                        
                        # Extract state management configuration
                        state_enabled = (
                            agent_config.get("state_enabled") or
                            node_data.get("stateEnabled") or
                            False
                        )
                        
                        state_operation = (
                            agent_config.get("state_operation") or
                            node_data.get("stateOperation") or
                            "passthrough"
                        )
                        
                        output_format = (
                            agent_config.get("output_format") or
                            node_data.get("outputFormat") or
                            "text"
                        )
                        
                        chain_key = (
                            agent_config.get("chain_key") or
                            node_data.get("chainKey") or
                            ""
                        )
                        
                        # CRITICAL FIX: Extract node-specific query
                        node_query = (
                            agent_config.get("query") or
                            node_data.get("query") or
                            ""
                        )
                        
                        agent_nodes.append({
                            "node_id": node.get("id"),
                            "agent_name": agent_name,
                            "agent_config": {
                                "name": agent_name,
                                "role": agent_info.get("role", ""),
                                "system_prompt": agent_info.get("system_prompt", ""),
                                # Filter out hardcoded model specs but preserve legitimate config
                                "config": self._filter_agent_config(agent_info.get("config", {}))
                            },
                            "custom_prompt": custom_prompt,
                            "query": node_query,  # Add node-specific query
                            "tools": tools,
                            "context": context,
                            "configured_timeout": configured_timeout,
                            "position": node.get("position", {}),
                            "state_enabled": state_enabled,
                            "state_operation": state_operation,
                            "output_format": output_format,
                            "chain_key": chain_key
                        })
        
        # Log the conversion results
        logger.info(f"[WORKFLOW CONVERSION] Found {len(agent_nodes)} agent nodes")
        for i, agent_node in enumerate(agent_nodes):
            state_info = ""
            if agent_node.get("state_enabled", False):
                state_info = f" [STATE: {agent_node.get('state_operation', 'passthrough')}, FORMAT: {agent_node.get('output_format', 'text')}]"
            logger.info(f"[WORKFLOW CONVERSION] Agent {i+1}: {agent_node['agent_name']} (Node: {agent_node['node_id']}){state_info}")
        
        # Determine execution pattern from edges (check workflow config first)
        execution_pattern = self._analyze_execution_pattern(agent_nodes, edges, workflow_config)
        
        # Prepare query from input or message
        query = message or input_data.get("query", "") if input_data else ""
        if not query and input_data:
            # Try to construct query from input data
            query = f"Process the following data: {json.dumps(input_data, indent=2)}"
        
        result = {
            "agents": agent_nodes,
            "state_nodes": state_nodes,
            "output_node": output_node,
            "pattern": execution_pattern,
            "query": query,
            "input_data": input_data,
            "edges": edges
        }
        
        # Add enhanced workflow metadata if available
        if hasattr(workflow_config, 'get') and workflow_config.get('connectivity_type') == '4-way':
            result.update({
                "execution_sequence": workflow_config.get('execution_sequence', []),
                "node_relationships": workflow_config.get('node_relationships', {}),
                "enhanced_edges": [edge for edge in workflow_config.get('edges', []) if 'connectivity_info' in edge],
                "connectivity_type": workflow_config.get('connectivity_type'),
                "workflow_version": workflow_config.get('version', '1.0')
            })
            
            logger.info(f"[WORKFLOW CONVERSION] Enhanced workflow metadata added - version {result['workflow_version']}")
        
        return result
    
    def _filter_agent_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out hardcoded model specifications and other unwanted defaults from agent config"""
        if not config:
            return {}
        
        # Create filtered config excluding hardcoded model specs
        filtered_config = {}
        
        # Explicitly exclude known hardcoded values
        excluded_keys = {
            "model",  # Removes "claude-3-sonnet-20240229" and other hardcoded models
            "temperature",  # Remove default temperature unless user-configured
            "timeout",  # Remove default timeout unless user-configured  
        }
        
        for key, value in config.items():
            if key not in excluded_keys:
                filtered_config[key] = value
        
        logger.debug(f"[CONFIG FILTER] Original config keys: {list(config.keys())}")
        logger.debug(f"[CONFIG FILTER] Filtered config keys: {list(filtered_config.keys())}")
        
        return filtered_config
    
    def _analyze_execution_pattern(self, agent_nodes: List[Dict], edges: List[Dict], workflow_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Analyze workflow structure to determine execution pattern
        
        FIXED: Check for explicit execution_sequence first, then fall back to heuristics
        Default to sequential execution to resolve user complaint about parallel execution
        Only use parallel when explicitly designed with multiple disconnected chains
        """
        
        # CRITICAL FIX: Check for explicit execution sequence first
        if workflow_config and workflow_config.get('execution_sequence'):
            execution_sequence = workflow_config.get('execution_sequence', [])
            # Count agent nodes in the sequence
            agent_nodes_in_sequence = [node_id for node_id in execution_sequence 
                                     if any(agent['node_id'] == node_id for agent in agent_nodes)]
            if len(agent_nodes_in_sequence) > 1:
                logger.info(f"[EXECUTION PATTERN] Using explicit execution sequence: {len(agent_nodes_in_sequence)} agents in sequence")
                return "sequential"
        
        if len(agent_nodes) <= 1:
            return "single"
        
        # CRITICAL FIX: Default to sequential execution
        # The original logic incorrectly defaulted to parallel for multiple starting nodes
        # User specifically complained: "it should have execute 1 at a time, finish and move on instead of running all 3"
        
        # Build dependency graph
        dependencies = {}
        for node in agent_nodes:
            dependencies[node["node_id"]] = []
        
        for edge in edges:
            target = edge.get("target")
            source = edge.get("source")
            if target and source and target in dependencies:
                dependencies[target].append(source)
        
        # Check for truly independent parallel chains
        # Only return "parallel" if there are multiple completely isolated chains
        isolated_chains = 0
        visited = set()
        
        for node_id in dependencies:
            if node_id not in visited:
                # Check if this node is part of an isolated chain
                chain_nodes = self._find_connected_nodes(node_id, dependencies, edges)
                if len(chain_nodes) > 0:
                    isolated_chains += 1
                    visited.update(chain_nodes)
        
        # Intelligent execution pattern determination
        # Check for truly independent parallel chains vs dependent sequential chains
        
        # For single node, return single
        if len(agent_nodes) == 1:
            return "single"
        
        # Calculate dependency ratio
        total_possible_connections = len(agent_nodes) * (len(agent_nodes) - 1)
        actual_connections = len(edges)
        dependency_ratio = actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        # If low dependency ratio and multiple starting nodes, allow parallel
        starting_nodes = [node for node in agent_nodes if node["node_id"] not in [edge.get("target") for edge in edges]]
        
        if len(starting_nodes) > 1 and dependency_ratio < 0.3:
            logger.info(f"[EXECUTION PATTERN] Parallel execution for {len(agent_nodes)} agents (dependency_ratio: {dependency_ratio:.2f})")
            return "parallel"
        else:
            logger.info(f"[EXECUTION PATTERN] Sequential execution for {len(agent_nodes)} agents (dependency_ratio: {dependency_ratio:.2f})")
            return "sequential"
    
    def _find_connected_nodes(self, start_node: str, dependencies: Dict, edges: List[Dict]) -> List[str]:
        """Find all nodes connected to start_node (helper for execution pattern analysis)"""
        connected = set()
        to_visit = [start_node]
        
        while to_visit:
            current = to_visit.pop()
            if current in connected:
                continue
            connected.add(current)
            
            # Find nodes connected via edges (both directions)
            for edge in edges:
                source = edge.get("source")
                target = edge.get("target")
                
                if source == current and target not in connected:
                    to_visit.append(target)
                elif target == current and source not in connected:
                    to_visit.append(source)
        
        return list(connected)
    
    async def _execute_agent_plan(
        self,
        agent_plan: Dict[str, Any],
        workflow_id: int,
        execution_id: str,
        execution_trace=None,
        workflow_state: WorkflowState = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute the agent plan using multi-agent system"""
        
        agents = agent_plan["agents"]
        state_nodes = agent_plan.get("state_nodes", [])
        pattern = agent_plan["pattern"]
        query = agent_plan["query"]
        workflow_edges = agent_plan.get("edges", [])
        
        # Process initial state operations before agent execution
        if state_nodes and workflow_state:
            self._process_state_operations(state_nodes, workflow_state, edges=[])
        
        # Prepare agents list for multi-agent system
        selected_agents = []
        for agent_node in agents:
            agent_config = agent_node["agent_config"]
            # Fix: Ensure agent names are properly extracted to avoid scoping issues
            # Use the extracted agent_name from the workflow node as primary source
            final_agent_name = agent_node.get("agent_name") or agent_config["name"]
            
            selected_agents.append({
                "name": final_agent_name,
                "agent_name": final_agent_name,  # Add both keys for compatibility
                "role": agent_config["role"],
                "system_prompt": self._build_system_prompt(agent_node),
                "tools": agent_node["tools"],
                "config": agent_config.get("config", {}),
                "node_id": agent_node["node_id"],
                # Pass through the configured timeout from workflow node
                "configured_timeout": agent_node.get("configured_timeout", 60),
                # Enhanced state management metadata for tracing
                "state_enabled": agent_node.get("state_enabled", False),
                "state_operation": agent_node.get("state_operation", "passthrough"),
                "output_format": agent_node.get("output_format", "text"),
                "chain_key": agent_node.get("chain_key", "")
            })
        
        # Yield agent selection info
        yield {
            "type": "agents_selected",
            "agents": [{"name": agent["name"], "role": agent["role"], "node_id": agent["node_id"]} 
                      for agent in selected_agents],
            "pattern": pattern
        }
        
        # Create agent execution sequence span to group all agent executions
        agent_execution_span = None
        if execution_trace:
            try:
                agent_execution_span = self.tracer.create_span(
                    execution_trace,
                    name="agent-execution-sequence",
                    metadata={
                        "operation": "agent_execution_sequence", 
                        "pattern": pattern,
                        "agent_count": len(selected_agents),
                        "sequence_type": pattern
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to create agent execution span: {e}")
        
        # Execute agents sequentially or in parallel based on pattern
        try:
            if pattern == "parallel":
                # Execute agents in parallel
                async for result in self._execute_agents_parallel(selected_agents, query, workflow_id, execution_id, execution_trace, workflow_state, agent_plan):
                    yield result
            else:
                # Execute agents sequentially (default) with enhanced workflow support
                # CRITICAL FIX: Always pass main execution_trace, not agent_execution_span
                # The DynamicMultiAgentSystem needs the main trace for generations
                
                # Pass agent_plan to sequential execution for enhanced features
                async for result in self._execute_agents_sequential(
                    selected_agents, query, workflow_id, execution_id, execution_trace, 
                    workflow_state, workflow_edges, agent_plan
                ):
                    yield result
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Agent execution failed: {e}")
            yield {
                "type": "execution_error",
                "error": str(e),
                "workflow_id": workflow_id,
                "execution_id": execution_id
            }
        finally:
            # End agent execution span
            if agent_execution_span:
                try:
                    agent_execution_span.end(
                        output={
                            "pattern": pattern,
                            "agents_executed": len(selected_agents),
                            "execution_completed": True
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to end agent execution span: {e}")
    
    def _reorder_agents_by_sequence(self, agents: List[Dict[str, Any]], execution_sequence: List[str]) -> List[Dict[str, Any]]:
        """Reorder agents based on predefined execution sequence from 4-way connectivity workflow"""
        if not execution_sequence:
            return agents
            
        # Create mapping from node_id to agent
        agent_map = {agent['node_id']: agent for agent in agents}
        reordered_agents = []
        
        # Add agents in sequence order if they exist
        for node_id in execution_sequence:
            if node_id in agent_map:
                reordered_agents.append(agent_map[node_id])
                del agent_map[node_id]  # Remove from map to avoid duplicates
        
        # Add any remaining agents not in sequence at the end
        reordered_agents.extend(agent_map.values())
        
        logger.info(f"[AGENT WORKFLOW] Reordered {len(reordered_agents)} agents based on execution sequence")
        return reordered_agents
    
    async def _execute_agents_sequential(
        self,
        agents: List[Dict[str, Any]],
        query: str,
        workflow_id: int,
        execution_id: str,
        execution_trace=None,
        workflow_state: WorkflowState = None,
        workflow_edges: List[Dict[str, Any]] = None,
        agent_plan: Dict[str, Any] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agents sequentially using state-based workflow execution with 4-way connectivity support"""
        
        # Check if workflow has predefined execution sequence from 4-way connectivity
        if agent_plan and agent_plan.get('execution_sequence'):
            execution_sequence = agent_plan.get('execution_sequence', [])
            logger.info(f"[AGENT WORKFLOW] Using predefined execution sequence: {execution_sequence}")
            # Reorder agents based on execution sequence
            agents = self._reorder_agents_by_sequence(agents, execution_sequence)
        
        # STATE-BASED EXECUTION: Each agent receives clean workflow state + previous outputs
        # No context accumulation - agents get structured state objects instead
        agent_outputs = {}
        workflow_edges = workflow_edges or []
        
        for i, agent in enumerate(agents):
            agent_name = agent["name"]
            
            # Update resource monitoring for agent execution
            self.resource_monitor.update_agent_count(workflow_id, execution_id, executed=1, active=1)
            
            # Don't create custom agent spans - let the multi-agent system handle this properly
            # The DynamicMultiAgentSystem will create the correct agent span structure
            agent_execution_span = execution_trace  # Pass the main trace as parent
            
            yield {
                "type": "agent_execution_start",
                "agent_name": agent_name,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "agent_index": i,
                "total_agents": len(agents),
                "sequence_display": f"{i + 1}/{len(agents)}",
                "state_enabled": agent.get("state_enabled", False),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            try:
                # STATE-BASED EXECUTION: Build clean agent input from workflow state with 4-way connectivity support
                enhanced_edges = agent_plan.get('enhanced_edges', []) if agent_plan else []
                node_relationships = agent_plan.get('node_relationships', {}) if agent_plan else {}
                
                agent_input = self._build_agent_state_input(
                    agent=agent,
                    agent_index=i,
                    workflow_state=workflow_state,
                    agent_outputs=agent_outputs,
                    user_query=query,
                    workflow_edges=workflow_edges,
                    enhanced_edges=enhanced_edges,
                    node_relationships=node_relationships
                )
                
                # Create agent prompt using state-based input format
                agent_prompt = self._create_state_based_prompt(
                    agent=agent,
                    agent_input=agent_input,
                    agent_index=i,
                    total_agents=len(agents)
                )
                
                # Execute agent with state-based input
                result = await self._execute_single_agent(
                    agent, agent_prompt, workflow_id, execution_id, 
                    execution_trace, workflow_state, i
                )
                agent_output = result.get("output", "")
                
                # Clean agent output for inter-agent communication (remove thinking tags)
                clean_output = self._clean_output_for_state_passing(agent_output)
                
                # Store agent output in clean format (no truncation needed with state-based approach)
                agent_outputs[agent_name] = agent_output  # Keep original for UI display
                
                # Update workflow state with agent output
                if workflow_state:
                    # Store agent output in structured format
                    agent_result = {
                        "agent_name": agent_name,
                        "node_id": agent["node_id"],
                        "output": clean_output,  # Use cleaned output for state passing
                        "raw_output": agent_output,  # Keep original for UI/debugging
                        "tools_used": result.get("tools_used", []),
                        "timestamp": datetime.utcnow().isoformat(),
                        "index": i
                    }
                    
                    # Store by agent name
                    workflow_state.set_state(f"agent_output_{agent_name}", agent_result, f"agent_{i+1}_result")
                    
                    # Store by node_id for edge-based dependency lookup
                    workflow_state.set_state(f"node_output_{agent['node_id']}", agent_result, f"node_{agent['node_id']}_result")
                    
                    # Update execution summary for next agents
                    execution_summary = workflow_state.get_state("execution_summary") or []
                    execution_summary.append({
                        "agent": agent_name,
                        "completed": True,
                        "index": i
                    })
                    workflow_state.set_state("execution_summary", execution_summary, "progress_tracking")
                
                # Fix: Don't create custom agent spans - let DynamicMultiAgentSystem handle this
                # The main execution trace is sufficient for automation workflows
                # Individual agent spans will be created by the multi-agent system itself
                pass
                
                # Parse the agent prompt to extract structured sections for frontend
                agent_input_sections = self._parse_agent_prompt_sections(agent_prompt)
                
                yield {
                    "type": "agent_execution_complete",
                    "agent_name": agent_name,
                    "input": agent_prompt,  # Provide full prompt for detailed analysis
                    "input_summary": agent_prompt[:200] + "..." if len(agent_prompt) > 200 else agent_prompt,
                    "input_sections": agent_input_sections,  # Structured sections for UI parsing
                    "output": result.get("output", ""),
                    "tools_used": result.get("tools_used", []),
                    "state_enabled": agent.get("state_enabled", False),
                    "state_operation": agent.get("state_operation", "passthrough"),
                    "output_format": agent.get("output_format", "text"),
                    "chain_key": agent.get("chain_key", ""),
                    "chain_data": None,  # State-based execution doesn't use chain data
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "agent_index": i,
                    "total_agents": len(agents),
                    "sequence_display": f"{i + 1}/{len(agents)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"[AGENT WORKFLOW] Agent {agent_name} failed: {e}")
                
                # Fix: Don't create custom agent spans - let DynamicMultiAgentSystem handle this
                # Agent errors will be captured by the multi-agent system's own tracing
                pass
                
                yield {
                    "type": "agent_execution_error",
                    "agent_name": agent_name,
                    "error": str(e),
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "agent_index": i,
                    "total_agents": len(agents),
                    "sequence_display": f"{i + 1}/{len(agents)}"
                }
        
        # Create final synthesis span - SIMPLIFIED
        synthesis_span = None
        if execution_trace:
            try:
                synthesis_span = self.tracer.create_span(
                    execution_trace,
                    name="workflow-synthesis",
                    metadata={
                        "operation": "workflow_synthesis",
                        "agent_count": len(agent_outputs),
                        "total_agents": len(agents),
                        "enhanced_agents_used": len([a for a in agents if a.get("state_enabled", False)])
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to create synthesis span: {e}")
        
        # Yield final response with OutputNode configuration
        output_node = agent_plan.get("output_node") if agent_plan else None
        final_response = self._synthesize_agent_outputs(agent_outputs, query, output_node)
        
        # Fix: Use correct Langfuse API - spans use .end() method
        if synthesis_span:
            try:
                synthesis_span.end(
                    output={
                        "final_response_length": len(final_response),
                        "synthesis_method": "automated" if len(agent_outputs) > 1 else "direct",
                        "agent_outputs_combined": len(agent_outputs)
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to end synthesis span: {e}")
        
        yield {
            "type": "workflow_result",
            "response": final_response,
            "agent_outputs": agent_outputs,
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "output_config": output_node
        }
    
    async def _execute_agents_parallel(
        self,
        agents: List[Dict[str, Any]],
        query: str,
        workflow_id: int,
        execution_id: str,
        execution_trace=None,
        workflow_state: WorkflowState = None,
        agent_plan: Dict[str, Any] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agents in parallel, then combine results"""
        
        # Start all agents
        for i, agent in enumerate(agents):
            yield {
                "type": "agent_execution_start",
                "agent_name": agent["name"],
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "agent_index": i,
                "total_agents": len(agents),
                "sequence_display": f"{i + 1}/{len(agents)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Execute all agents in parallel
        tasks = []
        for i, agent in enumerate(agents):
            # Build comprehensive prompt for parallel execution
            user_message = workflow_state.get_state("user_message") if workflow_state else None
            input_data = workflow_state.get_state("input_data") if workflow_state else None
            
            # Construct user query section
            user_query_section = []
            if user_message:
                user_query_section.append(f"User Message: {user_message}")
            if query and query != user_message:
                user_query_section.append(f"Workflow Query: {query}")
            if input_data:
                user_query_section.append(f"Input Data: {json.dumps(input_data, indent=2)}")
            
            user_query_text = "\n".join(user_query_section) if user_query_section else query
            
            agent_prompt = f"""{agent['system_prompt']}

USER REQUEST:
{user_query_text}

Please process this request independently and provide your analysis."""
            
            task = self._execute_single_agent(agent, agent_prompt, workflow_id, execution_id, execution_trace, workflow_state, i)
            tasks.append((agent["name"], task))
        
        # Collect results as they complete
        agent_outputs = {}
        for task_index, (agent_name, task) in enumerate(tasks):
            try:
                result = await task
                agent_output = result.get("output", "")
                agent_outputs[agent_name] = agent_output
                
                # Store agent output in workflow state
                if workflow_state:
                    workflow_state.set_state(f"agent_output_{agent_name}", agent_output, f"parallel_agent_{agent_name}")
                
                yield {
                    "type": "agent_execution_complete",
                    "agent_name": agent_name,
                    "input": f"Query: {query}",
                    "output": result.get("output", ""),
                    "tools_used": result.get("tools_used", []),
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "agent_index": task_index,
                    "total_agents": len(agents),
                    "sequence_display": f"{task_index + 1}/{len(agents)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"[AGENT WORKFLOW] Parallel agent {agent_name} failed: {e}")
                yield {
                    "type": "agent_execution_error",
                    "agent_name": agent_name,
                    "error": str(e),
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "agent_index": task_index,
                    "total_agents": len(agents),
                    "sequence_display": f"{task_index + 1}/{len(agents)}"
                }
        
        # Yield final response with OutputNode configuration
        output_node = agent_plan.get("output_node") if agent_plan else None
        final_response = self._synthesize_agent_outputs(agent_outputs, query, output_node)
        yield {
            "type": "workflow_result",
            "response": final_response,
            "agent_outputs": agent_outputs,
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "output_config": output_node
        }
    
    async def _execute_single_agent(
        self,
        agent: Dict[str, Any],
        prompt: str,
        workflow_id: int,
        execution_id: str,
        execution_trace=None,
        workflow_state: WorkflowState = None,
        agent_index: int = 0
    ) -> Dict[str, Any]:
        """Execute a single agent using the proven multi-agent system"""
        
        try:
            # Try different possible field names for agent name
            agent_name = (
                agent.get("name", "") or           # Primary field from _execute_agent_plan
                agent.get("agent_name", "") or 
                agent.get("agentName", "") or
                agent.get("node", {}).get("agent_name", "") or
                ""
            )
            
            logger.debug(f"[AGENT WORKFLOW] Extracted agent name: '{agent_name}'")
            
            if not agent_name:
                logger.error(f"[AGENT WORKFLOW] No agent name found in: {list(agent.keys())}")
                raise ValueError(f"Agent name is required. Available keys: {list(agent.keys())}")
                
            logger.info(f"[AGENT WORKFLOW] Executing agent: {agent_name}")
            
            # Use the pre-created DynamicMultiAgentSystem instance
            dynamic_agent_system = self.dynamic_agent_system
            
            # Get the full agent data from cache
            agent_info = get_agent_by_name(agent_name)
            if not agent_info:
                raise ValueError(f"Agent '{agent_name}' not found in cache")
            
            # CRITICAL FIX: Override cached agent config with workflow-specific configuration
            # Extract workflow config from multiple possible locations
            workflow_node = agent.get("node", {})  # Main workflow node config
            agent_config_data = agent.get("agent_config", {})
            workflow_context = agent.get("context", {})
            available_tools = agent.get("tools", []) or workflow_node.get("tools", [])
            
            # Create a copy of agent_info to avoid modifying the cache
            agent_info = agent_info.copy()
            
            # Override tools with workflow config
            if available_tools:
                agent_info["tools"] = available_tools
                logger.info(f"[AGENT WORKFLOW] Overriding agent {agent_name} tools with workflow config: {available_tools}")
            else:
                logger.info(f"[AGENT WORKFLOW] Using default agent {agent_name} tools from cache: {agent_info.get('tools', [])}")
            
            # Override agent config with workflow-specific values
            if "config" not in agent_info:
                agent_info["config"] = {}
            
            # Override model if specified in workflow
            workflow_model = (
                workflow_node.get("model") or
                workflow_context.get("model") or 
                agent_config_data.get("model")
            )
            if workflow_model:
                agent_info["config"]["model"] = workflow_model
                logger.info(f"[AGENT WORKFLOW] Overriding agent {agent_name} model with workflow config: {workflow_model}")
                
            # Override temperature if specified in workflow  
            workflow_temperature = (
                workflow_node.get("temperature") or
                workflow_context.get("temperature") or
                agent_config_data.get("temperature")
            )
            if workflow_temperature is not None:
                agent_info["config"]["temperature"] = workflow_temperature
                logger.info(f"[AGENT WORKFLOW] Overriding agent {agent_name} temperature with workflow config: {workflow_temperature}")
                
            # Override max_tokens if specified in workflow
            workflow_max_tokens = (
                workflow_node.get("max_tokens") or
                workflow_context.get("max_tokens") or
                agent_config_data.get("max_tokens")
            )
            if workflow_max_tokens:
                agent_info["config"]["max_tokens"] = workflow_max_tokens
                logger.info(f"[AGENT WORKFLOW] Overriding agent {agent_name} max_tokens with workflow config: {workflow_max_tokens}")
            
            # Use configured timeout from workflow node, with fallback to reasonable default
            effective_timeout = agent.get("configured_timeout", 60)
            logger.info(f"[AGENT WORKFLOW] Using configured timeout: {effective_timeout}s for agent {agent_name}")
            
            context = {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "agent_config": agent,
                "tools": available_tools,
                "temperature": agent.get("context", {}).get("temperature", 0.7),
                "timeout": effective_timeout  # Pass configured timeout to agent system
            }
            
            # Execute using the dynamic agent system with enhanced tracking
            final_response = None
            tools_used = []
            llm_usage = {}
            model_info = "unknown"
            start_time = datetime.utcnow()
            
            # CRITICAL FIX: Pass the correct parent span to DynamicMultiAgentSystem
            # This should be the agent-execution-sequence span, not the main trace
            async for event in dynamic_agent_system.execute_agent(
                agent_name=agent_name,
                agent_data=agent_info,
                query=prompt,
                context=context,
                parent_trace_or_span=execution_trace  # This is the agent-execution-sequence span now
            ):
                event_type = event.get("type", "")
                
                # Handle events from DynamicMultiAgentSystem
                if event_type == "agent_complete":
                    final_response = event.get("content", "")
                    logger.info(f"[AGENT WORKFLOW] Agent {agent_name} completed with response length: {len(final_response) if final_response else 0}")
                    
                    # CRITICAL FIX: Match multi-agent mode - cleanup MCP subprocesses after each agent
                    try:
                        logger.info(f"[AGENT WORKFLOW] Cleaning up MCP subprocesses after agent {agent_name} (matching multi-agent mode)")
                        
                        # Clean up MCP subprocesses between agents (like multi-agent mode does)
                        from app.core.unified_mcp_service import cleanup_mcp_subprocesses
                        await cleanup_mcp_subprocesses()
                        
                        # Update resource monitoring
                        self.resource_monitor.update_agent_count(workflow_id, execution_id, executed=1)
                        
                        logger.info(f"[AGENT WORKFLOW] Inter-agent cleanup completed for {agent_name}")
                    except Exception as cleanup_error:
                        logger.warning(f"[AGENT WORKFLOW] Inter-agent cleanup failed for {agent_name}: {cleanup_error}")
                    
                    # Extract usage and model information for cost tracking
                    llm_usage = event.get("usage", {})
                    model_info = event.get("model", "unknown")
                    
                elif event_type == "tool_call":
                    tool_call_info = {
                        "tool": event.get("tool", ""),
                        "input": event.get("input", {}),
                        "output": event.get("output", {}),
                        "duration": event.get("duration", 0),
                        "success": event.get("success", True),
                        "name": event.get("tool", "")
                    }
                    tools_used.append(tool_call_info)
                    
                    # REMOVED: Don't create custom tool spans - DynamicMultiAgentSystem handles its own tracing
                    # The DynamicMultiAgentSystem will create proper agent and tool spans within its own trace context
                    pass
                    
                elif event_type == "agent_error":
                    error_msg = event.get("error", "Unknown error")
                    logger.error(f"[AGENT WORKFLOW] Agent {agent_name} error: {error_msg}")
                    final_response = f"Agent execution failed: {error_msg}"
                    
                # Log critical events only
                if event_type in ['error', 'tool_call_failed']:
                    logger.debug(f"[AGENT WORKFLOW] Event: {event_type} - Agent: {event.get('agent', 'unknown')}")
            
            if final_response is None:
                final_response = "No response generated"
            
            # Calculate response time
            response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
            return {
                "output": final_response,
                "tools_used": tools_used,
                "success": True,
                "agent_name": agent_name,
                "llm_usage": llm_usage,  # Include usage for cost tracking
                "model": model_info,  # Include model info for generation tracking
                "response_time_ms": response_time_ms  # Include timing info
            }
            
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Single agent execution failed: {e}")
            logger.debug(f"Agent {agent_name} execution failed: {e}")
            return {
                "output": f"Agent execution failed: {str(e)}",
                "tools_used": [],
                "success": False,
                "error": str(e),
                "agent_name": agent.get("agent_name", "unknown"),
                "llm_usage": {},  # Empty usage for failed executions
                "model": "unknown",
                "response_time_ms": None
            }
    
    def _format_agent_output_for_chaining(
        self, 
        agent_output: str, 
        agent_node: Dict[str, Any], 
        previous_chain_data: Any = None
    ) -> Any:
        """Format agent output for direct chaining to next agent"""
        
        if not agent_node.get("state_enabled", False):
            return agent_output
        
        logger.info(f"[ENHANCED CHAINING] Formatting output for {agent_node['agent_name']} with operation: {agent_node.get('state_operation', 'passthrough')}, format: {agent_node.get('output_format', 'text')}")
        
        output_format = agent_node.get("output_format", "text") 
        state_operation = agent_node.get("state_operation", "passthrough")
        chain_key = agent_node.get("chain_key", "")
        
        # Format the output based on output_format setting
        if output_format == "text":
            formatted_output = agent_output
        elif output_format == "structured":
            formatted_output = {
                "content": agent_output,
                "agent": agent_node["agent_name"],
                "timestamp": datetime.utcnow().isoformat()
            }
        elif output_format == "context":
            formatted_output = {
                "response": agent_output,
                "agent_name": agent_node["agent_name"],
                "context": agent_node.get("context", {}),
                "chain_key": chain_key
            }
        elif output_format == "full":
            formatted_output = {
                "output": agent_output,
                "agent_name": agent_node["agent_name"],
                "tools": agent_node.get("tools", []),
                "context": agent_node.get("context", {}),
                "state_operation": state_operation,
                "chain_key": chain_key
            }
        else:
            formatted_output = agent_output
        
        # Apply state operation if there's previous chain data
        if previous_chain_data is not None and state_operation != "passthrough":
            if state_operation == "merge":
                if isinstance(previous_chain_data, dict) and isinstance(formatted_output, dict):
                    # Merge dictionaries
                    merged_data = previous_chain_data.copy()
                    merged_data.update(formatted_output)
                    return merged_data
                else:
                    # Merge as strings
                    return f"{previous_chain_data}\n\n{formatted_output}"
            
            elif state_operation == "replace":
                return formatted_output
            
            elif state_operation == "append":
                if isinstance(previous_chain_data, list):
                    previous_chain_data.append(formatted_output)
                    return previous_chain_data
                else:
                    return [previous_chain_data, formatted_output]
        
        return formatted_output
    
    def _synthesize_agent_outputs(self, agent_outputs: Dict[str, str], original_query: str, output_node: Optional[Dict] = None) -> str:
        """Return final agent output (synthesizer) with OutputNode configuration formatting"""
        
        if not agent_outputs:
            return "No agent outputs to synthesize."
        
        # Get OutputNode configuration with defaults
        output_config = output_node or {}
        output_format = output_config.get("output_format", "text")
        include_metadata = output_config.get("include_metadata", False)
        include_tool_calls = output_config.get("include_tool_calls", False)
        
        # For single agent, return clean output
        if len(agent_outputs) == 1:
            agent_output = list(agent_outputs.values())[0]
            return self._format_output(agent_output, output_format, include_metadata, include_tool_calls, original_query)
        
        # For multiple agents, return the LAST agent's output (which should be the synthesizer)
        # The last agent in a properly designed workflow should have already synthesized all previous outputs
        final_agent_output = list(agent_outputs.values())[-1]
        
        # Only combine all outputs if explicitly requested via include_metadata
        if include_metadata:
            synthesis = f"Based on analysis from {len(agent_outputs)} AI agents:\n\n"
            for agent_name, output in agent_outputs.items():
                synthesis += f"**{agent_name}:**\n{output}\n\n"
            synthesis += f"**Final Synthesis:**\n{final_agent_output}"
        else:
            # Return only the final synthesizer output (clean, no concatenation)
            synthesis = final_agent_output
        
        return self._format_output(synthesis, output_format, include_metadata, include_tool_calls, original_query)
    
    def _format_output(self, content: str, output_format: str, include_metadata: bool, include_tool_calls: bool, original_query: str) -> str:
        """Format output according to OutputNode configuration"""
        
        # Clean tool calls if not requested
        if not include_tool_calls:
            content = self._remove_tool_calls(content)
        
        # Apply format-specific processing
        if output_format == "markdown":
            return self._format_as_markdown(content)
        elif output_format == "json":
            return self._format_as_json(content, original_query, include_metadata)
        elif output_format == "html":
            return self._format_as_html(content)
        else:  # text format
            return content
    
    def _remove_tool_calls(self, content: str) -> str:
        """Remove tool call information from content"""
        import re
        # Remove common tool call patterns
        patterns = [
            r'\*\*Tools Used.*?\*\*',
            r'Tool\s+called:.*?\n',
            r'Function\s+called:.*?\n',
            r'\[Tool:\s+.*?\]',
            r'Using\s+tool:.*?\n'
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content.strip()
    
    def _format_as_markdown(self, content: str) -> str:
        """Format content as clean markdown"""
        # Ensure proper markdown structure
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Convert **text:** to markdown headers if appropriate
                if line.endswith(':') and line.startswith('**') and line.endswith('**:'):
                    # Convert **Agent Name:** to ## Agent Name
                    header_text = line[2:-3]  # Remove ** and :
                    formatted_lines.append(f"## {header_text}\n")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append('')
        
        return '\n'.join(formatted_lines).strip()
    
    def _format_as_json(self, content: str, query: str, include_metadata: bool) -> str:
        """Format content as JSON structure"""
        import json
        
        result = {
            "content": content,
            "query": query if include_metadata else None
        }
        
        if include_metadata:
            from datetime import datetime
            result["generated_at"] = datetime.now().isoformat()
            result["format"] = "json"
        
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    def _format_as_html(self, content: str) -> str:
        """Format content as HTML"""
        import re
        
        # Convert markdown-style formatting to HTML
        html_content = content
        
        # Convert **text** to <strong>text</strong>
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
        
        # Convert line breaks to <br> and paragraphs
        paragraphs = html_content.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            if para.strip():
                # Replace single line breaks with <br>
                para = para.replace('\n', '<br>')
                formatted_paragraphs.append(f'<p>{para}</p>')
        
        return '\n'.join(formatted_paragraphs)
    
    def _process_state_operations(self, state_nodes: List[Dict], workflow_state: WorkflowState, edges: List[Dict]):
        """Process state node operations in the correct order"""
        if not state_nodes or not workflow_state:
            return
        
        try:
            # Sort state nodes by execution order based on edges
            # For now, process them in the order they appear
            # TODO: Implement proper topological sorting based on edges
            
            for state_node in state_nodes:
                operation = state_node.get("state_operation", "merge")
                state_keys = state_node.get("state_keys", [])
                state_values = state_node.get("state_values", {})
                checkpoint_name = state_node.get("checkpoint_name", "")
                node_id = state_node.get("node_id", "")
                
                logger.info(f"[STATE OPERATION] Processing {operation} for node {node_id}")
                
                if operation == "merge":
                    for key in state_keys:
                        if key in state_values:
                            workflow_state.merge_state(key, state_values[key], checkpoint_name)
                
                elif operation == "set":
                    for key in state_keys:
                        if key in state_values:
                            workflow_state.set_state(key, state_values[key], checkpoint_name)
                
                elif operation == "get":
                    # Get operation doesn't modify state, just logs current values
                    for key in state_keys:
                        current_value = workflow_state.get_state(key)
                        logger.info(f"[STATE GET] {key}: {current_value}")
                
                elif operation == "clear":
                    if state_keys:
                        workflow_state.clear_state(state_keys, checkpoint_name)
                    else:
                        workflow_state.clear_state(None, checkpoint_name)
                
                # Create checkpoint if specified
                if checkpoint_name:
                    workflow_state.create_checkpoint(checkpoint_name)
                    
        except Exception as e:
            logger.error(f"[STATE OPERATION] Error processing state operations: {e}")
    
    async def get_workflow_execution_status(
        self, 
        workflow_id: int, 
        execution_id: str
    ) -> Dict[str, Any]:
        """Get current execution status"""
        try:
            workflow_state = workflow_redis.get_workflow_state(workflow_id, execution_id)
            return workflow_state or {"status": "not_found"}
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Failed to get execution status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cancel_workflow_execution(
        self, 
        workflow_id: int, 
        execution_id: str
    ) -> bool:
        """Cancel running workflow execution"""
        try:
            # Update state to cancelled
            cancel_state = {
                "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat()
            }
            workflow_redis.update_workflow_state(workflow_id, execution_id, cancel_state)
            
            logger.info(f"[AGENT WORKFLOW] Cancelled execution {execution_id} for workflow {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Failed to cancel execution: {e}")
            return False
    
    def _build_agent_state_input(
        self,
        agent: Dict[str, Any],
        agent_index: int,
        workflow_state: WorkflowState,
        agent_outputs: Dict[str, str],
        user_query: str,
        workflow_edges: List[Dict[str, Any]],
        enhanced_edges: List[Dict[str, Any]] = None,
        node_relationships: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build clean state-based input for agent execution"""
        
        # Get initial workflow data
        user_message = workflow_state.get_state("user_message") if workflow_state else None
        input_data = workflow_state.get_state("input_data") if workflow_state else None
        
        # Build agent input object (clean state format)
        agent_input = {
            "user_request": {
                "message": user_message,
                "query": user_query,
                "input_data": input_data
            },
            "workflow_context": {
                "current_agent": {
                    "name": agent["name"],
                    "role": agent["role"],
                    "node_id": agent["node_id"],
                    "index": agent_index
                },
                "total_agents": len(workflow_state.get_state("execution_summary") or []) if workflow_state else 0,
                "execution_id": workflow_state.execution_id if workflow_state else None
            },
            "previous_results": []
        }
        
        # Enhanced dependency resolution with 4-way connectivity support
        if workflow_edges and workflow_state:
            # Use enhanced edges if available (from 4-way connectivity)
            edges_to_use = enhanced_edges or workflow_edges
            
            # Get specific dependencies from workflow edges with enhanced handle information
            dependencies = self._get_agent_dependencies(agent["node_id"], edges_to_use)
            
            # Use node_relationships for enhanced dependency resolution if available
            if node_relationships and agent["node_id"] in node_relationships:
                relationship_info = node_relationships[agent["node_id"]]
                input_connections = relationship_info.get("inputs", [])
                
                logger.info(f"[4-WAY CONNECTIVITY] Agent {agent['name']} has {len(input_connections)} input connections")
                
                for connection in input_connections:
                    source_node_id = connection.get("source_node")
                    source_handle = connection.get("source_handle")
                    target_handle = connection.get("target_handle")
                    
                    # Enhanced dependency tracking with handle information
                    dep_result = workflow_state.get_state(f"node_output_{source_node_id}")
                    if dep_result:
                        # Enhance result with connection metadata
                        enhanced_result = dep_result.copy() if isinstance(dep_result, dict) else {"output": dep_result}
                        enhanced_result["connection_info"] = {
                            "source_handle": source_handle,
                            "target_handle": target_handle,
                            "connection_type": "4-way"
                        }
                        agent_input["previous_results"].append(enhanced_result)
            else:
                # Standard dependency resolution
                for dep_node_id in dependencies:
                    # Find the agent name/index for this dependency node
                    dep_result = workflow_state.get_state(f"node_output_{dep_node_id}")
                    if dep_result:
                        agent_input["previous_results"].append(dep_result)
                    else:
                        # Fallback: try to find by agent name mapping
                        for existing_agent_name in agent_outputs.keys():
                            agent_result = workflow_state.get_state(f"agent_output_{existing_agent_name}")
                            if agent_result and isinstance(agent_result, dict):
                                # Check if this agent result matches the dependency node
                                if agent_result.get("node_id") == dep_node_id:
                                    agent_input["previous_results"].append(agent_result)
                                    break
        elif agent_index > 0 and workflow_state:
            # Fallback to sequential for workflows without explicit edges
            for i in range(agent_index):
                previous_result = workflow_state.get_state(f"agent_output_{list(agent_outputs.keys())[i] if i < len(agent_outputs) else f'agent_{i}'}")
                if previous_result:
                    # Add only structured result, not raw text
                    if isinstance(previous_result, dict):
                        agent_input["previous_results"].append(previous_result)
                    else:
                        # Legacy format - convert to structured
                        agent_input["previous_results"].append({
                            "agent_name": f"agent_{i}",
                            "output": str(previous_result),
                            "index": i
                        })
        
        return agent_input
    
    def _create_state_based_prompt(
        self,
        agent: Dict[str, Any],
        agent_input: Dict[str, Any],
        agent_index: int,
        total_agents: int
    ) -> str:
        """Create agent prompt using state-based input format"""
        
        system_prompt = agent["system_prompt"]
        user_request = agent_input["user_request"]
        workflow_context = agent_input["workflow_context"]
        previous_results = agent_input.get("previous_results", [])
        
        # Build clean, structured prompt
        prompt_parts = [
            f"ROLE: {agent['role']}",
            f"SYSTEM: {system_prompt}",
            "",
            "=== WORKFLOW CONTEXT ===",
            f"Agent Position: {agent_index + 1} of {total_agents}",
            f"Current Agent: {workflow_context['current_agent']['name']}",
            "",
            "=== USER REQUEST ===",
            f"Message: {user_request.get('message', '')}",
            f"Query: {user_request.get('query', '')}",
        ]
        
        if user_request.get('input_data'):
            prompt_parts.extend([
                "Input Data:",
                json.dumps(user_request['input_data'], indent=2)
            ])
        
        # Add dependency-based results (edge-based, not sequential)
        if previous_results:
            prompt_parts.extend([
                "",
                "=== WORKFLOW DEPENDENCIES ===",
                f"This agent receives input from {len(previous_results)} dependency agent(s):",
            ])
            for result in previous_results:  # Show all dependencies, not just last 2
                agent_name = result.get("agent_name", "Unknown Agent")
                node_id = result.get("node_id", "unknown")
                output = result.get("output", "")[:500]  # Limit to 500 chars
                prompt_parts.extend([
                    f"Dependency: {agent_name} (Node: {node_id})",
                    f"Output: {output}",
                    "---"
                ])
        
        # Add task instruction based on position
        prompt_parts.extend([
            "",
            "=== TASK ===",
        ])
        
        if agent_index == 0:
            prompt_parts.append("As the first agent, analyze the user request from your role's perspective. Provide a focused response.")
        elif agent_index == total_agents - 1:
            prompt_parts.append("As the final agent, synthesize the previous results and provide a comprehensive conclusion.")
        else:
            prompt_parts.append("Based on the user request and previous results, provide your analysis from your role's perspective.")
        
        return "\n".join(prompt_parts)
    
    def _get_agent_dependencies(self, node_id: str, workflow_edges: List[Dict[str, Any]]) -> List[str]:
        """Get list of agent node IDs that this agent depends on"""
        dependencies = []
        for edge in workflow_edges:
            if edge.get("target") == node_id:
                dependencies.append(edge.get("source"))
        return dependencies
    
    def _clean_output_for_state_passing(self, agent_output: str) -> str:
        """Clean agent output for inter-agent communication by removing thinking tags and extracting final analysis"""
        if not agent_output:
            return ""
        
        import re
        
        # Remove thinking tags completely for state passing
        clean_content = re.sub(r'<think>.*?</think>', '', agent_output, flags=re.DOTALL)
        clean_content = re.sub(r'</?think>', '', clean_content)
        
        # Clean up extra whitespace
        clean_content = '\n'.join(line.strip() for line in clean_content.split('\n') if line.strip())
        clean_content = clean_content.strip()
        
        # If the cleaned content is too short, try extracting content differently
        if len(clean_content) < 100:
            # Look for thinking content that might contain the actual analysis
            thinking_match = re.search(r'<think>(.*?)</think>', agent_output, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
                # If thinking content is substantial, consider it might be the main content
                if len(thinking_content) > len(clean_content) * 2:
                    clean_content = thinking_content
        
        return clean_content if clean_content else agent_output
    
    def _parse_agent_prompt_sections(self, agent_prompt: str) -> Dict[str, Any]:
        """Parse agent prompt into structured sections for frontend display"""
        if not agent_prompt:
            return {"has_dependencies": False, "role": "", "dependency_results": "", "user_request": ""}
        
        lines = agent_prompt.split('\n')
        role = ""
        dependency_results = ""
        user_request = ""
        in_dependency_section = False
        in_user_request_section = False
        
        for line in lines:
            line_clean = line.strip()
            
            # Extract role
            if line.startswith('ROLE:'):
                role = line.replace('ROLE:', '').strip()
            
            # Detect sections
            if 'WORKFLOW DEPENDENCIES' in line or 'DEPENDENCY RESULTS' in line:
                in_dependency_section = True
                in_user_request_section = False
                continue
            elif 'USER REQUEST' in line or 'ORIGINAL QUERY' in line:
                in_user_request_section = True
                in_dependency_section = False
                continue
            elif line.startswith('TASK:') or line.startswith('INSTRUCTIONS:'):
                in_dependency_section = False
                in_user_request_section = False
                continue
            
            # Capture content
            if in_dependency_section and line_clean:
                dependency_results += line + '\n'
            elif in_user_request_section and line_clean:
                user_request += line + '\n'
        
        return {
            "has_dependencies": bool(dependency_results.strip()),
            "role": role,
            "dependency_results": dependency_results.strip(),
            "user_request": user_request.strip()
        }
    
    def _build_system_prompt(self, agent_node: Dict[str, Any]) -> str:
        """
        Build system prompt with proper priority:
        1. Combine workflow custom_prompt + query 
        2. Fallback to agent database system_prompt
        3. Final fallback to default prompt
        """
        # Build combined workflow prompt from custom_prompt + query
        workflow_prompt_parts = []
        if agent_node.get("custom_prompt"):
            workflow_prompt_parts.append(agent_node["custom_prompt"])
        if agent_node.get("query"):
            workflow_prompt_parts.append(agent_node["query"])
        
        combined_workflow_prompt = "\n\n".join(workflow_prompt_parts) if workflow_prompt_parts else ""
        
        # Apply priority chain: workflow prompt -> agent database -> default
        return (
            combined_workflow_prompt or 
            agent_node.get("agent_config", {}).get("system_prompt") or 
            "You are a helpful assistant."
        )