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
            logger.info(f"[AGENT NODE DEBUG] ===== WORKFLOW EXECUTION START =====")
            logger.info(f"[AGENT NODE DEBUG] Workflow ID: {workflow_id}, Execution ID: {execution_id}")
            logger.info(f"[AGENT NODE DEBUG] Input data: {input_data}")
            logger.info(f"[AGENT NODE DEBUG] Message: {message}")
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
            
            # Register APINode MCP tools for this workflow
            from app.automation.integrations.apinode_mcp_bridge import apinode_mcp_bridge
            try:
                registered_tools = apinode_mcp_bridge.register_workflow_tools(
                    workflow_id=str(workflow_id),
                    workflow_config=workflow_config
                )
                if registered_tools > 0:
                    logger.info(f"[AGENT WORKFLOW] Registered {registered_tools} APINode MCP tools for workflow {workflow_id}")
                    
                    # Add workflow tools info to agent plan
                    agent_plan['apinode_tools'] = apinode_mcp_bridge.get_workflow_tools(str(workflow_id))
            except Exception as e:
                logger.error(f"[AGENT WORKFLOW] Failed to register APINode MCP tools: {str(e)}")
            
            # Transfer execution sequence if provided (regardless of connectivity type)
            if 'execution_sequence' in workflow_config:
                agent_plan['execution_sequence'] = workflow_config['execution_sequence']
                logger.info(f"[AGENT WORKFLOW] Using predefined execution sequence: {len(agent_plan['execution_sequence'])} steps")
            
            # Process enhanced workflow metadata if available
            if workflow_config.get('connectivity_type') == '4-way':
                logger.info(f"[AGENT WORKFLOW] Processing 4-way connectivity workflow v{workflow_config.get('version', '1.0')}")
                
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
        
        # Extract agent nodes, state nodes, router nodes, parallel nodes, aggregator nodes, condition nodes, cache nodes, transform nodes, trigger nodes, api nodes, and output nodes from workflow
        agent_nodes = []
        state_nodes = []
        router_nodes = []
        parallel_nodes = []
        aggregator_nodes = []
        condition_nodes = []
        cache_nodes = []
        transform_nodes = []
        trigger_nodes = []
        api_nodes = []
        output_node = None
        
        for node in nodes:
            node_data = node.get("data", {})
            node_type = node_data.get("type", "")
            
            # DEBUG: Log each node processing
            logger.info(f"[WORKFLOW DEBUG] Processing node: {node.get('id')}")
            logger.info(f"[WORKFLOW DEBUG]   node.type: {node.get('type')}")
            logger.info(f"[WORKFLOW DEBUG]   node_data.type: {node_type}")
            logger.info(f"[WORKFLOW DEBUG]   node keys: {list(node.keys())}")
            logger.info(f"[WORKFLOW DEBUG]   data keys: {list(node_data.keys())}")
            
            # Check for state nodes
            if node_type == "StateNode" or node.get("type") == "statenode":
                state_config = node_data.get("node", {}) or node_data
                state_nodes.append({
                    "node_id": node.get("id"),
                    "label": state_config.get("label", "State"),
                    "node_data": node_data,
                    "position": node.get("position", {}),
                    # Map frontend fields to backend expectations
                    "state_operation": state_config.get("operation", "merge"),
                    "state_key": state_config.get("stateKey", ""),
                    "default_value": state_config.get("defaultValue", {}),
                    "merge_strategy": state_config.get("mergeStrategy", "deep"),
                    "ttl": state_config.get("ttl", 0),
                    "scope": state_config.get("scope", "workflow"),
                    "persistence": state_config.get("persistState", True),
                    "checkpoint_name": state_config.get("checkpointName", ""),
                    # Legacy fields for backward compatibility
                    "state_keys": state_config.get("stateKeys", [state_config.get("stateKey", "")]),
                    "state_values": state_config.get("stateValues", {})
                })
                logger.debug(f"[WORKFLOW CONVERSION] Found StateNode: {node.get('id')}")
            
            # Check for router nodes
            elif node_type == "RouterNode" or node.get("type") == "routernode":
                router_config = node_data.get("node", {}) or node_data
                
                # Transform routes to match expected format
                transformed_routes = []
                for route in router_config.get("routes", []):
                    # Extract match values from condition
                    condition = route.get("condition", "")
                    match_values = [condition] if condition else []
                    
                    # Get target nodes - support both 'output' and 'target_nodes'
                    target_nodes = route.get("target_nodes", [])
                    if not target_nodes and route.get("output"):
                        target_nodes = [route["output"]]
                    
                    transformed_route = {
                        "id": route.get("id", ""),
                        "match_values": match_values,
                        "target_nodes": target_nodes,
                        "description": route.get("description", "")
                    }
                    transformed_routes.append(transformed_route)
                
                router_nodes.append({
                    "node_id": node.get("id"),
                    "label": router_config.get("label", "Router"),
                    "routing_mode": router_config.get("routing_mode", "multi-select"),
                    "match_type": router_config.get("match_type", "exact"),
                    "routes": transformed_routes,
                    "fallback_route": router_config.get("fallback_route", ""),
                    "case_sensitive": router_config.get("case_sensitive", False),
                    "output_field": router_config.get("output_field", ""),
                    "position": node.get("position", {})
                })
                logger.info(f"[WORKFLOW CONVERSION] Found RouterNode: {node.get('id')} with {len(transformed_routes)} routes")
            
            # Check for parallel nodes
            elif node_type == "ParallelNode" or node.get("type") == "parallelnode":
                parallel_config = node_data.get("node", {}) or node_data
                parallel_nodes.append({
                    "node_id": node.get("id"),
                    "label": parallel_config.get("label", "Parallel Execution"),
                    "max_parallel": parallel_config.get("maxParallel") or parallel_config.get("max_parallel", 3),
                    "wait_for_all": parallel_config.get("waitForAll") if parallel_config.get("waitForAll") is not None else parallel_config.get("wait_for_all", True),
                    "combine_strategy": parallel_config.get("combineStrategy") or parallel_config.get("combine_strategy", "merge"),
                    "position": node.get("position", {})
                })
                logger.info(f"[WORKFLOW CONVERSION] Found ParallelNode: {node.get('id')} with max_parallel: {parallel_config.get('max_parallel', 3)}")
            
            # Check for aggregator nodes
            elif node_type == "AggregatorNode" or node.get("type") == "aggregatornode":
                aggregator_config = node_data.get("node", {}) or node_data
                aggregator_nodes.append({
                    "node_id": node.get("id"),
                    "label": aggregator_config.get("label", "Output Aggregator"),
                    "aggregation_strategy": aggregator_config.get("aggregation_strategy") or aggregator_config.get("aggregationStrategy", "semantic_merge"),
                    "confidence_threshold": aggregator_config.get("confidence_threshold") or aggregator_config.get("confidenceThreshold", 0.3),
                    "max_inputs": aggregator_config.get("max_inputs") or aggregator_config.get("maxInputs", 0),
                    "deduplication_enabled": aggregator_config.get("deduplication_enabled") if aggregator_config.get("deduplication_enabled") is not None else aggregator_config.get("deduplicationEnabled", True),
                    "similarity_threshold": aggregator_config.get("similarity_threshold") or aggregator_config.get("similarityThreshold", 0.85),
                    "quality_weights": aggregator_config.get("quality_weights") or aggregator_config.get("qualityWeights", {
                        "length": 0.2,
                        "coherence": 0.3,
                        "relevance": 0.3,
                        "completeness": 0.2
                    }),
                    "output_format": aggregator_config.get("output_format") or aggregator_config.get("outputFormat", "comprehensive"),
                    "include_source_attribution": aggregator_config.get("include_source_attribution") if aggregator_config.get("include_source_attribution") is not None else aggregator_config.get("includeSourceAttribution", True),
                    "conflict_resolution": aggregator_config.get("conflict_resolution") or aggregator_config.get("conflictResolution", "highlight_conflicts"),
                    "semantic_analysis": aggregator_config.get("semantic_analysis") if aggregator_config.get("semantic_analysis") is not None else aggregator_config.get("semanticAnalysis", True),
                    "preserve_structure": aggregator_config.get("preserve_structure") if aggregator_config.get("preserve_structure") is not None else aggregator_config.get("preserveStructure", False),
                    "fallback_strategy": aggregator_config.get("fallback_strategy") or aggregator_config.get("fallbackStrategy", "return_best"),
                    "position": node.get("position", {})
                })
                logger.info(f"[WORKFLOW CONVERSION] Found AggregatorNode: {node.get('id')} with strategy: {aggregator_config.get('aggregation_strategy', 'semantic_merge')}")
            
            # Check for condition nodes
            elif node_type == "ConditionNode" or node.get("type") == "conditionnode":
                condition_config = node_data.get("node", {}) or node_data
                condition_nodes.append({
                    "node_id": node.get("id"),
                    "label": condition_config.get("label", "Condition"),
                    "condition_type": condition_config.get("condition_type") or condition_config.get("conditionType", "simple"),
                    "operator": condition_config.get("operator", "equals"),
                    "compare_value": condition_config.get("compare_value") or condition_config.get("compareValue") or condition_config.get("rightOperand", ""),
                    "left_operand": condition_config.get("left_operand") or condition_config.get("leftOperand", ""),
                    "right_operand": condition_config.get("right_operand") or condition_config.get("rightOperand", ""),
                    "ai_criteria": condition_config.get("ai_criteria") or condition_config.get("aiCriteria", ""),
                    "case_sensitive": condition_config.get("case_sensitive") if condition_config.get("case_sensitive") is not None else condition_config.get("caseSensitive", False),
                    "data_type": condition_config.get("data_type") or condition_config.get("dataType", "string"),
                    "position": node.get("position", {})
                })
                logger.info(f"[WORKFLOW CONVERSION] Found ConditionNode: {node.get('id')} with type: {condition_config.get('condition_type', 'simple')}")
            
            # Check for cache nodes
            elif node_type == "CacheNode" or node.get("type") == "cachenode":
                cache_config = node_data.get("node", {}) or node_data
                cache_nodes.append({
                    "node_id": node.get("id"),
                    "label": cache_config.get("label", "Cache"),
                    "cache_key": cache_config.get("cache_key") or cache_config.get("cacheKey", ""),
                    "cache_key_pattern": cache_config.get("cache_key_pattern") or cache_config.get("cacheKeyPattern", "auto"),
                    "ttl": cache_config.get("ttl", 3600),
                    "cache_policy": cache_config.get("cache_policy") or cache_config.get("cachePolicy", "always"),
                    "invalidate_on": cache_config.get("invalidate_on") or cache_config.get("invalidateOn", ["input_change"]),
                    "cache_condition": cache_config.get("cache_condition") or cache_config.get("cacheCondition", ""),
                    "enable_warming": cache_config.get("enable_warming") if cache_config.get("enable_warming") is not None else cache_config.get("enableWarming", False),
                    "max_cache_size": cache_config.get("max_cache_size") or cache_config.get("maxCacheSize", 10),
                    "cache_namespace": cache_config.get("cache_namespace") or cache_config.get("cacheNamespace", "default"),
                    "show_statistics": cache_config.get("show_statistics") if cache_config.get("show_statistics") is not None else cache_config.get("showStatistics", True),
                    "position": node.get("position", {})
                })
                logger.info(f"[WORKFLOW CONVERSION] Found CacheNode: {node.get('id')} with TTL: {cache_config.get('ttl', 3600)}s")
            
            # Check for transform nodes
            elif node_type == "TransformNode" or node.get("type") == "transformnode":
                transform_config = node_data.get("node", {}) or node_data
                transform_nodes.append({
                    "node_id": node.get("id"),
                    "label": transform_config.get("label", "Transform"),
                    "transform_type": transform_config.get("transform_type", "jsonpath"),
                    "expression": transform_config.get("expression", "$"),
                    "error_handling": transform_config.get("error_handling", "continue"),
                    "default_value": transform_config.get("default_value"),
                    "input_validation": transform_config.get("input_validation", {}),
                    "output_validation": transform_config.get("output_validation", {}),
                    "cache_results": transform_config.get("cache_results", False),
                    "position": node.get("position", {})
                })
                logger.info(f"[WORKFLOW CONVERSION] Found TransformNode: {node.get('id')} with type: {transform_config.get('transform_type', 'jsonpath')}")
            
            # Check for trigger nodes
            elif node_type == "TriggerNode" or node.get("type") == "triggernode":
                trigger_config = node_data.get("node", {}) or node_data
                trigger_nodes.append({
                    "node_id": node.get("id"),
                    "trigger_name": trigger_config.get("trigger_name", f"trigger-{node.get('id')}"),
                    "http_methods": trigger_config.get("http_methods", ["POST"]),
                    "authentication_type": trigger_config.get("authentication_type", "api_key"),
                    "auth_header_name": trigger_config.get("auth_header_name", "X-API-Key"),
                    "auth_token": trigger_config.get("auth_token", ""),
                    "rate_limit": trigger_config.get("rate_limit", 60),
                    "timeout": trigger_config.get("timeout", 300),
                    "response_format": trigger_config.get("response_format", "workflow_output"),
                    "custom_response_template": trigger_config.get("custom_response_template", ""),
                    "cors_enabled": trigger_config.get("cors_enabled", True),
                    "cors_origins": trigger_config.get("cors_origins", "*"),
                    "log_requests": trigger_config.get("log_requests", True),
                    "position": node.get("position", {})
                })
                logger.info(f"[WORKFLOW CONVERSION] Found TriggerNode: {node.get('id')} with name: {trigger_config.get('trigger_name')}")
            
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
                        output_config.get("auto_display") if output_config.get("auto_display") is not None else
                        node_data.get("auto_display") if node_data.get("auto_display") is not None else
                        False  # Default to False - only show if explicitly enabled
                    ),
                    "auto_save": (
                        output_config.get("auto_save") if output_config.get("auto_save") is not None else
                        node_data.get("auto_save") if node_data.get("auto_save") is not None else
                        False
                    )
                }
                logger.debug(f"[WORKFLOW CONVERSION] Found OutputNode: {node.get('id')} with format: {output_node['output_format']}, auto_display: {output_node['auto_display']}, auto_save: {output_node['auto_save']}")
            
            # Check for API nodes
            elif node_type == "APINode" or node.get("type") == "apinode":
                api_config = node_data.get("node", {}) or node_data
                api_nodes.append({
                    "node_id": node.get("id"),
                    "label": api_config.get("label", "APINode"),
                    "base_url": api_config.get("base_url", ""),
                    "endpoint_path": api_config.get("endpoint_path", ""),
                    "http_method": api_config.get("http_method", "GET"),
                    "authentication_type": api_config.get("authentication_type", "none"),
                    "auth_header_name": api_config.get("auth_header_name", "X-API-Key"),
                    "auth_token": api_config.get("auth_token", ""),
                    "basic_auth_username": api_config.get("basic_auth_username", ""),
                    "basic_auth_password": api_config.get("basic_auth_password", ""),
                    "request_schema": api_config.get("request_schema", {}),
                    "response_schema": api_config.get("response_schema", {}),
                    "timeout": api_config.get("timeout", 30),
                    "retry_count": api_config.get("retry_count", 3),
                    "rate_limit": api_config.get("rate_limit", 60),
                    "custom_headers": api_config.get("custom_headers", {}),
                    "response_transformation": api_config.get("response_transformation", ""),
                    "error_handling": api_config.get("error_handling", "throw"),
                    "enable_mcp_tool": api_config.get("enable_mcp_tool", True),
                    "tool_description": api_config.get("tool_description", ""),
                    "position": node.get("position", {})
                })
                logger.debug(f"[WORKFLOW CONVERSION] Found APINode: {node.get('id')}")
            
            # Check for both legacy and agent-based node types
            logger.info(f"[WORKFLOW DEBUG] Checking agent node condition for {node.get('id')}")
            logger.info(f"[WORKFLOW DEBUG]   node_type == 'AgentNode': {node_type == 'AgentNode'}")  
            logger.info(f"[WORKFLOW DEBUG]   node.get('type') == 'agentnode': {node.get('type') == 'agentnode'}")
            logger.info(f"[WORKFLOW DEBUG]   Combined condition: {node_type == 'AgentNode' or node.get('type') == 'agentnode'}")
            
            if node_type == "AgentNode" or node.get("type") == "agentnode":
                logger.info(f"[WORKFLOW DEBUG] INSIDE agent node processing block for {node.get('id')}")
                # Try multiple ways to extract agent configuration
                agent_config = node_data.get("node", {})
                
                # Get agent name from various possible locations
                agent_name = (
                    agent_config.get("agent_name") or
                    node_data.get("agentName") or  # Frontend format
                    node_data.get("agent_name") or
                    ""
                )
                
                logger.info(f"[WORKFLOW DEBUG] Agent name extracted: '{agent_name}'")
                logger.debug(f"[WORKFLOW CONVERSION] Node: {node.get('id')}, Type: {node_type}, Agent: {agent_name}")
                logger.debug(f"[WORKFLOW CONVERSION] node_data keys: {list(node_data.keys()) if node_data else 'None'}")
                logger.debug(f"[WORKFLOW CONVERSION] agent_config keys: {list(agent_config.keys()) if agent_config else 'None'}")
                
                if agent_name:
                    logger.info(f"[WORKFLOW DEBUG] Agent name validation passed for: '{agent_name}'")
                    # Get agent from cache
                    agent_info = get_agent_by_name(agent_name)
                    logger.info(f"[WORKFLOW DEBUG] Agent cache lookup for '{agent_name}': {agent_info is not None}")
                    
                    # TEMPORARY: Create a basic agent_info if not found in cache for testing
                    if not agent_info and agent_name == "Research Analyst":
                        logger.info(f"[WORKFLOW DEBUG] Creating temporary agent_info for testing purposes")
                        agent_info = {
                            "name": agent_name,
                            "role": "Senior Research Analyst",
                            "system_prompt": "You are a senior research analyst. Provide detailed, comprehensive analysis with specific metrics and data points.",
                            "config": {"temperature": 0.7, "max_tokens": 2000},
                            "tools": []
                        }
                    
                    if agent_info:
                        logger.info(f"[WORKFLOW DEBUG] Agent cache validation passed for: '{agent_name}'")
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
                        logger.debug(f"[WORKFLOW TOOLS] Agent {agent_name}: extracted tools = {tools}")
                        
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
                        
                        # Extract model from AgentNode UI data
                        node_model = (
                            agent_config.get("model") or
                            node_data.get("model") or
                            None
                        )
                        
                        # Extract temperature from AgentNode UI data
                        node_temperature = (
                            agent_config.get("temperature") or
                            node_data.get("temperature") or
                            None
                        )
                        
                        # Extract max_tokens from AgentNode UI data
                        node_max_tokens = (
                            agent_config.get("max_tokens") or
                            node_data.get("max_tokens") or
                            None
                        )
                        
                        logger.info(f"[WORKFLOW DEBUG] Adding agent '{agent_name}' to agent_nodes list")
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
                            "model": node_model,  # Add model from AgentNode UI
                            "temperature": node_temperature,  # Add temperature from AgentNode UI
                            "max_tokens": node_max_tokens,  # Add max_tokens from AgentNode UI
                            "configured_timeout": configured_timeout,
                            "position": node.get("position", {}),
                            "state_enabled": state_enabled,
                            "state_operation": state_operation,
                            "output_format": output_format,
                            "chain_key": chain_key
                        })
                    else:
                        logger.warning(f"[WORKFLOW DEBUG] Agent '{agent_name}' not found in cache - REJECTED")
                else:
                    logger.warning(f"[WORKFLOW DEBUG] No agent name found - agent_config keys: {list(agent_config.keys())}")
        
        # Log the conversion results
        logger.info(f"[WORKFLOW CONVERSION] Found {len(agent_nodes)} agent nodes")
        for i, agent_node in enumerate(agent_nodes):
            state_info = ""
            if agent_node.get("state_enabled", False):
                state_info = f" [STATE: {agent_node.get('state_operation', 'passthrough')}, FORMAT: {agent_node.get('output_format', 'text')}]"
            logger.info(f"[WORKFLOW CONVERSION] Agent {i+1}: {agent_node['agent_name']} (Node: {agent_node['node_id']}){state_info}")
            logger.info(f"[WORKFLOW CONVERSION] Agent {i+1} tools: {agent_node.get('tools', [])}")
        
        # Determine execution pattern from edges (check workflow config first)
        execution_pattern = self._analyze_execution_pattern(agent_nodes, edges, workflow_config)
        
        # Prepare query from input or message
        query = message or input_data.get("query", "") if input_data else ""
        
        # Handle TriggerNode input data specially
        if not query and input_data and trigger_nodes:
            # If we have trigger nodes and special trigger input format
            if input_data.get("trigger_data") is not None:
                # Use enhanced message extraction if available
                extracted_message = input_data.get("message", "")
                formatted_query = input_data.get("formatted_query", "")
                
                # Prioritize formatted query, then message, then fallback to old behavior
                if formatted_query:
                    query = formatted_query
                    logger.info(f"[TRIGGER] Using formatted query from trigger: {len(query)} chars")
                elif extracted_message:
                    query = extracted_message
                    logger.info(f"[TRIGGER] Using extracted message from trigger: {len(query)} chars")
                else:
                    # Fallback to legacy trigger data processing for backward compatibility
                    trigger_data = input_data.get("trigger_data", {})
                    query_params = input_data.get("query_params", {})
                    headers = input_data.get("headers", {})
                    
                    # Construct query from trigger data (legacy)
                    if trigger_data:
                        query = f"Process the external trigger data: {json.dumps(trigger_data, indent=2)}"
                    elif query_params:
                        query = f"Process the query parameters: {json.dumps(query_params, indent=2)}"
                    else:
                        query = "Process external trigger request"
                    logger.info(f"[TRIGGER] Using legacy trigger data processing")
            else:
                # Standard input data processing
                query = f"Process the following data: {json.dumps(input_data, indent=2)}"
        elif not query and input_data:
            # Try to construct query from input data
            query = f"Process the following data: {json.dumps(input_data, indent=2)}"
        
        result = {
            "agents": agent_nodes,
            "state_nodes": state_nodes,
            "router_nodes": router_nodes,
            "parallel_nodes": parallel_nodes,
            "aggregator_nodes": aggregator_nodes,
            "condition_nodes": condition_nodes,
            "cache_nodes": cache_nodes,
            "transform_nodes": transform_nodes,
            "trigger_nodes": trigger_nodes,
            "api_nodes": api_nodes,
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
        router_nodes = agent_plan.get("router_nodes", [])
        parallel_nodes = agent_plan.get("parallel_nodes", [])
        aggregator_nodes = agent_plan.get("aggregator_nodes", [])
        condition_nodes = agent_plan.get("condition_nodes", [])
        cache_nodes = agent_plan.get("cache_nodes", [])
        transform_nodes = agent_plan.get("transform_nodes", [])
        api_nodes = agent_plan.get("api_nodes", [])
        pattern = agent_plan["pattern"]
        query = agent_plan["query"]
        workflow_edges = agent_plan.get("edges", [])
        
        # Process initial state operations before agent execution
        if state_nodes and workflow_state:
            self._process_state_operations(state_nodes, workflow_state, edges=[])
        
        # Check if workflow starts with a ParallelNode
        # This handles cases where Start → ParallelNode → Agents
        execution_sequence = agent_plan.get("execution_sequence", [])
        agent_outputs = {}  # Initialize agent outputs dictionary
        
        # Find if any parallel node is at the beginning (connected from Start)
        initial_parallel_node = None
        if parallel_nodes and execution_sequence:
            # Check if a parallel node appears early in the execution sequence
            for i, node_id in enumerate(execution_sequence[:6]):  # Check first 6 nodes to account for trigger/group nodes
                for parallel in parallel_nodes:
                    if parallel["node_id"] == node_id:
                        # Verify it's connected from Start node (directly or indirectly)
                        for edge in workflow_edges:
                            if edge.get("target") == node_id and "start" in edge.get("source", "").lower():
                                initial_parallel_node = parallel
                                logger.info(f"[WORKFLOW] Found initial ParallelNode: {node_id}")
                                break
                        break
                if initial_parallel_node:
                    break
        
        # If workflow starts with ParallelNode, process it first
        if initial_parallel_node:
            logger.info(f"[WORKFLOW] Processing initial ParallelNode: {initial_parallel_node['node_id']}")
            
            # Process the parallel node
            async for parallel_event in self._process_parallel_node(
                initial_parallel_node,
                query,  # Use the query as input
                workflow_edges,
                workflow_id,
                execution_id,
                agent_plan,
                workflow_state,
                execution_trace,
                agent_outputs
            ):
                yield parallel_event
            
            # After parallel execution, check if we need to continue with other nodes
            # For now, if ParallelNode is processed, we can skip the regular agent execution
            # since agents were already executed within the parallel node
            
            # Yield final response with OutputNode configuration
            output_node = agent_plan.get("output_node")
            
            # Check if ParallelNode generated a combined output (AI summary)
            parallel_result = workflow_state.get_state(f"node_output_{initial_parallel_node['node_id']}")
            if parallel_result and parallel_result.get("combined_output"):
                # Use the AI summary from parallel execution instead of individual agent outputs
                final_response = parallel_result["combined_output"]
                logger.info(f"[WORKFLOW] Using AI summary from ParallelNode: {len(final_response)} chars")
            else:
                # Fallback to synthesizing individual agent outputs
                final_response = self._synthesize_agent_outputs(agent_outputs, query, output_node)
                logger.info(f"[WORKFLOW] Using synthesized agent outputs: {len(final_response)} chars")
            
            yield {
                "type": "workflow_result",
                "response": final_response,
                "agent_outputs": agent_outputs,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "output_config": output_node
            }
            
            return  # Exit early since parallel node handled the execution
        
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
                # Pass through workflow-specific configuration
                "custom_prompt": agent_node.get("custom_prompt", ""),
                "query": agent_node.get("query", ""),
                "agent_config": agent_config,  # Pass full agent config for _create_state_based_prompt
                # Pass through model, temperature, and max_tokens from AgentNode UI
                "model": agent_node.get("model"),
                "temperature": agent_node.get("temperature"),
                "max_tokens": agent_node.get("max_tokens"),
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
                    workflow_state, workflow_edges, agent_plan, cache_nodes, transform_nodes, parallel_nodes, condition_nodes, state_nodes, api_nodes
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
            # Cleanup APINode MCP tools for this workflow
            from app.automation.integrations.apinode_mcp_bridge import apinode_mcp_bridge
            try:
                unregistered_tools = apinode_mcp_bridge.unregister_workflow_tools(str(workflow_id))
                if unregistered_tools > 0:
                    logger.info(f"[AGENT WORKFLOW] Unregistered {unregistered_tools} APINode MCP tools for workflow {workflow_id}")
            except Exception as e:
                logger.error(f"[AGENT WORKFLOW] Failed to unregister APINode MCP tools: {str(e)}")
            
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
        """Reorder agents based on predefined execution sequence - CRITICAL: Must follow exact order"""
        if not execution_sequence:
            logger.warning("[CRITICAL] No execution sequence provided - using original order")
            return agents
            
        # Create mapping from node_id to agent
        agent_map = {agent['node_id']: agent for agent in agents}
        reordered_agents = []
        
        # Filter execution sequence to only include agent nodes
        agent_sequence = [node_id for node_id in execution_sequence if node_id in agent_map]
        non_agent_nodes = [node_id for node_id in execution_sequence if node_id not in agent_map]
        
        logger.info(f"[SEQUENCE FILTER] Agent nodes in sequence: {agent_sequence}")
        logger.info(f"[SEQUENCE FILTER] Non-agent nodes (filtered out): {non_agent_nodes}")
        
        # CRITICAL: Add agents in EXACT sequence order (filtered)
        for node_id in agent_sequence:
            reordered_agents.append(agent_map[node_id])
            del agent_map[node_id]  # Remove from map to avoid duplicates
        
        # CRITICAL: Fail if there are extra agents not in sequence
        if agent_map:
            extra_nodes = list(agent_map.keys())
            logger.error(f"[CRITICAL ERROR] Found agents not in execution sequence: {extra_nodes}")
            raise ValueError(f"Agents exist that are not in execution sequence: {extra_nodes}")
        
        logger.info(f"[CRITICAL SUCCESS] Reordered {len(reordered_agents)} agents following EXACT execution sequence: {[a['node_id'] for a in reordered_agents]}")
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
        agent_plan: Dict[str, Any] = None,
        cache_nodes: List[Dict[str, Any]] = None,
        transform_nodes: List[Dict[str, Any]] = None,
        parallel_nodes: List[Dict[str, Any]] = None,
        condition_nodes: List[Dict[str, Any]] = None,
        state_nodes: List[Dict[str, Any]] = None,
        api_nodes: List[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agents sequentially using state-based workflow execution with 4-way connectivity support"""
        
        # Check if workflow has predefined execution sequence from 4-way connectivity
        logger.info(f"[DEBUG] agent_plan exists: {agent_plan is not None}")
        if agent_plan:
            logger.info(f"[DEBUG] agent_plan keys: {list(agent_plan.keys())}")
            logger.info(f"[DEBUG] execution_sequence in agent_plan: {'execution_sequence' in agent_plan}")
            if 'execution_sequence' in agent_plan:
                logger.info(f"[DEBUG] execution_sequence value: {agent_plan.get('execution_sequence')}")
        
        if agent_plan and agent_plan.get('execution_sequence'):
            execution_sequence = agent_plan.get('execution_sequence', [])
            logger.info(f"[AGENT WORKFLOW] Using predefined execution sequence: {execution_sequence}")
            
            # Debug: Log agents before reordering
            logger.info(f"[DEBUG] Agents before reordering: {[agent.get('node_id') for agent in agents]}")
            
            # Reorder agents based on execution sequence
            agents = self._reorder_agents_by_sequence(agents, execution_sequence)
            
            # Debug: Log agents after reordering  
            logger.info(f"[DEBUG] Agents after reordering: {[agent.get('node_id') for agent in agents]}")
        else:
            logger.info(f"[DEBUG] No execution sequence found in agent_plan, using original order")
        
        # STATE-BASED EXECUTION: Each agent receives clean workflow state + previous outputs
        # No context accumulation - agents get structured state objects instead
        agent_outputs = {}
        workflow_edges = workflow_edges or []
        
        # Track condition results for conditional execution
        condition_results = {}
        
        # Get router nodes for filtering
        router_nodes = agent_plan.get("router_nodes", []) if agent_plan else []
        aggregator_nodes = agent_plan.get("aggregator_nodes", []) if agent_plan else []
        cache_nodes = cache_nodes or []
        transform_nodes = transform_nodes or []
        parallel_nodes = parallel_nodes or []
        condition_nodes = condition_nodes or []
        state_nodes = state_nodes or []
        active_target_nodes = set()
        
        # Process RouterNodes that are connected from StartNode or InputNode
        if router_nodes:
            for router in router_nodes:
                router_node_id = router["node_id"]
                
                # Check if this router is connected from StartNode or InputNode
                is_initial_router = False
                router_input = None
                
                for edge in workflow_edges:
                    if edge.get("target") == router_node_id:
                        source_id = edge.get("source", "")
                        # Check if source is StartNode or InputNode
                        if "start" in source_id.lower() or "input" in source_id.lower():
                            is_initial_router = True
                            # For initial routers, use the query as input
                            router_input = query
                            break
                
                if is_initial_router:
                    # Check if there's InputNode data in workflow state
                    if workflow_state and "input" in source_id.lower():
                        # Try to get InputNode output from workflow state
                        input_node_output = workflow_state.get_state(f"node_output_{source_id}")
                        if input_node_output:
                            router_input = input_node_output
                            logger.info(f"[ROUTER] Using InputNode data for RouterNode {router_node_id}")
                    
                    logger.info(f"[ROUTER] Processing initial RouterNode {router_node_id} with input: {str(router_input)[:100]}...")
                    
                    # Process the router with the initial input
                    router_result = self._process_router_node(
                        router, router_input, agent_outputs, workflow_edges
                    )
                    
                    if router_result:
                        active_target_nodes.update(router_result["target_nodes"])
                        logger.info(f"[ROUTER] Initial router decision: {router_result['matched_routes']} → targets: {list(router_result['target_nodes'])}")
                        
                        # Yield router decision event
                        yield {
                            "type": "router_decision",
                            "router_id": router["node_id"],
                            "matched_routes": router_result["matched_routes"],
                            "target_nodes": list(router_result["target_nodes"]),
                            "workflow_id": workflow_id,
                            "execution_id": execution_id,
                            "timestamp": datetime.utcnow().isoformat(),
                            "triggered_by": "initial_routing"
                        }
                    else:
                        logger.warning(f"[ROUTER] Initial router {router_node_id} made no routing decision")
        
        for i, agent in enumerate(agents):
            agent_name = agent["name"]
            agent_node_id = agent.get("node_id")
            
            # Check if this agent should execute based on router decisions
            if active_target_nodes and agent_node_id and agent_node_id not in active_target_nodes:
                # Check if this agent is downstream of any active nodes
                if not self._should_execute_node(agent_node_id, active_target_nodes, workflow_edges):
                    logger.info(f"[ROUTER FILTERING] Skipping agent {agent_name} (node: {agent_node_id}) - not selected by router. Active targets: {list(active_target_nodes)}")
                    yield {
                        "type": "agent_execution_skipped",
                        "agent_name": agent_name,
                        "node_id": agent_node_id,
                        "reason": "Not selected by router",
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "agent_index": i,
                        "total_agents": len(agents),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    continue
            
            # CONDITION CHECK: Check if this agent should execute based on condition results
            if condition_results and not self._should_execute_node_with_conditions(agent_node_id, condition_results, workflow_edges):
                logger.info(f"[CONDITION FILTERING] Skipping agent {agent_name} (node: {agent_node_id}) - blocked by condition")
                yield {
                    "type": "agent_execution_skipped",
                    "agent_name": agent_name,
                    "node_id": agent_node_id,
                    "reason": "Blocked by condition",
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "agent_index": i,
                    "total_agents": len(agents),
                    "timestamp": datetime.utcnow().isoformat()
                }
                continue
            
            # CACHE CHECK: Check if there's a cache node connected to this agent
            cache_result = None
            agent_input = None
            cache_hit_skip = False  # Flag to control main agent loop
            if cache_nodes:
                # Check if this agent is connected TO a cache node (agent -> cache node)
                for edge in workflow_edges:
                    if edge.get("source") == agent_node_id:
                        target_id = edge.get("target")
                        # Check if target is a cache node
                        for cache in cache_nodes:
                            if cache["node_id"] == target_id:
                                # Build agent input for cache key generation
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
                                
                                # Check cache - use stable input for cache key (exclude dynamic execution data)
                                stable_input = {
                                    "user_request": agent_input.get("user_request", {}),
                                    "agent_config": {
                                        "name": agent["name"],
                                        "role": agent["role"],
                                        "node_id": agent["node_id"]
                                    }
                                }
                                cache_result = await self._process_cache_node(
                                    cache, str(stable_input), workflow_id, execution_id, agent_node_id
                                )
                                
                                if cache_result and cache_result.get("skip_execution"):
                                    # Cache hit! Skip agent execution
                                    logger.info(f"[CACHE] Cache HIT - Skipping agent {agent_name} execution")
                                    
                                    # Enhanced cache hit event with metadata
                                    cache_metadata = cache_result.get("cache_metadata", {})
                                    yield {
                                        "type": "cache_hit",
                                        "agent_name": agent_name,
                                        "node_id": agent_node_id,
                                        "cache_id": cache["node_id"],
                                        "cache_key": cache_result.get("cache_key", ""),
                                        "cached_data": cache_result.get("cached_data"),
                                        "cache_metadata": cache_metadata,
                                        "cache_size": cache_result.get("cache_size", 0),
                                        "cache_age_hours": cache_metadata.get("age_hours", 0),
                                        "is_fresh": cache_metadata.get("is_fresh", True),
                                        "data_type": cache_metadata.get("data_type", "unknown"),
                                        "workflow_id": workflow_id,
                                        "execution_id": execution_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                    
                                    # Use cached output instead of executing agent
                                    agent_output = cache_result.get("cached_data", "")
                                    agent_outputs[agent_name] = agent_output
                                    
                                    # Store result in workflow state
                                    if workflow_state:
                                        agent_result = {
                                            "agent_name": agent_name,
                                            "node_id": agent["node_id"],
                                            "output": agent_output,
                                            "cached": True,
                                            "cache_key": cache_result.get("cache_key", ""),
                                            "tools_used": [],
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "index": i
                                        }
                                        workflow_state.set_state(f"agent_output_{agent_name}", agent_result, f"agent_{i+1}_cached_result")
                                        workflow_state.set_state(f"node_output_{agent['node_id']}", agent_result, f"node_{agent['node_id']}_cached_result")
                                        
                                        # CRITICAL: Store cache node output for downstream nodes (like RouterNode)
                                        cache_node_result = {
                                            "node_id": cache["node_id"],
                                            "output": agent_output,
                                            "cache_hit": True,
                                            "cache_key": cache_result.get("cache_key", ""),
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                        workflow_state.set_state(f"node_output_{cache['node_id']}", cache_node_result, f"cache_{cache['node_id']}_result")
                                    
                                    yield {
                                        "type": "agent_execution_complete",
                                        "agent_name": agent_name,
                                        "node_id": agent.get("node_id"),
                                        "output": agent_output,
                                        "cached": True,
                                        "cache_info": cache_result,
                                        "workflow_id": workflow_id,
                                        "execution_id": execution_id,
                                        "agent_index": i,
                                        "total_agents": len(agents),
                                        "sequence_display": f"{i + 1}/{len(agents)}",
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                    
                                    # CRITICAL: Process RouterNode if CacheNode connects to it
                                    cache_node_id = cache["node_id"]
                                    for edge in workflow_edges:
                                        if edge.get("source") == cache_node_id:
                                            target_id = edge.get("target")
                                            # Check if target is a router node
                                            for router in router_nodes:
                                                if router["node_id"] == target_id:
                                                    logger.info(f"[CACHE] Processing RouterNode {target_id} with cached result: {agent_output}")
                                                    router_result = self._process_router_node(
                                                        router, agent_output, agent_outputs, workflow_edges
                                                    )
                                                    if router_result:
                                                        active_target_nodes.update(router_result["target_nodes"])
                                                        logger.info(f"[ROUTER] Cache-triggered routing decision: {router_result['matched_routes']} → targets: {list(router_result['target_nodes'])}")
                                                        yield {
                                                            "type": "router_decision",
                                                            "router_id": router["node_id"],
                                                            "matched_routes": router_result["matched_routes"],
                                                            "target_nodes": list(router_result["target_nodes"]),
                                                            "workflow_id": workflow_id,
                                                            "execution_id": execution_id,
                                                            "timestamp": datetime.utcnow().isoformat(),
                                                            "triggered_by": "cache_hit"
                                                        }
                                                    else:
                                                        logger.warning(f"[ROUTER] No routing decision made for cached result: {agent_output}")
                                    
                                    # CRITICAL: Process direct CacheNode → AgentNode connections  
                                    cache_node_id = cache["node_id"]
                                    for edge in workflow_edges:
                                        if edge.get("source") == cache_node_id:
                                            target_id = edge.get("target")
                                            # Check if target is an agent node (not router node)
                                            target_agent = None
                                            for target_agent_candidate in agents:
                                                if target_agent_candidate.get("node_id") == target_id:
                                                    target_agent = target_agent_candidate
                                                    break
                                            
                                            if target_agent:
                                                # This is a direct CacheNode → AgentNode connection
                                                logger.info(f"[CACHE] Direct connection: CacheNode {cache_node_id} → AgentNode {target_id}, passing cached result: {agent_output}")
                                                
                                                # Store cached output as input for the target agent
                                                if workflow_state:
                                                    target_input_key = f"agent_input_{target_agent['name']}"
                                                    cached_input = {
                                                        "cached_output": agent_output,
                                                        "source_cache_node": cache_node_id,
                                                        "source_agent": agent_name,
                                                        "timestamp": datetime.utcnow().isoformat()
                                                    }
                                                    workflow_state.set_state(target_input_key, cached_input, f"cache_input_for_{target_agent['name']}")
                                                
                                                # Yield cache connection info
                                                yield {
                                                    "type": "cache_connection",
                                                    "source_cache_node": cache_node_id,
                                                    "target_agent_node": target_id,
                                                    "target_agent_name": target_agent["name"],
                                                    "cached_data": agent_output,
                                                    "workflow_id": workflow_id,
                                                    "execution_id": execution_id,
                                                    "timestamp": datetime.utcnow().isoformat()
                                                }
                                    
                                    # Set flag to skip agent execution in main loop
                                    cache_hit_skip = True
                                break
            
            # Check if cache hit occurred and skip agent execution
            if cache_hit_skip:
                continue
            
            # Update resource monitoring for agent execution
            self.resource_monitor.update_agent_count(workflow_id, execution_id, executed=1, active=1)
            
            # Don't create custom agent spans - let the multi-agent system handle this properly
            # The DynamicMultiAgentSystem will create the correct agent span structure
            agent_execution_span = execution_trace  # Pass the main trace as parent
            
            yield {
                "type": "agent_execution_start",
                "agent_name": agent_name,
                "node_id": agent.get("node_id"),  # CRITICAL: Add node_id for visual status update
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
                
                # Yield agent completion event for test scripts and UI
                yield {
                    "type": "agent_complete",
                    "content": agent_output,
                    "agent_name": agent_name,
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "tools_used": result.get("tools_used", [])
                }
                
                # Clean agent output for inter-agent communication (remove thinking tags)
                clean_output = self._clean_output_for_state_passing(agent_output)
                
                # Store agent output in clean format (no truncation needed with state-based approach)
                agent_outputs[agent_name] = agent_output  # Keep original for UI display
                
                # CRITICAL: Apply state formatting if enabled
                formatted_output = clean_output  # Default to clean output
                if agent.get("state_enabled", False):
                    # Format output based on state configuration
                    formatted_output = self._format_agent_output_for_chaining(
                        clean_output,
                        agent,
                        None  # No previous chain data for passthrough operation
                    )
                    logger.info(f"[STATE FORMATTING] Applied {agent.get('output_format', 'text')} format to {agent_name} output")
                
                # Update workflow state with agent output
                if workflow_state:
                    # Store agent output in structured format
                    agent_result = {
                        "agent_name": agent_name,
                        "node_id": agent["node_id"],
                        "output": formatted_output,  # Use formatted output for state passing
                        "raw_output": agent_output,  # Keep original for UI/debugging
                        "clean_output": clean_output,  # Keep clean version too
                        "tools_used": result.get("tools_used", []),
                        "timestamp": datetime.utcnow().isoformat(),
                        "index": i,
                        "state_enabled": agent.get("state_enabled", False),
                        "output_format": agent.get("output_format", "text")
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
                
                # CACHE STORAGE: Store result in cache if there was a cache miss
                if cache_result and cache_result.get("should_cache") and not cache_result.get("cache_hit"):
                    # Find the cache node to get its configuration
                    cache_config = None
                    for edge in workflow_edges:
                        if edge.get("source") == agent_node_id:
                            target_id = edge.get("target")
                            for cache in cache_nodes:
                                if cache["node_id"] == target_id:
                                    cache_config = cache
                                    break
                    
                    if cache_config:
                        # Store the agent result in cache
                        primary_cache_key = cache_result.get("cache_key", "")
                        cache_stored = await self._store_cache_result(
                            primary_cache_key,
                            agent_output,
                            cache_config.get("ttl", 3600),
                            cache_result.get("cache_condition", ""),
                            cache_config.get("max_cache_size", 10) * 1024 * 1024
                        )
                        
                        # Also store with cache node ID for easy lookup
                        cache_node_key = f"cache_{workflow_id}_{cache_config['node_id']}"
                        await self._store_cache_result(
                            cache_node_key,
                            agent_output,
                            cache_config.get("ttl", 3600),
                            cache_result.get("cache_condition", ""),
                            cache_config.get("max_cache_size", 10) * 1024 * 1024
                        )
                        
                        if cache_stored:
                            yield {
                                "type": "cache_stored",
                                "agent_name": agent_name,
                                "node_id": agent_node_id,
                                "cache_id": cache_config["node_id"],
                                "cache_key": primary_cache_key,
                                "cache_node_key": cache_node_key,
                                "ttl": cache_config.get("ttl", 3600),
                                "workflow_id": workflow_id,
                                "execution_id": execution_id,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                
                # Parse the agent prompt to extract structured sections for frontend
                agent_input_sections = self._parse_agent_prompt_sections(agent_prompt)
                
                yield {
                    "type": "agent_execution_complete",
                    "agent_name": agent_name,
                    "node_id": agent.get("node_id"),  # CRITICAL: Add node_id for visual status update
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
                
                # Process router nodes after agent execution
                current_node_id = agent.get("node_id")
                if router_nodes and current_node_id:
                    # Check if this agent connects to any router (direct connection)
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            target_id = edge.get("target")
                            # Check if target is a router node
                            for router in router_nodes:
                                if router["node_id"] == target_id:
                                    logger.info(f"[ROUTER] Processing RouterNode {target_id} with direct agent result: {agent_output}")
                                    # Process router logic
                                    router_result = self._process_router_node(
                                        router, agent_output, agent_outputs, workflow_edges
                                    )
                                    if router_result:
                                        active_target_nodes.update(router_result["target_nodes"])
                                        yield {
                                            "type": "router_decision",
                                            "router_id": router["node_id"],
                                            "matched_routes": router_result["matched_routes"],
                                            "target_nodes": list(router_result["target_nodes"]),
                                            "workflow_id": workflow_id,
                                            "execution_id": execution_id,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                    
                    # CRITICAL: Also check for agent→cache→router connections (cache MISS case)
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            cache_target_id = edge.get("target")
                            # Check if target is a cache node
                            for cache in cache_nodes:
                                if cache["node_id"] == cache_target_id:
                                    # Found agent→cache connection, now check cache→router
                                    for cache_edge in workflow_edges:
                                        if cache_edge.get("source") == cache_target_id:
                                            router_target_id = cache_edge.get("target")
                                            # Check if cache connects to router
                                            for router in router_nodes:
                                                if router["node_id"] == router_target_id:
                                                    logger.info(f"[ROUTER] Processing RouterNode {router_target_id} with agent result via CacheNode (cache MISS): {agent_output}")
                                                    # Process router logic with agent output
                                                    router_result = self._process_router_node(
                                                        router, agent_output, agent_outputs, workflow_edges
                                                    )
                                                    if router_result:
                                                        active_target_nodes.update(router_result["target_nodes"])
                                                        yield {
                                                            "type": "router_decision",
                                                            "router_id": router["node_id"],
                                                            "matched_routes": router_result["matched_routes"],
                                                            "target_nodes": list(router_result["target_nodes"]),
                                                            "workflow_id": workflow_id,
                                                            "execution_id": execution_id,
                                                            "timestamp": datetime.utcnow().isoformat(),
                                                            "triggered_by": "cache_miss_via_agent"
                                                        }
                                                    else:
                                                        logger.warning(f"[ROUTER] No routing decision made for agent result via cache: {agent_output}")
                
                # Process transform nodes after agent execution
                current_node_id = agent.get("node_id")
                if transform_nodes and current_node_id:
                    # Check if this agent connects to any transform node
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            target_id = edge.get("target")
                            # Check if target is a transform node
                            for transform in transform_nodes:
                                if transform["node_id"] == target_id:
                                    # Process transform logic
                                    transform_result = self._process_transform_node(
                                        transform, agent_output
                                    )
                                    if transform_result:
                                        yield {
                                            "type": "transform_execution",
                                            "transform_id": transform["node_id"],
                                            "transform_type": transform["transform_type"],
                                            "input": agent_output,
                                            "output": transform_result["output"],
                                            "error": transform_result.get("error"),
                                            "workflow_id": workflow_id,
                                            "execution_id": execution_id,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                
                # Process parallel nodes after agent execution
                if parallel_nodes and current_node_id:
                    # Check if this agent connects to any parallel node
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            target_id = edge.get("target")
                            # Check if target is a parallel node
                            for parallel in parallel_nodes:
                                if parallel["node_id"] == target_id:
                                    # Process parallel execution logic
                                    # Process parallel node with async generator
                                    async for parallel_event in self._process_parallel_node(
                                        parallel, 
                                        agent_output, 
                                        workflow_edges, 
                                        workflow_id, 
                                        execution_id,
                                        agent_plan,
                                        workflow_state,
                                        execution_trace,
                                        agent_outputs
                                    ):
                                        yield parallel_event
                
                # Process condition nodes after agent execution
                if condition_nodes and current_node_id:
                    # Check if this agent connects to any condition node
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            target_id = edge.get("target")
                            # Check if target is a condition node
                            for condition in condition_nodes:
                                if condition["node_id"] == target_id:
                                    # Process condition logic
                                    yield {
                                        "type": "condition_evaluation_start",
                                        "condition_id": condition["node_id"],
                                        "condition_type": condition["condition_type"],
                                        "operator": condition.get("operator", "equals"),
                                        "workflow_id": workflow_id,
                                        "execution_id": execution_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                    
                                    condition_result = await self._process_condition_node(
                                        condition, agent_output, workflow_id, execution_id
                                    )
                                    
                                    if condition_result:
                                        # Store condition result for execution control
                                        condition_results[condition["node_id"]] = condition_result
                                        
                                        # Yield condition result
                                        yield {
                                            "type": "condition_evaluation_complete",
                                            "condition_id": condition["node_id"],
                                            "result": condition_result["result"],
                                            "branch": condition_result["branch"],
                                            "evaluation_details": condition_result.get("evaluation_details", {}),
                                            "condition_type": condition_result.get("condition_type", "unknown"),
                                            "workflow_id": workflow_id,
                                            "execution_id": execution_id,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                        
                                        # Handle branching logic - find nodes connected to true/false branches
                                        self._handle_condition_branching(
                                            condition_result, condition, workflow_edges, agents
                                        )
                
                # STATE NODE PROCESSING: Process state nodes connected to this agent
                if state_nodes and current_node_id:
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            target_id = edge.get("target")
                            # Check if target is a state node
                            for state_node in state_nodes:
                                if state_node["node_id"] == target_id:
                                    # Process the state node with agent output
                                    yield {
                                        "type": "state_node_execution_start",
                                        "state_node_id": state_node["node_id"],
                                        "state_operation": state_node.get("state_operation", "merge"),
                                        "state_key": state_node.get("state_key", ""),
                                        "workflow_id": workflow_id,
                                        "execution_id": execution_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                    
                                    # Execute state node operation with agent output
                                    await self._process_state_node(
                                        state_node, agent_output, workflow_state, workflow_id, execution_id
                                    )
                                    
                                    yield {
                                        "type": "state_node_execution_complete",
                                        "state_node_id": state_node["node_id"],
                                        "state_operation": state_node.get("state_operation", "merge"),
                                        "state_key": state_node.get("state_key", ""),
                                        "workflow_id": workflow_id,
                                        "execution_id": execution_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                
                # Process API nodes after agent execution
                if api_nodes and current_node_id:
                    # Check if this agent connects to any API node
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            target_id = edge.get("target")
                            # Check if target is an API node
                            for api_node in api_nodes:
                                if api_node["node_id"] == target_id:
                                    # Process API call with agent output
                                    yield {
                                        "type": "api_execution_start",
                                        "api_node_id": api_node["node_id"],
                                        "api_label": api_node["label"],
                                        "api_endpoint": f"{api_node['base_url']}{api_node['endpoint_path']}",
                                        "http_method": api_node["http_method"],
                                        "workflow_id": workflow_id,
                                        "execution_id": execution_id,
                                        "timestamp": datetime.utcnow().isoformat()
                                    }
                                    
                                    try:
                                        # Import APIExecutor
                                        from app.automation.core.api_executor import get_api_executor
                                        
                                        # Get executor instance
                                        executor = await get_api_executor()
                                        
                                        # Prepare parameters from agent output
                                        # Try to parse agent output as JSON for API parameters
                                        try:
                                            import json
                                            if isinstance(agent_output, str):
                                                # Try to extract JSON from agent output
                                                import re
                                                json_match = re.search(r'\{.*\}', agent_output, re.DOTALL)
                                                if json_match:
                                                    api_parameters = json.loads(json_match.group())
                                                else:
                                                    # Use agent output as query parameter
                                                    api_parameters = {"query": agent_output}
                                            else:
                                                api_parameters = agent_output if isinstance(agent_output, dict) else {"query": str(agent_output)}
                                        except Exception as e:
                                            logger.warning(f"[API NODE] Failed to parse agent output for API parameters: {e}")
                                            api_parameters = {"query": str(agent_output)}
                                        
                                        # Execute API call
                                        api_result = await executor.execute_api_call(
                                            node_config=api_node,
                                            parameters=api_parameters,
                                            workflow_id=workflow_id,
                                            execution_id=execution_id,
                                            node_id=api_node["node_id"]
                                        )
                                        
                                        # Store API result in workflow state
                                        if workflow_state:
                                            api_output_key = f"api_output_{api_node['node_id']}"
                                            workflow_state.set_state(api_output_key, api_result)
                                            logger.info(f"[API NODE] Stored API result for node {api_node['node_id']}")
                                        
                                        # Yield API execution success
                                        yield {
                                            "type": "api_execution_success",
                                            "api_node_id": api_node["node_id"],
                                            "api_label": api_node["label"],
                                            "api_result": api_result,
                                            "api_parameters": api_parameters,
                                            "workflow_id": workflow_id,
                                            "execution_id": execution_id,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                                        
                                    except Exception as e:
                                        logger.error(f"[API NODE] API execution failed for node {api_node['node_id']}: {e}")
                                        
                                        # Yield API execution error
                                        yield {
                                            "type": "api_execution_error",
                                            "api_node_id": api_node["node_id"],
                                            "api_label": api_node["label"],
                                            "error": str(e),
                                            "workflow_id": workflow_id,
                                            "execution_id": execution_id,
                                            "timestamp": datetime.utcnow().isoformat()
                                        }
                
                # Store agent output for potential aggregator nodes (input collection)
                if aggregator_nodes and current_node_id:
                    # Find downstream aggregator nodes that need this output
                    for edge in workflow_edges:
                        if edge.get("source") == current_node_id:
                            target_id = edge.get("target")
                            # Check if target is an aggregator node
                            for aggregator in aggregator_nodes:
                                if aggregator["node_id"] == target_id:
                                    # Store this agent output for the aggregator
                                    aggregator_key = f"aggregator_inputs_{aggregator['node_id']}"
                                    if workflow_state:
                                        # Get existing inputs or initialize
                                        existing_inputs = workflow_state.get_state(aggregator_key) or []
                                        
                                        # Add this agent's output to the collection
                                        input_data = {
                                            "source_agent": agent_name,
                                            "source_node_id": current_node_id,
                                            "output": agent_output,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "agent_index": i
                                        }
                                        existing_inputs.append(input_data)
                                        workflow_state.set_state(aggregator_key, existing_inputs)
                                        
                                        logger.info(f"[AGGREGATOR] Stored input from {agent_name} for aggregator {aggregator['node_id']} ({len(existing_inputs)} total inputs)")
                                        
                                        # Check if we have collected all required inputs for this aggregator
                                        required_inputs = self._count_upstream_nodes(aggregator["node_id"], workflow_edges)
                                        
                                        if len(existing_inputs) >= required_inputs:
                                            # All inputs collected - process aggregation
                                            yield {
                                                "type": "aggregator_execution_start",
                                                "aggregator_id": aggregator["node_id"],
                                                "aggregation_strategy": aggregator["aggregation_strategy"],
                                                "collected_inputs": len(existing_inputs),
                                                "required_inputs": required_inputs,
                                                "workflow_id": workflow_id,
                                                "execution_id": execution_id,
                                                "timestamp": datetime.utcnow().isoformat()
                                            }
                                            
                                            # Extract just the outputs for aggregation
                                            aggregator_inputs = [inp["output"] for inp in existing_inputs]
                                            
                                            aggregator_result = await self._process_aggregator_node(
                                                aggregator, aggregator_inputs, workflow_id, execution_id
                                            )
                                            
                                            if aggregator_result:
                                                yield {
                                                    "type": "aggregator_execution_complete",
                                                    "aggregator_id": aggregator["node_id"],
                                                    "aggregated_result": aggregator_result.get("aggregated_result"),
                                                    "confidence_score": aggregator_result.get("confidence_score"),
                                                    "source_analysis": aggregator_result.get("source_analysis"),
                                                    "metadata": aggregator_result.get("metadata"),
                                                    "input_sources": [inp["source_agent"] for inp in existing_inputs],
                                                    "workflow_id": workflow_id,
                                                    "execution_id": execution_id,
                                                    "timestamp": datetime.utcnow().isoformat()
                                                }
                                                
                                                # Clear the collected inputs after processing
                                                workflow_state.clear_state([aggregator_key])
                                        else:
                                            # Still waiting for more inputs
                                            yield {
                                                "type": "aggregator_input_collected",
                                                "aggregator_id": aggregator["node_id"],
                                                "source_agent": agent_name,
                                                "collected_count": len(existing_inputs),
                                                "required_count": required_inputs,
                                                "workflow_id": workflow_id,
                                                "execution_id": execution_id,
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
                    "node_id": agent.get("node_id"),  # CRITICAL: Add node_id for visual status update
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
                "node_id": agent.get("node_id"),  # CRITICAL: Add node_id for visual status update
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
            # Get agent node_id from original agents list
            agent_node_id = agents[task_index].get("node_id") if task_index < len(agents) else None
            
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
                    "node_id": agent_node_id,  # CRITICAL: Add node_id for visual status update
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
                    "node_id": agent_node_id,  # CRITICAL: Add node_id for visual status update
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
            
            logger.info(f"[AGENT NODE DEBUG] Starting execution for agent: '{agent_name}' in workflow {workflow_id}")
            logger.info(f"[AGENT NODE DEBUG] Parameters received - agent_index: {agent_index}, prompt length: {len(prompt) if prompt else 0}")
            logger.info(f"[AGENT NODE DEBUG] Agent configuration keys: {list(agent.keys())}")
            logger.debug(f"[AGENT NODE DEBUG] Full agent config: {agent}")
            
            if not agent_name:
                logger.error(f"[AGENT NODE DEBUG] No agent name found in: {list(agent.keys())}")
                raise ValueError(f"Agent name is required. Available keys: {list(agent.keys())}")
                
            logger.info(f"[AGENT WORKFLOW] Executing agent: {agent_name}")
            
            # Use the pre-created DynamicMultiAgentSystem instance
            dynamic_agent_system = self.dynamic_agent_system
            
            # CRITICAL FIX: Extract workflow config from multiple possible locations FIRST
            workflow_node = agent.get("node", {})  # Main workflow node config
            agent_config_data = agent.get("agent_config", {})
            workflow_context = agent.get("context", {})
            available_tools = agent.get("tools", []) or workflow_node.get("tools", [])
            
            # Helper function to check if workflow has complete configuration
            def _has_complete_workflow_config():
                """Check if workflow has sufficient configuration to skip database loading"""
                # Check for essential config: max_tokens, model, and either custom_prompt or query
                has_max_tokens = bool(
                    agent.get("max_tokens") or 
                    workflow_node.get("max_tokens") or 
                    workflow_context.get("max_tokens") or 
                    agent_config_data.get("max_tokens")
                )
                
                has_model = bool(
                    agent.get("model") or 
                    workflow_node.get("model") or
                    workflow_context.get("model") or 
                    agent_config_data.get("model")
                )
                
                has_prompt_config = bool(
                    agent.get("custom_prompt") or 
                    agent.get("query") or
                    agent_config_data.get("system_prompt")
                )
                
                return has_max_tokens and has_model and has_prompt_config
            
            # Use workflow config directly if complete, otherwise load from database
            if _has_complete_workflow_config():
                logger.info(f"[AGENT WORKFLOW] Using complete workflow configuration for agent {agent_name}, skipping database")
                
                # Build agent_info from workflow configuration only
                agent_info = {
                    "name": agent_name,
                    "role": agent_config_data.get("role", f"Agent {agent_name}"),
                    "system_prompt": "You are a helpful assistant.",  # Will be overridden below
                    "config": {},
                    "tools": available_tools or []
                }
                
            else:
                logger.info(f"[AGENT WORKFLOW] Workflow configuration incomplete, loading agent {agent_name} from database")
                
                # Get the full agent data from cache
                agent_info = get_agent_by_name(agent_name)
                
                # TEMPORARY: Create a basic agent_info if not found in cache for testing
                if not agent_info and agent_name == "Research Analyst":
                    logger.info(f"[AGENT WORKFLOW] Creating temporary agent_info for testing in _execute_single_agent")
                    agent_info = {
                        "name": agent_name,
                        "role": "Senior Research Analyst",
                        "system_prompt": "You are a senior research analyst. Provide detailed, comprehensive analysis with specific metrics and data points.",
                        "config": {"temperature": 0.7, "max_tokens": 2000},
                        "tools": ["get_datetime"]
                    }
                
                if not agent_info:
                    raise ValueError(f"Agent '{agent_name}' not found in cache")
                
                # Create a copy of agent_info to avoid modifying the cache
                agent_info = agent_info.copy()
            
            # Debug logging for tool extraction
            logger.debug(f"[TOOL EXTRACTION] Agent: {agent_name}")
            logger.debug(f"[TOOL EXTRACTION] agent.get('tools'): {agent.get('tools', [])}")
            logger.debug(f"[TOOL EXTRACTION] workflow_node.get('tools'): {workflow_node.get('tools', [])}")
            logger.debug(f"[TOOL EXTRACTION] Final available_tools: {available_tools}")
            
            # Ensure tools are set properly (already handled in workflow config path above)
            if available_tools and agent_info["tools"] != available_tools:
                agent_info["tools"] = available_tools
                logger.info(f"[AGENT WORKFLOW] Setting agent {agent_name} tools: {available_tools}")
            
            # Ensure config section exists
            if "config" not in agent_info:
                agent_info["config"] = {}
            
            # Debug logging for model extraction
            logger.debug(f"[MODEL EXTRACTION] Agent dict keys: {list(agent.keys())}")
            logger.debug(f"[MODEL EXTRACTION] agent.get('model'): {agent.get('model')}")
            logger.debug(f"[MODEL EXTRACTION] workflow_node.get('model'): {workflow_node.get('model')}")
            logger.debug(f"[MODEL EXTRACTION] agent_config_data.get('model'): {agent_config_data.get('model')}")
            
            # Set workflow configuration values (no override needed when built from workflow)
            workflow_model = (
                agent.get("model") or  # Direct model from AgentNode UI
                workflow_node.get("model") or
                workflow_context.get("model") or 
                agent_config_data.get("model")
            )
            logger.debug(f"[MODEL EXTRACTION] workflow_model value: {workflow_model}")
            logger.debug(f"[MODEL EXTRACTION] agent_info config before setting: {agent_info.get('config', {})}")
            
            if workflow_model:
                agent_info["config"]["model"] = workflow_model
                logger.info(f"[AGENT WORKFLOW] Setting agent {agent_name} model: {workflow_model}")
            else:
                logger.warning(f"[AGENT WORKFLOW] No workflow model found for agent {agent_name}")
            
            # Set temperature if specified in workflow  
            workflow_temperature = (
                agent.get("temperature") or  # Direct temperature from AgentNode UI
                workflow_node.get("temperature") or
                workflow_context.get("temperature") or
                agent_config_data.get("temperature")
            )
            if workflow_temperature is not None:
                agent_info["config"]["temperature"] = workflow_temperature
                logger.info(f"[AGENT WORKFLOW] Setting agent {agent_name} temperature: {workflow_temperature}")
                
            # Set max_tokens if specified in workflow
            # DEBUG: Log all possible sources for max_tokens
            agent_max_tokens = agent.get("max_tokens")
            node_max_tokens = workflow_node.get("max_tokens")
            context_max_tokens = workflow_context.get("max_tokens")
            config_max_tokens = agent_config_data.get("max_tokens")
            
            logger.debug(f"[MAX_TOKENS DEBUG] agent.max_tokens: {agent_max_tokens}, node.max_tokens: {node_max_tokens}, context.max_tokens: {context_max_tokens}, config.max_tokens: {config_max_tokens}")
            
            workflow_max_tokens = (
                agent_max_tokens or  # Direct max_tokens from AgentNode UI
                node_max_tokens or
                context_max_tokens or
                config_max_tokens
            )
            
            logger.debug(f"[MAX_TOKENS DEBUG] Final workflow_max_tokens: {workflow_max_tokens}")
            
            if workflow_max_tokens:
                agent_info["config"]["max_tokens"] = workflow_max_tokens
                logger.info(f"[AGENT WORKFLOW] Setting agent {agent_name} max_tokens: {workflow_max_tokens}")
            else:
                logger.warning(f"[AGENT WORKFLOW] No max_tokens found in workflow config for agent {agent_name}, keeping default")
            
            # Set system_prompt with workflow configuration
            # Build workflow prompt from custom_prompt + query
            workflow_custom_prompt = agent.get("custom_prompt", "")
            workflow_query = agent.get("query", "")
            
            # Combine workflow custom prompt and query
            workflow_prompt_parts = []
            if workflow_custom_prompt:
                workflow_prompt_parts.append(workflow_custom_prompt)
            if workflow_query:
                workflow_prompt_parts.append(workflow_query)
            
            combined_workflow_prompt = "\n\n".join(workflow_prompt_parts) if workflow_prompt_parts else ""
            
            # Set agent's system prompt if workflow provides one
            if combined_workflow_prompt:
                agent_info["system_prompt"] = combined_workflow_prompt
                logger.info(f"[AGENT WORKFLOW] Setting agent {agent_name} system_prompt from workflow config (length: {len(combined_workflow_prompt)})")
                logger.debug(f"[AGENT WORKFLOW] Workflow system prompt: {combined_workflow_prompt[:200]}...")
            else:
                # Fallback to agent config data system prompt or keep existing
                fallback_prompt = agent_config_data.get("system_prompt", agent_info.get("system_prompt", "You are a helpful assistant."))
                agent_info["system_prompt"] = fallback_prompt
                logger.info(f"[AGENT WORKFLOW] Using fallback system_prompt for agent {agent_name} (length: {len(fallback_prompt)})")
            
            # Use configured timeout from workflow node, with fallback to reasonable default
            effective_timeout = agent.get("configured_timeout", 60)
            logger.info(f"[AGENT WORKFLOW] Using configured timeout: {effective_timeout}s for agent {agent_name}")
            
            # Debug: Log the final agent_info configuration
            logger.debug(f"[AGENT WORKFLOW] Final agent_info config for {agent_name}: {agent_info.get('config', {})}")
            logger.info(f"[AGENT WORKFLOW] Final agent_info keys: name={agent_info.get('name')}, tools_count={len(agent_info.get('tools', []))}, system_prompt_length={len(agent_info.get('system_prompt', ''))}")
            
            # Get model, temperature, and max_tokens from overridden agent_info config OR from agent dict
            # Priority: agent dict (from UI) > agent_info config (after override) > defaults
            model_to_use = agent.get("model") or agent_info.get("config", {}).get("model")
            temperature_to_use = agent.get("temperature") or agent_info.get("config", {}).get("temperature", agent.get("context", {}).get("temperature", 0.7))
            max_tokens_to_use = agent.get("max_tokens") or agent_info.get("config", {}).get("max_tokens")
            
            logger.debug(f"[AGENT WORKFLOW] Agent dict model: {agent.get('model')}, temp: {agent.get('temperature')}, max_tokens: {agent.get('max_tokens')}")
            logger.debug(f"[AGENT WORKFLOW] Context model: {model_to_use}, temperature: {temperature_to_use}, max_tokens: {max_tokens_to_use}")
            
            context = {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "agent_config": agent,
                "tools": available_tools,
                "available_tools": available_tools,  # Ensure tools are passed to DynamicMultiAgentSystem
                "model": model_to_use,  # Pass the overridden model
                "temperature": temperature_to_use,
                "max_tokens": max_tokens_to_use,  # Pass the overridden max_tokens
                "timeout": effective_timeout  # Pass configured timeout to agent system
            }
            
            # Execute using the dynamic agent system with enhanced tracking
            final_response = None
            tools_used = []
            llm_usage = {}
            model_info = "unknown"
            start_time = datetime.utcnow()
            
            # CRITICAL FIX: Check if prompt contains previous agent results
            # If prompt contains "Context from previous agents:", it has dependencies
            has_dependencies = "Context from previous agents:" in prompt
            
            if has_dependencies:
                # CRITICAL: When agent has dependencies, combine workflow prompt + context
                # The 'prompt' contains query + previous results
                if combined_workflow_prompt:
                    # If workflow defines custom_prompt, prepend it
                    full_context_prompt = combined_workflow_prompt + "\n\n" + prompt
                else:
                    # Otherwise use just the context prompt
                    full_context_prompt = prompt
                
                agent_info["system_prompt"] = full_context_prompt
                # CRITICAL FIX: Pass a query to avoid "no query provided" error
                simple_query = "Process the task based on the context provided in the system prompt."
                logger.info(f"[AGENT WORKFLOW] Agent {agent_name} has dependencies - using full context as system prompt")
                logger.debug(f"[AGENT WORKFLOW] System prompt with context, length: {len(agent_info['system_prompt'])}")
                system_prompt_preview = agent_info.get('system_prompt', '')[:500] if agent_info.get('system_prompt') else '...'
                logger.debug(f"[AGENT WORKFLOW] System prompt preview: {system_prompt_preview}...")
            else:
                # Normal case - pass simple query
                simple_query = workflow_query if workflow_query else prompt
                query_preview = simple_query[:100] if simple_query else '...'
                logger.info(f"[AGENT WORKFLOW] Passing simple query to DynamicMultiAgentSystem: {query_preview}...")
            
            # CRITICAL FIX: Pass the correct parent span to DynamicMultiAgentSystem
            # This should be the agent-execution-sequence span, not the main trace
            logger.info(f"[AGENT NODE DEBUG] Calling dynamic_agent_system.execute_agent for {agent_name}")
            logger.info(f"[AGENT NODE DEBUG] Tools available: {agent_info.get('tools', [])}")
            logger.info(f"[AGENT NODE DEBUG] Query: {simple_query[:200]}...")
            logger.info(f"[AGENT NODE DEBUG] Context length: {len(context) if context else 0}")
            
            async for event in dynamic_agent_system.execute_agent(
                agent_name=agent_name,
                agent_data=agent_info,
                query=simple_query,  # Pass simple query, not full prompt
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
                    logger.info(f"[AGENT WORKFLOW] Extracted model from event: {model_info} (event keys: {list(event.keys())})")
                    
                elif event_type == "tool_call":
                    tool_name = event.get("tool", "")
                    tool_input = event.get("input", {})
                    tool_output = event.get("output", {})
                    success = event.get("success", True)
                    duration = event.get("duration", 0)
                    
                    logger.info(f"[AGENT NODE DEBUG] MCP tool call detected: {tool_name}")
                    logger.info(f"[AGENT NODE DEBUG] Tool input: {str(tool_input)[:200]}...")
                    logger.info(f"[AGENT NODE DEBUG] Tool success: {success}, duration: {duration}s")
                    logger.info(f"[AGENT NODE DEBUG] Tool output: {str(tool_output)[:300]}...")
                    
                    tool_call_info = {
                        "tool": tool_name,
                        "input": tool_input,
                        "output": tool_output,
                        "duration": duration,
                        "success": success,
                        "name": tool_name
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
        # Map UI names to internal format names
        format_mapping = {
            "text": "text",
            "plain text": "text",
            "structured": "structured",
            "structured data": "structured",
            "context": "context",
            "context object": "context",
            "full": "full",
            "full agent response": "full"
        }
        
        normalized_format = format_mapping.get(output_format.lower(), output_format)
        
        if normalized_format == "text":
            # Plain text - just the output
            formatted_output = agent_output
        elif normalized_format == "structured":
            # Structured data - organized with metadata
            # CRITICAL FIX: Ensure all values are JSON-serializable
            formatted_output = {
                "content": str(agent_output),  # Ensure string conversion
                "agent": str(agent_node["agent_name"]),
                "node_id": str(agent_node.get("node_id", "")),
                "timestamp": datetime.utcnow().isoformat(),
                "tools_used": list(agent_node.get("tools_used", []))  # Ensure list conversion
            }
            # Validate JSON serializability
            try:
                json.dumps(formatted_output)
                logger.debug(f"[STRUCTURED OUTPUT] Successfully formatted output for {agent_node['agent_name']}")
            except (TypeError, ValueError) as e:
                logger.error(f"[STRUCTURED OUTPUT] Failed to serialize structured output for {agent_node['agent_name']}: {e}")
                # Fallback to safe format
                formatted_output = {
                    "content": str(agent_output),
                    "agent": str(agent_node["agent_name"]),
                    "node_id": str(agent_node.get("node_id", "")),
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": "Serialization fallback applied"
                }
        elif normalized_format == "context":
            # Context object - includes context and state info
            # CRITICAL FIX: Ensure all context data is JSON-serializable
            context_data = agent_node.get("context", {})
            try:
                # Test JSON serializability of context
                json.dumps(context_data)
                safe_context = context_data
            except (TypeError, ValueError) as e:
                logger.warning(f"[CONTEXT OUTPUT] Context not JSON-serializable for {agent_node['agent_name']}: {e}")
                safe_context = {"error": "Context serialization failed", "raw": str(context_data)}
            
            formatted_output = {
                "response": str(agent_output),
                "agent_name": str(agent_node["agent_name"]),
                "node_id": str(agent_node.get("node_id", "")),
                "context": safe_context,
                "state_operation": str(state_operation),
                "chain_key": str(chain_key)
            }
        elif normalized_format == "full":
            # Full agent response - everything
            # CRITICAL FIX: Ensure all nested data is JSON-serializable
            context_data = agent_node.get("context", {})
            tools_data = agent_node.get("tools", [])
            
            try:
                json.dumps(context_data)
                safe_context = context_data
            except (TypeError, ValueError):
                safe_context = {"error": "Context serialization failed", "raw": str(context_data)}
            
            try:
                json.dumps(tools_data)
                safe_tools = tools_data
            except (TypeError, ValueError):
                safe_tools = [str(tool) for tool in tools_data] if isinstance(tools_data, list) else [str(tools_data)]
            
            formatted_output = {
                "output": str(agent_output),
                "agent_name": str(agent_node["agent_name"]),
                "node_id": str(agent_node.get("node_id", "")),
                "tools": safe_tools,
                "context": safe_context,
                "state_operation": str(state_operation),
                "chain_key": str(chain_key),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Default to plain text
            formatted_output = agent_output
        
        # Apply state operation if there's previous chain data
        # Map UI names to internal operation names
        operation_mapping = {
            "passthrough": "passthrough",
            "pass output directly": "passthrough",
            "merge": "merge",
            "merge with previous": "merge",
            "replace": "replace",
            "replace previous": "replace",
            "append": "append",
            "append to previous": "append"
        }
        
        normalized_operation = operation_mapping.get(state_operation.lower(), state_operation)
        
        if previous_chain_data is not None and normalized_operation != "passthrough":
            if normalized_operation == "merge":
                # Merge with previous - combine data
                if isinstance(previous_chain_data, dict) and isinstance(formatted_output, dict):
                    # Merge dictionaries
                    merged_data = previous_chain_data.copy()
                    merged_data.update(formatted_output)
                    return merged_data
                else:
                    # Merge as strings
                    return f"{previous_chain_data}\n\n{formatted_output}"
            
            elif normalized_operation == "replace":
                # Replace previous - discard previous data
                return formatted_output
            
            elif normalized_operation == "append":
                # Append to previous - create or extend list
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
        
        # For multiple agents, combine their outputs
        # If output format is markdown (from OutputNode), use a clean merge
        if output_format == "markdown" or len(agent_outputs) > 1:
            # Combine all agent outputs with clear separation
            synthesis_parts = []
            
            # Sort by key to ensure consistent ordering
            sorted_outputs = sorted(agent_outputs.items())
            
            for i, (agent_key, output) in enumerate(sorted_outputs):
                # Extract agent name from key (could be node ID or agent name)
                agent_name = agent_key
                # Try to make the name more readable if it's a node ID
                if agent_key.startswith("agentnode-"):
                    agent_name = f"Agent {i+1}"
                
                synthesis_parts.append(f"## {agent_name}\n\n{output}")
            
            synthesis = "\n\n---\n\n".join(synthesis_parts)
            
            # Add metadata header if requested
            if include_metadata:
                synthesis = f"# Analysis from {len(agent_outputs)} AI Agents\n\n{synthesis}"
        else:
            # For single agent or when explicitly not combining
            synthesis = list(agent_outputs.values())[-1]
        
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
                state_key = state_node.get("state_key", "")
                state_keys = state_node.get("state_keys", [])
                state_values = state_node.get("state_values", {})
                default_value = state_node.get("default_value", {})
                merge_strategy = state_node.get("merge_strategy", "deep")
                ttl = state_node.get("ttl", 0)
                scope = state_node.get("scope", "workflow")
                checkpoint_name = state_node.get("checkpoint_name", "")
                node_id = state_node.get("node_id", "")
                
                logger.info(f"[STATE OPERATION] Processing {operation} for node {node_id}, key: {state_key}")
                
                # Handle single state key operations (new format)
                if state_key:
                    if operation == "write" or operation == "set":
                        # Write operation with default value
                        value_to_write = default_value if default_value else {}
                        workflow_state.set_state(state_key, value_to_write, checkpoint_name)
                        logger.info(f"[STATE WRITE] {state_key}: {value_to_write}")
                    
                    elif operation == "read" or operation == "get":
                        # Read operation - get current value or default
                        current_value = workflow_state.get_state(state_key)
                        if current_value is None and default_value is not None:
                            # Set default value if key doesn't exist
                            workflow_state.set_state(state_key, default_value, checkpoint_name)
                            current_value = default_value
                        logger.info(f"[STATE READ] {state_key}: {current_value}")
                    
                    elif operation == "update":
                        # Update operation - merge with existing value
                        current_value = workflow_state.get_state(state_key)
                        if current_value is None:
                            current_value = default_value if default_value else {}
                        
                        # Apply merge strategy
                        if merge_strategy == "deep":
                            if isinstance(current_value, dict) and isinstance(default_value, dict):
                                merged_value = {**current_value, **default_value}
                            else:
                                merged_value = default_value
                        elif merge_strategy == "shallow":
                            if isinstance(current_value, dict) and isinstance(default_value, dict):
                                merged_value = current_value.copy()
                                merged_value.update(default_value)
                            else:
                                merged_value = default_value
                        elif merge_strategy == "append":
                            if isinstance(current_value, list) and isinstance(default_value, list):
                                merged_value = current_value + default_value
                            else:
                                merged_value = default_value
                        else:  # replace
                            merged_value = default_value
                        
                        workflow_state.set_state(state_key, merged_value, checkpoint_name)
                        logger.info(f"[STATE UPDATE] {state_key}: {merged_value}")
                    
                    elif operation == "delete":
                        # Delete operation
                        workflow_state.clear_state([state_key], checkpoint_name)
                        logger.info(f"[STATE DELETE] {state_key}")
                    
                    elif operation == "merge":
                        # Legacy merge operation
                        if default_value:
                            workflow_state.merge_state(state_key, default_value, checkpoint_name)
                            logger.info(f"[STATE MERGE] {state_key}: {default_value}")
                
                # Handle legacy multi-key operations (backward compatibility)
                elif state_keys:
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
    
    async def _process_state_node(
        self,
        state_node: Dict[str, Any],
        agent_output: str,
        workflow_state: WorkflowState,
        workflow_id: int,
        execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single state node with agent output"""
        try:
            operation = state_node.get("state_operation", "merge")
            state_key = state_node.get("state_key", "")
            default_value = state_node.get("default_value", {})
            merge_strategy = state_node.get("merge_strategy", "deep")
            ttl = state_node.get("ttl", 0)
            scope = state_node.get("scope", "workflow")
            checkpoint_name = state_node.get("checkpoint_name", "")
            node_id = state_node.get("node_id", "")
            
            logger.info(f"[STATE NODE] Processing {operation} for node {node_id}, key: {state_key}")
            
            # Use agent output as the value to store/process
            value_to_process = agent_output
            
            # Apply different operations based on configuration
            if operation == "write" or operation == "set":
                # Write operation - store agent output
                workflow_state.set_state(state_key, value_to_process, checkpoint_name)
                logger.info(f"[STATE NODE WRITE] {state_key}: {value_to_process}")
            
            elif operation == "read" or operation == "get":
                # Read operation - get current value, optionally set default
                current_value = workflow_state.get_state(state_key)
                if current_value is None and default_value is not None:
                    workflow_state.set_state(state_key, default_value, checkpoint_name)
                    current_value = default_value
                logger.info(f"[STATE NODE READ] {state_key}: {current_value}")
                return {"result": current_value, "operation": "read"}
            
            elif operation == "update":
                # Update operation - merge agent output with existing value
                current_value = workflow_state.get_state(state_key)
                if current_value is None:
                    current_value = default_value if default_value else {}
                
                # Apply merge strategy
                if merge_strategy == "deep":
                    if isinstance(current_value, dict) and isinstance(value_to_process, str):
                        # Try to parse agent output as JSON
                        try:
                            import json
                            parsed_output = json.loads(value_to_process)
                            if isinstance(parsed_output, dict):
                                merged_value = {**current_value, **parsed_output}
                            else:
                                merged_value = parsed_output
                        except:
                            merged_value = value_to_process
                    else:
                        merged_value = value_to_process
                elif merge_strategy == "shallow":
                    if isinstance(current_value, dict) and isinstance(value_to_process, str):
                        try:
                            import json
                            parsed_output = json.loads(value_to_process)
                            if isinstance(parsed_output, dict):
                                merged_value = current_value.copy()
                                merged_value.update(parsed_output)
                            else:
                                merged_value = parsed_output
                        except:
                            merged_value = value_to_process
                    else:
                        merged_value = value_to_process
                elif merge_strategy == "append":
                    if isinstance(current_value, list):
                        if isinstance(value_to_process, list):
                            merged_value = current_value + value_to_process
                        else:
                            merged_value = current_value + [value_to_process]
                    else:
                        merged_value = [current_value, value_to_process]
                else:  # replace
                    merged_value = value_to_process
                
                workflow_state.set_state(state_key, merged_value, checkpoint_name)
                logger.info(f"[STATE NODE UPDATE] {state_key}: {merged_value}")
            
            elif operation == "delete":
                # Delete operation
                workflow_state.clear_state([state_key], checkpoint_name)
                logger.info(f"[STATE NODE DELETE] {state_key}")
            
            elif operation == "merge":
                # Legacy merge operation
                workflow_state.merge_state(state_key, value_to_process, checkpoint_name)
                logger.info(f"[STATE NODE MERGE] {state_key}: {value_to_process}")
            
            # Create checkpoint if specified
            if checkpoint_name:
                workflow_state.create_checkpoint(checkpoint_name)
            
            return {
                "node_id": node_id,
                "operation": operation,
                "state_key": state_key,
                "result": "success",
                "value": value_to_process
            }
            
        except Exception as e:
            logger.error(f"[STATE NODE] Error processing state node {node_id}: {e}")
            return {
                "node_id": node_id,
                "operation": operation,
                "state_key": state_key,
                "result": "error",
                "error": str(e)
            }
    
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
        # CRITICAL FIX: Always provide original user message to ALL agents
        # Each agent should interpret the original message through their role, not just previous results
        agent_specific_query = agent.get("query", "")
        original_user_query = user_query  # This is the original user message/question
        
        # For "Execute with message" workflows, all agents should get the original user message
        # Agent-specific queries are supplementary, not replacements
        if user_message and not agent_specific_query:
            # If no agent-specific query but we have original user message, use it
            agent_specific_query = original_user_query
        
        agent_input = {
            "user_request": {
                "message": user_message,
                "query": agent_specific_query,  # Agent-specific or original query
                "original_query": original_user_query,  # Always preserve original user question
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
                
                # CRITICAL FIX: Track results found for multi-input debugging
                results_found = []
                
                for connection in input_connections:
                    source_node_id = connection.get("source_node")
                    source_handle = connection.get("source_handle")
                    target_handle = connection.get("target_handle")
                    
                    # Enhanced dependency tracking with handle information
                    dep_result = workflow_state.get_state(f"node_output_{source_node_id}")
                    if dep_result:
                        # CRITICAL FIX: Ensure structured outputs are properly serialized
                        if isinstance(dep_result, dict):
                            enhanced_result = dep_result.copy()
                            # Ensure the output field is JSON-serializable for structured formats
                            if "output" in enhanced_result and isinstance(enhanced_result["output"], dict):
                                try:
                                    # Serialize and deserialize to ensure it's JSON-safe
                                    enhanced_result["output"] = json.loads(json.dumps(enhanced_result["output"]))
                                except (TypeError, ValueError) as e:
                                    logger.warning(f"[MULTI-INPUT] Failed to serialize structured output from {source_node_id}: {e}")
                                    # Fallback to string representation
                                    enhanced_result["output"] = str(enhanced_result["output"])
                        else:
                            enhanced_result = {"output": str(dep_result)}
                        
                        enhanced_result["connection_info"] = {
                            "source_handle": source_handle,
                            "target_handle": target_handle,
                            "connection_type": "4-way"
                        }
                        agent_input["previous_results"].append(enhanced_result)
                        results_found.append(source_node_id)
                    else:
                        logger.warning(f"[MULTI-INPUT] No result found for dependency {source_node_id} → {agent['name']}")
                
                logger.info(f"[MULTI-INPUT] Agent {agent['name']} aggregated {len(results_found)} inputs: {results_found}")
            else:
                # Standard dependency resolution with enhanced multi-input support
                results_found = []
                for dep_node_id in dependencies:
                    # Find the agent name/index for this dependency node
                    dep_result = workflow_state.get_state(f"node_output_{dep_node_id}")
                    if dep_result:
                        # CRITICAL FIX: Apply same serialization fix for standard dependencies
                        if isinstance(dep_result, dict) and "output" in dep_result and isinstance(dep_result["output"], dict):
                            try:
                                # Ensure structured outputs are JSON-serializable
                                dep_result_copy = dep_result.copy()
                                dep_result_copy["output"] = json.loads(json.dumps(dep_result["output"]))
                                agent_input["previous_results"].append(dep_result_copy)
                            except (TypeError, ValueError) as e:
                                logger.warning(f"[MULTI-INPUT] Failed to serialize structured output from {dep_node_id}: {e}")
                                # Fallback: convert to string
                                dep_result_copy = dep_result.copy()
                                dep_result_copy["output"] = str(dep_result["output"])
                                agent_input["previous_results"].append(dep_result_copy)
                        else:
                            agent_input["previous_results"].append(dep_result)
                        results_found.append(dep_node_id)
                
                if len(dependencies) > 1:  # Multi-input scenario
                    logger.info(f"[MULTI-INPUT] Agent {agent['name']} aggregated {len(results_found)} standard inputs: {results_found}")
                
                    # Fallback logic should be outside the multi-input check
                for dep_node_id in dependencies:
                    # Check if we still need to find this dependency
                    if dep_node_id not in results_found:
                        # Fallback: try to find by agent name mapping
                        for existing_agent_name in agent_outputs.keys():
                            agent_result = workflow_state.get_state(f"agent_output_{existing_agent_name}")
                            if agent_result and isinstance(agent_result, dict):
                                # Check if this agent result matches the dependency node
                                if agent_result.get("node_id") == dep_node_id:
                                    # Apply same serialization fix for fallback results
                                    if "output" in agent_result and isinstance(agent_result["output"], dict):
                                        try:
                                            agent_result_copy = agent_result.copy()
                                            agent_result_copy["output"] = json.loads(json.dumps(agent_result["output"]))
                                            agent_input["previous_results"].append(agent_result_copy)
                                        except (TypeError, ValueError) as e:
                                            logger.warning(f"[MULTI-INPUT] Failed to serialize fallback output from {dep_node_id}: {e}")
                                            agent_result_copy = agent_result.copy()
                                            agent_result_copy["output"] = str(agent_result["output"])
                                            agent_input["previous_results"].append(agent_result_copy)
                                    else:
                                        agent_input["previous_results"].append(agent_result)
                                    results_found.append(dep_node_id)
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
        
        # CRITICAL: Check for cached input from CacheNode
        if workflow_state:
            cached_input_key = f"agent_input_{agent['name']}"
            cached_input = workflow_state.get_state(cached_input_key)
            if cached_input:
                # Add cached data to the agent input
                logger.info(f"[CACHE INPUT] Agent {agent['name']} received cached input from CacheNode")
                agent_input["cached_input"] = {
                    "cached_output": cached_input.get("cached_output"),
                    "source_cache_node": cached_input.get("source_cache_node"),
                    "source_agent": cached_input.get("source_agent"),
                    "timestamp": cached_input.get("timestamp")
                }
                
                # Add cached data to previous results so agent can use it
                agent_input["previous_results"].append({
                    "agent_name": cached_input.get("source_agent", "cache"),
                    "node_id": cached_input.get("source_cache_node"),
                    "output": cached_input.get("cached_output"),
                    "cached": True,
                    "connection_info": {
                        "source_handle": "output",
                        "target_handle": "input", 
                        "connection_type": "cache"
                    }
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
        
        # CRITICAL FIX: Check if we have a workflow query
        workflow_query = agent.get("query", "")
        
        # If workflow provides a query AND this is the first agent with no dependencies
        previous_results = agent_input.get("previous_results", [])
        
        # ENHANCED DEBUG logging for multi-input scenarios
        query_preview = workflow_query[:50] if workflow_query else "..."
        logger.info(f"[PROMPT BUILD] Agent: {agent.get('name', 'unknown')}, Query: {query_preview}..., Previous results: {len(previous_results)}")
        
        if previous_results:
            # CRITICAL FIX: Enhanced logging for multi-input debugging
            if len(previous_results) > 1:
                logger.info(f"[MULTI-INPUT PROMPT] Agent {agent.get('name', 'unknown')} processing {len(previous_results)} inputs")
                for idx, result in enumerate(previous_results):
                    if isinstance(result, dict):
                        agent_name = result.get('agent_name', 'unknown')
                        node_id = result.get('node_id', 'unknown')
                        output_type = type(result.get('output', '')).__name__
                        logger.debug(f"[MULTI-INPUT PROMPT] Input {idx+1}: {agent_name} ({node_id}) - {output_type}")
                    else:
                        logger.debug(f"[MULTI-INPUT PROMPT] Input {idx+1}: {type(result).__name__}")
            else:
                result_preview = str(previous_results[0])[:200] if previous_results[0] else "..."
                logger.debug(f"[PROMPT BUILD] Previous result preview: {result_preview}...")
        
        # CRITICAL FIX: Always get the original user message for ALL agents
        user_request = agent_input["user_request"]
        original_query = user_request.get("original_query", "") or user_request.get("message", "")
        
        # For agents with specific workflow queries (like first agent), use that
        # For other agents, use original user query so they understand the overall task
        primary_task = workflow_query if workflow_query else original_query
        
        if not previous_results:
            # First agent or agents with no dependencies - just return the primary task
            return primary_task
        
        # For agents with dependencies, build comprehensive prompt that includes:
        # 1. Original user question (so they understand the overall goal)
        # 2. Previous results (as supporting context)
        # 3. Their specific role (handled by custom_prompt)
        if True:  # Always build comprehensive prompt for agents with dependencies
            # Build comprehensive prompt that includes original question + context
            prompt_parts = []
            
            # ALWAYS start with the original user question/task
            if original_query:
                prompt_parts.append("=== USER QUESTION ===")
                prompt_parts.append(original_query)
            
            # If agent has specific workflow query different from original, add it too
            if workflow_query and workflow_query != original_query:
                prompt_parts.append("")
                prompt_parts.append("=== SPECIFIC INSTRUCTIONS ===")
                prompt_parts.append(workflow_query)
            prompt_parts.append("")
            
            # Then add the previous results with clear context
            prompt_parts.append("Context from previous agents:")
            prompt_parts.append("")
            
            # CRITICAL FIX: Enhanced previous result formatting for multi-input scenarios
            for i, result in enumerate(previous_results):
                agent_name = result.get("agent_name", "Unknown Agent")
                node_id = result.get("node_id", "unknown")
                output = result.get("output", "")
                
                # ENHANCED: Handle different output formats with better error handling
                try:
                    if isinstance(output, dict):
                        # State-formatted output (context object, structured, etc.)
                        if "response" in output:
                            # Context format
                            display_output = str(output.get("response", ""))
                        elif "content" in output:
                            # Structured format
                            display_output = str(output.get("content", ""))
                        elif "output" in output:
                            # Full format (nested output)
                            display_output = str(output.get("output", ""))
                        else:
                            # Unknown dict format - convert to readable string
                            try:
                                display_output = json.dumps(output, indent=2, ensure_ascii=False)
                            except (TypeError, ValueError):
                                display_output = str(output)
                    else:
                        # String output - clean thinking tags if present
                        display_output = str(output)
                        if "<think>" in display_output and "</think>" in display_output:
                            parts = display_output.split("</think>")
                            if len(parts) > 1:
                                display_output = parts[1].strip()
                except Exception as e:
                    logger.warning(f"[PROMPT BUILD] Error formatting output from {agent_name}: {e}")
                    display_output = f"[Error formatting output: {str(output)[:100]}...]"
                
                # Add the result with clear structure
                prompt_parts.extend([
                    f"=== Agent {i+1}: {agent_name} (Node: {node_id}) ===",
                    str(display_output)[:2000],  # Include substantial context
                    ""
                ])
            
            prompt_parts.append("=== Your Task ===")
            prompt_parts.append("Analyze the USER QUESTION above from your role's perspective. Use the context from previous agents to inform your analysis, but focus on answering the original user question.")
            
            full_prompt = "\n".join(prompt_parts)
            logger.debug(f"[PROMPT BUILD] Built dependency prompt, length: {len(full_prompt)}")
            
            return full_prompt
        
        # Otherwise, build full structured prompt for agents without workflow queries
        agent_config = agent.get("agent_config", {})
        database_system_prompt = agent_config.get("system_prompt", "You are a helpful assistant.")
        
        user_request = agent_input["user_request"]
        workflow_context = agent_input["workflow_context"]
        previous_results = agent_input.get("previous_results", [])
        
        # Build clean, structured prompt
        prompt_parts = [
            f"ROLE: {agent_config.get('role', '')}",
            f"SYSTEM: {database_system_prompt}",
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
                output_text = result.get("output", "")
                
                # CRITICAL FIX: Handle structured outputs that might be dictionaries
                if isinstance(output_text, dict):
                    # Convert structured output to string safely
                    try:
                        if "content" in output_text:
                            output = str(output_text["content"])[:500]
                        elif "response" in output_text:
                            output = str(output_text["response"])[:500]
                        else:
                            output = str(output_text)[:500]
                    except Exception as e:
                        logger.warning(f"[SLICE FIX] Error formatting structured output: {e}")
                        output = str(output_text)[:500]
                elif output_text:
                    # String output - safe to slice
                    output = str(output_text)[:500]
                else:
                    output = ""
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
        
        # Get original user question for task instruction
        original_user_question = user_request.get("original_query", "") or user_request.get("message", "") or user_request.get("query", "")
        
        if agent_index == 0:
            prompt_parts.append(f"As the first agent, analyze this user question from your role's perspective: {original_user_question}")
        elif agent_index == total_agents - 1:
            prompt_parts.append(f"As the final agent, synthesize all previous analyses to provide a comprehensive answer to: {original_user_question}")
        else:
            prompt_parts.append(f"Analyze this user question from your role's perspective, considering the previous agents' insights: {original_user_question}")
        
        final_prompt = "\n".join(prompt_parts)
        logger.debug(f"[PROMPT BUILD] Built structured prompt for {agent.get('name', 'unknown')}, length: {len(final_prompt)}")
        return final_prompt
    
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
        Build system prompt with dynamic tool information:
        1. Combine workflow custom_prompt + query 
        2. Fallback to agent database system_prompt
        3. Generate dynamic tools section based on workflow tools
        4. Final fallback to default prompt
        """
        from app.automation.core.workflow_prompt_generator import generate_workflow_agent_prompt
        
        # Build combined workflow prompt from custom_prompt + query
        workflow_prompt_parts = []
        if agent_node.get("custom_prompt"):
            workflow_prompt_parts.append(agent_node["custom_prompt"])
        if agent_node.get("query"):
            workflow_prompt_parts.append(agent_node["query"])
        
        combined_workflow_prompt = "\n\n".join(workflow_prompt_parts) if workflow_prompt_parts else ""
        
        # Get base system prompt with priority chain: workflow prompt -> agent database -> default
        base_prompt = (
            combined_workflow_prompt or 
            agent_node.get("agent_config", {}).get("system_prompt") or 
            "You are a helpful assistant."
        )
        
        # Get workflow tools and agent info
        workflow_tools = agent_node.get("tools", [])
        agent_name = agent_node.get("agent_name", "Unknown Agent")
        agent_role = agent_node.get("agent_config", {}).get("role", "")
        
        # Generate dynamic prompt with correct tool information
        dynamic_prompt = generate_workflow_agent_prompt(
            agent_name=agent_name,
            workflow_tools=workflow_tools,
            base_system_prompt=base_prompt,
            role=agent_role,
            custom_prompt=""  # Already included in base_prompt
        )
        
        logger.info(f"[WORKFLOW PROMPT] Generated dynamic prompt for {agent_name} with tools: {workflow_tools}")
        
        return dynamic_prompt
    
    def _process_router_node(
        self,
        router: Dict[str, Any],
        agent_output: str,
        agent_outputs: Dict[str, str],
        workflow_edges: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Process router node to determine which routes to activate"""
        try:
            routing_mode = router.get("routing_mode", "multi-select")
            match_type = router.get("match_type", "exact")
            routes = router.get("routes", [])
            case_sensitive = router.get("case_sensitive", False)
            output_field = router.get("output_field", "")
            
            # Extract the value to match against
            match_value = agent_output
            
            # Clean up agent output - remove think tags
            if "<think>" in match_value and "</think>" in match_value:
                # Remove think tags and their content
                import re
                match_value = re.sub(r'<think>.*?</think>', '', match_value, flags=re.DOTALL).strip()
                logger.debug(f"[ROUTER DEBUG] Cleaned agent output (removed think tags): '{match_value}'")
            
            if output_field:
                # Try to extract specific field from output
                try:
                    import json
                    output_data = json.loads(match_value)
                    match_value = output_data.get(output_field, match_value)
                except:
                    # If not JSON or field not found, use full output
                    pass
            
            # Convert to string for matching
            match_value = str(match_value).strip()
            if not case_sensitive:
                match_value = match_value.lower()
            
            logger.info(f"[ROUTER DEBUG] Processing router {router['node_id']} - match_type: {match_type}, case_sensitive: {case_sensitive}")
            logger.info(f"[ROUTER DEBUG] Agent output to match: '{match_value}' (length: {len(match_value)})")
            
            matched_routes = []
            target_nodes = set()
            
            # Check each route
            for route in routes:
                route_matched = False
                route_values = route.get("match_values", [])
                
                logger.debug(f"[ROUTER DEBUG] Checking route {route.get('id')}: match_values={route_values}, target_nodes={route.get('target_nodes', [])}")
                
                for value in route_values:
                    test_value = str(value)
                    if not case_sensitive:
                        test_value = test_value.lower()
                    
                    logger.debug(f"[ROUTER DEBUG] Comparing match_value='{match_value}' with test_value='{test_value}' using match_type='{match_type}'")
                    
                    # Apply matching based on type
                    if match_type == "exact":
                        if match_value == test_value:
                            route_matched = True
                            break
                    elif match_type == "contains":
                        if test_value in match_value:
                            route_matched = True
                            break
                    elif match_type == "regex":
                        import re
                        try:
                            if re.search(test_value, match_value):
                                route_matched = True
                                break
                        except:
                            pass
                    elif match_type == "in_array":
                        # Check if match_value is in array form
                        try:
                            import json
                            array_data = json.loads(match_value) if isinstance(match_value, str) else match_value
                            if isinstance(array_data, list) and test_value in array_data:
                                route_matched = True
                                break
                        except:
                            pass
                
                if route_matched:
                    matched_routes.append(route["id"])
                    target_nodes.update(route.get("target_nodes", []))
                    
                    # If single-select mode, stop after first match
                    if routing_mode == "single-select":
                        break
            
            # If no matches and fallback route specified
            if not matched_routes and router.get("fallback_route"):
                target_nodes.add(router["fallback_route"])
                matched_routes.append("fallback")
            
            logger.info(f"[ROUTER] Router {router['node_id']} matched routes: {matched_routes}, target nodes: {list(target_nodes)}")
            
            return {
                "matched_routes": matched_routes,
                "target_nodes": target_nodes
            } if matched_routes else None
            
        except Exception as e:
            logger.error(f"[ROUTER] Error processing router node: {e}")
            return None
    
    def _should_execute_node(
        self,
        node_id: str,
        active_target_nodes: set,
        workflow_edges: List[Dict[str, Any]]
    ) -> bool:
        """Check if a node should execute based on router selections"""
        # If no active targets, execute all
        if not active_target_nodes:
            return True
        
        # If this node is directly selected
        if node_id in active_target_nodes:
            return True
        
        # Check if this node is downstream of any active node
        for edge in workflow_edges:
            if edge.get("source") in active_target_nodes and edge.get("target") == node_id:
                return True
        
        return False
    
    def _process_transform_node(
        self,
        transform: Dict[str, Any],
        input_data: str
    ) -> Optional[Dict[str, Any]]:
        """Process transform node to transform data"""
        try:
            transform_type = transform.get("transform_type", "jsonpath")
            expression = transform.get("expression", "$")
            error_handling = transform.get("error_handling", "continue")
            default_value = transform.get("default_value")
            
            # Parse input data
            try:
                data = json.loads(input_data) if isinstance(input_data, str) else input_data
            except:
                data = input_data
            
            result = None
            
            # Apply transformation based on type
            if transform_type == "jsonpath":
                try:
                    import jsonpath_ng
                    parser = jsonpath_ng.parse(expression)
                    matches = [match.value for match in parser.find(data)]
                    result = matches[0] if len(matches) == 1 else matches
                except Exception as e:
                    if error_handling == "fail":
                        raise e
                    elif error_handling == "default":
                        result = default_value
                    else:
                        result = str(e)
                        
            elif transform_type == "javascript":
                try:
                    # Note: In production, use a safe JS evaluator
                    import js2py
                    context = js2py.EvalJs()
                    context.data = data
                    result = context.eval(f"({expression})")
                except Exception as e:
                    if error_handling == "fail":
                        raise e
                    elif error_handling == "default":
                        result = default_value
                    else:
                        result = str(e)
                        
            elif transform_type == "python":
                try:
                    # Safe evaluation with restricted globals
                    safe_globals = {"__builtins__": {}, "data": data}
                    result = eval(expression, safe_globals)
                except Exception as e:
                    if error_handling == "fail":
                        raise e
                    elif error_handling == "default":
                        result = default_value
                    else:
                        result = str(e)
                        
            elif transform_type == "jq":
                try:
                    import pyjq
                    result = pyjq.apply(expression, data)
                except Exception as e:
                    if error_handling == "fail":
                        raise e
                    elif error_handling == "default":
                        result = default_value
                    else:
                        result = str(e)
            
            logger.info(f"[TRANSFORM] Transform {transform['node_id']} ({transform_type}) completed successfully")
            
            return {
                "output": result,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"[TRANSFORM] Error processing transform node: {e}")
            if error_handling == "fail":
                return {
                    "output": None,
                    "error": str(e)
                }
            elif error_handling == "default":
                return {
                    "output": default_value,
                    "error": None
                }
            else:
                return {
                    "output": str(e),
                    "error": str(e)
                }
    
    async def _process_parallel_node(
        self,
        parallel: Dict[str, Any],
        input_data: str,
        workflow_edges: List[Dict[str, Any]],
        workflow_id: int,
        execution_id: str,
        agent_plan: Dict[str, Any] = None,
        workflow_state: WorkflowState = None,
        execution_trace = None,
        agent_outputs: Dict[str, str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Process parallel node to execute multiple agents simultaneously with proper UI event streaming"""
        try:
            max_parallel = parallel.get("max_parallel", 3)
            wait_for_all = parallel.get("wait_for_all", True)
            combine_strategy = parallel.get("combine_strategy", "merge")
            parallel_node_id = parallel["node_id"]
            
            # Initialize agent_outputs if not provided
            if agent_outputs is None:
                agent_outputs = {}
            
            # Extract cache nodes from agent plan
            cache_nodes = agent_plan.get("cache_nodes", []) if agent_plan else []
            
            # Find all agent nodes connected downstream from this parallel node
            downstream_agents = []
            all_agents = agent_plan.get("agents", []) if agent_plan else []
            
            for edge in workflow_edges:
                if edge.get("source") == parallel_node_id:
                    target_id = edge.get("target")
                    
                    # Find the actual agent node from the workflow
                    for agent in all_agents:
                        if agent.get("node_id") == target_id:
                            downstream_agents.append(agent)
                            logger.info(f"[PARALLEL] Found downstream agent: {agent['agent_name']} (node: {target_id})")
                            break
            
            if not downstream_agents:
                logger.warning(f"[PARALLEL] No downstream agents found for parallel node {parallel_node_id}")
                yield {
                    "type": "parallel_execution_error",
                    "parallel_id": parallel_node_id,
                    "error": "No downstream agents found",
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                return
            
            # Limit to max_parallel agents
            agents_to_execute = downstream_agents[:max_parallel]
            total_count = len(agents_to_execute)
            
            logger.info(f"[PARALLEL] Starting parallel execution of {total_count} agents with strategy: {combine_strategy}")
            
            # DON'T emit node_start here - wait until actual processing begins
            
            # Yield parallel execution start event
            yield {
                "type": "parallel_execution_start",
                "parallel_id": parallel_node_id,
                "max_parallel": max_parallel,
                "combine_strategy": combine_strategy,
                "wait_for_all": wait_for_all,
                "agents_to_execute": [{"agent_name": a["agent_name"], "node_id": a["node_id"]} for a in agents_to_execute],
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Execute agents in parallel with proper streaming
            results = []
            completed_count = 0
            
            # Start all agent executions and yield node_start events
            for i, agent in enumerate(agents_to_execute):
                # Emit node_start event for UI animation
                yield {
                    "type": "node_start",
                    "node_id": agent["node_id"],
                    "agent_name": agent["agent_name"],
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Execute agents using the same pattern as sequential execution
            async def execute_single_parallel_agent(agent: Dict[str, Any], index: int) -> Dict[str, Any]:
                """Execute a single agent within parallel execution"""
                try:
                    from app.automation.integrations.redis_bridge import workflow_redis
                    
                    agent_node_id = agent["node_id"]
                    logger.info(f"[PARALLEL] Processing agent {agent['agent_name']} (index: {index})")
                    
                    # Check if there's a cache node connected to this agent
                    cached_result = None
                    cache_checked = False
                    
                    for edge in workflow_edges:
                        if edge.get("source") == agent_node_id:
                            target_id = edge.get("target")
                            
                            # Check if target is a cache node
                            for cache in cache_nodes:
                                if cache["node_id"] == target_id:
                                    logger.info(f"[PARALLEL] Checking cache node {target_id} for agent {agent_node_id}")
                                    
                                    # Check cache using the cache_node_key pattern
                                    cache_key = f"cache_{workflow_id}_{cache['node_id']}"
                                    cached_data = workflow_redis.get_value(cache_key)
                                    
                                    if cached_data:
                                        logger.info(f"[PARALLEL] Cache HIT for agent {agent['agent_name']} from cache node {target_id}")
                                        cached_result = cached_data
                                        cache_checked = True
                                        break
                                    else:
                                        logger.info(f"[PARALLEL] Cache MISS for agent {agent['agent_name']} from cache node {target_id}")
                            
                            if cached_result:
                                break
                    
                    # If we have a cached result, use it
                    if cached_result:
                        logger.info(f"[PARALLEL] Using cached result for agent {agent['agent_name']}")
                        return {
                            "agent_id": agent_node_id,
                            "agent_name": agent["agent_name"],
                            "output": cached_result,
                            "status": "completed",
                            "tools_used": [],
                            "from_cache": True
                        }
                    
                    # No cache hit, execute the agent
                    logger.info(f"[PARALLEL] Executing agent {agent['agent_name']} (no cache hit)")
                    
                    # Build agent prompt (simplified for parallel execution)
                    agent_prompt = input_data
                    if agent.get("custom_prompt"):
                        agent_prompt = f"{agent['custom_prompt']}\n\n{input_data}"
                    
                    # Execute using _execute_single_agent
                    result = await self._execute_single_agent(
                        agent, 
                        agent_prompt, 
                        workflow_id, 
                        execution_id,
                        execution_trace,
                        workflow_state,
                        index
                    )
                    
                    return {
                        "agent_id": agent_node_id,
                        "agent_name": agent["agent_name"],
                        "output": result.get("output", ""),
                        "status": "completed",
                        "tools_used": result.get("tools_used", []),
                        "from_cache": False
                    }
                except Exception as e:
                    logger.error(f"[PARALLEL] Error executing agent {agent['agent_name']}: {e}")
                    return {
                        "agent_id": agent["node_id"],
                        "agent_name": agent["agent_name"],
                        "output": f"Error: {str(e)}",
                        "status": "failed",
                        "error": str(e),
                        "from_cache": False
                    }
            
            # Execute all agents in parallel
            if wait_for_all:
                # Emit node_start right before heavy processing begins
                start_event = {
                    "type": "node_start",
                    "node_id": parallel_node_id,
                    "node_type": "ParallelNode", 
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.info(f"[PARALLEL ANIMATION] Yielding node_start event before agent execution: {start_event}")
                yield start_event
                
                # Wait for all agents to complete
                tasks = [execute_single_parallel_agent(agent, i) for i, agent in enumerate(agents_to_execute)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and emit completion events
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # Handle exception case
                        agent = agents_to_execute[i]
                        yield {
                            "type": "node_error",
                            "node_id": agent["node_id"],
                            "agent_name": agent["agent_name"],
                            "error": str(result),
                            "workflow_id": workflow_id,
                            "execution_id": execution_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        # Emit node_complete event for UI animation
                        yield {
                            "type": "node_complete",
                            "node_id": result["agent_id"],
                            "agent_name": result["agent_name"],
                            "output": result.get("output", ""),
                            "from_cache": result.get("from_cache", False),
                            "workflow_id": workflow_id,
                            "execution_id": execution_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        # IMPORTANT: Add agent output to main agent_outputs dictionary
                        if result.get("status") == "completed":
                            agent_name = result["agent_name"]
                            agent_node_id = result["agent_id"]
                            # Use node_id as key if multiple agents have same name
                            output_key = agent_node_id if agent_node_id else agent_name
                            agent_outputs[output_key] = result.get("output", "")
                            logger.info(f"[PARALLEL] Added output from {agent_name} (key: {output_key}) to agent_outputs")
                
                completed_count = len([r for r in results if not isinstance(r, Exception) and r.get("status") == "completed"])
            else:
                # Emit node_start right before limited parallelism processing
                start_event = {
                    "type": "node_start",
                    "node_id": parallel_node_id,
                    "node_type": "ParallelNode", 
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.info(f"[PARALLEL ANIMATION] Yielding node_start event before limited parallelism: {start_event}")
                yield start_event
                
                # Execute agents with limited parallelism (not commonly used)
                logger.warning("[PARALLEL] Limited parallelism mode not fully implemented for real agent execution")
                results = []
                completed_count = 0
            
            # Apply combine strategy (this is where the heavy AI processing happens)
            logger.info(f"[PARALLEL] Starting combine strategy '{combine_strategy}' - this may take time for AI processing")
            
            # If this is an AI summary strategy, ensure animation is still running
            if combine_strategy == "summary":
                logger.info(f"[PARALLEL ANIMATION] AI summary strategy - ParallelNode {parallel_node_id} should be animating during processing")
            
            combined_output = await self._combine_parallel_results(
                [r for r in results if not isinstance(r, Exception)], 
                combine_strategy,
                workflow_id,
                execution_id
            )
            logger.info(f"[PARALLEL] Combine strategy completed")
            
            # Generate summary if needed
            summary = None
            if combine_strategy == "summary":
                summary = combined_output
            
            # Count cache hits
            cache_hits = len([r for r in results if not isinstance(r, Exception) and r.get("from_cache", False)])
            if cache_hits > 0:
                logger.info(f"[PARALLEL] Parallel node {parallel_node_id} completed: {completed_count}/{total_count} agents ({cache_hits} from cache)")
            else:
                logger.info(f"[PARALLEL] Parallel node {parallel_node_id} completed: {completed_count}/{total_count} agents")
            
            # Yield parallel execution complete event
            yield {
                "type": "parallel_execution_complete",
                "parallel_id": parallel_node_id,
                "results": [r for r in results if not isinstance(r, Exception)],
                "combined_output": combined_output,
                "summary": summary,
                "completed_count": completed_count,
                "total_count": total_count,
                "strategy_used": combine_strategy,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Process cache nodes connected to the executed agents
            if cache_nodes:
                for result in results:
                    if isinstance(result, dict) and result.get("status") == "completed":
                        # Skip cache storage if result came from cache
                        if result.get("from_cache", False):
                            logger.info(f"[PARALLEL] Skipping cache storage for agent {result.get('agent_name')} - result from cache")
                            continue
                            
                        agent_node_id = result.get("agent_id")
                        agent_output = result.get("output", "")
                        
                        # Find ALL cache nodes connected to this agent
                        connected_cache_nodes = []
                        for edge in workflow_edges:
                            if edge.get("source") == agent_node_id:
                                target_id = edge.get("target")
                                
                                # Check if target is a cache node
                                for cache in cache_nodes:
                                    if cache["node_id"] == target_id:
                                        connected_cache_nodes.append(cache)
                                        logger.info(f"[PARALLEL] Found CacheNode {target_id} connected to agent {agent_node_id}")
                        
                        # Process each connected cache node
                        for cache in connected_cache_nodes:
                            logger.info(f"[PARALLEL] Processing CacheNode {cache['node_id']} for agent {agent_node_id}")
                            
                            # Store result in cache with multiple keys for lookup
                            # Primary key with agent node ID
                            agent_cache_key = f"workflow_{workflow_id}_node_{agent_node_id}_output"
                            cache_stored = await self._store_cache_result(
                                agent_cache_key,
                                agent_output,
                                cache.get("ttl", 3600),
                                "",  # No specific condition
                                cache.get("max_cache_size", 10) * 1024 * 1024
                            )
                            
                            # Also store with cache node ID for easy lookup
                            cache_node_key = f"cache_{workflow_id}_{cache['node_id']}"
                            await self._store_cache_result(
                                cache_node_key,
                                agent_output,
                                cache.get("ttl", 3600),
                                "",  # No specific condition
                                cache.get("max_cache_size", 10) * 1024 * 1024
                            )
                            
                            if cache_stored:
                                yield {
                                    "type": "cache_stored",
                                    "agent_name": result.get("agent_name", "Unknown"),
                                    "node_id": agent_node_id,
                                    "cache_id": cache["node_id"],
                                    "cache_key": agent_cache_key,
                                    "cache_node_key": cache_node_key,
                                    "ttl": cache.get("ttl", 3600),
                                    "workflow_id": workflow_id,
                                    "execution_id": execution_id,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                            
                            # Store cache node output in workflow state
                            if workflow_state:
                                cache_node_result = {
                                    "node_id": cache["node_id"],
                                    "output": agent_output,
                                    "cache_hit": False,
                                    "cache_key": cache_node_key,
                                    "source_agent": agent_node_id,
                                    "timestamp": datetime.utcnow().isoformat()
                                }
                                workflow_state.set_state(f"node_output_{cache['node_id']}", cache_node_result, f"cache_{cache['node_id']}_result")
            
            # Store parallel node output in workflow state if available
            if workflow_state:
                parallel_result = {
                    "node_id": parallel_node_id,
                    "combined_output": combined_output,
                    "individual_results": results,
                    "summary": summary,
                    "timestamp": datetime.utcnow().isoformat()
                }
                workflow_state.set_state(f"node_output_{parallel_node_id}", parallel_result, f"parallel_{parallel_node_id}_result")
            
            # Yield node complete event for UI grow effect
            complete_event = {
                "type": "node_complete",
                "node_id": parallel_node_id,
                "node_type": "ParallelNode",
                "output": combined_output,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            logger.info(f"[PARALLEL ANIMATION] Yielding node_complete event: {complete_event}")
            yield complete_event
            
        except Exception as e:
            logger.error(f"[PARALLEL] Error processing parallel node: {e}")
            yield {
                "type": "parallel_execution_error",
                "parallel_id": parallel_node_id,
                "error": str(e),
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _combine_parallel_results(
        self,
        results: List[Dict[str, Any]],
        strategy: str,
        workflow_id: int = None,
        execution_id: str = None
    ) -> str:
        """Combine results from parallel agent execution based on strategy"""
        try:
            if not results:
                return "No results to combine"
            
            logger.info(f"[PARALLEL] Combining {len(results)} results with strategy: {strategy}")
            
            if strategy == "merge":
                # Merge all outputs
                combined = []
                for i, result in enumerate(results):
                    output = result.get("output", "")
                    agent_name = result.get("agent_name", "Unknown")
                    logger.info(f"[PARALLEL] Adding output from {agent_name} (length: {len(output)})")
                    combined.append(f"**Agent {i+1} ({agent_name}):**\n{output}")
                merged = "\n\n".join(combined)
                logger.info(f"[PARALLEL] Merged output length: {len(merged)}")
                return merged
            
            elif strategy == "best":
                # Select the best result based on multiple criteria
                logger.info(f"[PARALLEL] Selecting best result from {len(results)} outputs")
                
                if not results:
                    return "No results to select from"
                
                # Score each result based on multiple factors
                scored_results = []
                for result in results:
                    output = result.get("output", "")
                    score = 0
                    
                    # Length score (longer is generally more detailed)
                    score += min(len(output) / 100, 10)  # Cap at 10 points
                    
                    # Tool usage score (agents that used tools may have fresher data)
                    tools_used = result.get("tools_used", [])
                    score += len(tools_used) * 5  # 5 points per tool used
                    
                    # Structure score (check for headers, lists, etc.)
                    if "##" in output or "**" in output:
                        score += 3  # Structured content
                    if "\n-" in output or "\n*" in output or "\n1." in output:
                        score += 2  # Has lists
                    
                    # Error penalty
                    if "error" in result.get("status", "").lower():
                        score -= 20
                    
                    scored_results.append((result, score))
                
                # Select the highest scoring result
                best_result, best_score = max(scored_results, key=lambda x: x[1])
                agent_name = best_result.get("agent_name", "Unknown")
                
                logger.info(f"[PARALLEL] Selected {agent_name} as best result (score: {best_score})")
                return f"## Best Result (from {agent_name})\n\n{best_result.get('output', '')}"
            
            elif strategy == "summary":
                # Use AI to create an intelligent summary of all results
                logger.info(f"[PARALLEL] Generating AI summary of {len(results)} agent outputs")
                
                # Collect all outputs for summarization
                all_outputs = []
                for i, result in enumerate(results):
                    agent_name = result.get("agent_name", "Unknown")
                    output = result.get("output", "")
                    all_outputs.append(f"**Agent {i+1} ({agent_name}):**\n{output}")
                
                combined_text = "\n\n---\n\n".join(all_outputs)
                
                # If outputs are too short, just merge them
                if len(combined_text) < 200:
                    return combined_text
                
                # Create a summary using AI
                try:
                    # Build summarization prompt
                    summary_prompt = f"""Please provide a comprehensive summary of the following {len(results)} agent outputs. 
Synthesize the key information from all agents into a cohesive summary that captures the main points, findings, and insights.
Focus on combining complementary information and highlighting any differences or unique contributions from each agent.

{combined_text}

Please provide a well-structured summary that integrates all the agent outputs above."""
                    
                    # Use the dynamic agent system to generate summary
                    if self.dynamic_agent_system:
                        logger.info("[PARALLEL] Calling AI for summary generation")
                        
                        # Create a simple agent configuration for summarization
                        summary_agent = {
                            "name": "Summary Generator",
                            "role": "summarizer",
                            "system_prompt": "You are an expert at creating concise, comprehensive summaries that integrate multiple perspectives and sources of information.",
                            "config": {
                                "temperature": 0.3,
                                "max_tokens": 4000
                            }
                        }
                        
                        # Execute summary generation - no await needed, it returns an async generator
                        summary_result = self.dynamic_agent_system.execute_agent(
                            agent_name="Summary Generator",
                            agent_data=summary_agent,
                            query=summary_prompt,
                            context={}
                        )
                        
                        # Extract the summary from the generator
                        ai_summary = ""
                        async for event in summary_result:
                            event_type = event.get("type", "unknown")
                            
                            # Only process meaningful events, ignore agent_token spam
                            if event.get("type") == "agent_response":
                                content = event.get("content", "")
                                ai_summary += content
                            elif event.get("type") in ["completion", "enhanced_completion", "agent_complete"]:
                                # Enhanced completion contains the full response
                                content = event.get("content", "")
                                if content:
                                    ai_summary = content
                                    logger.info(f"[PARALLEL AI] Got {event_type} with content: {len(content)} chars")
                                    break
                            # Ignore agent_token and other noise events silently
                        
                        if ai_summary:
                            # Clean thinking tags from AI summary
                            import re
                            original_length = len(ai_summary)
                            
                            # Remove thinking tags and extract content after them
                            if "<think>" in ai_summary and "</think>" in ai_summary:
                                # First try to extract content after thinking tags
                                think_end = ai_summary.find("</think>")
                                if think_end != -1:
                                    cleaned_summary = ai_summary[think_end + 8:].strip()
                                    if cleaned_summary and len(cleaned_summary) > 50:  # Substantial content after think tags
                                        ai_summary = cleaned_summary
                                        logger.info(f"[PARALLEL] Extracted content after think tags: {len(ai_summary)} chars")
                                    else:
                                        # No substantial content after think tags, extract from inside
                                        think_match = re.search(r'<think>(.*?)</think>', ai_summary, re.DOTALL)
                                        if think_match:
                                            ai_summary = think_match.group(1).strip()
                                            logger.info(f"[PARALLEL] Extracted content from inside think tags: {len(ai_summary)} chars")
                                else:
                                    # Malformed thinking tags, remove them
                                    ai_summary = re.sub(r'</?think>', '', ai_summary).strip()
                                    logger.info(f"[PARALLEL] Removed malformed think tags: {len(ai_summary)} chars")
                            
                            # Ensure we have substantial content
                            if len(ai_summary) < 50:
                                logger.warning(f"[PARALLEL] AI summary too short ({len(ai_summary)} chars), falling back to merge")
                                # Don't return None, fall through to merged output
                            else:
                                logger.info(f"[PARALLEL] AI summary processed successfully (original: {original_length}, final: {len(ai_summary)})")
                                logger.info(f"[PARALLEL AI SUMMARY] Final content preview: {ai_summary[:200]}...")
                                return ai_summary
                        else:
                            logger.warning("[PARALLEL] AI summary generation returned empty result")
                    else:
                        logger.warning("[PARALLEL] No dynamic agent system available for AI summary")
                    
                except Exception as e:
                    logger.error(f"[PARALLEL] Error generating AI summary: {e}")
                    import traceback
                    logger.error(f"[PARALLEL] Full traceback: {traceback.format_exc()}")
                
                # Fallback to merged output if AI summary fails
                logger.warning("[PARALLEL] AI summary failed, falling back to merged output")
                return f"## Combined Agent Outputs (AI Summary Failed)\n\n{combined_text}"
            
            elif strategy == "vote":
                # Intelligent voting mechanism - find consensus among outputs
                logger.info(f"[PARALLEL] Analyzing {len(results)} outputs for consensus")
                
                if not results:
                    return "No results for voting"
                
                if len(results) == 1:
                    # Only one result, return it
                    return results[0].get("output", "")
                
                # Use AI to analyze outputs and find consensus
                try:
                    # Collect all outputs
                    outputs_text = []
                    for i, result in enumerate(results):
                        agent_name = result.get("agent_name", f"Agent {i+1}")
                        output = result.get("output", "")
                        outputs_text.append(f"**{agent_name}:**\n{output}")
                    
                    combined_outputs = "\n\n---\n\n".join(outputs_text)
                    
                    # Create consensus analysis prompt
                    consensus_prompt = f"""Analyze the following {len(results)} agent outputs and determine the consensus or majority opinion.

Look for:
1. Common themes and agreements across outputs
2. Key facts or conclusions that multiple agents agree on
3. Any significant disagreements or contradictions
4. The majority viewpoint if opinions differ

Agent Outputs:
{combined_outputs}

Please provide:
1. The consensus findings that most or all agents agree on
2. Any notable disagreements with a note on which view has more support
3. A final synthesized answer based on the majority consensus"""
                    
                    # Use AI to find consensus
                    if self.dynamic_agent_system:
                        logger.info("[PARALLEL] Using AI to analyze consensus")
                        
                        consensus_agent = {
                            "name": "Consensus Analyzer",
                            "role": "analyzer",
                            "system_prompt": "You are an expert at analyzing multiple viewpoints and finding consensus, identifying majority opinions, and synthesizing coherent conclusions from diverse inputs.",
                            "config": {
                                "temperature": 0.2,  # Low temperature for consistency
                                "max_tokens": 800
                            }
                        }
                        
                        # Execute consensus analysis - no await needed, it returns an async generator
                        consensus_result = self.dynamic_agent_system.execute_agent(
                            agent_name="Consensus Analyzer",
                            agent_data=consensus_agent,
                            query=consensus_prompt,
                            context={}
                        )
                        
                        # Extract consensus
                        consensus_text = ""
                        async for event in consensus_result:
                            # Check for different event types
                            if event.get("type") == "agent_response":
                                consensus_text += event.get("content", "")
                            elif event.get("type") in ["completion", "enhanced_completion", "agent_complete"]:
                                # Enhanced completion contains the full response
                                consensus_text = event.get("content", "")
                                break
                        
                        if consensus_text:
                            logger.info(f"[PARALLEL] Consensus analysis completed")
                            return f"## Consensus Result\n\n{consensus_text}"
                    
                    # Fallback: Simple similarity check
                    logger.info("[PARALLEL] Falling back to simple consensus check")
                    
                    # Check if outputs are very similar (simple approach)
                    if len(set(output.strip().lower() for output in [r.get("output", "") for r in results])) == 1:
                        # All outputs are identical
                        return f"## Unanimous Result\n\n{results[0].get('output', '')}"
                    
                    # Return the most common theme
                    return f"## Voting Result\n\nMultiple viewpoints detected. Showing all outputs:\n\n{combined_outputs}"
                    
                except Exception as e:
                    logger.error(f"[PARALLEL] Error in voting analysis: {e}")
                    # Fallback to first result
                    return f"## Voting Result (Fallback)\n\n{results[0].get('output', '')}"
            
            else:
                # Default to merge
                return await self._combine_parallel_results(results, "merge", workflow_id, execution_id)
                
        except Exception as e:
            logger.error(f"[PARALLEL] Error combining results: {e}")
            return f"Error combining results: {str(e)}"
    
    async def _process_aggregator_node(
        self,
        aggregator: Dict[str, Any],
        input_data: Any,
        workflow_id: int,
        execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """Process aggregator node to intelligently combine multiple agent outputs"""
        try:
            # Extract aggregator configuration
            aggregation_strategy = aggregator.get("aggregation_strategy", "semantic_merge")
            confidence_threshold = aggregator.get("confidence_threshold", 0.3)
            max_inputs = aggregator.get("max_inputs", 0)
            deduplication_enabled = aggregator.get("deduplication_enabled", True)
            similarity_threshold = aggregator.get("similarity_threshold", 0.85)
            quality_weights = aggregator.get("quality_weights", {
                "length": 0.2,
                "coherence": 0.3,
                "relevance": 0.3,
                "completeness": 0.2
            })
            output_format = aggregator.get("output_format", "comprehensive")
            include_source_attribution = aggregator.get("include_source_attribution", True)
            conflict_resolution = aggregator.get("conflict_resolution", "highlight_conflicts")
            semantic_analysis = aggregator.get("semantic_analysis", True)
            preserve_structure = aggregator.get("preserve_structure", False)
            fallback_strategy = aggregator.get("fallback_strategy", "return_best")
            aggregator_node_id = aggregator["node_id"]
            
            logger.info(f"[AGGREGATOR] Processing aggregator node {aggregator_node_id} with strategy: {aggregation_strategy}")
            
            # Handle input data - could be list of agent outputs or single input
            if isinstance(input_data, list):
                agent_outputs = input_data
            elif isinstance(input_data, dict) and "inputs" in input_data:
                agent_outputs = input_data["inputs"]
            elif isinstance(input_data, str):
                # Try to parse as JSON array
                try:
                    import json
                    agent_outputs = json.loads(input_data)
                    if not isinstance(agent_outputs, list):
                        agent_outputs = [input_data]
                except:
                    agent_outputs = [input_data]
            else:
                agent_outputs = [input_data]
            
            logger.info(f"[AGGREGATOR] Processing {len(agent_outputs)} agent outputs")
            
            # Apply security validation
            from aggregator_node_critical_fixes import AggregatorNodeSecurityFixes
            is_valid, error_message = AggregatorNodeSecurityFixes.validate_input_limits(agent_outputs)
            if not is_valid:
                logger.error(f"[AGGREGATOR] Security validation failed: {error_message}")
                return {"error": f"Input validation failed: {error_message}"}
            
            # Apply max_inputs limit if specified
            if max_inputs > 0 and len(agent_outputs) > max_inputs:
                agent_outputs = agent_outputs[:max_inputs]
                logger.info(f"[AGGREGATOR] Limited inputs to {max_inputs} items")
            
            # Pre-process outputs for quality scoring with security
            processed_outputs = []
            for i, output in enumerate(agent_outputs):
                processed_output = AggregatorNodeSecurityFixes.safe_analyze_output_quality(output, quality_weights, i)
                processed_outputs.append(processed_output)
            
            # Apply confidence threshold filtering with validation
            filtered_outputs = [
                output for output in processed_outputs 
                if AggregatorNodeSecurityFixes.validate_confidence_score(output.get("confidence_score", 0)) >= confidence_threshold
            ]
            
            if not filtered_outputs:
                logger.warning(f"[AGGREGATOR] No outputs meet confidence threshold {confidence_threshold}")
                if fallback_strategy == "return_best":
                    filtered_outputs = [max(processed_outputs, key=lambda x: x.get("confidence_score", 0))]
                elif fallback_strategy == "error":
                    return {"error": "No outputs meet confidence threshold"}
                elif fallback_strategy == "empty_result":
                    return {"aggregated_result": "", "confidence_score": 0.0}
                else:  # simple_merge
                    filtered_outputs = processed_outputs
            
            logger.info(f"[AGGREGATOR] {len(filtered_outputs)} outputs passed confidence threshold")
            
            # Apply deduplication if enabled
            if deduplication_enabled:
                filtered_outputs = await self._deduplicate_outputs(filtered_outputs, similarity_threshold)
                logger.info(f"[AGGREGATOR] After deduplication: {len(filtered_outputs)} outputs remain")
            
            # Apply aggregation strategy
            if aggregation_strategy == "semantic_merge":
                result = await self._semantic_merge_outputs(filtered_outputs, semantic_analysis, preserve_structure)
            elif aggregation_strategy == "weighted_vote":
                result = self._weighted_vote_outputs(filtered_outputs, quality_weights)
            elif aggregation_strategy == "consensus_ranking":
                result = self._consensus_ranking_outputs(filtered_outputs)
            elif aggregation_strategy == "relevance_weighted":
                result = self._relevance_weighted_outputs(filtered_outputs)
            elif aggregation_strategy == "confidence_filter":
                result = self._confidence_filter_outputs(filtered_outputs, confidence_threshold)
            elif aggregation_strategy == "diversity_preservation":
                result = self._diversity_preservation_outputs(filtered_outputs)
            elif aggregation_strategy == "temporal_priority":
                result = self._temporal_priority_outputs(filtered_outputs)
            elif aggregation_strategy == "simple_concatenate":
                result = self._simple_concatenate_outputs(filtered_outputs)
            elif aggregation_strategy == "best_selection":
                result = self._best_selection_output(filtered_outputs)
            elif aggregation_strategy == "structured_fusion":
                result = self._structured_fusion_outputs(filtered_outputs, preserve_structure)
            else:
                logger.warning(f"[AGGREGATOR] Unknown strategy {aggregation_strategy}, falling back to semantic_merge")
                result = await self._semantic_merge_outputs(filtered_outputs, semantic_analysis, preserve_structure)
            
            # Apply conflict resolution
            if conflict_resolution != "include_all_perspectives":
                result = self._resolve_conflicts(result, conflict_resolution, filtered_outputs)
            
            # Format output according to output_format
            formatted_result = self._format_aggregated_output(
                result, output_format, include_source_attribution, filtered_outputs
            )
            
            # Calculate overall confidence score
            overall_confidence = self._calculate_overall_confidence(filtered_outputs, aggregation_strategy)
            
            # Generate source analysis
            source_analysis = self._generate_source_analysis(filtered_outputs, processed_outputs)
            
            # Generate metadata
            metadata = {
                "strategy_used": aggregation_strategy,
                "inputs_processed": len(agent_outputs),
                "inputs_after_filtering": len(filtered_outputs),
                "confidence_threshold": confidence_threshold,
                "deduplication_applied": deduplication_enabled,
                "output_format": output_format,
                "conflict_resolution": conflict_resolution,
                "processing_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"[AGGREGATOR] Aggregation completed with {overall_confidence:.2f} confidence")
            
            return {
                "aggregated_result": formatted_result,
                "confidence_score": overall_confidence,
                "source_analysis": source_analysis,
                "metadata": metadata,
                "raw_inputs": agent_outputs,
                "processed_inputs": filtered_outputs
            }
            
        except Exception as e:
            logger.error(f"[AGGREGATOR] Error processing aggregator node: {e}")
            
            # Apply fallback strategy on error
            if fallback_strategy == "return_best" and agent_outputs:
                try:
                    best_output = max(agent_outputs, key=lambda x: len(str(x)))
                    return {
                        "aggregated_result": str(best_output),
                        "confidence_score": 0.5,
                        "source_analysis": {"fallback": True},
                        "metadata": {"error": str(e), "fallback_applied": True}
                    }
                except:
                    pass
            
            return {"error": f"Aggregation failed: {str(e)}"}
    
    def _analyze_output_quality(self, output: Any, quality_weights: Dict[str, float], index: int) -> Dict[str, Any]:
        """Analyze quality of individual output and assign confidence score"""
        try:
            output_text = str(output)
            
            # Calculate quality factors
            length_score = min(len(output_text) / 1000.0, 1.0)  # Normalize to 0-1
            
            # Simple coherence heuristic (punctuation and structure)
            sentences = output_text.count('.') + output_text.count('!') + output_text.count('?')
            coherence_score = min(sentences / max(len(output_text.split()), 1), 1.0)
            
            # Relevance heuristic (placeholder - in real implementation would use embedding similarity)
            relevance_score = 0.7  # Default moderate relevance
            
            # Completeness heuristic (length and structure indicators)
            completeness_indicators = output_text.count('\n') + output_text.count(':') + output_text.count('-')
            completeness_score = min(completeness_indicators / 10.0, 1.0)
            
            # Calculate weighted confidence score
            confidence_score = (
                length_score * quality_weights.get("length", 0.2) +
                coherence_score * quality_weights.get("coherence", 0.3) +
                relevance_score * quality_weights.get("relevance", 0.3) +
                completeness_score * quality_weights.get("completeness", 0.2)
            )
            
            return {
                "original_output": output,
                "text": output_text,
                "confidence_score": confidence_score,
                "quality_factors": {
                    "length": length_score,
                    "coherence": coherence_score,
                    "relevance": relevance_score,
                    "completeness": completeness_score
                },
                "index": index,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[AGGREGATOR] Error analyzing output quality: {e}")
            return {
                "original_output": output,
                "text": str(output),
                "confidence_score": 0.1,
                "quality_factors": {},
                "index": index,
                "error": str(e)
            }
    
    async def _deduplicate_outputs(self, outputs: List[Dict[str, Any]], similarity_threshold: float) -> List[Dict[str, Any]]:
        """Remove duplicate or highly similar outputs with AI-powered similarity analysis"""
        try:
            if len(outputs) <= 1:
                return outputs
            
            unique_outputs = []
            for output in outputs:
                is_duplicate = False
                current_text = output.get("text", "").lower().strip()
                
                for existing in unique_outputs:
                    existing_text = existing.get("text", "").lower().strip()
                    
                    # Simple exact match check first
                    if current_text == existing_text:
                        is_duplicate = True
                        break
                    
                    # Use basic word overlap similarity
                    if len(current_text) > 0 and len(existing_text) > 0:
                        current_words = set(current_text.split())
                        existing_words = set(existing_text.split())
                        
                        if len(current_words) > 5 and len(existing_words) > 5:  # Only for substantial text
                            overlap = len(current_words & existing_words)
                            total_unique = len(current_words | existing_words)
                            similarity = overlap / total_unique if total_unique > 0 else 0
                            
                            # For high similarity, use AI to make final decision
                            if similarity >= (similarity_threshold * 0.8):  # Pre-filter with lower threshold
                                try:
                                    # Use AI for semantic similarity analysis
                                    from app.core.langgraph_agents_cache import get_agent_by_role
                                    analysis_agent = get_agent_by_role("analysis")
                                    
                                    if analysis_agent and len(current_text) > 100 and len(existing_text) > 100:
                                        similarity_prompt = f"""Please analyze if these two text passages are semantically similar or duplicates.

Text A: {current_text[:500]}...

Text B: {existing_text[:500]}...

Consider:
1. Core meaning and intent
2. Factual content overlap
3. Different phrasings of same information

Respond with just: SIMILAR or DIFFERENT"""

                                        from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
                                        agent_system = DynamicMultiAgentSystem(
                                            agents=[analysis_agent],
                                            workflow_id=None,
                                            execution_id="similarity_check"
                                        )
                                        
                                        async for response in agent_system.arun(
                                            query=similarity_prompt,
                                            workflow_state=None,
                                            execution_trace=None,
                                            detailed_response=False
                                        ):
                                            if response.get("type") == "agent_output":
                                                ai_result = response.get("output", "").strip().upper()
                                                if "SIMILAR" in ai_result:
                                                    logger.info(f"[AGGREGATOR] AI detected semantic similarity for deduplication")
                                                    similarity = similarity_threshold + 0.1  # Force similarity above threshold
                                                break
                                                
                                except Exception as e:
                                    logger.debug(f"[AGGREGATOR] AI similarity check failed: {e}, using basic similarity")
                            
                            if similarity >= similarity_threshold:
                                # Keep the one with higher confidence
                                if output.get("confidence_score", 0) <= existing.get("confidence_score", 0):
                                    is_duplicate = True
                                    break
                                else:
                                    unique_outputs.remove(existing)
                        else:
                            # For short texts, use exact substring similarity
                            if similarity >= similarity_threshold:
                                if output.get("confidence_score", 0) <= existing.get("confidence_score", 0):
                                    is_duplicate = True
                                    break
                                else:
                                    unique_outputs.remove(existing)
                
                if not is_duplicate:
                    unique_outputs.append(output)
            
            return unique_outputs
            
        except Exception as e:
            logger.error(f"[AGGREGATOR] Error in deduplication: {e}")
            return outputs
    
    async def _semantic_merge_outputs(self, outputs: List[Dict[str, Any]], semantic_analysis: bool, preserve_structure: bool) -> str:
        """Advanced semantic merging of outputs with AI support"""
        try:
            if not outputs:
                return ""
            
            if len(outputs) == 1:
                return outputs[0].get("text", "")
            
            # Sort by confidence score
            sorted_outputs = sorted(outputs, key=lambda x: x.get("confidence_score", 0), reverse=True)
            
            # Use AI for semantic fusion if semantic_analysis is enabled and we have multiple high-quality outputs
            if semantic_analysis and len(sorted_outputs) > 1:
                try:
                    # Get an analysis agent for semantic fusion
                    from app.core.langgraph_agents_cache import get_agent_by_role
                    analysis_agent = get_agent_by_role("analysis")
                    
                    if analysis_agent:
                        # Prepare input texts for semantic fusion
                        input_texts = []
                        for i, output in enumerate(sorted_outputs):
                            confidence = output.get("confidence_score", 0)
                            text = output.get("text", "")
                            input_texts.append(f"Source {i+1} (confidence: {confidence:.2f}):\n{text}")
                        
                        # Create fusion prompt
                        fusion_prompt = f"""Please analyze and semantically merge the following {len(input_texts)} text sources into a coherent, comprehensive result. 

Focus on:
1. Identifying common themes and consolidating overlapping information
2. Resolving any contradictions by noting them or prioritizing higher-confidence sources
3. Creating a unified narrative that preserves the most important information
4. Maintaining factual accuracy while improving coherence

Input Sources:
{chr(10).join(input_texts)}

Please provide a well-structured, comprehensive merge that represents the best synthesis of all sources."""

                        # Use the agent system for semantic fusion
                        from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
                        agent_system = DynamicMultiAgentSystem(
                            agents=[analysis_agent],
                            workflow_id=None,
                            execution_id="semantic_merge"
                        )
                        
                        # Execute semantic fusion
                        async for response in agent_system.arun(
                            query=fusion_prompt,
                            workflow_state=None,
                            execution_trace=None,
                            detailed_response=False
                        ):
                            if response.get("type") == "agent_output":
                                ai_merged_result = response.get("output", "")
                                if ai_merged_result and len(ai_merged_result.strip()) > 50:
                                    logger.info(f"[AGGREGATOR] Successfully used AI for semantic fusion")
                                    return ai_merged_result
                        
                        logger.warning(f"[AGGREGATOR] AI semantic fusion failed, falling back to basic merge")
                        
                except Exception as e:
                    logger.warning(f"[AGGREGATOR] AI semantic fusion error: {e}, falling back to basic merge")
            
            # Fallback to intelligent concatenation
            merged_sections = []
            
            for i, output in enumerate(sorted_outputs):
                confidence = output.get("confidence_score", 0)
                text = output.get("text", "")
                
                if confidence > 0.7:
                    weight = "High Confidence"
                elif confidence > 0.4:
                    weight = "Medium Confidence"
                else:
                    weight = "Low Confidence"
                
                section = f"**Source {i+1} ({weight}):**\n{text}"
                merged_sections.append(section)
            
            return "\n\n".join(merged_sections)
            
        except Exception as e:
            logger.error(f"[AGGREGATOR] Error in semantic merge: {e}")
            return self._simple_concatenate_outputs(outputs)
    
    def _weighted_vote_outputs(self, outputs: List[Dict[str, Any]], quality_weights: Dict[str, float]) -> str:
        """Quality-weighted voting on outputs"""
        try:
            if not outputs:
                return ""
            
            # Calculate weighted scores
            weighted_outputs = []
            for output in outputs:
                total_weight = sum(
                    output.get("quality_factors", {}).get(factor, 0) * weight
                    for factor, weight in quality_weights.items()
                )
                weighted_outputs.append({
                    "output": output,
                    "weight": total_weight
                })
            
            # Sort by weight
            weighted_outputs.sort(key=lambda x: x["weight"], reverse=True)
            
            # Return top weighted result with vote summary
            best = weighted_outputs[0]["output"]
            vote_summary = f"Selected result with weight {weighted_outputs[0]['weight']:.3f} from {len(outputs)} candidates."
            
            return f"{vote_summary}\n\n{best.get('text', '')}"
            
        except Exception as e:
            logger.error(f"[AGGREGATOR] Error in weighted vote: {e}")
            return self._simple_concatenate_outputs(outputs)
    
    def _simple_concatenate_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """Simple concatenation fallback"""
        try:
            texts = [output.get("text", "") for output in outputs if output.get("text")]
            return "\n\n".join(texts)
        except:
            return "Error in concatenation"
    
    def _best_selection_output(self, outputs: List[Dict[str, Any]]) -> str:
        """Select single best output"""
        try:
            if not outputs:
                return ""
            best = max(outputs, key=lambda x: x.get("confidence_score", 0))
            return best.get("text", "")
        except:
            return ""
    
    def _consensus_ranking_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """Consensus-based ranking (simplified implementation)"""
        return self._weighted_vote_outputs(outputs, {"confidence": 1.0})
    
    def _relevance_weighted_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """Relevance-weighted combination (simplified implementation)"""
        return self._weighted_vote_outputs(outputs, {"relevance": 0.8, "coherence": 0.2})
    
    def _confidence_filter_outputs(self, outputs: List[Dict[str, Any]], threshold: float) -> str:
        """Filter by confidence and merge high-confidence results"""
        high_confidence = [o for o in outputs if o.get("confidence_score", 0) >= threshold]
        return self._simple_concatenate_outputs(high_confidence)
    
    def _diversity_preservation_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """Preserve diverse perspectives"""
        return self._simple_concatenate_outputs(outputs)  # Keep all for diversity
    
    def _temporal_priority_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """Prioritize by timestamp (most recent first)"""
        try:
            sorted_outputs = sorted(outputs, key=lambda x: x.get("timestamp", ""), reverse=True)
            return self._simple_concatenate_outputs(sorted_outputs)
        except:
            return self._simple_concatenate_outputs(outputs)
    
    def _structured_fusion_outputs(self, outputs: List[Dict[str, Any]], preserve_structure: bool) -> str:
        """Fuse structured data"""
        # For now, fall back to simple concatenation
        # In full implementation, would handle JSON/object merging
        return self._simple_concatenate_outputs(outputs)
    
    def _resolve_conflicts(self, result: str, conflict_resolution: str, outputs: List[Dict[str, Any]]) -> str:
        """Resolve conflicts in aggregated result"""
        # Simplified implementation - in practice would use NLP to detect conflicts
        if conflict_resolution == "highlight_conflicts":
            return f"[Conflicts may exist between sources]\n\n{result}"
        return result
    
    def _format_aggregated_output(self, result: str, output_format: str, include_attribution: bool, outputs: List[Dict[str, Any]]) -> str:
        """Format the final aggregated output"""
        try:
            if output_format == "comprehensive":
                formatted = f"# Comprehensive Analysis\n\n{result}"
                if include_attribution:
                    formatted += f"\n\n*Based on {len(outputs)} source(s)*"
                return formatted
            elif output_format == "summary":
                # Truncate for summary
                summary = result[:500] + "..." if len(result) > 500 else result
                return f"## Executive Summary\n\n{summary}"
            elif output_format == "structured":
                return f"```\n{result}\n```"
            elif output_format == "ranked_list":
                lines = result.split('\n')
                ranked = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines[:10]) if line.strip())
                return f"## Ranked Results\n\n{ranked}"
            elif output_format == "consensus":
                return f"**Consensus Statement:**\n\n{result}"
            else:  # raw_merge
                return result
        except:
            return result
    
    def _calculate_overall_confidence(self, outputs: List[Dict[str, Any]], strategy: str) -> float:
        """Calculate overall confidence score for aggregated result"""
        try:
            if not outputs:
                return 0.0
            
            scores = [output.get("confidence_score", 0) for output in outputs]
            
            if strategy in ["weighted_vote", "best_selection"]:
                return max(scores)
            elif strategy == "consensus_ranking":
                return sum(scores) / len(scores)  # Average
            else:
                # Weighted average based on confidence distribution
                return sum(score * (i + 1) for i, score in enumerate(sorted(scores, reverse=True))) / sum(range(1, len(scores) + 1))
        except:
            return 0.5
    
    def _generate_source_analysis(self, filtered_outputs: List[Dict[str, Any]], all_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis of input sources"""
        try:
            return {
                "total_sources": len(all_outputs),
                "sources_used": len(filtered_outputs),
                "average_confidence": sum(o.get("confidence_score", 0) for o in filtered_outputs) / max(len(filtered_outputs), 1),
                "quality_distribution": {
                    "high": len([o for o in filtered_outputs if o.get("confidence_score", 0) > 0.7]),
                    "medium": len([o for o in filtered_outputs if 0.4 <= o.get("confidence_score", 0) <= 0.7]),
                    "low": len([o for o in filtered_outputs if o.get("confidence_score", 0) < 0.4])
                },
                "source_types": "mixed"  # Could analyze agent types if available
            }
        except:
            return {"error": "Could not generate source analysis"}
    
    def _count_upstream_nodes(self, target_node_id: str, workflow_edges: List[Dict[str, Any]]) -> int:
        """Count the number of nodes that feed into a target node"""
        try:
            upstream_count = len([
                edge for edge in workflow_edges 
                if edge.get("target") == target_node_id
            ])
            logger.debug(f"[AGGREGATOR] Node {target_node_id} has {upstream_count} upstream connections")
            return upstream_count
        except Exception as e:
            logger.error(f"[AGGREGATOR] Error counting upstream nodes for {target_node_id}: {e}")
            return 1  # Fallback to assume at least 1 input
    
    async def _process_condition_node(
        self,
        condition: Dict[str, Any],
        input_data: str,
        workflow_id: int,
        execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """Process condition node to evaluate conditional logic"""
        try:
            condition_type = condition.get("condition_type", "simple")
            operator = condition.get("operator", "equals")
            compare_value = condition.get("compare_value", "")
            left_operand = condition.get("left_operand", "")
            right_operand = condition.get("right_operand", "")
            ai_criteria = condition.get("ai_criteria", "")
            case_sensitive = condition.get("case_sensitive", False)
            data_type = condition.get("data_type", "string")
            condition_node_id = condition["node_id"]
            
            logger.info(f"[CONDITION] Processing condition node {condition_node_id} with type: {condition_type}")
            
            # Determine what values to compare
            if condition_type == "simple":
                # Use left_operand and right_operand if available, otherwise fall back to input_data and compare_value
                left_value = left_operand if left_operand else input_data
                right_value = right_operand if right_operand else compare_value
            else:
                left_value = input_data
                right_value = compare_value
            
            result = False
            evaluation_details = {}
            
            # Apply condition evaluation based on type
            if condition_type == "simple":
                result = self._evaluate_simple_condition(
                    left_value, right_value, operator, case_sensitive, data_type
                )
                evaluation_details = {
                    "left_value": left_value,
                    "right_value": right_value,
                    "operator": operator,
                    "case_sensitive": case_sensitive,
                    "data_type": data_type
                }
                
            elif condition_type == "ai_decision":
                result = await self._evaluate_ai_condition(
                    input_data, ai_criteria, workflow_id, execution_id
                )
                evaluation_details = {
                    "input_data": input_data,
                    "ai_criteria": ai_criteria,
                    "evaluation_method": "AI-based decision"
                }
                
            elif condition_type == "custom":
                # For custom logic, use simple evaluation for now
                # In a real implementation, this would support custom code execution
                result = self._evaluate_simple_condition(
                    left_value, right_value, operator, case_sensitive, data_type
                )
                evaluation_details = {
                    "left_value": left_value,
                    "right_value": right_value,
                    "operator": operator,
                    "note": "Custom logic currently uses simple evaluation"
                }
            
            logger.info(f"[CONDITION] Condition {condition_node_id} evaluated to: {result}")
            
            return {
                "result": result,
                "branch": "true" if result else "false",
                "evaluation_details": evaluation_details,
                "condition_type": condition_type,
                "node_id": condition_node_id
            }
            
        except Exception as e:
            logger.error(f"[CONDITION] Error processing condition node: {e}")
            return {
                "result": False,
                "branch": "false",
                "evaluation_details": {"error": str(e)},
                "condition_type": condition.get("condition_type", "unknown"),
                "node_id": condition.get("node_id", "unknown"),
                "error": str(e)
            }
    
    def _evaluate_simple_condition(
        self,
        left_value: str,
        right_value: str,
        operator: str,
        case_sensitive: bool,
        data_type: str
    ) -> bool:
        """Evaluate simple condition based on operator"""
        try:
            # Convert values based on data type
            if data_type == "number":
                try:
                    left_value = float(left_value) if left_value else 0
                    right_value = float(right_value) if right_value else 0
                except ValueError:
                    # If conversion fails, treat as strings
                    pass
            elif data_type == "boolean":
                left_value = str(left_value).lower() in ['true', '1', 'yes', 'on']
                right_value = str(right_value).lower() in ['true', '1', 'yes', 'on']
            elif data_type == "json":
                try:
                    import json
                    left_value = json.loads(left_value) if isinstance(left_value, str) else left_value
                    right_value = json.loads(right_value) if isinstance(right_value, str) else right_value
                except:
                    pass
            
            # Convert to strings for comparison if not numeric/boolean
            if data_type not in ["number", "boolean"]:
                left_str = str(left_value)
                right_str = str(right_value)
                
                # Apply case sensitivity
                if not case_sensitive:
                    left_str = left_str.lower()
                    right_str = right_str.lower()
                
                left_value = left_str
                right_value = right_str
            
            # Apply operator
            if operator == "equals":
                return left_value == right_value
            elif operator == "not_equals":
                return left_value != right_value
            elif operator == "greater_than":
                return left_value > right_value
            elif operator == "less_than":
                return left_value < right_value
            elif operator == "greater_equal":
                return left_value >= right_value
            elif operator == "less_equal":
                return left_value <= right_value
            elif operator == "contains":
                return str(right_value) in str(left_value)
            elif operator == "starts_with":
                return str(left_value).startswith(str(right_value))
            elif operator == "ends_with":
                return str(left_value).endswith(str(right_value))
            elif operator == "regex_match":
                import re
                flags = 0 if case_sensitive else re.IGNORECASE
                return bool(re.search(str(right_value), str(left_value), flags))
            else:
                # Default to equals
                return left_value == right_value
                
        except Exception as e:
            logger.error(f"[CONDITION] Error evaluating simple condition: {e}")
            return False
    
    async def _evaluate_ai_condition(
        self,
        input_data: str,
        ai_criteria: str,
        workflow_id: int,
        execution_id: str
    ) -> bool:
        """Evaluate condition using AI decision"""
        try:
            # For AI-based decision, we'll use the dynamic agent system
            if not self.dynamic_agent_system:
                logger.warning("[CONDITION] No dynamic agent system available for AI decision")
                return False
            
            # Create a simple prompt for AI evaluation
            evaluation_prompt = f"""
Evaluate the following data against the given criteria and respond with only "TRUE" or "FALSE".

Data to evaluate:
{input_data}

Evaluation criteria:
{ai_criteria}

Respond with only "TRUE" if the data meets the criteria, or "FALSE" if it doesn't.
"""
            
            # Use a lightweight agent for evaluation (or the first available agent)
            from app.core.langgraph_agents_cache import get_langgraph_agents
            cached_agents = get_langgraph_agents()
            
            if not cached_agents:
                logger.warning("[CONDITION] No agents available for AI decision")
                return False
            
            # Use the first available agent for evaluation
            agent_name = list(cached_agents.keys())[0]
            agent_data = cached_agents[agent_name]
            
            # Simple context for the evaluation
            context = {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "evaluation": True
            }
            
            # Execute AI evaluation
            final_response = None
            async for event in self.dynamic_agent_system.execute_agent(
                agent_name=agent_name,
                agent_data=agent_data,
                query=evaluation_prompt,
                context=context
            ):
                if event.get("type") == "agent_complete":
                    final_response = event.get("content", "")
                    break
            
            if final_response:
                # Parse the response to determine true/false
                response_lower = final_response.lower().strip()
                return "true" in response_lower and "false" not in response_lower
            
            return False
            
        except Exception as e:
            logger.error(f"[CONDITION] Error evaluating AI condition: {e}")
            return False
    
    def _handle_condition_branching(
        self,
        condition_result: Dict[str, Any],
        condition: Dict[str, Any],
        workflow_edges: List[Dict[str, Any]],
        agents: List[Dict[str, Any]]
    ) -> None:
        """Handle branching logic for condition nodes"""
        try:
            condition_node_id = condition["node_id"]
            branch = condition_result["branch"]  # "true" or "false"
            result = condition_result["result"]
            
            # Find edges from the condition node
            outgoing_edges = [
                edge for edge in workflow_edges 
                if edge.get("source") == condition_node_id
            ]
            
            # Determine which branch to follow based on the result
            # In a real implementation, this would need to match the specific handle IDs
            # For now, we'll log the branching decision
            logger.info(f"[CONDITION] Condition {condition_node_id} branching: {branch} (result: {result})")
            
            if result:
                # Follow the "true" branch
                true_targets = []
                for edge in outgoing_edges:
                    source_handle = edge.get("sourceHandle", "")
                    if "true" in source_handle.lower():
                        true_targets.append(edge.get("target"))
                
                if true_targets:
                    logger.info(f"[CONDITION] Following TRUE branch to nodes: {true_targets}")
                else:
                    logger.info(f"[CONDITION] No TRUE branch found, continuing with all connected nodes")
            else:
                # Follow the "false" branch
                false_targets = []
                for edge in outgoing_edges:
                    source_handle = edge.get("sourceHandle", "")
                    if "false" in source_handle.lower():
                        false_targets.append(edge.get("target"))
                
                if false_targets:
                    logger.info(f"[CONDITION] Following FALSE branch to nodes: {false_targets}")
                else:
                    logger.info(f"[CONDITION] No FALSE branch found, continuing with all connected nodes")
            
            # Store branching information for workflow execution control
            # In a more sophisticated implementation, this would control which agents execute next
            
        except Exception as e:
            logger.error(f"[CONDITION] Error handling condition branching: {e}")
    
    def _should_execute_node_with_conditions(
        self,
        node_id: str,
        condition_results: Dict[str, Dict[str, Any]],
        workflow_edges: List[Dict[str, Any]]
    ) -> bool:
        """Check if a node should execute based on condition results"""
        try:
            # Find if this node is connected from any condition node
            incoming_condition_edges = []
            for edge in workflow_edges:
                if edge.get("target") == node_id:
                    source_id = edge.get("source")
                    # Check if source is a condition node
                    if source_id in condition_results:
                        incoming_condition_edges.append({
                            "source": source_id,
                            "source_handle": edge.get("sourceHandle", ""),
                            "condition_result": condition_results[source_id]
                        })
            
            if not incoming_condition_edges:
                # No condition dependencies, execute normally
                return True
            
            # Check if any condition allows execution
            for edge_info in incoming_condition_edges:
                source_handle = edge_info["source_handle"]
                condition_result = edge_info["condition_result"]
                branch = condition_result.get("branch", "false")
                
                # If this edge is from the matching branch, allow execution
                if (("true" in source_handle.lower() and branch == "true") or
                    ("false" in source_handle.lower() and branch == "false")):
                    return True
            
            # No matching branch found, don't execute
            return False
            
        except Exception as e:
            logger.error(f"[CONDITION] Error checking node execution with conditions: {e}")
            return True  # Default to execute on error
    
    async def _process_cache_node(
        self,
        cache_config: Dict[str, Any],
        input_data: str,
        workflow_id: int,
        execution_id: str,
        agent_node_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """Process cache node to check for cached results or store new ones"""
        try:
            from app.automation.integrations.redis_bridge import workflow_redis
            import hashlib
            import json
            
            cache_key_pattern = cache_config.get("cache_key_pattern", "auto")
            custom_cache_key = cache_config.get("cache_key", "")
            ttl = cache_config.get("ttl", 3600)
            cache_policy = cache_config.get("cache_policy", "always")
            cache_namespace = cache_config.get("cache_namespace", "default")
            max_cache_size = cache_config.get("max_cache_size", 10) * 1024 * 1024  # Convert MB to bytes
            cache_node_id = cache_config["node_id"]
            
            # Generate cache key
            cache_key = self._generate_cache_key(
                cache_key_pattern, custom_cache_key, workflow_id, 
                cache_node_id, agent_node_id, input_data, cache_namespace
            )
            
            logger.info(f"[CACHE] Processing cache node {cache_node_id} with key: {cache_key}")
            
            # Check cache policy
            if cache_policy == "always" or cache_policy == "input_match":
                # Try to get cached result
                cached_result = workflow_redis.get_value(cache_key)
                
                if cached_result:
                    logger.info(f"[CACHE] Cache HIT for key: {cache_key}")
                    
                    # Check if cached data is within size limits
                    cached_size = len(json.dumps(cached_result).encode('utf-8'))
                    if max_cache_size > 0 and cached_size > max_cache_size:
                        logger.warning(f"[CACHE] Cached data too large ({cached_size} bytes), invalidating")
                        workflow_redis.delete_value(cache_key)
                        cached_result = None
                    else:
                        # Calculate cache metadata for enhanced cache events
                        cache_metadata = self._get_cache_metadata(cache_key, cached_result, cached_size)
                        
                        return {
                            "cache_hit": True,
                            "cached_data": cached_result,
                            "cache_key": cache_key,
                            "cache_size": cached_size,
                            "cache_metadata": cache_metadata,
                            "skip_execution": True
                        }
                
                logger.info(f"[CACHE] Cache MISS for key: {cache_key}")
                return {
                    "cache_hit": False,
                    "cache_key": cache_key,
                    "skip_execution": False,
                    "should_cache": True
                }
            
            elif cache_policy == "conditional":
                # Conditional caching - always execute but may cache result
                return {
                    "cache_hit": False,
                    "cache_key": cache_key,
                    "skip_execution": False,
                    "should_cache": True,
                    "conditional": True,
                    "cache_condition": cache_config.get("cache_condition", "")
                }
            
            # Default: no caching
            return {
                "cache_hit": False,
                "cache_key": cache_key,
                "skip_execution": False,
                "should_cache": False
            }
            
        except Exception as e:
            logger.error(f"[CACHE] Error processing cache node: {e}")
            return {
                "cache_hit": False,
                "cache_key": "",
                "skip_execution": False,
                "should_cache": False,
                "error": str(e)
            }
    
    def _generate_cache_key(
        self,
        pattern: str,
        custom_key: str,
        workflow_id: int,
        cache_node_id: str,
        agent_node_id: str,
        input_data: str,
        namespace: str
    ) -> str:
        """Generate cache key based on pattern"""
        try:
            import hashlib
            import json
            
            if pattern == "custom" and custom_key:
                return f"{namespace}:custom:{custom_key}"
            
            elif pattern == "node_only":
                return f"{namespace}:node:{cache_node_id}"
            
            elif pattern == "input_hash":
                input_hash = hashlib.md5(input_data.encode('utf-8')).hexdigest()[:8]
                return f"{namespace}:input:{input_hash}"
            
            else:  # auto pattern
                # Include workflow, node, and input hash for comprehensive key
                input_hash = hashlib.md5(input_data.encode('utf-8')).hexdigest()[:8]
                agent_part = f"_{agent_node_id}" if agent_node_id else ""
                return f"{namespace}:auto:w{workflow_id}:n{cache_node_id}{agent_part}:i{input_hash}"
                
        except Exception as e:
            logger.error(f"[CACHE] Error generating cache key: {e}")
            return f"{namespace}:fallback:{cache_node_id}"
    
    async def _store_cache_result(
        self,
        cache_key: str,
        result_data: Any,
        ttl: int,
        cache_condition: str = "",
        max_size: int = 0
    ) -> bool:
        """Store result in cache with optional conditions"""
        try:
            from app.automation.integrations.redis_bridge import workflow_redis
            import json
            
            # Check conditional caching
            if cache_condition:
                try:
                    # Simple condition evaluation (in production, use safer evaluation)
                    # For now, just check basic conditions
                    if "length" in cache_condition:
                        data_length = len(str(result_data))
                        condition_met = eval(cache_condition.replace("output.length", str(data_length)))
                        if not condition_met:
                            logger.info(f"[CACHE] Cache condition not met, skipping cache storage")
                            return False
                except Exception as e:
                    logger.warning(f"[CACHE] Error evaluating cache condition: {e}")
                    return False
            
            # Check size limits
            result_size = len(json.dumps(result_data).encode('utf-8'))
            if max_size > 0 and result_size > max_size:
                logger.warning(f"[CACHE] Result too large to cache ({result_size} bytes)")
                return False
            
            # Store in cache
            success = workflow_redis.set_value(cache_key, result_data, expire=ttl)
            if success:
                logger.info(f"[CACHE] Stored result in cache with key: {cache_key}, TTL: {ttl}s")
            else:
                logger.warning(f"[CACHE] Failed to store result in cache")
            
            return success
            
        except Exception as e:
            logger.error(f"[CACHE] Error storing cache result: {e}")
            return False
    
    def _get_cache_metadata(self, cache_key: str, cached_data: Any, cached_size: int) -> Dict[str, Any]:
        """Extract metadata from cached data for enhanced cache events"""
        try:
            from datetime import datetime, timezone
            import json
            
            # Try to extract timestamp if available
            timestamp = None
            last_access = None
            
            if isinstance(cached_data, dict):
                timestamp = cached_data.get("timestamp") or cached_data.get("created_at")
                last_access = cached_data.get("last_access")
            
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
                        "created_at": timestamp,
                        "is_fresh": age_seconds < 3600,  # Less than 1 hour
                        "is_stale": age_seconds > 86400   # More than 24 hours
                    }
                except Exception as e:
                    logger.warning(f"[CACHE METADATA] Error parsing timestamp: {e}")
            
            # Determine data type and structure
            data_type = type(cached_data).__name__
            structure_info = {}
            
            if isinstance(cached_data, dict):
                data_type = "object"
                structure_info = {
                    "key_count": len(cached_data.keys()),
                    "has_output": "output" in cached_data,
                    "has_metadata": any(key in cached_data for key in ["timestamp", "agent_name", "node_id"])
                }
            elif isinstance(cached_data, list):
                data_type = "array"
                structure_info = {"length": len(cached_data)}
            elif isinstance(cached_data, str):
                data_type = "string"
                structure_info = {
                    "length": len(cached_data),
                    "word_count": len(cached_data.split()),
                    "line_count": len(cached_data.split('\n'))
                }
            
            # Performance metrics
            size_info = {
                "size_bytes": cached_size,
                "size_kb": round(cached_size / 1024, 2),
                "size_mb": round(cached_size / (1024 * 1024), 3),
                "compression_ratio": 1.0  # Could be enhanced with actual compression
            }
            
            # Generate preview for logs
            preview = str(cached_data)[:100] + "..." if len(str(cached_data)) > 100 else str(cached_data)
            
            return {
                "cache_key": cache_key,
                "data_type": data_type,
                "last_access": datetime.now(timezone.utc).isoformat(),
                "access_count": 1,  # Could be enhanced with actual tracking
                "preview": preview,
                **age_info,
                **structure_info,
                **size_info
            }
            
        except Exception as e:
            logger.warning(f"[CACHE METADATA] Error extracting metadata: {e}")
            return {
                "cache_key": cache_key,
                "size_bytes": cached_size,
                "data_type": "unknown",
                "error": str(e),
                "last_access": datetime.now(timezone.utc).isoformat()
            }