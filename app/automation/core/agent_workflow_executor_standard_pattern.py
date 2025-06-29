"""
Agent Workflow Executor - Following Standard Chat Mode Pattern
Simplified to match the exact Langfuse tracing pattern used in standard chat
"""
import logging
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_name
from app.automation.integrations.redis_bridge import workflow_redis
from app.core.langfuse_integration import get_tracer
from app.automation.core.workflow_state import WorkflowState, workflow_state_manager

logger = logging.getLogger(__name__)

class AgentWorkflowExecutor:
    """Executes visual workflows using standard chat mode Langfuse pattern"""
    
    def __init__(self):
        self.tracer = get_tracer()
    
    async def execute_agent_workflow(
        self,
        workflow_id: int,
        execution_id: str,
        workflow_config: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        trace=None  # Follow standard chat mode - receive trace from API endpoint
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute agent workflow following standard chat mode pattern
        """
        
        try:
            logger.info(f"[AGENT WORKFLOW] Starting workflow {workflow_id}, execution {execution_id}")
            
            # Use trace from API endpoint (like standard chat mode)
            if trace:
                logger.info(f"[LANGFUSE] Using trace from API endpoint for workflow {workflow_id}")
            
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
            
            # Convert visual workflow to agent execution plan
            agent_plan = self._convert_workflow_to_agent_plan(workflow_config, input_data, message)
            
            # Log plan (like standard chat mode - simple logging)
            logger.info(f"[WORKFLOW PLAN] Pattern: {agent_plan['pattern']}, Agents: {len(agent_plan['agents'])}, Enhanced: {len([a for a in agent_plan['agents'] if a.get('state_enabled', False)])}")
            
            # Yield agent plan
            yield {
                "type": "agent_plan",
                "agents": agent_plan["agents"],
                "execution_pattern": agent_plan["pattern"]
            }
            
            # Execute agents using multi-agent system (pass trace like standard chat)
            async for update in self._execute_agent_plan(
                agent_plan, 
                workflow_id, 
                execution_id, 
                trace,  # Pass trace from API endpoint
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
            
            # Simple completion logging (like standard chat mode)
            logger.info(f"[LANGFUSE] Automation workflow {workflow_id} completed successfully")
            
            yield {
                "type": "workflow_complete",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "completed"
            }
            
            # Clean up workflow state from memory after completion
            if workflow_state_obj:
                workflow_state_manager.remove_state(workflow_id, execution_id)
            
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Execution failed: {e}")
            
            # Simple error logging (like standard chat mode)
            logger.error(f"[LANGFUSE] Automation workflow {workflow_id} failed: {e}")
            
            # Update error state
            error_state = {
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e)
            }
            workflow_redis.update_workflow_state(workflow_id, execution_id, error_state)
            
            yield {
                "type": "workflow_error",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "error": str(e)
            }
    
    async def _execute_agent_plan(
        self,
        agent_plan: Dict[str, Any],
        workflow_id: int,
        execution_id: str,
        trace=None,  # Receive trace like standard chat mode
        workflow_state: WorkflowState = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agent plan following standard chat mode pattern"""
        
        agents = agent_plan["agents"]
        pattern = agent_plan["pattern"]
        
        yield {
            "type": "agents_selected",
            "agents": [{"name": agent["agent_name"]} for agent in agents],
            "pattern": pattern,
            "workflow_id": workflow_id,
            "execution_id": execution_id
        }
        
        if pattern == "hierarchical":
            # Sequential execution with state chaining
            async for update in self._execute_sequential_agents(
                agents, 
                agent_plan.get("query", ""), 
                workflow_id, 
                execution_id, 
                trace,  # Pass trace
                workflow_state
            ):
                yield update
        
        elif pattern == "parallel":
            # Parallel execution
            async for update in self._execute_parallel_agents(
                agents, 
                agent_plan.get("query", ""), 
                workflow_id, 
                execution_id, 
                trace,  # Pass trace
                workflow_state
            ):
                yield update
        
        else:
            # Single agent execution
            if agents:
                agent = agents[0]
                agent_name = agent["agent_name"]
                
                yield {
                    "type": "agent_execution_start",
                    "agent_name": agent_name,
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                try:
                    # Execute single agent (pass trace like standard chat)
                    result = await self._execute_single_agent(
                        agent, 
                        agent_plan.get("query", ""), 
                        workflow_id, 
                        execution_id, 
                        trace,  # Pass trace
                        workflow_state
                    )
                    
                    yield {
                        "type": "agent_execution_complete",
                        "agent_name": agent_name,
                        "output": result.get("output", ""),
                        "tools_used": result.get("tools_used", []),
                        "workflow_id": workflow_id,
                        "execution_id": execution_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"[AGENT WORKFLOW] Single agent {agent_name} failed: {e}")
                    yield {
                        "type": "agent_execution_error",
                        "agent_name": agent_name,
                        "error": str(e),
                        "workflow_id": workflow_id,
                        "execution_id": execution_id
                    }
    
    async def _execute_sequential_agents(
        self,
        agents: List[Dict[str, Any]],
        query: str,
        workflow_id: int,
        execution_id: str,
        trace=None,  # Follow standard chat mode pattern
        workflow_state: WorkflowState = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agents sequentially following standard chat mode pattern"""
        
        context = query
        agent_outputs = {}
        chain_data = None  # For direct agent-to-agent chaining
        
        for i, agent in enumerate(agents):
            agent_name = agent["agent_name"]
            
            # Simple agent execution logging (like standard chat mode)
            logger.info(f"[AGENT EXECUTION] Starting {agent_name} (index: {i}, state: {agent.get('state_enabled', False)})")
            
            yield {
                "type": "agent_execution_start",
                "agent_name": agent_name,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "agent_index": i,
                "state_enabled": agent.get("state_enabled", False),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            try:
                # Build prompt for agent (simplified logic)
                user_message = workflow_state.get_state("user_message") if workflow_state else None
                input_data = workflow_state.get_state("input_data") if workflow_state else None
                
                # Build agent prompt based on position
                if i == 0:
                    # First agent
                    agent_prompt = f"""{agent['system_prompt']}

USER REQUEST:
{user_message or query}

Please process this request and provide your analysis."""
                
                elif agent.get("state_enabled", False) and chain_data is not None:
                    # Enhanced agent with chaining
                    agent_prompt = f"""{agent['system_prompt']}

ORIGINAL USER REQUEST:
{user_message or query}

INPUT FROM PREVIOUS AGENT:
{json.dumps(chain_data, indent=2) if isinstance(chain_data, dict) else str(chain_data)}

Please process the input from the previous agent and provide your analysis."""
                
                else:
                    # Traditional sequential agent
                    previous_context = "\n".join([
                        f"**{prev_agent['agent_name']}:** {agent_outputs.get(prev_agent['agent_name'], '')}"
                        for prev_agent in agents[:i]
                    ])
                    
                    agent_prompt = f"""{agent['system_prompt']}

ORIGINAL USER REQUEST:
{user_message or query}

PREVIOUS AGENT OUTPUTS:
{previous_context}

Please analyze the above information and provide your contribution."""
                
                # Execute agent (pass trace like standard chat mode)
                result = await self._execute_single_agent(agent, agent_prompt, workflow_id, execution_id, trace, workflow_state)
                agent_output = result.get("output", "")
                agent_outputs[agent_name] = agent_output
                
                # Simple completion logging (like standard chat mode)
                logger.info(f"[AGENT EXECUTION] Completed {agent_name}: {len(agent_output)} chars, {len(result.get('tools_used', []))} tools")
                
                # Handle state management for enhanced agents (simplified)
                formatted_output = None
                if agent.get("state_enabled", False):
                    # Format output for potential chaining to next agent
                    formatted_output = self._format_agent_output_for_chaining(
                        agent_output, 
                        agent, 
                        chain_data
                    )
                    
                    # Update chain data for next agent
                    chain_data = formatted_output
                    
                    # Store state
                    if workflow_state:
                        workflow_state.set_state(f"agent_output_{agent_name}", agent_output, f"agent_{i+1}_output")
                        workflow_state.set_state(f"agent_chain_{agent_name}", formatted_output, f"agent_{i+1}_chain")
                        
                        # Store in chain key if specified
                        chain_key = agent.get("chain_key")
                        if chain_key:
                            workflow_state.set_state(chain_key, formatted_output, f"chain_key_{chain_key}")
                    
                    logger.info(f"[STATE MANAGEMENT] Enhanced chaining for {agent_name}: {agent.get('state_operation', 'passthrough')}")
                else:
                    # Traditional workflow
                    if workflow_state:
                        workflow_state.set_state(f"agent_output_{agent_name}", agent_output, f"agent_{i+1}_output")
                
                # Simple success logging (like standard chat mode)
                logger.info(f"[AGENT EXECUTION] {agent_name} completed successfully")
                
                yield {
                    "type": "agent_execution_complete",
                    "agent_name": agent_name,
                    "input": agent_prompt[:200] + "..." if len(agent_prompt) > 200 else agent_prompt,
                    "output": result.get("output", ""),
                    "tools_used": result.get("tools_used", []),
                    "state_enabled": agent.get("state_enabled", False),
                    "chain_data": formatted_output,  # Will be None for non-state-enabled agents
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"[AGENT WORKFLOW] Agent {agent_name} failed: {e}")
                
                yield {
                    "type": "agent_execution_error",
                    "agent_name": agent_name,
                    "error": str(e),
                    "workflow_id": workflow_id,
                    "execution_id": execution_id
                }
        
        # Yield final response
        final_response = self._synthesize_agent_outputs(agent_outputs, query)
        yield {
            "type": "workflow_result",
            "response": final_response,
            "agent_outputs": agent_outputs,
            "workflow_id": workflow_id,
            "execution_id": execution_id
        }
    
    async def _execute_single_agent(
        self,
        agent: Dict[str, Any],
        prompt: str,
        workflow_id: int,
        execution_id: str,
        trace=None,  # Receive trace like standard chat mode
        workflow_state: WorkflowState = None
    ) -> Dict[str, Any]:
        """Execute a single agent following standard chat mode pattern"""
        
        try:
            # Get agent name
            agent_name = (
                agent.get("name", "") or 
                agent.get("agent_name", "") or 
                agent.get("agentName", "") or
                agent.get("node", {}).get("agent_name", "") or
                ""
            )
            
            logger.info(f"[AGENT WORKFLOW] Executing agent: {agent_name}")
            
            # Initialize the Dynamic Multi-Agent System
            dynamic_agent_system = DynamicMultiAgentSystem()
            
            # Get the full agent data from cache
            agent_info = get_agent_by_name(agent_name)
            if not agent_info:
                raise ValueError(f"Agent '{agent_name}' not found in cache")
            
            # Prepare context (like standard chat mode)
            context = {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "agent_config": agent,
                "tools": agent.get("tools", []),
                "temperature": agent.get("context", {}).get("temperature", 0.7),
                "timeout": agent.get("context", {}).get("timeout", 45)
            }
            
            # Execute using the dynamic agent system (pass trace like standard chat)
            final_response = None
            tools_used = []
            start_time = datetime.utcnow()
            
            async for event in dynamic_agent_system.execute_agent(
                agent_name=agent_name,
                agent_data=agent_info,
                query=prompt,
                context=context,
                parent_trace_or_span=trace  # Pass trace like standard chat mode
            ):
                event_type = event.get("type", "")
                
                # Handle events from DynamicMultiAgentSystem
                if event_type == "agent_complete":
                    final_response = event.get("content", "")
                    logger.info(f"[AGENT WORKFLOW] Agent {agent_name} completed with response length: {len(final_response) if final_response else 0}")
                    
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
                    
                elif event_type == "agent_error":
                    error_msg = event.get("error", "Unknown error")
                    logger.error(f"[AGENT WORKFLOW] Agent {agent_name} error: {error_msg}")
                    final_response = f"Agent execution failed: {error_msg}"
                
                # Log events for debugging
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
                "response_time_ms": response_time_ms
            }
            
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Single agent execution failed: {e}")
            return {
                "output": f"Agent execution failed: {str(e)}",
                "tools_used": [],
                "success": False,
                "error": str(e),
                "agent_name": agent.get("agent_name", "unknown"),
                "response_time_ms": None
            }
    
    # ... (other methods remain the same as original implementation)
    # Copy from original: _convert_workflow_to_agent_plan, _format_agent_output_for_chaining, etc.