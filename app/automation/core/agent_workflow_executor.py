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

from app.llm.ollama import OllamaLLM
from app.llm.base import LLMConfig
from app.core.langgraph_agents_cache import get_langgraph_agents, get_agent_by_name
from app.automation.integrations.redis_bridge import workflow_redis
from app.core.langfuse_integration import get_tracer

logger = logging.getLogger(__name__)

class AgentWorkflowExecutor:
    """Executes visual workflows using agent-based approach with LangGraph"""
    
    def __init__(self):
        self.tracer = get_tracer()
    
    async def execute_agent_workflow(
        self,
        workflow_id: int,
        execution_id: str,
        workflow_config: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute agent-based workflow with real-time streaming
        
        Args:
            workflow_id: Workflow identifier
            execution_id: Unique execution identifier
            workflow_config: Visual workflow configuration (nodes and edges)
            input_data: Optional input data for workflow
            message: Optional message to trigger workflow
            
        Yields:
            Real-time execution updates with agent progress
        """
        execution_trace = None
        
        try:
            logger.info(f"[AGENT WORKFLOW] Starting workflow {workflow_id}, execution {execution_id}")
            
            # Create execution trace
            if self.tracer.is_enabled():
                try:
                    execution_trace = self.tracer.create_automation_execution_trace(
                        workflow_id=workflow_id,
                        execution_id=execution_id,
                        input_data=input_data
                    )
                except Exception as e:
                    logger.warning(f"Failed to create execution trace: {e}")
            
            # Initialize workflow state
            workflow_state = {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "input_data": input_data,
                "message": message,
                "agent_states": {},
                "execution_log": []
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
            
            # Yield agent plan
            yield {
                "type": "agent_plan",
                "agents": agent_plan["agents"],
                "execution_pattern": agent_plan["pattern"]
            }
            
            # Execute agents using multi-agent system
            async for update in self._execute_agent_plan(
                agent_plan, 
                workflow_id, 
                execution_id, 
                execution_trace
            ):
                yield update
            
            # Final status
            final_state = {
                "status": "completed",
                "completed_at": datetime.utcnow().isoformat()
            }
            workflow_redis.update_workflow_state(workflow_id, execution_id, final_state)
            
            yield {
                "type": "workflow_complete",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Execution failed: {e}")
            
            # Update error state
            error_state = {
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": str(e)
            }
            workflow_redis.update_workflow_state(workflow_id, execution_id, error_state)
            
            # End execution trace with error
            if execution_trace:
                try:
                    self.tracer.end_span_with_result(execution_trace, None, success=False, error=str(e))
                except Exception:
                    pass
            
            yield {
                "type": "workflow_error",
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "error": str(e)
            }
    
    def _convert_workflow_to_agent_plan(
        self, 
        workflow_config: Dict[str, Any], 
        input_data: Optional[Dict[str, Any]], 
        message: Optional[str]
    ) -> Dict[str, Any]:
        """Convert visual workflow configuration to agent execution plan"""
        
        nodes = workflow_config.get("nodes", [])
        edges = workflow_config.get("edges", [])
        
        # Extract agent nodes from workflow
        agent_nodes = []
        for node in nodes:
            node_data = node.get("data", {})
            node_type = node_data.get("type", "")
            
            if node_type == "AgentNode":
                agent_config = node_data.get("node", {})
                agent_name = agent_config.get("agent_name")
                
                if agent_name:
                    # Get agent from cache
                    agent_info = get_agent_by_name(agent_name)
                    if agent_info:
                        agent_nodes.append({
                            "node_id": node.get("id"),
                            "agent_name": agent_name,
                            "agent_config": agent_info,
                            "custom_prompt": agent_config.get("custom_prompt", ""),
                            "tools": agent_config.get("tools", agent_info.get("tools", [])),
                            "position": node.get("position", {})
                        })
        
        # Determine execution pattern from edges
        execution_pattern = self._analyze_execution_pattern(agent_nodes, edges)
        
        # Prepare query from input or message
        query = message or input_data.get("query", "") if input_data else ""
        if not query and input_data:
            # Try to construct query from input data
            query = f"Process the following data: {json.dumps(input_data, indent=2)}"
        
        return {
            "agents": agent_nodes,
            "pattern": execution_pattern,
            "query": query,
            "input_data": input_data
        }
    
    def _analyze_execution_pattern(self, agent_nodes: List[Dict], edges: List[Dict]) -> str:
        """Analyze workflow structure to determine execution pattern"""
        
        if len(agent_nodes) <= 1:
            return "single"
        
        # Build dependency graph
        dependencies = {}
        for node in agent_nodes:
            dependencies[node["node_id"]] = []
        
        for edge in edges:
            target = edge.get("target")
            source = edge.get("source")
            if target and source and target in dependencies:
                dependencies[target].append(source)
        
        # Check for parallel execution (nodes with no dependencies)
        parallel_nodes = [node_id for node_id, deps in dependencies.items() if not deps]
        
        if len(parallel_nodes) > 1:
            return "parallel"
        elif len(parallel_nodes) == 1 and len(agent_nodes) > 1:
            return "sequential"
        else:
            return "hierarchical"
    
    async def _execute_agent_plan(
        self,
        agent_plan: Dict[str, Any],
        workflow_id: int,
        execution_id: str,
        execution_trace=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute the agent plan using multi-agent system"""
        
        agents = agent_plan["agents"]
        pattern = agent_plan["pattern"]
        query = agent_plan["query"]
        
        # Prepare agents list for multi-agent system
        selected_agents = []
        for agent_node in agents:
            agent_config = agent_node["agent_config"]
            selected_agents.append({
                "name": agent_config["name"],
                "role": agent_config["role"],
                "system_prompt": agent_node.get("custom_prompt") or agent_config["system_prompt"],
                "tools": agent_node["tools"],
                "config": agent_config.get("config", {}),
                "node_id": agent_node["node_id"]
            })
        
        # Yield agent selection info
        yield {
            "type": "agents_selected",
            "agents": [{"name": agent["name"], "role": agent["role"], "node_id": agent["node_id"]} 
                      for agent in selected_agents],
            "pattern": pattern
        }
        
        # Execute agents sequentially or in parallel based on pattern
        try:
            if pattern == "parallel":
                # Execute agents in parallel
                async for result in self._execute_agents_parallel(selected_agents, query, workflow_id, execution_id):
                    yield result
            else:
                # Execute agents sequentially (default)
                async for result in self._execute_agents_sequential(selected_agents, query, workflow_id, execution_id):
                    yield result
                
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Agent execution failed: {e}")
            yield {
                "type": "execution_error",
                "error": str(e),
                "workflow_id": workflow_id,
                "execution_id": execution_id
            }
    
    async def _execute_agents_sequential(
        self,
        agents: List[Dict[str, Any]],
        query: str,
        workflow_id: int,
        execution_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute agents one after another, building context"""
        
        context = query
        agent_outputs = {}
        
        for i, agent in enumerate(agents):
            agent_name = agent["name"]
            
            yield {
                "type": "agent_execution_start",
                "agent_name": agent_name,
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "agent_index": i,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            try:
                # Build prompt with context from previous agents
                if i == 0:
                    agent_prompt = f"{agent['system_prompt']}\n\nUser Query: {query}"
                else:
                    previous_context = "\n".join([
                        f"Agent {prev_agent['name']}: {agent_outputs.get(prev_agent['name'], '')}"
                        for prev_agent in agents[:i]
                    ])
                    agent_prompt = f"{agent['system_prompt']}\n\nOriginal Query: {query}\n\nPrevious Agent Outputs:\n{previous_context}\n\nPlease analyze and contribute to this discussion."
                
                # Execute agent
                result = await self._execute_single_agent(agent, agent_prompt, workflow_id, execution_id)
                agent_outputs[agent_name] = result.get("output", "")
                
                yield {
                    "type": "agent_execution_complete",
                    "agent_name": agent_name,
                    "input": agent_prompt[:200] + "..." if len(agent_prompt) > 200 else agent_prompt,
                    "output": result.get("output", ""),
                    "tools_used": result.get("tools_used", []),
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
    
    async def _execute_agents_parallel(
        self,
        agents: List[Dict[str, Any]],
        query: str,
        workflow_id: int,
        execution_id: str
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
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Execute all agents in parallel
        tasks = []
        for agent in agents:
            agent_prompt = f"{agent['system_prompt']}\n\nUser Query: {query}"
            task = self._execute_single_agent(agent, agent_prompt, workflow_id, execution_id)
            tasks.append((agent["name"], task))
        
        # Collect results as they complete
        agent_outputs = {}
        for agent_name, task in tasks:
            try:
                result = await task
                agent_outputs[agent_name] = result.get("output", "")
                
                yield {
                    "type": "agent_execution_complete",
                    "agent_name": agent_name,
                    "input": f"Query: {query}",
                    "output": result.get("output", ""),
                    "tools_used": result.get("tools_used", []),
                    "workflow_id": workflow_id,
                    "execution_id": execution_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"[AGENT WORKFLOW] Parallel agent {agent_name} failed: {e}")
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
        execution_id: str
    ) -> Dict[str, Any]:
        """Execute a single agent with LLM"""
        
        try:
            # Get agent configuration
            config = agent.get("config", {})
            model = config.get("model", "qwen3:30b-a3b")
            
            # Create LLM config
            llm_config = LLMConfig(
                model=model,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
                timeout=config.get("timeout", 45)
            )
            
            # Initialize LLM
            llm = OllamaLLM(config=llm_config)
            
            # Generate response
            response = await llm.agenerate(prompt)
            
            return {
                "output": response,
                "tools_used": [],  # Tools would be handled separately in a full implementation
                "success": True
            }
            
        except Exception as e:
            logger.error(f"[AGENT WORKFLOW] Single agent execution failed: {e}")
            return {
                "output": f"Agent execution failed: {str(e)}",
                "tools_used": [],
                "success": False,
                "error": str(e)
            }
    
    def _synthesize_agent_outputs(self, agent_outputs: Dict[str, str], original_query: str) -> str:
        """Synthesize final response from all agent outputs"""
        
        if not agent_outputs:
            return "No agent outputs to synthesize."
        
        if len(agent_outputs) == 1:
            return list(agent_outputs.values())[0]
        
        # Combine multiple agent outputs
        synthesis = f"Based on analysis from {len(agent_outputs)} AI agents:\n\n"
        
        for agent_name, output in agent_outputs.items():
            synthesis += f"**{agent_name}:**\n{output}\n\n"
        
        synthesis += f"**Summary:** The agents have provided complementary perspectives on: {original_query}"
        
        return synthesis
    
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