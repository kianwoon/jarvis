"""
Collaboration Pattern Executor - handles different agent collaboration patterns
"""
import asyncio
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CollaborationExecutor:
    """Executes agents according to different collaboration patterns"""
    
    async def execute_agents(self, pattern: str, agent_tasks: List[Tuple[str, Any]], 
                           state: Dict, agent_start_times: Dict):
        """
        Execute agents based on collaboration pattern
        
        Args:
            pattern: 'parallel', 'sequential', or 'hierarchical'
            agent_tasks: List of (agent_name, agent_generator) tuples
            state: Agent state dict to update
            agent_start_times: Dict of agent start times
            
        Yields:
            Events from agent execution
        """
        print(f"[DEBUG] CollaborationExecutor.execute_agents called with pattern={pattern}")
        print(f"[DEBUG] CollaborationExecutor received {len(agent_tasks)} tasks: {[task[0] for task in agent_tasks]}")
        
        if pattern == "sequential":
            print(f"[DEBUG] CollaborationExecutor: Using sequential execution")
            async for event in self._execute_sequential(agent_tasks, state, agent_start_times):
                yield event
        elif pattern == "hierarchical":
            print(f"[DEBUG] CollaborationExecutor: Using hierarchical execution")
            async for event in self._execute_hierarchical(agent_tasks, state, agent_start_times):
                yield event
        else:  # parallel
            print(f"[DEBUG] CollaborationExecutor: Using parallel execution")
            async for event in self._execute_parallel(agent_tasks, state, agent_start_times):
                yield event
        
        print(f"[DEBUG] CollaborationExecutor.execute_agents finished for pattern={pattern}")
    
    async def _execute_sequential(self, agent_tasks: List[Tuple[str, Any]], 
                                state: Dict, agent_start_times: Dict):
        """Execute agents one after another"""
        logger.info(f"[SEQUENTIAL] Executing {len(agent_tasks)} agents sequentially")
        print(f"[DEBUG] _execute_sequential: Processing {len(agent_tasks)} agents")
        
        for agent_name, agent_gen in agent_tasks:
            logger.info(f"[SEQUENTIAL] Starting agent: {agent_name}")
            print(f"[DEBUG] _execute_sequential: About to iterate over generator for {agent_name}")
            print(f"[DEBUG] _execute_sequential: Generator type for {agent_name}: {type(agent_gen)}")
            
            # Execute agent and collect response
            agent_response = None
            event_count = 0
            print(f"[DEBUG] _execute_sequential: Starting async iteration for {agent_name}")
            async for event in agent_gen:
                event_count += 1
                if isinstance(event, dict):
                    # Update metrics and state
                    if event.get("type") == "agent_complete":
                        agent_response = event.get("content", "")
                        print(f"[DEBUG] CollaborationExecutor: Agent {agent_name} completed with content length: {len(agent_response)}")
                        print(f"[DEBUG] CollaborationExecutor: Content preview: {agent_response[:100]!r}")
                        self._update_agent_state(event, state, agent_start_times)
                    elif event.get("type") == "agent_token":
                        # Tokens are forwarded without logging for performance
                    yield event
                else:
                    logger.warning(f"[SEQUENTIAL] Agent {agent_name} yielded non-dict: {type(event)}")
            
            # Make response available to next agents
            if agent_response and len(agent_tasks) > 1:
                state["previous_agent_response"] = {
                    "agent": agent_name,
                    "response": agent_response
                }
            
            logger.info(f"[SEQUENTIAL] Completed agent: {agent_name}")
    
    async def _execute_hierarchical(self, agent_tasks: List[Tuple[str, Any]], 
                                  state: Dict, agent_start_times: Dict):
        """Execute agents based on dependencies"""
        logger.info(f"[HIERARCHICAL] Executing {len(agent_tasks)} agents with dependencies")
        
        # For now, execute like sequential but with dependency awareness
        # TODO: Implement proper dependency resolution
        dependencies = state.get("agent_dependencies", {})
        execution_order = state.get("execution_order", [task[0] for task in agent_tasks])
        
        # Reorder tasks based on execution order
        ordered_tasks = []
        task_map = {name: gen for name, gen in agent_tasks}
        
        for agent_name in execution_order:
            if agent_name in task_map:
                ordered_tasks.append((agent_name, task_map[agent_name]))
        
        # Execute in order
        async for event in self._execute_sequential(ordered_tasks, state, agent_start_times):
            yield event
    
    async def _execute_parallel(self, agent_tasks: List[Tuple[str, Any]], 
                              state: Dict, agent_start_times: Dict):
        """Execute agents in parallel with queue management"""
        from app.core.agent_queue import agent_queue
        
        # Use queue for more than 2 agents
        if len(agent_tasks) > 2:
            logger.info(f"[PARALLEL] Using agent queue for {len(agent_tasks)} agents")
            async for event in agent_queue.execute_agents_parallel(agent_tasks):
                if isinstance(event, dict):
                    if event.get("type") == "agent_complete":
                        self._update_agent_state(event, state, agent_start_times)
                    yield event
        else:
            # Direct parallel execution for 2 or fewer agents
            logger.info(f"[PARALLEL] Direct execution for {len(agent_tasks)} agents")
            async for event in self._direct_parallel_execution(agent_tasks, state, agent_start_times):
                yield event
    
    async def _direct_parallel_execution(self, agent_tasks: List[Tuple[str, Any]], 
                                       state: Dict, agent_start_times: Dict):
        """Direct parallel execution without queue"""
        event_queue = asyncio.Queue()
        active_tasks = []
        
        async def agent_worker(name: str, gen):
            try:
                async for event in gen:
                    if isinstance(event, dict):
                        await event_queue.put(event)
            except Exception as e:
                await event_queue.put({
                    "type": "agent_error",
                    "agent": name,
                    "error": str(e)
                })
            finally:
                await event_queue.put({"type": "agent_done", "agent": name})
        
        # Start all tasks
        for name, gen in agent_tasks:
            task = asyncio.create_task(agent_worker(name, gen))
            active_tasks.append(task)
        
        # Process events
        completed = 0
        total = len(agent_tasks)
        
        while completed < total:
            event = await event_queue.get()
            if event.get("type") == "agent_done":
                completed += 1
            else:
                if event.get("type") == "agent_complete":
                    self._update_agent_state(event, state, agent_start_times)
                yield event
        
        await asyncio.gather(*active_tasks, return_exceptions=True)
    
    def _update_agent_state(self, event: Dict, state: Dict, agent_start_times: Dict):
        """Update agent state and metrics"""
        agent_name = event.get("agent")
        if not agent_name:
            return
            
        # Update agent outputs
        if agent_name not in state["agent_outputs"]:
            state["agent_outputs"][agent_name] = {}
        state["agent_outputs"][agent_name]["response"] = event.get("content", "")
        
        # Update metrics
        if agent_name in agent_start_times:
            end_time = datetime.now()
            duration = (end_time - agent_start_times[agent_name]).total_seconds()
            
            state["agent_metrics"][agent_name]["end_time"] = end_time.isoformat()
            state["agent_metrics"][agent_name]["duration"] = duration
            state["agent_metrics"][agent_name]["status"] = "completed"
            
            # Add timing to event
            event["duration"] = duration
            event["end_time"] = end_time.isoformat()