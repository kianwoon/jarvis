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
            print(f"[DEBUG] CollaborationExecutor: Using sequential execution (MacBook optimized: 1 agent at a time)")
            async for event in self._execute_sequential(agent_tasks, state, agent_start_times):
                yield event
        elif pattern == "hierarchical":
            print(f"[DEBUG] CollaborationExecutor: Using hierarchical execution (will run sequentially)")
            async for event in self._execute_hierarchical(agent_tasks, state, agent_start_times):
                yield event
        else:  # parallel - but actually sequential with max_concurrent=1
            print(f"[DEBUG] CollaborationExecutor: Using 'parallel' execution (limited to 1 concurrent agent for MacBook)")
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
            agent_completed = False
            agent_had_error = False
            
            print(f"[DEBUG] _execute_sequential: Starting async iteration for {agent_name}")
            
            try:
                async for event in agent_gen:
                    event_count += 1
                    if isinstance(event, dict):
                        event_type = event.get("type")
                        
                        # Handle completion (including timeouts)
                        if event_type == "agent_complete":
                            agent_response = event.get("content", "")
                            agent_completed = True
                            is_timeout = event.get("timeout", False)
                            
                            if is_timeout:
                                print(f"[WARNING] CollaborationExecutor: Agent {agent_name} completed with timeout after {event.get('timeout_duration', 'unknown')}s")
                            else:
                                print(f"[DEBUG] CollaborationExecutor: Agent {agent_name} completed successfully with content length: {len(agent_response)}")
                            
                            print(f"[DEBUG] CollaborationExecutor: Content preview: {agent_response[:100]!r}")
                            self._update_agent_state(event, state, agent_start_times)
                            
                        elif event_type == "agent_error":
                            agent_had_error = True
                            print(f"[ERROR] CollaborationExecutor: Agent {agent_name} had error: {event.get('error', 'Unknown error')}")
                            # Still update state for tracking
                            self._update_agent_state(event, state, agent_start_times)
                            
                        elif event_type == "agent_token":
                            # Tokens are forwarded without logging for performance
                            pass
                            
                        yield event
                    else:
                        logger.warning(f"[SEQUENTIAL] Agent {agent_name} yielded non-dict: {type(event)}")
                        
            except Exception as e:
                print(f"[ERROR] CollaborationExecutor: Exception during {agent_name} execution: {e}")
                agent_had_error = True
                # Create error event if none was yielded
                if not agent_completed:
                    error_event = {
                        "type": "agent_error",
                        "agent": agent_name,
                        "error": f"Execution exception: {str(e)}"
                    }
                    self._update_agent_state(error_event, state, agent_start_times)
                    yield error_event
            
            # Make response available to next agents (even if partial due to timeout)
            if agent_response and len(agent_tasks) > 1:
                state["previous_agent_response"] = {
                    "agent": agent_name,
                    "response": agent_response,
                    "completed": agent_completed,
                    "had_error": agent_had_error
                }
                
                # Emit communication event for UI flow diagram
                # Find current agent index and check if there's a next one
                current_index = -1
                for i, (task_name, _) in enumerate(agent_tasks):
                    if task_name == agent_name:
                        current_index = i
                        break
                
                if current_index >= 0 and current_index < len(agent_tasks) - 1:
                    next_agent = agent_tasks[current_index + 1][0]
                    # Extract key info from response for communication message
                    response_preview = agent_response[:200] + "..." if len(agent_response) > 200 else agent_response
                    if not response_preview:
                        response_preview = "Analysis and findings passed to next agent"
                    
                    yield {
                        "type": "agent_communication",
                        "from_agent": agent_name,
                        "to_agent": next_agent,
                        "message": f"Passing analysis: {response_preview[:100]}..."
                    }
            
            status = "completed" if agent_completed else ("error" if agent_had_error else "unknown")
            logger.info(f"[SEQUENTIAL] Finished agent: {agent_name} (status: {status}, events: {event_count})")
    
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
        
        print(f"[DEBUG] Hierarchical execution - Original tasks: {list(task_map.keys())}")
        print(f"[DEBUG] Hierarchical execution - Execution order: {execution_order}")
        
        for agent_name in execution_order:
            if agent_name in task_map:
                ordered_tasks.append((agent_name, task_map[agent_name]))
                print(f"[DEBUG] Hierarchical execution - Added {agent_name} to ordered_tasks")
            else:
                print(f"[WARNING] Hierarchical execution - Agent {agent_name} in execution_order but not in task_map")
        
        print(f"[DEBUG] Hierarchical execution - Final ordered_tasks: {[name for name, _ in ordered_tasks]}")
        
        if len(ordered_tasks) != len(agent_tasks):
            print(f"[WARNING] Hierarchical execution - Task count mismatch: {len(ordered_tasks)} ordered vs {len(agent_tasks)} original")
            
            # Add any missing tasks that weren't in execution_order
            for agent_name, gen in agent_tasks:
                if not any(name == agent_name for name, _ in ordered_tasks):
                    ordered_tasks.append((agent_name, gen))
                    print(f"[DEBUG] Hierarchical execution - Added missing task: {agent_name}")
        
        # Ensure we execute ALL original tasks, even if execution_order is incomplete
        final_ordered_tasks = ordered_tasks if ordered_tasks else agent_tasks
        print(f"[DEBUG] Hierarchical execution - Executing {len(final_ordered_tasks)} tasks: {[name for name, _ in final_ordered_tasks]}")
        
        # Execute in order
        async for event in self._execute_sequential(final_ordered_tasks, state, agent_start_times):
            yield event
    
    async def _execute_parallel(self, agent_tasks: List[Tuple[str, Any]], 
                              state: Dict, agent_start_times: Dict):
        """Execute agents in parallel with queue management"""
        from app.core.agent_queue import agent_queue
        
        # Always use agent queue for parallel execution to ensure consistency
        logger.info(f"[PARALLEL] Using agent queue for {len(agent_tasks)} agents (max_concurrent=1 for MacBook stability)")
        
        try:
            async for event in agent_queue.execute_agents_parallel(agent_tasks):
                if isinstance(event, dict):
                    event_type = event.get("type")
                    agent_name = event.get("agent")
                    
                    if event_type == "agent_complete":
                        self._update_agent_state(event, state, agent_start_times)
                        print(f"[DEBUG] Parallel execution: Agent {agent_name} completed")
                    elif event_type == "agent_error":
                        self._update_agent_state(event, state, agent_start_times)
                        print(f"[WARNING] Parallel execution: Agent {agent_name} had error")
                    
                    yield event
                else:
                    logger.warning(f"[PARALLEL] Received non-dict event: {type(event)}")
                    
        except Exception as e:
            logger.error(f"[PARALLEL] Queue execution failed: {e}")
            # Fallback to direct execution if queue fails
            logger.info(f"[PARALLEL] Falling back to direct execution for {len(agent_tasks)} agents")
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