"""
Agent Execution Queue for managing parallel agent execution with resource constraints
"""
import asyncio
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AgentExecutionQueue:
    """
    Manages agent execution with concurrency control to prevent LLM overload
    while maintaining the benefits of parallel collaboration patterns
    """
    
    def __init__(self, max_concurrent: int = 2):
        """
        Initialize the execution queue
        
        Args:
            max_concurrent: Maximum number of agents that can execute simultaneously
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_agents: Dict[str, datetime] = {}
        self.queue_metrics = {
            "total_queued": 0,
            "total_executed": 0,
            "total_errors": 0,
            "average_wait_time": 0
        }
    
    async def execute_agent(self, agent_name: str, agent_func: Callable, *args, **kwargs):
        """
        Execute an agent function with queue management
        
        Args:
            agent_name: Name of the agent
            agent_func: Async generator function to execute
            *args, **kwargs: Arguments to pass to the agent function
            
        Yields:
            Events from the agent execution
        """
        queue_start = datetime.now()
        self.queue_metrics["total_queued"] += 1
        
        # Wait for available slot
        async with self.semaphore:
            wait_time = (datetime.now() - queue_start).total_seconds()
            
            # Update average wait time
            current_avg = self.queue_metrics["average_wait_time"]
            total_executed = self.queue_metrics["total_executed"]
            new_avg = (current_avg * total_executed + wait_time) / (total_executed + 1)
            self.queue_metrics["average_wait_time"] = new_avg
            
            logger.info(f"[QUEUE] Agent {agent_name} starting execution after {wait_time:.2f}s wait")
            
            # Track active agent
            self.active_agents[agent_name] = datetime.now()
            
            try:
                # Execute agent
                async for event in agent_func(*args, **kwargs):
                    yield event
                    
                self.queue_metrics["total_executed"] += 1
                
            except Exception as e:
                logger.error(f"[QUEUE] Agent {agent_name} failed: {e}")
                self.queue_metrics["total_errors"] += 1
                yield {
                    "type": "agent_error",
                    "agent": agent_name,
                    "error": str(e)
                }
                
            finally:
                # Remove from active agents
                if agent_name in self.active_agents:
                    del self.active_agents[agent_name]
                    
                logger.info(f"[QUEUE] Agent {agent_name} completed. Active agents: {list(self.active_agents.keys())}")
    
    async def execute_agents_parallel(self, agent_tasks: List[tuple]):
        """
        Execute multiple agents in parallel with queue management
        
        Args:
            agent_tasks: List of (agent_name, agent_generator) tuples
            
        Yields:
            Events from all agents
        """
        # Create queue for collecting events
        event_queue = asyncio.Queue()
        active_tasks = []
        
        async def agent_worker(name: str, gen):
            """Worker that executes agent and puts events in queue"""
            try:
                async for event in self.execute_agent(name, lambda: gen):
                    if isinstance(event, dict):
                        await event_queue.put(event)
                    else:
                        logger.warning(f"Agent {name} yielded non-dict event: {type(event)}")
            except Exception as e:
                logger.error(f"Agent worker {name} failed: {e}")
                await event_queue.put({
                    "type": "agent_error",
                    "agent": name,
                    "error": str(e)
                })
            finally:
                await event_queue.put({"type": "agent_done", "agent": name})
        
        # Start all agent tasks
        for name, gen in agent_tasks:
            task = asyncio.create_task(agent_worker(name, gen))
            active_tasks.append(task)
        
        # Track completed agents
        completed_agents = 0
        total_agents = len(agent_tasks)
        
        # Process events from queue
        while completed_agents < total_agents:
            event = await event_queue.get()
            
            if event.get("type") == "agent_done":
                completed_agents += 1
                logger.info(f"[QUEUE] Progress: {completed_agents}/{total_agents} agents completed")
            else:
                yield event
        
        # Wait for all tasks to complete
        await asyncio.gather(*active_tasks, return_exceptions=True)
        
        # Log final metrics
        logger.info(f"[QUEUE] Execution complete. Metrics: {self.queue_metrics}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue performance metrics"""
        return {
            **self.queue_metrics,
            "currently_active": len(self.active_agents),
            "active_agents": list(self.active_agents.keys())
        }

# Global instance for shared use
# Increased from 2 to 3 to better handle multi-agent scenarios with 5+ agents
agent_queue = AgentExecutionQueue(max_concurrent=3)