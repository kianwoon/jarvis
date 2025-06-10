"""
Enhanced Pipeline Executor with Full I/O Tracking

This module extends the pipeline execution to track complete input/output
data for each agent, enabling better debugging and pipeline editing.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, asdict

from app.core.redis_client import get_redis_client
from app.core.db import SessionLocal
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class AgentInput:
    """Structured agent input data"""
    query: str
    context: Dict[str, Any]
    previous_outputs: List[Dict[str, Any]]
    tools_available: List[str]
    pipeline_context: Dict[str, Any]


@dataclass
class ToolExecution:
    """Tool execution details"""
    tool: str
    input: Any
    output: Any
    duration: float
    error: Optional[str] = None


@dataclass
class AgentOutput:
    """Structured agent output data"""
    response: str
    structured_data: Optional[Dict[str, Any]] = None
    tools_used: List[ToolExecution] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class EnhancedPipelineExecutor:
    """Enhanced pipeline executor with full I/O tracking"""
    
    def __init__(self, pipeline_id: str, execution_id: str):
        self.pipeline_id = pipeline_id
        self.execution_id = execution_id
        self.redis_client = get_redis_client()
        self.agent_inputs = {}
        self.agent_outputs = {}
        
    async def publish_agent_io_update(self, agent_name: str, status: str,
                                     input_data: Optional[AgentInput] = None,
                                     output_data: Optional[AgentOutput] = None,
                                     error: Optional[str] = None):
        """Publish detailed agent I/O update to Redis"""
        
        update_data = {
            "agent_name": agent_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add input data if starting
        if input_data and status == "running":
            self.agent_inputs[agent_name] = input_data
            update_data["input"] = asdict(input_data)
            
        # Add output data if completed
        if output_data and status == "completed":
            self.agent_outputs[agent_name] = output_data
            update_data["output"] = {
                "response": output_data.response,
                "structured_data": output_data.structured_data,
                "tools_used": [asdict(t) for t in (output_data.tools_used or [])],
                "tokens_used": output_data.tokens_used,
                "cost": output_data.cost,
            }
            
        # Add error if failed
        if error and status == "error":
            update_data["error"] = error
            
        # Include timing information
        if agent_name in self.agent_inputs:
            update_data["started_at"] = self.agent_inputs[agent_name].pipeline_context.get("started_at")
            if status in ["completed", "error"]:
                started = update_data["started_at"]
                if started:
                    duration = (datetime.now() - datetime.fromisoformat(started)).total_seconds()
                    update_data["execution_time"] = duration
                    update_data["completed_at"] = datetime.now().isoformat()
        
        # Publish to Redis
        if self.redis_client:
            channel = f"pipeline_execution:{self.execution_id}"
            message = {
                "type": "agent_io_update",
                "payload": {
                    "agent": agent_name,
                    "update": update_data
                }
            }
            self.redis_client.publish(channel, json.dumps(message))
            
    async def store_agent_io_history(self, agent_name: str):
        """Store complete agent I/O in database for history"""
        
        if agent_name not in self.agent_inputs or agent_name not in self.agent_outputs:
            return
            
        input_data = self.agent_inputs[agent_name]
        output_data = self.agent_outputs[agent_name]
        
        db = SessionLocal()
        try:
            # Store in new detailed execution table
            db.execute(
                text("""
                INSERT INTO pipeline_agent_executions 
                (execution_id, agent_name, input_data, output_data, 
                 tools_used, tokens_used, cost, execution_time, created_at)
                VALUES (:execution_id, :agent_name, :input_data, :output_data,
                        :tools_used, :tokens_used, :cost, :execution_time, :created_at)
                """),
                {
                    "execution_id": self.execution_id,
                    "agent_name": agent_name,
                    "input_data": json.dumps(asdict(input_data)),
                    "output_data": json.dumps({
                        "response": output_data.response,
                        "structured_data": output_data.structured_data
                    }),
                    "tools_used": json.dumps([asdict(t) for t in (output_data.tools_used or [])]),
                    "tokens_used": output_data.tokens_used,
                    "cost": output_data.cost,
                    "execution_time": input_data.pipeline_context.get("execution_time"),
                    "created_at": datetime.now()
                }
            )
            db.commit()
        except Exception as e:
            logger.error(f"Failed to store agent I/O history: {e}")
            db.rollback()
        finally:
            db.close()
            
    async def replay_from_agent(self, agent_name: str, modified_input: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Replay pipeline execution from a specific agent with modified input"""
        
        # Load pipeline configuration
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                SELECT agent_sequence FROM pipeline_templates 
                WHERE id = :pipeline_id
                """),
                {"pipeline_id": self.pipeline_id}
            ).first()
            
            if not result:
                yield {
                    "type": "error",
                    "data": {"error": "Pipeline not found"}
                }
                return
                
            agent_sequence = json.loads(result[0])
            
            # Find the starting point
            start_index = None
            for i, agent_info in enumerate(agent_sequence):
                if agent_info["agent"] == agent_name:
                    start_index = i
                    break
                    
            if start_index is None:
                yield {
                    "type": "error", 
                    "data": {"error": f"Agent {agent_name} not found in pipeline"}
                }
                return
                
            # Create new execution ID for replay
            replay_execution_id = f"{self.execution_id}_replay_{int(time.time())}"
            
            # Execute from the specified agent onwards
            for i in range(start_index, len(agent_sequence)):
                agent_info = agent_sequence[i]
                current_agent = agent_info["agent"]
                
                # Use modified input for the starting agent
                if i == start_index:
                    input_data = AgentInput(**modified_input)
                else:
                    # Use output from previous agent
                    prev_agent = agent_sequence[i-1]["agent"]
                    if prev_agent in self.agent_outputs:
                        input_data = AgentInput(
                            query=modified_input.get("query", ""),
                            context=modified_input.get("context", {}),
                            previous_outputs=[{
                                "agent": prev_agent,
                                "output": self.agent_outputs[prev_agent].response
                            }],
                            tools_available=agent_info.get("tools", []),
                            pipeline_context={"started_at": datetime.now().isoformat()}
                        )
                        
                yield {
                    "type": "replay_progress",
                    "data": {
                        "execution_id": replay_execution_id,
                        "current_agent": current_agent,
                        "agent_index": i,
                        "total_agents": len(agent_sequence)
                    }
                }
                
        finally:
            db.close()


async def create_enhanced_executor(pipeline_id: str, execution_id: str) -> EnhancedPipelineExecutor:
    """Factory function to create enhanced executor"""
    executor = EnhancedPipelineExecutor(pipeline_id, execution_id)
    return executor