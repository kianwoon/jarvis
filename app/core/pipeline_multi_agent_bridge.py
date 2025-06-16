"""
Bridge between Pipeline Execution and Enhanced Multi-Agent System

This module connects the pipeline execution infrastructure with the
enhanced multi-agent system for seamless integration.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, asdict

from app.langchain.enhanced_multi_agent_system import EnhancedMultiAgentSystem
from app.agents.agent_contracts import AgentContract, create_agent_contract
from app.core.langgraph_agents_cache import get_agent_by_name
from app.core.db import SessionLocal
from app.core.redis_client import get_redis_client
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


class PipelineMultiAgentBridge:
    """Bridges pipeline execution with multi-agent system"""
    
    def __init__(self, pipeline_id: str, execution_id: str, trace=None):
        self.pipeline_id = pipeline_id
        self.execution_id = execution_id
        self.trace = trace
        self.multi_agent_system = None
        self.agent_contracts = {}
        self.redis_client = get_redis_client()
        self.agent_inputs = {}
        self.agent_outputs = {}
        self.agent_start_times = {}
        
    async def initialize(self):
        """Initialize the bridge with pipeline configuration"""
        
        # Create enhanced multi-agent system
        self.multi_agent_system = EnhancedMultiAgentSystem(
            conversation_id=self.execution_id,
            trace=self.trace
        )
        
        # Load pipeline configuration from database
        db = SessionLocal()
        try:
            # Get pipeline template
            result = db.execute(
                text("""
                SELECT * FROM pipeline_templates 
                WHERE id = :pipeline_id AND is_active = true
                """),
                {"pipeline_id": self.pipeline_id}
            ).first()
            
            if result:
                self.pipeline_config = dict(result._mapping)
                self.agent_sequence = json.loads(self.pipeline_config.get("agent_sequence", "[]"))
                
                # Load agent templates and create contracts
                await self._load_agent_contracts()
        finally:
            db.close()
    
    async def _load_agent_contracts(self):
        """Load agent contracts from templates"""
        
        db = SessionLocal()
        try:
            for agent_info in self.agent_sequence:
                agent_name = agent_info.get("agent")
                
                # Get agent template
                template = db.execute(
                    text("""
                    SELECT * FROM agent_templates 
                    WHERE name = :name
                    """),
                    {"name": agent_name}
                ).first()
                
                if template:
                    # Create agent contract from template
                    template_dict = dict(template._mapping)
                    contract = create_agent_contract(
                        name=template_dict["name"],
                        description=template_dict["description"],
                        instructions=template_dict["default_instructions"],
                        capabilities=json.loads(template_dict["capabilities"]),
                        expected_input=json.loads(template_dict["expected_input"]),
                        output_format=json.loads(template_dict["output_format"]),
                        tools=agent_info.get("tools", [])
                    )
                    
                    self.agent_contracts[agent_name] = contract
        finally:
            db.close()
    
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
            self.agent_start_times[agent_name] = datetime.now()
            update_data["input"] = asdict(input_data)
            update_data["started_at"] = self.agent_start_times[agent_name].isoformat()
            
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
            
            # Calculate execution time
            if agent_name in self.agent_start_times:
                duration = (datetime.now() - self.agent_start_times[agent_name]).total_seconds()
                update_data["execution_time"] = duration
                update_data["completed_at"] = datetime.now().isoformat()
            
        # Add error if failed
        if error and status == "error":
            update_data["error"] = error
            if agent_name in self.agent_start_times:
                duration = (datetime.now() - self.agent_start_times[agent_name]).total_seconds()
                update_data["execution_time"] = duration
                update_data["failed_at"] = datetime.now().isoformat()
        
        
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
    
    async def execute_pipeline(self, initial_query: str, 
                             context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute pipeline using enhanced multi-agent system"""
        
        if not self.multi_agent_system:
            await self.initialize()
        
        self.start_time = datetime.now()
        
        # Create pipeline configuration for multi-agent system
        pipeline_config = {
            "pipeline": {
                "id": self.pipeline_id,
                "name": self.pipeline_config.get("name"),
                "agents": self.agent_sequence
            },
            "agents": self.agent_sequence,
            "context": context or {}
        }
        
        # Track execution progress
        total_agents = len(self.agent_sequence)
        completed_agents = 0
        current_agent_idx = 0
        previous_outputs = []
        
        # Execute through multi-agent system
        async for event in self.multi_agent_system.execute(
            initial_query, 
            mode="sequential",
            config=pipeline_config
        ):
            # Transform events for pipeline execution format
            event_type = event.get("event", "")
            
            if event_type == "agent_start":
                # New event type for agent starting
                agent_data = event.get("data", {})
                agent_name = agent_data.get("agent")
                
                if agent_name and current_agent_idx < len(self.agent_sequence):
                    agent_info = self.agent_sequence[current_agent_idx]
                    
                    # Create input data
                    input_data = AgentInput(
                        query=initial_query if current_agent_idx == 0 else previous_outputs[-1]["output"] if previous_outputs else initial_query,
                        context=context or {},
                        previous_outputs=previous_outputs.copy(),
                        tools_available=agent_info.get("tools", []),
                        pipeline_context={
                            "pipeline_id": self.pipeline_id,
                            "execution_id": self.execution_id,
                            "agent_index": current_agent_idx,
                            "total_agents": total_agents
                        }
                    )
                    
                    # Publish agent starting with input
                    await self.publish_agent_io_update(agent_name, "running", input_data=input_data)
                    
            elif event_type == "streaming":
                # Forward streaming tokens
                yield {
                    "type": "agent_token",
                    "data": {
                        "execution_id": self.execution_id,
                        "agent": event["data"].get("agent"),
                        "content": event["data"].get("content", "")
                    }
                }
                
            elif event_type == "agent_complete":
                completed_agents += 1
                agent_data = event.get("data", {})
                agent_name = agent_data.get("agent")
                
                # Create output data
                output_data = AgentOutput(
                    response=agent_data.get("response", ""),
                    structured_data=agent_data.get("parsed_response"),
                    tools_used=[],  # TODO: Extract from agent execution
                    tokens_used=agent_data.get("tokens_used"),
                    cost=agent_data.get("cost")
                )
                
                # Publish agent completion with output
                await self.publish_agent_io_update(agent_name, "completed", output_data=output_data)
                
                # Store for next agent's input
                previous_outputs.append({
                    "agent": agent_name,
                    "output": agent_data.get("response", "")
                })
                current_agent_idx += 1
                
                
                # Yield completion event
                yield {
                    "type": "agent_complete",
                    "data": {
                        "execution_id": self.execution_id,
                        "agent": agent_name,
                        "progress": (completed_agents / total_agents) * 100,
                        "response": agent_data.get("response"),
                        "duration": agent_data.get("duration", 0)
                    }
                }
                
            elif event_type == "pipeline_complete":
                # Mark execution as complete
                await self._mark_execution_complete()
                
                yield {
                    "type": "pipeline_complete",
                    "data": {
                        "execution_id": self.execution_id,
                        "pipeline_id": self.pipeline_id,
                        "summary": event["data"].get("execution_summary", {}),
                        "total_duration": (datetime.now() - self.start_time).total_seconds()
                    }
                }
                
            elif event_type == "error":
                yield {
                    "type": "error",
                    "data": {
                        "execution_id": self.execution_id,
                        "error": event["data"].get("error", "Unknown error")
                    }
                }
    
    async def _mark_execution_complete(self):
        """Mark pipeline execution as complete"""
        
        db = SessionLocal()
        try:
            db.execute(
                text("""
                UPDATE pipeline_executions 
                SET status = 'completed', completed_at = :completed_at
                WHERE id = :execution_id
                """),
                {
                    "execution_id": self.execution_id,
                    "completed_at": datetime.now()
                }
            )
            db.commit()
        finally:
            db.close()


async def execute_pipeline_with_agents(pipeline_id: str, execution_id: str,
                                      query: str, context: Optional[Dict[str, Any]] = None):
    """Helper function to execute a pipeline with multi-agent system"""
    
    bridge = PipelineMultiAgentBridge(pipeline_id, execution_id)
    
    async for event in bridge.execute_pipeline(query, context):
        yield event