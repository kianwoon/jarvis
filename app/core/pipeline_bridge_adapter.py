"""
Pipeline Bridge Adapter

Adapts the current pipeline structure to work with the PipelineMultiAgentBridge
for proper I/O tracking.
"""

import json
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from app.core.pipeline_multi_agent_bridge import (
    PipelineMultiAgentBridge, 
    AgentInput, 
    AgentOutput,
    ToolExecution
)
from app.langchain.enhanced_multi_agent_system import EnhancedMultiAgentSystem
from app.core.db import SessionLocal
from sqlalchemy import text

logger = logging.getLogger(__name__)


class PipelineBridgeAdapter:
    """Adapter to use PipelineMultiAgentBridge with current pipeline structure"""
    
    def __init__(self, pipeline_config: Dict[str, Any], execution_id: str):
        self.pipeline_config = pipeline_config
        self.execution_id = execution_id
        self.pipeline_id = str(pipeline_config.get("id", "unknown"))
        self.bridge = PipelineMultiAgentBridge(self.pipeline_id, execution_id)
        
    async def execute_with_io_tracking(
        self, 
        query: str, 
        agents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute pipeline with full I/O tracking"""
        
        # Override the bridge's initialize to use our config
        self.bridge.pipeline_config = self.pipeline_config
        self.bridge.agent_sequence = [
            {
                "agent": agent.get("agent_name"),
                "tools": agent.get("config", {}).get("tools", []),
                "role": agent.get("config", {}).get("role", ""),
                "config": agent.get("config", {})
            }
            for agent in agents
        ]
        
        # Initialize multi-agent system
        self.bridge.multi_agent_system = EnhancedMultiAgentSystem(
            conversation_id=self.execution_id
        )
        
        # Create pipeline configuration for execution
        pipeline_config = {
            "pipeline": {
                "id": self.pipeline_id,
                "name": self.pipeline_config.get("name", "Pipeline"),
                "agents": self.bridge.agent_sequence
            },
            "agents": self.bridge.agent_sequence,
            "context": context or {}
        }
        
        # Track execution
        total_agents = len(agents)
        completed_agents = 0
        current_agent_idx = 0
        previous_outputs = []
        
        # Execute through multi-agent system
        async for event in self.bridge.multi_agent_system.execute(
            query, 
            mode="sequential",
            config=pipeline_config
        ):
            event_type = event.get("event", "")
            
            if event_type == "agent_start":
                # Enhanced with I/O tracking
                agent_data = event.get("data", {})
                agent_name = agent_data.get("agent")
                
                if agent_name and current_agent_idx < len(agents):
                    agent_info = agents[current_agent_idx]
                    
                    # Create detailed input data
                    input_data = AgentInput(
                        query=query if current_agent_idx == 0 else previous_outputs[-1]["output"] if previous_outputs else query,
                        context=context or {},
                        previous_outputs=previous_outputs.copy(),
                        tools_available=agent_info.get("config", {}).get("tools", []),
                        pipeline_context={
                            "pipeline_id": self.pipeline_id,
                            "execution_id": self.execution_id,
                            "agent_index": current_agent_idx,
                            "total_agents": total_agents,
                            "started_at": datetime.now().isoformat()
                        }
                    )
                    
                    # Publish detailed I/O update
                    await self.bridge.publish_agent_io_update(
                        agent_name, 
                        "running", 
                        input_data=input_data
                    )
                
                yield event
                
            elif event_type == "streaming":
                # Forward streaming tokens
                yield {
                    "type": "agent_token",
                    "data": event.get("data", {})
                }
                
            elif event_type == "agent_complete":
                completed_agents += 1
                agent_data = event.get("data", {})
                agent_name = agent_data.get("agent")
                
                # Create detailed output data
                output_data = AgentOutput(
                    response=agent_data.get("response", ""),
                    structured_data=agent_data.get("parsed_response"),
                    tools_used=[],  # TODO: Extract from execution
                    tokens_used=agent_data.get("tokens_used"),
                    cost=agent_data.get("cost")
                )
                
                # Publish detailed I/O update
                await self.bridge.publish_agent_io_update(
                    agent_name, 
                    "completed", 
                    output_data=output_data
                )
                
                # Store for next agent
                previous_outputs.append({
                    "agent": agent_name,
                    "output": agent_data.get("response", "")
                })
                current_agent_idx += 1
                
                # Transform to expected format
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
                yield {
                    "type": "pipeline_complete",
                    "data": {
                        "execution_id": self.execution_id,
                        "pipeline_id": self.pipeline_id,
                        "summary": event.get("data", {}).get("execution_summary", {}),
                        "total_agents": total_agents
                    }
                }
                
            elif event_type == "error":
                # Publish error for current agent if applicable
                if current_agent_idx < len(agents):
                    agent_name = agents[current_agent_idx].get("agent_name")
                    await self.bridge.publish_agent_io_update(
                        agent_name,
                        "error",
                        error=event.get("data", {}).get("error", "Unknown error")
                    )
                
                yield {
                    "type": "error",
                    "data": event.get("data", {})
                }