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
from app.core.langfuse_integration import get_tracer

logger = logging.getLogger(__name__)


class PipelineBridgeAdapter:
    """Adapter to use PipelineMultiAgentBridge with current pipeline structure"""
    
    def __init__(self, pipeline_config: Dict[str, Any], execution_id: str, trace=None):
        self.pipeline_config = pipeline_config
        self.execution_id = execution_id
        self.pipeline_id = str(pipeline_config.get("id", "unknown"))
        self.trace = trace  # Store pipeline trace
        self.bridge = PipelineMultiAgentBridge(self.pipeline_id, execution_id, trace=trace)
        self.agent_spans = {}  # Track individual agent spans
        
    async def execute_with_io_tracking(
        self, 
        query: str, 
        agents: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute pipeline with full I/O tracking"""
        
        # Get tracer for Langfuse integration
        tracer = get_tracer()
        
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
            conversation_id=self.execution_id,
            trace=self.bridge.trace
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
        
        # Create pipeline workflow span if trace is available
        pipeline_workflow_span = None
        if self.trace and tracer.is_enabled():
            try:
                pipeline_workflow_span = tracer.create_multi_agent_workflow_span(
                    self.trace,
                    self.pipeline_config.get("collaboration_mode", "sequential"),
                    [agent.get("agent_name", "unknown") for agent in agents]
                )
                logger.info(f"[LANGFUSE] Created pipeline workflow span for pipeline {self.pipeline_id}")
            except Exception as e:
                logger.warning(f"[LANGFUSE] Failed to create pipeline workflow span: {e}")
        
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
                    
                    # Create agent generation for Langfuse tracking (LLM responses should be generations)
                    if self.trace and tracer.is_enabled():
                        try:
                            agent_input_query = query if current_agent_idx == 0 else previous_outputs[-1]["output"] if previous_outputs else query
                            agent_generation = tracer.create_generation_with_usage(
                                trace=self.trace,
                                name=f"agent-{agent_name}",
                                model="qwen3:30b-a3b",  # Get from agent config if available
                                input_text=agent_input_query,
                                metadata={
                                    "pipeline_id": self.pipeline_id,
                                    "execution_id": self.execution_id,
                                    "agent_index": current_agent_idx,
                                    "total_agents": total_agents,
                                    "tools_available": agent_info.get("config", {}).get("tools", []),
                                    "collaboration_mode": self.pipeline_config.get("collaboration_mode", "sequential"),
                                    "agent_role": agent_info.get("config", {}).get("role", "")
                                }
                            )
                            self.agent_spans[agent_name] = agent_generation
                            logger.info(f"[LANGFUSE] Created agent generation for {agent_name} in pipeline {self.pipeline_id}")
                        except Exception as e:
                            logger.warning(f"[LANGFUSE] Failed to create agent generation for {agent_name}: {e}")
                    
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
                
                # Extract response properly - check multiple possible fields
                agent_response = (
                    agent_data.get("response") or 
                    agent_data.get("content") or 
                    agent_data.get("output") or
                    ""
                )
                
                # End agent generation with results
                if agent_name in self.agent_spans and tracer.is_enabled():
                    try:
                        agent_generation = self.agent_spans[agent_name]
                        
                        # Estimate token usage for cost calculation
                        agent_input = query if current_agent_idx == 0 else previous_outputs[-1]["output"] if previous_outputs else query
                        usage = tracer.estimate_token_usage(agent_input, agent_response)
                        
                        # End generation with proper metadata
                        agent_generation.end(
                            output=agent_response,
                            usage_details=usage,
                            metadata={
                                "success": True,
                                "response_length": len(agent_response),
                                "tools_used": agent_data.get("tools_used", []),
                                "duration": agent_data.get("duration", 0),
                                "agent_index": current_agent_idx,
                                "completed_agents": completed_agents,
                                "pipeline_id": self.pipeline_id
                            }
                        )
                        logger.info(f"[LANGFUSE] Ended agent generation for {agent_name} in pipeline {self.pipeline_id}")
                        
                        # Remove from tracking
                        del self.agent_spans[agent_name]
                        
                    except Exception as e:
                        logger.warning(f"[LANGFUSE] Failed to end agent generation for {agent_name}: {e}")
                
                # Log for debugging using proper logger
                logger.info(f"[BRIDGE DEBUG] Agent {agent_name} response fields: {list(agent_data.keys())}")
                logger.info(f"[BRIDGE DEBUG] Agent {agent_name} response length: {len(agent_response)}")
                
                # Create detailed output data with safe handling
                try:
                    # Ensure all fields are properly formatted for Langfuse
                    structured_data = agent_data.get("parsed_response")
                    if structured_data is None:
                        structured_data = {}
                    
                    tools_used = agent_data.get("tools_used")
                    if tools_used is None:
                        tools_used = []
                    
                    tokens_used = agent_data.get("tokens_used")
                    if tokens_used is not None and not isinstance(tokens_used, (int, float)):
                        tokens_used = None
                    
                    cost = agent_data.get("cost")
                    if cost is not None and not isinstance(cost, (int, float)):
                        cost = None
                    
                    output_data = AgentOutput(
                        response=agent_response,
                        structured_data=structured_data,
                        tools_used=tools_used,
                        tokens_used=tokens_used,
                        cost=cost
                    )
                except Exception as e:
                    logger.warning(f"[BRIDGE DEBUG] Error creating AgentOutput for {agent_name}: {e}")
                    # Fallback to minimal output data
                    output_data = AgentOutput(
                        response=agent_response,
                        structured_data={},
                        tools_used=[],
                        tokens_used=None,
                        cost=None
                    )
                
                # Publish detailed I/O update with error handling
                try:
                    await self.bridge.publish_agent_io_update(
                        agent_name, 
                        "completed", 
                        output_data=output_data
                    )
                except Exception as e:
                    logger.warning(f"[BRIDGE DEBUG] Error publishing I/O update for {agent_name}: {e}")
                    # Continue execution even if publishing fails
                
                # Store for next agent with proper output
                previous_outputs.append({
                    "agent": agent_name,
                    "output": agent_response,
                    "content": agent_response  # Ensure both fields are available
                })
                current_agent_idx += 1
                
                # Transform to expected format with proper response field
                yield {
                    "type": "agent_complete",
                    "data": {
                        "execution_id": self.execution_id,
                        "agent": agent_name,
                        "progress": (completed_agents / total_agents) * 100,
                        "response": agent_response,
                        "content": agent_response,  # Add content field for compatibility
                        "duration": agent_data.get("duration", 0),
                        "tools_used": agent_data.get("tools_used", [])
                    }
                }
                
            elif event_type == "pipeline_complete":
                # End pipeline workflow span
                if pipeline_workflow_span and tracer.is_enabled():
                    try:
                        execution_summary = event.get("data", {}).get("execution_summary", {})
                        tracer.end_span_with_result(
                            pipeline_workflow_span,
                            {
                                "pipeline_id": self.pipeline_id,
                                "execution_id": self.execution_id,
                                "total_agents": total_agents,
                                "completed_agents": completed_agents,
                                "collaboration_mode": self.pipeline_config.get("collaboration_mode", "sequential"),
                                "execution_summary": execution_summary
                            },
                            success=True
                        )
                        logger.info(f"[LANGFUSE] Ended pipeline workflow span for pipeline {self.pipeline_id}")
                    except Exception as e:
                        logger.warning(f"[LANGFUSE] Failed to end pipeline workflow span: {e}")
                
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
                # End any active agent spans with error
                error_message = event.get("data", {}).get("error", "Unknown error")
                
                if current_agent_idx < len(agents):
                    agent_name = agents[current_agent_idx].get("agent_name")
                    
                    # End agent generation with error if exists
                    if agent_name in self.agent_spans and tracer.is_enabled():
                        try:
                            agent_generation = self.agent_spans[agent_name]
                            agent_generation.end(
                                output=f"Error: {error_message}",
                                metadata={
                                    "success": False,
                                    "error": error_message,
                                    "pipeline_id": self.pipeline_id
                                }
                            )
                            logger.info(f"[LANGFUSE] Ended agent generation with error for {agent_name}: {error_message}")
                            del self.agent_spans[agent_name]
                        except Exception as e:
                            logger.warning(f"[LANGFUSE] Failed to end agent generation with error for {agent_name}: {e}")
                    
                    # Publish error for I/O tracking
                    await self.bridge.publish_agent_io_update(
                        agent_name,
                        "error",
                        error=error_message
                    )
                
                # End pipeline workflow span with error
                if pipeline_workflow_span and tracer.is_enabled():
                    try:
                        tracer.end_span_with_result(
                            pipeline_workflow_span,
                            None,
                            success=False,
                            error=error_message
                        )
                        logger.info(f"[LANGFUSE] Ended pipeline workflow span with error: {error_message}")
                    except Exception as e:
                        logger.warning(f"[LANGFUSE] Failed to end pipeline workflow span with error: {e}")
                
                yield {
                    "type": "error",
                    "data": event.get("data", {})
                }