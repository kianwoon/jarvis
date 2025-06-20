"""
Clean Pipeline Executor

CRITICAL: This executor handles ONLY agentic pipeline execution without ANY multi-agent contamination.
It directly executes pipeline agents using ONLY pipeline_agents table configurations.
"""

import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from app.core.pipeline_agents_cache import get_pipeline_agent_config
from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem

logger = logging.getLogger(__name__)

class CleanPipelineExecutor:
    """Clean pipeline executor that NEVER uses multi-agent system components"""
    
    def __init__(self, pipeline_id: int, execution_id: str, trace=None):
        self.pipeline_id = pipeline_id
        self.execution_id = execution_id
        self.trace = trace
        logger.info(f"🔴 [CLEAN PIPELINE] Initialized clean executor for pipeline {pipeline_id}")
        logger.info(f"🔴 [CLEAN PIPELINE] CRITICAL: This executor will NEVER create multi-agent spans!")
    
    async def execute_agents_sequentially(
        self, 
        query: str, 
        agent_sequence: List[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute pipeline agents sequentially using ONLY pipeline_agents table.
        NO multi-agent system contamination.
        """
        
        total_agents = len(agent_sequence)
        previous_outputs = []
        
        logger.info(f"🔴 [CLEAN PIPELINE] Starting sequential execution of {total_agents} agents")
        
        for idx, agent_info in enumerate(agent_sequence):
            agent_name = agent_info.get("agent_name")
            
            logger.info(f"🔴 [CLEAN PIPELINE] Executing agent {idx + 1}/{total_agents}: {agent_name}")
            
            # Emit agent start event
            yield {
                "event": "agent_start",
                "data": {
                    "agent": agent_name,
                    "agent_index": idx,
                    "total_agents": total_agents,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Load agent config from pipeline_agents table ONLY
            agent_data = get_pipeline_agent_config(self.pipeline_id, agent_name)
            if not agent_data:
                error_msg = f"Pipeline agent {agent_name} not found in pipeline_agents table for pipeline {self.pipeline_id}"
                logger.error(f"🔴 [CLEAN PIPELINE] CRITICAL: {error_msg}")
                yield {
                    "event": "error",
                    "data": {"error": error_msg}
                }
                return
            
            logger.info(f"🔴 [CLEAN PIPELINE] Agent {agent_name} loaded with tools: {agent_data.get('tools', [])}")
            logger.info(f"🔴 [CLEAN PIPELINE] Agent {agent_name} system_prompt preview: {agent_data.get('system_prompt', '')[:200]}...")
            logger.info(f"🔴 [CLEAN PIPELINE] Agent {agent_name} full config keys: {list(agent_data.keys())}")
            
            # Build clean context for pipeline execution
            pipeline_context = {
                "pipeline_id": self.pipeline_id,
                "execution_id": self.execution_id,
                "agent_index": idx,
                "total_agents": total_agents,
                "previous_outputs": previous_outputs.copy(),
                "is_first_agent": idx == 0,
                "is_last_agent": idx == total_agents - 1
            }
            
            # Build the clean query for this agent  
            if idx == 0:
                # First agent gets the original query
                base_query = query if query and query.strip() else "Proceed with your configured task."
                agent_query = base_query
            else:
                # Subsequent agents get context from previous agents
                previous_context = "\n\n".join([
                    f"Previous Agent ({output['agent']}):\n{output['response']}"
                    for output in previous_outputs
                ])
                agent_query = f"Original Query: {query}\n\n{previous_context}\n\nYour task as {agent_name}:"
            
            # Execute agent using DynamicMultiAgentSystem (clean path)
            dynamic_system = DynamicMultiAgentSystem(trace=self.trace)
            
            # Build context that will trigger pipeline mode detection
            context = {
                "pipeline_context": pipeline_context,
                "agent_name": agent_name,
                "available_tools": agent_data.get("tools", []),
                "conversation_context": {},
                "agent_data": agent_data
            }
            
            start_time = datetime.now()
            agent_response = ""
            
            # Execute the agent
            
            async for event in dynamic_system.execute_agent(
                agent_name=agent_name,
                agent_data=agent_data,
                query=agent_query,
                context=context
            ):
                event_type = event.get("type")
                
                if event_type == "agent_complete":
                    agent_response = event.get("content", "")
                    duration = (datetime.now() - start_time).total_seconds()
                    tools_used = event.get("tools_used", [])
                    
                    
                    # Store output for next agent
                    previous_outputs.append({
                        "agent": agent_name,
                        "response": agent_response,
                        "duration": duration,
                        "tools_used": tools_used
                    })
                    
                    # Emit agent completion
                    yield {
                        "event": "agent_complete",
                        "data": {
                            "agent": agent_name,
                            "response": agent_response,
                            "duration": duration,
                            "agent_index": idx,
                            "total_agents": total_agents,
                            "tools_used": event.get("tools_used", []),
                            "avatar": event.get("avatar", "🤖"),
                            "description": agent_data.get("description", "")
                        }
                    }
                    
                elif event.get("type") == "agent_token":
                    # Forward streaming tokens
                    yield {
                        "event": "streaming",
                        "data": {
                            "agent": agent_name,
                            "content": event.get("token", "")
                        }
                    }
                    
                elif event.get("type") == "agent_error":
                    logger.error(f"🔴 [CLEAN PIPELINE] Agent {agent_name} error: {event.get('error')}")
                    yield {
                        "event": "error",
                        "data": {
                            "agent": agent_name,
                            "error": event.get("error", "Agent execution failed")
                        }
                    }
                    return
        
        # Pipeline complete
        logger.info(f"🔴 [CLEAN PIPELINE] Pipeline {self.pipeline_id} execution completed successfully")
        yield {
            "event": "pipeline_complete",
            "data": {
                "pipeline_id": self.pipeline_id,
                "execution_id": self.execution_id,
                "total_agents": total_agents,
                "completed_agents": len(previous_outputs),
                "final_output": previous_outputs[-1]["response"] if previous_outputs else "",
                "execution_summary": {
                    "agents_executed": [output["agent"] for output in previous_outputs],
                    "total_duration": sum(output["duration"] for output in previous_outputs),
                    "tools_used": [tool for output in previous_outputs for tool in output["tools_used"]]
                }
            }
        }