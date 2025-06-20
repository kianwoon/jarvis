"""
Enhanced Multi-Agent System with Structured Communication

Integrates agent contracts, orchestration, and communication protocols
"""
import asyncio
import json
import yaml
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator

logger = logging.getLogger(__name__)

# Import after logger is set up to avoid circular imports
try:
    from app.agents.agent_communication import AgentCommunicationProtocol, PipelineContextManager
    from app.agents.pipeline_orchestrator import PipelineOrchestrator
    from app.langchain.multi_agent_system_simple import MultiAgentSystem
except ImportError as e:
    logger.warning(f"Failed to import enhanced components: {e}")
    # Fallback imports
    AgentCommunicationProtocol = None
    PipelineContextManager = None
    PipelineOrchestrator = None
    from app.langchain.multi_agent_system_simple import MultiAgentSystem

class EnhancedMultiAgentSystem(MultiAgentSystem):
    """Enhanced version with structured communication"""
    
    def __init__(self, conversation_id: Optional[str] = None, trace=None):
        super().__init__(conversation_id, trace)
        self.communication_protocol = AgentCommunicationProtocol() if AgentCommunicationProtocol else None
        self.context_manager = None
        self.pipeline_orchestrator = None
        self.agent_behaviors = self._load_agent_behaviors()
        self.enhanced_available = AgentCommunicationProtocol is not None
        # Create agent_registry from parent's agents dict for compatibility
        self.agent_registry = self.agents if hasattr(self, 'agents') else {}
        
        # Tool executor is optional - system works fine without it
        self.tool_executor = None
        
    def _load_agent_behaviors(self) -> Dict[str, Any]:
        """Load agent behavior definitions"""
        try:
            import os
            behaviors_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "agents", "agent_behaviors.yaml"
            )
            with open(behaviors_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load agent behaviors: {e}")
            return {}
    
    async def execute(self, query: str, mode: str = "sequential", 
                     config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute pipeline with enhanced communication"""
        
        # Check if enhanced components are available
        if not self.enhanced_available:
            logger.warning("Enhanced components not available, falling back to standard execution")
            async for event in super().execute(query, mode, config):
                yield event
            return
        
        # Initialize context manager for this execution
        self.context_manager = PipelineContextManager() if PipelineContextManager else None
        
        # CRITICAL: Extract pipeline_id from config for proper mode detection
        self.current_pipeline_id = None
        if config and "pipeline" in config:
            pipeline_config = config["pipeline"]
            self.current_pipeline_id = pipeline_config.get("id")
            logger.info(f"🔴 [PIPELINE DETECTED] Pipeline ID extracted from config: {self.current_pipeline_id}")
            
            # Initialize orchestrator
            if PipelineOrchestrator:
                self.pipeline_orchestrator = PipelineOrchestrator(pipeline_config)
        else:
            logger.info(f"🔵 [MULTI-AGENT DETECTED] No pipeline config found, using multi-agent mode")
        
        # Get agent sequence
        agents = config.get("agents", []) if config else []
        if not agents and config and "pipeline" in config:
            agents = config["pipeline"].get("agents", [])
        
        # Execute based on mode
        if mode == "sequential" and self.context_manager:
            async for event in self._execute_sequential_enhanced(query, agents):
                yield event
        else:
            # Fall back to original implementation for other modes
            async for event in super().execute(query, mode, config):
                yield event
    
    async def _execute_sequential_enhanced(self, query: str, 
                                         agents: List[Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Enhanced sequential execution with structured communication"""
        
        total_agents = len(agents)
        
        for idx, agent in enumerate(agents):
            # Get agent info
            if isinstance(agent, dict):
                agent_name = agent.get("agent", agent.get("name", f"Agent_{idx}"))
                agent_config = agent
            else:
                agent_name = str(agent)
                agent_config = {"name": agent_name}
            
            # Get next agent info for context
            next_agent_info = None
            if idx < total_agents - 1:
                next_agent = agents[idx + 1]
                if isinstance(next_agent, dict):
                    next_agent_name = next_agent.get("agent", next_agent.get("name"))
                else:
                    next_agent_name = str(next_agent)
                    
                # Get expected input from behaviors
                next_behavior = self.agent_behaviors.get("agent_behaviors", {}).get(next_agent_name, {})
                next_agent_info = {
                    "name": next_agent_name,
                    "role": next_agent.get("role", next_behavior.get("description", "Process information")),
                    "expected_input": next_behavior.get("expected_input", "Structured data from previous agent")
                }
            
            # Get context for this agent
            base_context = self.context_manager.get_context_for_agent(
                agent_name, idx, total_agents, next_agent_info
            ) if self.context_manager else {}
            
            # CRITICAL: Override with actual pipeline_id if this is a pipeline execution
            pipeline_context = base_context.copy()
            if self.current_pipeline_id is not None:
                # Ensure pipeline_id is converted to int for consistency
                pipeline_id = int(self.current_pipeline_id) if isinstance(self.current_pipeline_id, str) else self.current_pipeline_id
                pipeline_context["pipeline_id"] = pipeline_id
                pipeline_context["agent_index"] = idx
                pipeline_context["total_agents"] = total_agents
                logger.info(f"🔴 [PIPELINE CONTEXT] Added pipeline_id {pipeline_id} (type: {type(pipeline_id)}) to context for agent {agent_name} (idx {idx}/{total_agents})")
            else:
                logger.info(f"🔵 [MULTI-AGENT CONTEXT] No pipeline_id for agent {agent_name}")
            
            # Create enhanced prompt
            # First check if there's a system_prompt in the agent's config (pipeline-specific)
            agent_specific_config = agent_config.get("config", {})
            base_prompt = agent_specific_config.get("system_prompt", "")
            
            # If not in config, check top-level agent_config
            if not base_prompt:
                base_prompt = agent_config.get("system_prompt", "")
            
            # Finally fall back to agent registry
            if not base_prompt and agent_name in self.agent_registry:
                base_prompt = self.agent_registry[agent_name].get("system_prompt", "")
                
            logger.debug(f"[ENHANCED] Agent {agent_name} prompt source: {'pipeline-specific' if agent_specific_config.get('system_prompt') else 'registry'}")
            
            # CRITICAL: For pipeline mode, use ONLY the system_prompt from pipeline_agents table
            # Do NOT add hardcoded templates that contaminate the clean prompt
            if self.current_pipeline_id is not None:
                # PIPELINE MODE: Use clean prompt from pipeline_agents table without contamination
                full_prompt = base_prompt
                logger.info(f"🔴 [PIPELINE CLEAN] Using ONLY pipeline_agents system_prompt for {agent_name} - NO hardcoded templates")
            else:
                # MULTI-AGENT MODE: Use enhanced instructions (hardcoded templates are OK here)
                enhanced_instructions = self.communication_protocol.create_agent_instruction(
                    agent_config, pipeline_context
                )
                full_prompt = f"{base_prompt}\n\n{enhanced_instructions}"
                logger.info(f"🔵 [MULTI-AGENT ENHANCED] Using enhanced instructions for {agent_name}")
            
            # Add query context - different handling for pipeline vs multi-agent mode
            if self.current_pipeline_id is not None:
                # PIPELINE MODE: Clean query passing - let pipeline_agents system_prompt handle formatting
                full_prompt += f"\n\n{query}"
                logger.info(f"🔴 [PIPELINE CLEAN] Added clean query to {agent_name} without hardcoded formatting")
            else:
                # MULTI-AGENT MODE: Enhanced query formatting
                if idx == 0:
                    full_prompt += f"\n\nUSER QUERY: {query}"
                else:
                    full_prompt += f"\n\nORIGINAL USER QUERY: {query}"
                    full_prompt += "\n\nYour task based on previous agents' work."
                logger.info(f"🔵 [MULTI-AGENT ENHANCED] Added enhanced query formatting to {agent_name}")
            
            logger.info(f"[ENHANCED] Executing agent {idx + 1}/{total_agents}: {agent_name}")
            
            # Emit agent_start event
            yield {
                "event": "agent_start",
                "data": {
                    "agent": agent_name,
                    "agent_index": idx,
                    "total_agents": total_agents,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Execute agent with enhanced prompt
            agent_state = {
                "query": full_prompt,
                "metadata": {
                    "pipeline_agent_config": agent_config,
                    "pipeline_context": pipeline_context,
                    "agent_name": agent_name
                }
            }
            
            # Call the agent
            async for event in self._execute_single_agent(agent_name, agent_state):
                # Parse the response if it's a completion event
                if event.get("event") == "agent_complete":
                    response = event.get("data", {}).get("response", "")
                    parsed_response = self.communication_protocol.parse_agent_response(response)
                    
                    # Add parsed response to context
                    self.context_manager.add_agent_output(agent_name, parsed_response)
                    
                    # Add parsing info to event
                    event["data"]["parsed_response"] = parsed_response
                    event["data"]["pipeline_position"] = f"{idx + 1}/{total_agents}"
                
                yield event
        
        # Yield final summary
        yield {
            "event": "pipeline_complete",
            "data": {
                "execution_summary": self.context_manager.context,
                "agent_count": total_agents,
                "mode": "sequential_enhanced"
            }
        }
    
    async def _execute_single_agent(self, agent_name: str, 
                                  state: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single agent using DynamicMultiAgentSystem"""
        
        # Import DynamicMultiAgentSystem
        from app.langchain.dynamic_agent_system import DynamicMultiAgentSystem
        from app.core.langgraph_agents_cache import get_agent_by_name
        from app.core.pipeline_agents_cache import get_pipeline_agent_config
        
        # Check if this is a pipeline agent first
        pipeline_context = state.get("metadata", {}).get("pipeline_context", {})
        pipeline_id = pipeline_context.get("pipeline_id")
        
        agent_data = None
        
        if pipeline_id:
            # PIPELINE MODE: Use ONLY pipeline_agents table
            logger.info(f"🔴 [PIPELINE MODE] STRICT SEPARATION: Loading config ONLY from pipeline_agents table for {agent_name} in pipeline {pipeline_id}")
            agent_data = get_pipeline_agent_config(pipeline_id, agent_name)
            if not agent_data:
                logger.error(f"🔴 [PIPELINE MODE] CRITICAL: No config found in pipeline_agents table for {agent_name} in pipeline {pipeline_id}")
                yield {
                    "event": "error",
                    "data": {"error": f"Pipeline agent {agent_name} not found in pipeline {pipeline_id}"}
                }
                return
            logger.info(f"🔴 [PIPELINE MODE] SUCCESS: Agent {agent_name} config loaded with tools: {agent_data.get('tools', [])}")
        else:
            # MULTI-AGENT MODE: Use ONLY langgraph_agents table
            logger.info(f"🔵 [MULTI-AGENT MODE] STRICT SEPARATION: Loading config ONLY from langgraph_agents table for {agent_name}")
            agent_data = get_agent_by_name(agent_name)
            if not agent_data:
                # Try with agent registry if we have it
                if hasattr(self, 'agent_registry') and agent_name in self.agent_registry:
                    agent_data = self.agent_registry[agent_name]
                else:
                    logger.error(f"🔵 [MULTI-AGENT MODE] Agent {agent_name} not found in langgraph_agents table")
                    yield {
                        "event": "error",
                        "data": {"error": f"Multi-agent {agent_name} not found"}
                    }
                    return
            logger.info(f"🔵 [MULTI-AGENT MODE] SUCCESS: Agent {agent_name} config loaded with tools: {agent_data.get('tools', [])}")
        
        # NO CONFIG MIXING - each mode uses its own table exclusively
        
        # Create dynamic system and execute agent
        dynamic_system = DynamicMultiAgentSystem(trace=self.trace)
        
        # Build enhanced context with pipeline information
        context = {
            "pipeline_context": state.get("metadata", {}).get("pipeline_context", {}),
            "agent_name": agent_name,
            "conversation_history": state.get("conversation_history", []),
            "previous_agents": self.context_manager.context.get("agent_outputs", {}) if self.context_manager else {}
        }
        
        # Add any tools from agent config
        if "tools" in agent_data:
            context["available_tools"] = agent_data["tools"]
        
        # Execute agent
        start_time = datetime.now()
        response_text = ""
        
        async for event in dynamic_system.execute_agent(
            agent_name,
            agent_data,
            state.get("query", ""),
            context=context
        ):
            # Transform event format to match expected output
            event_type = event.get("type", "")
            
            if event_type == "agent_complete":
                duration = (datetime.now() - start_time).total_seconds()
                response_text = event.get("content", "")
                
                # Ensure response_text is not empty - check multiple fields
                if not response_text.strip():
                    response_text = (
                        event.get("response") or
                        event.get("output") or
                        "Agent completed but produced no output"
                    )
                
                yield {
                    "event": "agent_complete",
                    "data": {
                        "agent": agent_name,
                        "response": response_text,
                        "content": response_text,  # Add both fields for compatibility
                        "reasoning": event.get("reasoning", ""),
                        "duration": duration,
                        "avatar": event.get("avatar", "🤖"),
                        "description": event.get("description", ""),
                        "timeout": event.get("timeout", False),
                        "tools_used": event.get("tools_used", [])
                    }
                }
            elif event_type == "agent_token":
                # Stream tokens as they come
                yield {
                    "event": "streaming",
                    "data": {
                        "agent": agent_name,
                        "content": event.get("token", "")
                    }
                }
            elif event_type == "agent_error":
                yield {
                    "event": "error",
                    "data": {
                        "agent": agent_name,
                        "error": event.get("error", "Unknown error")
                    }
                }
            else:
                # Pass through other events with proper format
                yield {
                    "event": event_type,
                    "data": event
                }

def create_pipeline_config(pipeline_name: str, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Helper to create a pipeline configuration"""
    
    return {
        "pipeline": {
            "name": pipeline_name,
            "agents": agents,
            "mode": "sequential"
        },
        "agents": agents
    }

# Example usage
CUSTOMER_SERVICE_PIPELINE = create_pipeline_config(
    "customer_service",
    [
        {
            "name": "email_reader",
            "role": "Read and extract customer emails",
            "tools": ["search_emails", "read_email"],
            "system_prompt": "You are an email reader agent. Extract all relevant information from emails."
        },
        {
            "name": "email_responder", 
            "role": "Compose and send appropriate responses",
            "tools": ["gmail_send"],
            "system_prompt": "You are an email responder agent. Compose professional responses based on the email content provided."
        }
    ]
)