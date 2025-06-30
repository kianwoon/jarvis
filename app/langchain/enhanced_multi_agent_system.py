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
    from app.langchain.multi_agent_system_simple import MultiAgentSystem
    from app.langchain.multi_agent_resource_monitor import get_multi_agent_resource_monitor
except ImportError as e:
    logger.warning(f"Failed to import enhanced components: {e}")
    # Fallback imports
    AgentCommunicationProtocol = None
    PipelineContextManager = None
    from app.langchain.multi_agent_system_simple import MultiAgentSystem
    get_multi_agent_resource_monitor = None

class EnhancedMultiAgentSystem(MultiAgentSystem):
    """Enhanced version with structured communication"""
    
    def __init__(self, conversation_id: Optional[str] = None, trace=None):
        super().__init__(conversation_id, trace)
        self.communication_protocol = AgentCommunicationProtocol() if AgentCommunicationProtocol else None
        
        # Initialize resource monitoring for enhanced multi-agent system
        if get_multi_agent_resource_monitor:
            self.resource_monitor = get_multi_agent_resource_monitor()
            self.resource_usage = None
        else:
            self.resource_monitor = None
            self.resource_usage = None
        self.context_manager = None
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
        
        # Start resource monitoring if available
        if self.resource_monitor:
            self.resource_usage = self.resource_monitor.start_session_monitoring(
                conversation_id=self.conversation_id,
                mode=mode
            )
            logger.info(f"Started resource monitoring for enhanced multi-agent session {self.conversation_id}")
        
        try:
            # Check if enhanced components are available
            if not self.enhanced_available:
                logger.warning("Enhanced components not available, falling back to standard execution")
                async for event in super().execute(query, mode, config):
                    yield event
                return
            
            # Initialize context manager for this execution
            self.context_manager = PipelineContextManager() if PipelineContextManager else None
        
            logger.info(f"🔵 [MULTI-AGENT MODE] Enhanced multi-agent system executing")
            
            # Get agent sequence
            agents = config.get("agents", []) if config else []
        
            # Execute based on mode
            if mode == "sequential" and self.context_manager:
                async for event in self._execute_sequential_enhanced(query, agents):
                    yield event
            else:
                # Fall back to original implementation for other modes
                async for event in super().execute(query, mode, config):
                    yield event
        
        except Exception as e:
            logger.error(f"Enhanced multi-agent execution failed: {e}")
            # Complete resource monitoring with error status
            if self.resource_monitor and self.resource_usage:
                try:
                    final_usage = self.resource_monitor.complete_session_monitoring(self.conversation_id)
                    if final_usage and final_usage.limits_exceeded:
                        logger.warning(f"Failed multi-agent session {self.conversation_id} exceeded resource limits: {final_usage.limits_exceeded}")
                except Exception as monitor_error:
                    logger.warning(f"Failed to complete resource monitoring: {monitor_error}")
            raise
        
        finally:
            # Always complete resource monitoring on exit
            if self.resource_monitor and self.resource_usage:
                try:
                    final_usage = self.resource_monitor.complete_session_monitoring(self.conversation_id)
                    if final_usage:
                        if final_usage.limits_exceeded:
                            logger.warning(f"Multi-agent session {self.conversation_id} exceeded resource limits: {final_usage.limits_exceeded}")
                        logger.info(f"Enhanced multi-agent session {self.conversation_id} completed - "
                                  f"Memory peak: {final_usage.memory_peak_mb:.1f}MB, "
                                  f"Agents: {final_usage.agents_executed}, "
                                  f"LLM calls: {final_usage.llm_calls_count}, "
                                  f"Tool calls: {final_usage.tool_calls_count}, "
                                  f"Duration: {final_usage.execution_duration_ms:.1f}ms")
                except Exception as monitor_error:
                    logger.warning(f"Failed to complete resource monitoring: {monitor_error}")
    
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
            
            # Multi-agent context
            agent_context = base_context.copy()
            agent_context["agent_index"] = idx
            agent_context["total_agents"] = total_agents
            logger.info(f"🔵 [MULTI-AGENT CONTEXT] Context for agent {agent_name} (idx {idx}/{total_agents})")
            
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
            
            # MULTI-AGENT MODE: Use enhanced instructions
            enhanced_instructions = self.communication_protocol.create_agent_instruction(
                agent_config, agent_context
            ) if self.communication_protocol else ""
            full_prompt = f"{base_prompt}\n\n{enhanced_instructions}"
            logger.info(f"🔵 [MULTI-AGENT ENHANCED] Using enhanced instructions for {agent_name}")
            
            # Add query context for multi-agent mode
            if idx == 0:
                full_prompt += f"\n\nUSER QUERY: {query}"
            else:
                full_prompt += f"\n\nORIGINAL USER QUERY: {query}"
                full_prompt += "\n\nYour task based on previous agents' work."
            logger.info(f"🔵 [MULTI-AGENT] Added query formatting to {agent_name}")
            
            logger.info(f"[ENHANCED] Executing agent {idx + 1}/{total_agents}: {agent_name}")
            
            # Record agent start in resource monitoring
            if self.resource_monitor:
                self.resource_monitor.record_agent_start(self.conversation_id, agent_name)
            
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
                    "agent_config": agent_config,
                    "agent_context": agent_context,
                    "agent_name": agent_name
                }
            }
            
            # Call the agent
            agent_success = False
            async for event in self._execute_single_agent(agent_name, agent_state):
                # Parse the response if it's a completion event
                if event.get("event") == "agent_complete":
                    agent_success = True
                    response = event.get("data", {}).get("response", "")
                    parsed_response = self.communication_protocol.parse_agent_response(response)
                    
                    # Add parsed response to context
                    self.context_manager.add_agent_output(agent_name, parsed_response)
                    
                    # Add parsing info to event
                    event["data"]["parsed_response"] = parsed_response
                    event["data"]["agent_position"] = f"{idx + 1}/{total_agents}"
                    
                    # Record agent completion in resource monitoring
                    if self.resource_monitor:
                        self.resource_monitor.record_agent_complete(self.conversation_id, agent_name, success=True)
                elif event.get("event") == "error":
                    agent_success = False
                    # Record agent failure in resource monitoring
                    if self.resource_monitor:
                        self.resource_monitor.record_agent_complete(self.conversation_id, agent_name, success=False)
                
                yield event
            
            # Ensure agent completion is recorded even if no explicit completion event
            if not agent_success and self.resource_monitor:
                self.resource_monitor.record_agent_complete(self.conversation_id, agent_name, success=False)
        
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
        
        # MULTI-AGENT MODE: Load config from langgraph_agents table
        logger.info(f"🔵 [MULTI-AGENT MODE] Loading config from langgraph_agents table for {agent_name}")
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
        
        # Get dynamic system from pool for better resource management
        from app.langchain.dynamic_agent_system import agent_instance_pool
        dynamic_system = await agent_instance_pool.get_or_create_instance(trace=self.trace)
        
        # Build enhanced context with pipeline information
        context = {
            "agent_context": state.get("metadata", {}).get("agent_context", {}),
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
        
        try:
            async for event in dynamic_system.execute_agent(
                agent_name,
                agent_data,
                state.get("query", ""),
                context=context
            ):
                # Transform event format to match expected output
                event_type = event.get("type", "")
                
                # Monitor LLM calls and tool calls for resource tracking
                if self.resource_monitor:
                    if event_type in ["llm_call", "generation_start", "generation_complete"]:
                        self.resource_monitor.record_llm_call(self.conversation_id, agent_name)
                    elif event_type in ["tool_call", "tool_execution", "mcp_call"]:
                        self.resource_monitor.record_tool_call(self.conversation_id, agent_name)
                
                if event_type == "agent_complete":
                    duration = (datetime.now() - start_time).total_seconds()
                    response_text = event.get("content", "")
                    tools_used = event.get("tools_used", [])
                    
                    # Record additional tool calls based on tools_used count
                    if self.resource_monitor and tools_used:
                        for _ in tools_used:
                            self.resource_monitor.record_tool_call(self.conversation_id, agent_name)
                    
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
                            "tools_used": tools_used
                        }
                    }
                elif event_type == "agent_token":
                    # Stream tokens as they come - first token indicates LLM call started
                    if self.resource_monitor and event.get("token"):
                        # Only record once per agent execution by checking if this is the first token
                        # This prevents multiple recordings for the same LLM call
                        if not hasattr(self, f'_llm_recorded_{agent_name}_{start_time.timestamp()}'):
                            self.resource_monitor.record_llm_call(self.conversation_id, agent_name)
                            setattr(self, f'_llm_recorded_{agent_name}_{start_time.timestamp()}', True)
                    
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
        finally:
            # Always release the instance back to the pool
            await agent_instance_pool.release_instance(dynamic_system)

