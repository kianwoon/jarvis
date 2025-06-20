"""
Pipeline Executor for Agentic Pipeline feature.
Handles pipeline execution with different collaboration modes.
"""
import asyncio
import json
import logging
import time
import redis
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.core.pipeline_manager import PipelineManager
from app.core.redis_client import get_redis_client
from app.core.pipeline_config import get_pipeline_settings
from app.langchain.multi_agent_system_simple import MultiAgentSystem
from app.langchain.collaboration_executor import CollaborationExecutor
from app.core.agent_queue import AgentExecutionQueue
from app.core.pipeline_multi_agent_bridge import execute_pipeline_with_agents
from app.core.pipeline_bridge_adapter import PipelineBridgeAdapter
from app.core.db import SessionLocal
from app.api.v1.endpoints.pipeline_execution_ws import (
    publish_status_update,
    publish_agent_start,
    publish_agent_complete,
    publish_agent_error,
    publish_log_entry,
    publish_metrics_update,
    publish_execution_complete,
    publish_execution_error
)

logger = logging.getLogger(__name__)
settings = get_pipeline_settings()

# Try to import enhanced components
try:
    from app.langchain.enhanced_multi_agent_system import EnhancedMultiAgentSystem
    from app.agents.agent_communication import AgentCommunicationProtocol, PipelineContextManager
    ENHANCED_SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced multi-agent system not available: {e}")
    EnhancedMultiAgentSystem = None
    AgentCommunicationProtocol = None
    PipelineContextManager = None
    ENHANCED_SYSTEM_AVAILABLE = False


class PipelineExecutor:
    """Executes agentic pipelines with different collaboration modes."""
    
    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.redis_client = None
        self.agent_queue = AgentExecutionQueue()
        self._execution_cache_prefix = "pipeline_execution"
        self.use_enhanced_system = ENHANCED_SYSTEM_AVAILABLE  # Use enhanced system if available
        self.current_pipeline_id = None  # Track current pipeline for bridge
    
    def _get_redis(self):
        """Get Redis client (lazy initialization)"""
        if self.redis_client is None:
            self.redis_client = get_redis_client()
        return self.redis_client
    
    async def execute_pipeline(
        self,
        pipeline_id: int,
        input_data: Dict[str, Any],
        trigger_type: str = "manual",
        existing_execution_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a pipeline."""
        # Force reload of caches to ensure we have latest configurations
        try:
            # Reload tools cache first
            from app.core.mcp_tools_cache import reload_enabled_mcp_tools
            reload_enabled_mcp_tools()
            logger.info("[SYNC] Reloaded MCP tools cache")
            
            # Then reload agent cache
            from app.core.langgraph_agents_cache import reload_langgraph_agents
            reload_langgraph_agents()
            logger.info("[SYNC] Reloaded langgraph agents cache before pipeline execution")
        except Exception as e:
            logger.warning(f"[SYNC] Failed to reload caches: {e}")
        
        # Get pipeline configuration
        pipeline = await self.pipeline_manager.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        if not pipeline["is_active"]:
            raise ValueError(f"Pipeline {pipeline['name']} is not active")
        
        # Use existing execution_id or create new one
        if existing_execution_id:
            execution_id = existing_execution_id
            # Update status to running
            await self.pipeline_manager.update_execution_status(
                execution_id=execution_id,
                status="running"
            )
        else:
            # Record execution start
            execution_id = await self.pipeline_manager.record_execution(
                pipeline_id=pipeline_id,
                trigger_type=trigger_type,
                status="running",
                input_data=input_data
            )
        
        # Store execution state in Redis
        execution_key = f"{self._execution_cache_prefix}:{execution_id}"
        redis_client = self._get_redis()
        if redis_client:
            redis_client.setex(
                execution_key,
                settings.PIPELINE_REDIS_TTL,
                json.dumps({
                    "pipeline_id": pipeline_id,
                    "status": "running",
                    "started_at": datetime.now().isoformat(),
                    "input_data": input_data
                })
            )
        
        # Initialize Langfuse tracing for agentic pipeline (with enhanced error handling)
        trace = None
        try:
            from app.core.langfuse_integration import get_tracer
            tracer = get_tracer()
            
            if not tracer._initialized:
                tracer.initialize()
            
            if tracer.is_enabled():
                # Sanitize metadata to prevent Langfuse errors
                safe_metadata = {
                    "pipeline_id": str(pipeline_id),
                    "pipeline_name": str(pipeline.get("name", "Unknown")),
                    "collaboration_mode": str(pipeline.get("collaboration_mode", "unknown")),
                    "trigger_type": str(trigger_type),
                    "execution_id": str(execution_id),
                    "agent_count": len(pipeline.get("agents", []))
                }
                
                trace = tracer.create_trace(
                    name=f"pipeline-workflow",
                    input=str(input_data.get("query", ""))[:1000],  # Limit input length
                    metadata=safe_metadata
                )
                logger.info(f"[TRACE] Created Langfuse trace for pipeline {pipeline_id} execution {execution_id}")
            else:
                logger.info("[TRACE] Langfuse tracer not enabled for pipeline")
        except Exception as e:
            logger.warning(f"[TRACE] Failed to create Langfuse trace for pipeline: {e}")
            # Continue without tracing if Langfuse fails
            trace = None
        
        # Store trace and pipeline_id for bridge usage
        self.current_trace = trace
        self.current_pipeline_id = pipeline_id
        
        try:
            # Publish initial status
            await publish_status_update(
                str(execution_id),
                "starting",
                0.0,
                None
            )
            await publish_log_entry(
                str(execution_id),
                "info",
                f"Starting execution of pipeline '{pipeline['name']}' in {pipeline['collaboration_mode']} mode"
            )
            
            # Execute based on collaboration mode
            result = await self._execute_by_mode(
                pipeline=pipeline,
                input_data=input_data,
                execution_id=execution_id
            )
            
            # Record successful completion
            await self.pipeline_manager.update_execution_status(
                execution_id=execution_id,
                status="completed",
                output_data=result,
                execution_metadata={
                    "execution_time": result.get("execution_time", 0),
                    "agents_used": result.get("agents_used", [])
                }
            )
            
            # Publish completion
            await publish_execution_complete(
                str(execution_id),
                result,
                result.get("execution_time", 0)
            )
            
            # Complete Langfuse trace with enhanced error handling
            if trace:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    if tracer.is_enabled():
                        # Sanitize output and metadata
                        safe_output = str(result.get("final_output", ""))[:2000]  # Limit output length
                        safe_metadata = {
                            "execution_status": "completed",
                            "execution_time": float(result.get("execution_time", 0)),
                            "total_agents": int(len(result.get("agents_used", []))),
                            "pipeline_id": str(pipeline_id)
                        }
                        
                        trace.update(
                            output=safe_output,
                            metadata=safe_metadata
                        )
                        
                        # Flush traces to ensure they appear in Langfuse UI
                        tracer.flush()
                        logger.info(f"[TRACE] Completed and flushed Langfuse trace for pipeline {pipeline_id}")
                except Exception as e:
                    logger.warning(f"[TRACE] Failed to complete Langfuse trace: {e}")
                    # Continue execution even if trace completion fails
            
            return {
                "execution_id": execution_id,
                "pipeline_id": pipeline_id,
                "status": "completed",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            
            # Record failure
            await self.pipeline_manager.update_execution_status(
                execution_id=execution_id,
                status="failed",
                error_message=str(e)
            )
            
            # Publish error
            await publish_execution_error(
                str(execution_id),
                str(e)
            )
            
            # Complete Langfuse trace with error (enhanced error handling)
            if trace:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    if tracer.is_enabled():
                        safe_metadata = {
                            "execution_status": "failed",
                            "error_message": str(e)[:500],  # Limit error message length
                            "pipeline_id": str(pipeline_id),
                            "execution_id": str(execution_id)
                        }
                        
                        trace.update(
                            output="",
                            metadata=safe_metadata
                        )
                        
                        # Flush traces to ensure they appear in Langfuse UI
                        tracer.flush()
                        logger.info(f"[TRACE] Completed and flushed Langfuse trace with error for pipeline {pipeline_id}")
                except Exception as trace_error:
                    logger.warning(f"[TRACE] Failed to complete Langfuse trace with error: {trace_error}")
                    # Continue execution even if trace completion fails
            
            raise
        
        finally:
            # Clean up Redis state
            if redis_client:
                redis_client.delete(execution_key)
    
    async def _execute_by_mode(
        self,
        pipeline: Dict[str, Any],
        input_data: Dict[str, Any],
        execution_id: int
    ) -> Dict[str, Any]:
        """Execute pipeline based on collaboration mode."""
        mode = pipeline["collaboration_mode"]
        agents = pipeline["agents"]
        
        if not agents:
            raise ValueError("Pipeline has no agents configured")
        
        # Create pipeline execution span with proper hierarchy
        pipeline_span = None
        if self.current_trace:
            try:
                from app.core.langfuse_integration import get_tracer
                tracer = get_tracer()
                if tracer.is_enabled():
                    agent_names = [agent.get("agent_name", "unknown") for agent in agents]
                    pipeline_span = tracer.create_pipeline_execution_span(
                        self.current_trace, 
                        self.current_pipeline_id, 
                        mode, 
                        agent_names
                    )
            except Exception as e:
                logger.warning(f"Failed to create pipeline execution span: {e}")
        
        # Add pipeline goal to input data if available
        if pipeline.get("goal"):
            input_data["pipeline_goal"] = pipeline["goal"]
        
        start_time = time.time()
        
        try:
            if mode == "sequential":
                result = await self._execute_sequential(
                    agents=agents,
                    input_data=input_data,
                    execution_id=execution_id,
                    pipeline_goal=pipeline.get("goal", ""),
                    pipeline_span=pipeline_span
                )
            elif mode == "parallel":
                result = await self._execute_parallel(
                    agents=agents,
                    input_data=input_data,
                    execution_id=execution_id,
                    pipeline_goal=pipeline.get("goal", ""),
                    pipeline_span=pipeline_span
                )
            elif mode == "hierarchical":
                result = await self._execute_hierarchical(
                    agents=agents,
                    input_data=input_data,
                    execution_id=execution_id,
                    pipeline_goal=pipeline.get("goal", ""),
                    pipeline_span=pipeline_span
                )
            elif mode in settings.get_collaboration_modes():
                # Handle additional collaboration modes
                if mode == "conditional":
                    result = await self._execute_conditional(
                        agents=agents,
                        input_data=input_data,
                        execution_id=execution_id,
                        pipeline_goal=pipeline.get("goal", ""),
                        pipeline_span=pipeline_span
                    )
                elif mode == "approval_gate":
                    result = await self._execute_approval_gate(
                        agents=agents,
                        input_data=input_data,
                        execution_id=execution_id,
                        pipeline_goal=pipeline.get("goal", ""),
                        pipeline_span=pipeline_span
                    )
                elif mode == "event_driven":
                    result = await self._execute_event_driven(
                        agents=agents,
                        input_data=input_data,
                        execution_id=execution_id,
                        pipeline_goal=pipeline.get("goal", ""),
                        pipeline_span=pipeline_span
                    )
                elif mode == "hybrid":
                    result = await self._execute_hybrid(
                        agents=agents,
                        input_data=input_data,
                        execution_id=execution_id,
                        pipeline_goal=pipeline.get("goal", ""),
                        pipeline_span=pipeline_span
                    )
                else:
                    # Fallback to sequential for unknown modes
                    logger.warning(f"Unimplemented collaboration mode: {mode}, falling back to sequential")
                    result = await self._execute_sequential(
                        agents=agents,
                        input_data=input_data,
                        execution_id=execution_id,
                        pipeline_goal=pipeline.get("goal", ""),
                        pipeline_span=pipeline_span
                    )
            else:
                raise ValueError(f"Unknown collaboration mode: {mode}")
        
            execution_time = time.time() - start_time
            
            # End pipeline span with success
            if pipeline_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tracer.end_span_with_result(
                        pipeline_span,
                        {
                            "mode": mode,
                            "execution_time": execution_time,
                            "agents_used": [agent["agent_name"] for agent in agents],
                            "result_type": type(result).__name__
                        },
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to end pipeline execution span: {e}")
            
            return {
                "mode": mode,
                "execution_time": execution_time,
                "agents_used": [agent["agent_name"] for agent in agents],
                "result": result
            }
        
        except Exception as e:
            # End pipeline span with error
            if pipeline_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tracer.end_span_with_result(pipeline_span, None, False, str(e))
                except Exception as span_e:
                    logger.warning(f"Failed to end pipeline execution span with error: {span_e}")
            raise
    
    async def _execute_sequential(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = "",
        pipeline_span=None
    ) -> Dict[str, Any]:
        """Execute agents sequentially using clean agentic pipeline executor."""
        # Sort agents by execution order
        sorted_agents = sorted(agents, key=lambda x: x.get("execution_order", 0))
        
        logger.info(f"[PURE PIPELINE] Using PureAgenticPipeline for execution {execution_id}")
        
        # Initialize PURE agentic pipeline executor
        from app.langchain.pure_agentic_pipeline import PureAgenticPipeline
        executor = PureAgenticPipeline(
            pipeline_id=self.current_pipeline_id,
            execution_id=str(execution_id),
            trace=pipeline_span or self.current_trace  # Use pipeline_span as parent if available
        )
        
        # Check if debug mode is enabled
        debug_mode = input_data.get("debug_mode", False)
        current_input = input_data.get("query", "")
        
        # Execute agents cleanly
        agent_outputs = []
        final_output = ""
        total_agents = len(sorted_agents)
        completed_agents = 0
        
        # Publish initial agent list for progress tracking
        await publish_log_entry(
            str(execution_id),
            "info",
            f"Pipeline will execute {total_agents} agents: {[a.get('agent_name') for a in sorted_agents]}"
        )
        
        # Execute through pure executor
        async for event in executor.execute_agents_sequentially(
            query=input_data.get("query", ""),
            agents=sorted_agents
        ):
            event_type = event.get("event", "")
            event_data = event.get("data", {})
            
            if event_type == "streaming":
                # Streaming tokens - can be forwarded if needed
                pass
                
            elif event_type == "agent_start":
                agent_name = event_data.get("agent")
                await publish_agent_start(str(execution_id), agent_name)
                await publish_log_entry(
                    str(execution_id),
                    "info",
                    f"Starting agent: {agent_name}"
                )
                # Update progress
                progress = completed_agents / total_agents
                await publish_status_update(
                    str(execution_id),
                    "running",
                    progress,
                    agent_name
                )
                
            elif event_type == "agent_complete":
                completed_agents += 1
                agent_name = event_data.get("agent")
                
                # Extract response from multiple possible fields
                response = (
                    event_data.get("response") or 
                    event_data.get("content") or 
                    event_data.get("output") or
                    ""
                )
                
                duration = event_data.get("duration", 0)
                progress = event_data.get("progress", (completed_agents / total_agents) * 100)
                
                # Publish agent completion via WebSocket
                await publish_agent_complete(str(execution_id), agent_name, response, duration)
                await publish_log_entry(
                    str(execution_id),
                    "info",
                    f"Completed agent: {agent_name} ({duration:.2f}s)"
                )
                
                # Update overall progress
                progress_ratio = completed_agents / total_agents
                next_agent = sorted_agents[completed_agents].get("agent_name") if completed_agents < total_agents else None
                await publish_status_update(
                    str(execution_id),
                    "running",
                    progress_ratio,
                    next_agent
                )
                
                # Enhanced debugging with response extraction details
                if debug_mode:
                    logger.info(f"\n{'-'*60}")
                    logger.info(f"[AGENT I/O] Agent: {agent_name} (#{completed_agents}/{total_agents})")
                    logger.info(f"[AGENT I/O] Event data fields: {list(event_data.keys())}")
                    logger.info(f"[AGENT I/O] Execution time: {duration:.2f}s")
                    logger.info(f"[AGENT I/O] Progress: {progress:.1f}%")
                    logger.info(f"[AGENT I/O] Input received:")
                    logger.info(f"  - Query: {current_input[:500]}...")
                    logger.info(f"[AGENT I/O] Output generated:")
                    logger.info(f"  - Response length: {len(response)} characters")
                    logger.info(f"  - Response preview: {response[:500]}...")
                    logger.info(f"  - Response source field: {'response' if event_data.get('response') else 'content' if event_data.get('content') else 'output' if event_data.get('output') else 'none'}")
                    
                    # Log tools if used
                    if "tools_used" in event_data and event_data["tools_used"]:
                        logger.info(f"[AGENT I/O] Tools used:")
                        for tool in event_data["tools_used"]:
                            logger.info(f"  - {tool}")
                    
                    logger.info(f"{'-'*60}\n")
                else:
                    logger.info(f"[BRIDGE] Agent {agent_name} completed ({completed_agents}/{total_agents}) - Response: {len(response)} chars")
                
                # Build agent output with proper response handling
                agent_output = {
                    "agent": agent_name,
                    "output": response,
                    "content": response,  # For compatibility
                    "execution_time": duration
                }
                
                # Add any additional data from event
                if "reasoning" in event_data:
                    agent_output["reasoning"] = event_data["reasoning"]
                if "tools_used" in event_data:
                    agent_output["tools_used"] = event_data["tools_used"]
                
                agent_outputs.append(agent_output)
                
                # Update final_output to be the last agent's response (only if non-empty)
                if response.strip():
                    final_output = response
                    logger.info(f"[BRIDGE] Updated final_output from {agent_name}: {len(final_output)} chars")
                else:
                    logger.warning(f"[BRIDGE] Agent {agent_name} produced empty response!")
                
                # Update current input for next agent (if any)
                current_input = response
                
                # Log what will be passed to next agent
                if debug_mode and completed_agents < total_agents:
                    logger.info(f"[AGENT HANDOFF] Preparing to pass output to next agent")
                    logger.info(f"[AGENT HANDOFF] Output length: {len(response)} chars")
                    logger.info(f"[AGENT HANDOFF] Output preview: {response[:500]}...")
                    if len(response) > 1000:
                        logger.info(f"[AGENT HANDOFF] Output contains email info? {'From:' in response or 'from:' in response}")
                        # Try to extract key info
                        import re
                        from_match = re.search(r'(?:From|from):\s*([^\n]+)', response)
                        subject_match = re.search(r'(?:Subject|subject):\s*([^\n]+)', response)
                        if from_match:
                            logger.info(f"[AGENT HANDOFF] Found From: {from_match.group(1)}")
                        if subject_match:
                            logger.info(f"[AGENT HANDOFF] Found Subject: {subject_match.group(1)}")
                
            elif event_type == "pipeline_complete":
                if debug_mode:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"[PIPELINE DEBUG] Pipeline execution completed")
                    logger.info(f"[PIPELINE DEBUG] Total agents executed: {completed_agents}")
                    logger.info(f"[PIPELINE DEBUG] Final output length: {len(final_output)} characters")
                    logger.info(f"{'='*80}\n")
                else:
                    logger.info(f"[BRIDGE] Pipeline execution completed with {completed_agents} agents")
                summary = event_data.get("summary", {})
                
            elif event_type == "error":
                error_msg = event_data.get("error", "Unknown error")
                logger.error(f"[BRIDGE] Pipeline execution error: {error_msg}")
                raise Exception(f"Pipeline execution failed: {error_msg}")
        
        # Debug logging
        if debug_mode:
            logger.info(f"\n[PIPELINE SUMMARY] Execution complete:")
            logger.info(f"  - Total agents: {len(agent_outputs)}")
            logger.info(f"  - Total execution time: {sum(a.get('execution_time', 0) for a in agent_outputs):.2f}s")
            logger.info(f"  - Final output preview: {final_output[:200]}...")
        else:
            logger.info(f"[BRIDGE] Returning {len(agent_outputs)} agent outputs")
        
        # Ensure we have a final_output - fallback to last agent's output if needed
        if not final_output.strip() and agent_outputs:
            for agent_output in reversed(agent_outputs):
                potential_output = agent_output.get("output") or agent_output.get("content")
                if potential_output and potential_output.strip():
                    final_output = potential_output
                    logger.info(f"[BRIDGE] Using fallback final_output from {agent_output.get('agent')}: {len(final_output)} chars")
                    break
        
        # Last resort - combine all agent outputs
        if not final_output.strip() and agent_outputs:
            final_output = "\n\n".join([
                f"**{agent.get('agent', 'Unknown Agent')}:**\n{agent.get('output') or agent.get('content') or 'No output'}"
                for agent in agent_outputs
            ])
            logger.info(f"[BRIDGE] Using combined agent outputs as final_output: {len(final_output)} chars")
        
        return {
            "agent_outputs": agent_outputs,
            "final_output": final_output,
            "total_agents": total_agents
        }
    
    async def _execute_sequential_standard(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = ""
    ) -> Dict[str, Any]:
        """Standard sequential execution (fallback when enhanced not available)"""
        return await self._execute_sequential_original(agents, input_data, execution_id, pipeline_goal)
    
    async def _execute_sequential_original(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = ""
    ) -> Dict[str, Any]:
        """Original sequential execution logic"""
        multi_agent = MultiAgentSystem(trace=self.current_trace)
        
        # Check if debug mode is enabled
        debug_mode = input_data.get("debug_mode", False)
        
        if debug_mode:
            logger.info(f"\n{'='*80}")
            logger.info(f"[PIPELINE DEBUG] Starting sequential execution (fallback mode)")
            logger.info(f"[PIPELINE DEBUG] Execution ID: {execution_id}")
            logger.info(f"[PIPELINE DEBUG] Total agents: {len(agents)}")
            logger.info(f"[PIPELINE DEBUG] Agent sequence: {[a['agent_name'] for a in agents]}")
            logger.info(f"{'='*80}\n")
        
        current_input = input_data.get("query", "")
        agent_outputs = []
        conversation_history = input_data.get("conversation_history", [])
        
        for idx, agent in enumerate(agents):
            agent_name = agent['agent_name']
            
            if debug_mode:
                logger.info(f"\n{'-'*60}")
                logger.info(f"[AGENT I/O] Starting agent: {agent_name} (#{idx+1}/{len(agents)})")
                logger.info(f"[AGENT I/O] Input:")
                logger.info(f"  - Current query: {current_input[:500]}...")
                logger.info(f"  - Previous outputs: {len(agent_outputs)}")
            else:
                logger.info(f"Executing agent: {agent_name}")
            
            # Update execution progress
            await self._update_progress(
                execution_id,
                f"Executing {agent_name}"
            )
            
            # Add pipeline goal to agent config
            agent_config = agent.get("config", {})
            if pipeline_goal:
                agent_config["pipeline_goal"] = pipeline_goal
            
            # Execute single agent
            agent_result = await multi_agent.execute_single_agent(
                agent_name=agent_name,
                query=current_input,
                conversation_history=conversation_history,
                previous_outputs=agent_outputs,
                agent_config=agent_config
            )
            
            # Include all relevant data from agent result
            agent_output = {
                "agent": agent_name,
                "output": agent_result.get("content", ""),
                "content": agent_result.get("content", ""),
                "reasoning": agent_result.get("reasoning", ""),
                "execution_time": agent_result.get("execution_time", 0)
            }
            
            # Include tools_used if available
            if "tools_used" in agent_result:
                agent_output["tools_used"] = agent_result["tools_used"]
            
            if debug_mode:
                logger.info(f"[AGENT I/O] Output:")
                logger.info(f"  - Response length: {len(agent_output['output'])} characters")
                logger.info(f"  - Response preview: {agent_output['output'][:500]}...")
                logger.info(f"  - Execution time: {agent_output['execution_time']:.2f}s")
                if agent_output.get("tools_used"):
                    logger.info(f"  - Tools used: {agent_output['tools_used']}")
                logger.info(f"{'-'*60}\n")
            
            agent_outputs.append(agent_output)
            
            # Use output as input for next agent
            if agent.get("config", {}).get("pass_output_to_next", True):
                current_input = agent_result.get("content", current_input)
        
        if debug_mode:
            logger.info(f"\n{'='*80}")
            logger.info(f"[PIPELINE DEBUG] Execution complete")
            logger.info(f"  - Total agents executed: {len(agent_outputs)}")
            logger.info(f"  - Total execution time: {sum(a.get('execution_time', 0) for a in agent_outputs):.2f}s")
            logger.info(f"  - Final output length: {len(agent_outputs[-1]['output']) if agent_outputs else 0} characters")
            logger.info(f"{'='*80}\n")
        
        return {
            "agent_outputs": agent_outputs,
            "final_output": agent_outputs[-1]["output"] if agent_outputs else "",
            "total_agents": len(agents)
        }
    
    async def _execute_parallel(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = "",
        pipeline_span=None
    ) -> Dict[str, Any]:
        """Execute agents in parallel."""
        multi_agent = MultiAgentSystem(trace=pipeline_span or self.current_trace)
        
        # Prepare agent names and configs
        agent_names = [agent["agent_name"] for agent in agents]
        agent_configs = {}
        for agent in agents:
            config = agent.get("config", {})
            if pipeline_goal:
                config["pipeline_goal"] = pipeline_goal
            agent_configs[agent["agent_name"]] = config
        
        # Update progress
        await self._update_progress(
            execution_id,
            f"Executing {len(agents)} agents in parallel"
        )
        
        # Execute using existing multi-agent system
        result = await multi_agent.execute_agents(
            query=input_data.get("query", ""),
            selected_agents=agent_names,
            conversation_history=input_data.get("conversation_history", []),
            execution_pattern="parallel",
            agent_configs=agent_configs
        )
        
        return {
            "agent_outputs": result.get("agent_outputs", []),
            "final_output": result.get("final_output", ""),
            "total_agents": len(agents)
        }
    
    async def _execute_hierarchical(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = "",
        pipeline_span=None
    ) -> Dict[str, Any]:
        """Execute agents in hierarchical mode."""
        # Identify lead agent (no parent)
        lead_agents = [a for a in agents if not a.get("parent_agent")]
        if not lead_agents:
            raise ValueError("No lead agent found for hierarchical execution")
        
        lead_agent = lead_agents[0]
        
        # Build hierarchy
        hierarchy = self._build_agent_hierarchy(agents)
        
        # Update progress
        await self._update_progress(
            execution_id,
            f"Executing hierarchical pipeline with lead: {lead_agent['agent_name']}"
        )
        
        # Execute hierarchically
        multi_agent = MultiAgentSystem(trace=pipeline_span or self.current_trace)
        
        # Prepare subordinate agents for lead
        subordinates = hierarchy.get(lead_agent["agent_name"], [])
        
        # Add pipeline goal to configs
        lead_config = lead_agent.get("config", {})
        if pipeline_goal:
            lead_config["pipeline_goal"] = pipeline_goal
            
        subordinate_configs = {}
        for agent in agents:
            if agent["agent_name"] in subordinates:
                config = agent.get("config", {})
                if pipeline_goal:
                    config["pipeline_goal"] = pipeline_goal
                subordinate_configs[agent["agent_name"]] = config
        
        result = await multi_agent.execute_hierarchical(
            query=input_data.get("query", ""),
            lead_agent=lead_agent["agent_name"],
            subordinate_agents=subordinates,
            conversation_history=input_data.get("conversation_history", []),
            lead_config=lead_config,
            subordinate_configs=subordinate_configs
        )
        
        return {
            "hierarchy": hierarchy,
            "agent_outputs": result.get("agent_outputs", []),
            "final_output": result.get("final_output", ""),
            "total_agents": len(agents)
        }
    
    def _build_agent_hierarchy(
        self,
        agents: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Build agent hierarchy from parent-child relationships."""
        hierarchy = {}
        
        for agent in agents:
            parent = agent.get("parent_agent")
            if parent:
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(agent["agent_name"])
        
        return hierarchy
    
    async def _update_progress(self, execution_id: int, message: str):
        """Update execution progress in Redis."""
        progress_key = f"{self._execution_cache_prefix}:progress:{execution_id}"
        redis_client = self._get_redis()
        if redis_client:
            redis_client.lpush(progress_key, json.dumps({
                "timestamp": datetime.now().isoformat(),
                "message": message
            }))
            redis_client.expire(progress_key, settings.PIPELINE_PROGRESS_TTL)
    
    async def get_execution_progress(
        self,
        execution_id: int
    ) -> List[Dict[str, Any]]:
        """Get execution progress messages."""
        progress_key = f"{self._execution_cache_prefix}:progress:{execution_id}"
        redis_client = self._get_redis()
        if redis_client:
            messages = redis_client.lrange(progress_key, 0, -1)
            return [
                json.loads(msg)
                for msg in reversed(messages)  # Return in chronological order
            ]
        return []
    
    async def cancel_execution(self, execution_id: int) -> bool:
        """Cancel a running execution."""
        # Update status to cancelled
        await self.pipeline_manager.update_execution_status(
            execution_id=execution_id,
            status="cancelled",
            error_message="Execution cancelled by user"
        )
        
        # Clean up Redis state
        execution_key = f"{self._execution_cache_prefix}:{execution_id}"
        progress_key = f"{self._execution_cache_prefix}:progress:{execution_id}"
        
        redis_client = self._get_redis()
        if redis_client:
            redis_client.delete(execution_key)
            redis_client.delete(progress_key)
        
        return True
    
    async def _execute_sequential_enhanced(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = ""
    ) -> Dict[str, Any]:
        """Execute agents sequentially with enhanced communication."""
        if not ENHANCED_SYSTEM_AVAILABLE:
            logger.warning("[ENHANCED] Enhanced system not available, falling back to standard execution")
            return await self._execute_sequential_standard(agents, input_data, execution_id, pipeline_goal)
        
        logger.info("🔴 [CLEAN PIPELINE] Using CLEAN pipeline executor - NO multi-agent contamination")
        
        # Initialize CLEAN agentic pipeline executor
        from app.langchain.agentic_pipeline_executor import AgenticPipelineExecutor
        pipeline_executor = AgenticPipelineExecutor(
            pipeline_id=pipeline["id"],
            execution_id=self.execution_id,
            trace=self.current_trace
        )
        
        # Get communication patterns from database
        communication_patterns = await self._load_communication_patterns(agents)
        
        # Prepare agent configurations with templates
        enhanced_agents = []
        for idx, agent in enumerate(agents):
            agent_name = agent["agent_name"]
            
            # Load agent template if available
            template = await self._load_agent_template(agent_name)
            
            # Merge template with agent config
            enhanced_config = {
                "name": agent_name,
                "role": agent.get("config", {}).get("role", template.get("description", "")),
                "tools": agent.get("config", {}).get("tools", []),
                "system_prompt": agent.get("config", {}).get("system_prompt", ""),
                "template": template,
                "position": idx,
                "total_agents": len(agents)
            }
            
            # Add communication pattern for next agent
            if idx < len(agents) - 1:
                next_agent = agents[idx + 1]["agent_name"]
                pattern = communication_patterns.get((agent_name, next_agent))
                if pattern:
                    enhanced_config["communication_pattern"] = pattern
            
            enhanced_agents.append(enhanced_config)
        
        # Create pipeline config for enhanced system
        pipeline_config = {
            "pipeline": {
                "name": f"pipeline_{execution_id}",
                "agents": enhanced_agents,
                "goal": pipeline_goal,
                "mode": "sequential"
            },
            "agents": enhanced_agents
        }
        
        # Execute with CLEAN pipeline executor
        agent_outputs = []
        final_output = ""
        
        # Execute through clean agentic pipeline executor
        async for event in pipeline_executor.execute_sequential_pipeline(
            query=input_data.get("query", ""),
            agents=[{"agent_name": agent["agent_name"], "config": agent.get("config", {})} for agent in agents]
        ):
            if event.get("event") == "agent_complete":
                data = event.get("data", {})
                agent_output = {
                    "agent": data.get("agent", "Unknown"),
                    "output": data.get("response", ""),
                    "content": data.get("response", ""),
                    "reasoning": data.get("reasoning", ""),
                    "execution_time": data.get("execution_time", 0),
                    "parsed_response": data.get("parsed_response", {})
                }
                
                if "tools_used" in data:
                    agent_output["tools_used"] = data["tools_used"]
                
                agent_outputs.append(agent_output)
                final_output = agent_output["output"]
                
                # Update progress  
                await self._update_progress(
                    execution_id,
                    f"Completed {data.get('agent')} ({data.get('agent_index', '')}/{data.get('total_agents', '')})"
                )
            
            elif event.get("event") == "pipeline_complete":
                summary = event.get("data", {}).get("execution_summary", {})
                logger.info(f"[ENHANCED] Pipeline completed with {len(agent_outputs)} agents")
        
        return {
            "agent_outputs": agent_outputs,
            "final_output": final_output,
            "total_agents": len(agents),
            "enhanced_execution": True
        }
    
    async def _load_agent_template(self, agent_name: str) -> Dict[str, Any]:
        """Load agent template from database."""
        from sqlalchemy import text
        db = SessionLocal()
        try:
            query = text("SELECT * FROM agent_templates WHERE name = :name")
            result = db.execute(query, {"name": agent_name}).fetchone()
            if result:
                return {
                    "name": result.name,
                    "description": result.description,
                    "capabilities": result.capabilities,
                    "expected_input": result.expected_input,
                    "output_format": result.output_format,
                    "default_instructions": result.default_instructions
                }
        except Exception as e:
            logger.warning(f"Failed to load template for {agent_name}: {e}")
        finally:
            db.close()
        return {}
    
    async def _load_communication_patterns(
        self, 
        agents: List[Dict[str, Any]]
    ) -> Dict[tuple, Dict[str, Any]]:
        """Load communication patterns for agent pairs."""
        from sqlalchemy import text
        patterns = {}
        db = SessionLocal()
        try:
            for i in range(len(agents) - 1):
                from_agent = agents[i]["agent_name"]
                to_agent = agents[i + 1]["agent_name"]
                
                query = text("""
                    SELECT * FROM agent_communication_patterns 
                    WHERE from_agent = :from_agent AND to_agent = :to_agent
                """)
                result = db.execute(query, {
                    "from_agent": from_agent,
                    "to_agent": to_agent
                }).fetchone()
                
                if result:
                    patterns[(from_agent, to_agent)] = {
                        "pattern_name": result.pattern_name,
                        "handoff_data": result.handoff_data,
                        "instructions_template": result.instructions_template,
                        "data_transformation": result.data_transformation
                    }
        except Exception as e:
            logger.warning(f"Failed to load communication patterns: {e}")
        finally:
            db.close()
        return patterns
    
    async def _execute_conditional(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = "",
        pipeline_span=None
    ) -> Dict[str, Any]:
        """Execute agents with conditional branching logic."""
        # TODO: Implement conditional execution logic
        logger.info("Conditional execution mode - falling back to sequential")
        return await self._execute_sequential(agents, input_data, execution_id, pipeline_goal, pipeline_span)
    
    async def _execute_approval_gate(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = "",
        pipeline_span=None
    ) -> Dict[str, Any]:
        """Execute agents with approval gates."""
        # TODO: Implement approval gate logic
        logger.info("Approval gate execution mode - falling back to sequential")
        return await self._execute_sequential(agents, input_data, execution_id, pipeline_goal, pipeline_span)
    
    async def _execute_event_driven(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = "",
        pipeline_span=None
    ) -> Dict[str, Any]:
        """Execute agents in event-driven mode."""
        # TODO: Implement event-driven execution logic
        logger.info("Event-driven execution mode - falling back to sequential")
        return await self._execute_sequential(agents, input_data, execution_id, pipeline_goal, pipeline_span)
    
    async def _execute_hybrid(
        self,
        agents: List[Dict[str, Any]],
        input_data: Dict[str, Any],
        execution_id: int,
        pipeline_goal: str = "",
        pipeline_span=None
    ) -> Dict[str, Any]:
        """Execute agents in hybrid mode (mixed collaboration patterns)."""
        # TODO: Implement hybrid execution logic
        logger.info("Hybrid execution mode - falling back to sequential")
        return await self._execute_sequential(agents, input_data, execution_id, pipeline_goal, pipeline_span)