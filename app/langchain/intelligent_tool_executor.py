"""
Intelligent Tool Executor

Executes planned tools with context awareness, adaptive re-planning,
and comprehensive error handling.
"""

import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass

from .intelligent_tool_planner import ExecutionPlan, ToolPlan, get_tool_planner

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Result of a tool execution"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None
    context_updates: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionContext:
    """Context that flows between tool executions"""
    task: str
    previous_results: List[ToolResult]
    user_context: Dict[str, Any]
    execution_metadata: Dict[str, Any]
    
    def get_result_by_tool(self, tool_name: str) -> Optional[ToolResult]:
        """Get result from a specific tool"""
        for result in self.previous_results:
            if result.tool_name == tool_name:
                return result
        return None
    
    def get_all_results_data(self) -> Dict[str, Any]:
        """Get all tool results as a dict"""
        return {result.tool_name: result.result for result in self.previous_results if result.success}

class IntelligentToolExecutor:
    """
    Executes tool plans with intelligent context management, error handling,
    and adaptive re-planning capabilities.
    """
    
    def __init__(self, trace=None):
        self.trace = trace
        self.planner = get_tool_planner()
        self.execution_span = None  # Store current execution workflow span
        
    async def execute_task_intelligently(
        self, 
        task: str, 
        context: Dict[str, Any] = None,
        stream_callback=None,
        mode: str = "standard",
        agent_name: str = None,
        pipeline_id: int = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a task using intelligent tool planning and execution
        
        Args:
            task: The task to accomplish
            context: Additional context (conversation history, etc.)
            stream_callback: Optional callback for streaming updates
            
        Yields:
            Dict with execution updates including tool results and progress
        """
        
        try:
            # Create planning span if trace is available
            planning_span = None
            if self.trace:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    if tracer.is_enabled():
                        # Get available tools count for metadata
                        available_tools = self.planner.get_enhanced_tool_metadata(mode, agent_name, pipeline_id)
                        available_tools_count = len(available_tools) if available_tools else 0
                        
                        planning_span = tracer.create_intelligent_tool_planning_span(
                            self.trace, task, mode, available_tools_count
                        )
                except Exception as e:
                    logger.warning(f"Failed to create planning span: {e}")
            
            # Create execution plan
            if stream_callback:
                stream_callback("🧠 Analyzing task and planning tool execution...")
            
            yield {
                "type": "planning_start",
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
            
            execution_plan = await self.planner.plan_tool_execution(task, context, mode, agent_name, pipeline_id)
            
            # End planning span
            if planning_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tracer.end_span_with_result(
                        planning_span, 
                        {
                            "tools_planned": len(execution_plan.tools) if execution_plan.tools else 0,
                            "reasoning": execution_plan.reasoning[:500] if execution_plan.reasoning else "",
                            "estimated_duration": execution_plan.estimated_duration
                        }, 
                        success=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to end planning span: {e}")
            
            # Create execution workflow span for tool execution
            if self.trace and execution_plan.tools:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    if tracer.is_enabled():
                        self.execution_span = tracer.create_tool_execution_workflow_span(
                            self.trace, len(execution_plan.tools), mode
                        )
                except Exception as e:
                    logger.warning(f"Failed to create execution workflow span: {e}")
            
            if not execution_plan.tools:
                yield {
                    "type": "no_tools_needed",
                    "reasoning": execution_plan.reasoning,
                    "task": task
                }
                return
            
            # Safety check for standard mode - prevent over-execution
            if mode == "standard" and len(execution_plan.tools) > 5:
                logger.warning(f"[SAFETY] Standard mode planned {len(execution_plan.tools)} tools, limiting to 5 for safety")
                # Keep only the first 5 tools
                execution_plan.tools = execution_plan.tools[:5]
                execution_plan.reasoning += f"\n[SAFETY LIMIT] Execution limited to first 5 tools to prevent over-execution."
            
            # Additional safety check for action tools in standard mode
            if mode == "standard":
                action_tools = ["send", "create", "delete", "update", "draft", "gmail_send", "jira_create"]
                safe_tools = []
                action_count = 0
                
                for tool_plan in execution_plan.tools:
                    # Check if tool is an action tool
                    is_action_tool = any(action in tool_plan.tool_name.lower() for action in action_tools)
                    
                    if is_action_tool:
                        action_count += 1
                        if action_count <= 1:  # Allow only 1 action tool in standard mode
                            safe_tools.append(tool_plan)
                        else:
                            logger.warning(f"[SAFETY] Skipping action tool {tool_plan.tool_name} - only 1 action tool allowed in standard mode")
                    else:
                        safe_tools.append(tool_plan)
                
                if len(safe_tools) != len(execution_plan.tools):
                    execution_plan.tools = safe_tools
                    execution_plan.reasoning += f"\n[SAFETY LIMIT] Restricted to 1 action tool to prevent unintended modifications."
            
            # CRITICAL SECURITY CHECK for pipeline mode - ONLY allow tools from pipeline_agents table
            if mode == "pipeline" and pipeline_id is not None and agent_name:
                from app.core.pipeline_agents_cache import validate_pipeline_agent_tools
                
                planned_tools = [tool_plan.tool_name for tool_plan in execution_plan.tools]
                logger.info(f"[PIPELINE SECURITY] Validating tools {planned_tools} for agent {agent_name} in pipeline {pipeline_id}")
                
                if not validate_pipeline_agent_tools(pipeline_id, agent_name, planned_tools):
                    logger.error(f"[PIPELINE SECURITY] BLOCKED: One or more tools not allowed for pipeline agent {agent_name}")
                    yield {
                        "type": "security_violation",
                        "error": f"Security violation: Tools {planned_tools} not authorized for pipeline agent {agent_name}",
                        "agent_name": agent_name,
                        "pipeline_id": pipeline_id,
                        "unauthorized_tools": planned_tools
                    }
                    return
                else:
                    logger.info(f"[PIPELINE SECURITY] All tools validated for pipeline agent {agent_name}")
            
            yield {
                "type": "plan_created",
                "plan": {
                    "reasoning": execution_plan.reasoning,
                    "tools_count": len(execution_plan.tools),
                    "tools": [{"name": t.tool_name, "purpose": t.purpose} for t in execution_plan.tools],
                    "estimated_duration": execution_plan.estimated_duration
                }
            }
            
            # Initialize execution context
            exec_context = ExecutionContext(
                task=task,
                previous_results=[],
                user_context=context or {},
                execution_metadata={
                    "started_at": datetime.now().isoformat(),
                    "plan_id": str(hash(execution_plan.reasoning)),
                    "total_tools": len(execution_plan.tools)
                }
            )
            
            # Execute tools in sequence
            for tool_index, tool_plan in enumerate(execution_plan.tools):
                try:
                    if stream_callback:
                        stream_callback(f"🔧 Executing {tool_plan.tool_name}: {tool_plan.purpose}")
                    
                    yield {
                        "type": "tool_start",
                        "tool_name": tool_plan.tool_name,
                        "purpose": tool_plan.purpose,
                        "index": tool_index + 1,
                        "total": len(execution_plan.tools)
                    }
                    
                    # Resolve parameters with context
                    resolved_params = self._resolve_parameters(tool_plan, exec_context)
                    
                    # Execute the tool with proper span hierarchy
                    tool_result = await self._execute_single_tool(
                        tool_plan.tool_name,
                        resolved_params,
                        exec_context,
                        parent_span=self.execution_span  # Pass execution workflow span as parent
                    )
                    
                    # Add result to context
                    exec_context.previous_results.append(tool_result)
                    
                    # Yield tool completion
                    yield {
                        "type": "tool_complete",
                        "tool_name": tool_plan.tool_name,
                        "success": tool_result.success,
                        "result": tool_result.result if tool_result.success else None,
                        "error": tool_result.error,
                        "execution_time": tool_result.execution_time,
                        "index": tool_index + 1,
                        "total": len(execution_plan.tools)
                    }
                    
                    # Handle tool failure
                    if not tool_result.success:
                        # Try fallback plan if available
                        if execution_plan.fallback_plan:
                            yield {
                                "type": "fallback_triggered",
                                "failed_tool": tool_plan.tool_name,
                                "error": tool_result.error
                            }
                            
                            # Execute fallback
                            async for fallback_event in self._execute_fallback_plan(
                                execution_plan.fallback_plan,
                                exec_context,
                                stream_callback
                            ):
                                yield fallback_event
                        else:
                            # Try adaptive re-planning
                            yield {
                                "type": "adaptive_replan",
                                "failed_tool": tool_plan.tool_name,
                                "remaining_tools": len(execution_plan.tools) - tool_index - 1
                            }
                            
                            async for replan_event in self._adaptive_replan(
                                execution_plan.tools[tool_index + 1:],
                                exec_context,
                                tool_result.error,
                                stream_callback,
                                mode,
                                agent_name,
                                pipeline_id
                            ):
                                yield replan_event
                        break
                        
                except Exception as e:
                    logger.error(f"[TOOL EXECUTOR] Tool execution failed: {e}")
                    yield {
                        "type": "tool_error",
                        "tool_name": tool_plan.tool_name,
                        "error": str(e),
                        "index": tool_index + 1,
                        "total": len(execution_plan.tools)
                    }
                    break
            
            # Execution complete
            successful_tools = [r for r in exec_context.previous_results if r.success]
            failed_tools = [r for r in exec_context.previous_results if not r.success]
            
            # End execution workflow span
            if self.execution_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tracer.end_span_with_result(
                        self.execution_span,
                        {
                            "successful_tools": len(successful_tools),
                            "failed_tools": len(failed_tools),
                            "total_execution_time": sum(r.execution_time or 0 for r in exec_context.previous_results),
                            "results_summary": {r.tool_name: "success" if r.success else "failed" for r in exec_context.previous_results}
                        },
                        success=len(failed_tools) == 0  # Success if no tools failed
                    )
                except Exception as e:
                    logger.warning(f"Failed to end execution workflow span: {e}")
            
            yield {
                "type": "execution_complete",
                "task": task,
                "successful_tools": len(successful_tools),
                "failed_tools": len(failed_tools),
                "total_execution_time": sum(r.execution_time or 0 for r in exec_context.previous_results),
                "results": {r.tool_name: r.result for r in successful_tools},
                "context": exec_context.get_all_results_data()
            }
            
        except Exception as e:
            logger.error(f"[TOOL EXECUTOR] Task execution failed: {e}")
            yield {
                "type": "execution_error",
                "task": task,
                "error": str(e)
            }
    
    def _resolve_parameters(self, tool_plan: ToolPlan, context: ExecutionContext) -> Dict[str, Any]:
        """Resolve tool parameters using context from previous tools"""
        resolved_params = tool_plan.parameters.copy()
        
        # If tool depends on previous results
        if tool_plan.depends_on:
            for dependency in tool_plan.depends_on:
                dep_result = context.get_result_by_tool(dependency)
                if dep_result and dep_result.success:
                    # Smart parameter resolution based on result type
                    if isinstance(dep_result.result, dict):
                        # If result is dict, try to map specific fields
                        for key, value in dep_result.result.items():
                            param_key = f"{dependency}_{key}"
                            if param_key in resolved_params:
                                resolved_params[param_key] = value
                    elif isinstance(dep_result.result, (str, int, float)):
                        # If result is primitive, use it directly
                        for param_key, param_value in resolved_params.items():
                            if param_value == f"${{{dependency}}}":
                                resolved_params[param_key] = dep_result.result
        
        # Resolve context keys
        if tool_plan.context_keys:
            for context_key in tool_plan.context_keys:
                if context_key in context.user_context:
                    # Replace placeholder in parameters
                    for param_key, param_value in resolved_params.items():
                        if isinstance(param_value, str) and f"${{{context_key}}}" in param_value:
                            resolved_params[param_key] = param_value.replace(
                                f"${{{context_key}}}", 
                                str(context.user_context[context_key])
                            )
        
        return resolved_params
    
    async def _execute_single_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any], 
        context: ExecutionContext,
        parent_span=None
    ) -> ToolResult:
        """Execute a single tool and return result"""
        import time
        start_time = time.time()
        
        try:
            # Create tool span with proper hierarchy
            tool_span = None
            if self.trace:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    if tracer.is_enabled():
                        # Use parent_span if provided, otherwise use trace
                        span_parent = parent_span if parent_span else self.trace
                        tool_span = tracer.create_tool_span(
                            span_parent, tool_name, parameters, parent_span=parent_span
                        )
                except Exception as e:
                    logger.warning(f"Failed to create tool span for {tool_name}: {e}")
            
            # Import the existing tool execution function
            from app.langchain.service import call_mcp_tool
            
            # Execute tool with tracing (skip span creation since we handle it here)
            result = call_mcp_tool(
                tool_name, 
                parameters, 
                trace=tool_span if tool_span else self.trace, 
                _skip_span_creation=True  # We handle span creation here
            )
            
            execution_time = time.time() - start_time
            
            # Check if result indicates success or failure
            if isinstance(result, dict) and "error" in result:
                # End tool span with error
                if tool_span:
                    try:
                        from app.core.langfuse_integration import get_tracer
                        tracer = get_tracer()
                        tracer.end_span_with_result(tool_span, result, False, result["error"])
                    except Exception as e:
                        logger.warning(f"Failed to end tool span for {tool_name}: {e}")
                
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result=result,
                    error=result["error"],
                    execution_time=execution_time
                )
            else:
                # End tool span with success
                if tool_span:
                    try:
                        from app.core.langfuse_integration import get_tracer
                        tracer = get_tracer()
                        tracer.end_span_with_result(tool_span, result, True)
                    except Exception as e:
                        logger.warning(f"Failed to end tool span for {tool_name}: {e}")
                
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    context_updates=self._extract_context_updates(result)
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[TOOL EXECUTOR] Tool {tool_name} execution failed: {e}")
            
            # End tool span with exception
            if tool_span:
                try:
                    from app.core.langfuse_integration import get_tracer
                    tracer = get_tracer()
                    tracer.end_span_with_result(tool_span, None, False, str(e))
                except Exception as span_e:
                    logger.warning(f"Failed to end tool span for {tool_name}: {span_e}")
            
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def _extract_context_updates(self, result: Any) -> Dict[str, Any]:
        """Extract useful context from tool results for future tools"""
        context_updates = {}
        
        if isinstance(result, dict):
            # Extract common useful fields
            for key in ["id", "url", "content", "data", "items", "results"]:
                if key in result:
                    context_updates[key] = result[key]
        elif isinstance(result, (str, int, float)):
            context_updates["last_result"] = result
        
        return context_updates
    
    async def _execute_fallback_plan(
        self, 
        fallback_tools: List[ToolPlan], 
        context: ExecutionContext,
        stream_callback=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute fallback plan when primary plan fails"""
        
        for tool_index, tool_plan in enumerate(fallback_tools):
            try:
                if stream_callback:
                    stream_callback(f"🔄 Fallback: {tool_plan.tool_name}")
                
                yield {
                    "type": "fallback_tool_start",
                    "tool_name": tool_plan.tool_name,
                    "purpose": tool_plan.purpose,
                    "index": tool_index + 1,
                    "total": len(fallback_tools)
                }
                
                resolved_params = self._resolve_parameters(tool_plan, context)
                tool_result = await self._execute_single_tool(
                    tool_plan.tool_name, 
                    resolved_params, 
                    context, 
                    parent_span=self.execution_span
                )
                context.previous_results.append(tool_result)
                
                yield {
                    "type": "fallback_tool_complete",
                    "tool_name": tool_plan.tool_name,
                    "success": tool_result.success,
                    "result": tool_result.result if tool_result.success else None,
                    "error": tool_result.error
                }
                
                if tool_result.success:
                    break  # Fallback succeeded, continue
                    
            except Exception as e:
                yield {
                    "type": "fallback_tool_error",
                    "tool_name": tool_plan.tool_name,
                    "error": str(e)
                }
    
    async def _adaptive_replan(
        self, 
        remaining_tools: List[ToolPlan], 
        context: ExecutionContext, 
        failure_reason: str,
        stream_callback=None,
        mode: str = "standard",
        agent_name: str = None,
        pipeline_id: int = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Adaptively re-plan remaining tools based on failure"""
        
        if not remaining_tools:
            return
        
        try:
            # Create new task focusing on what's left to do
            replan_task = f"""
            Original task: {context.task}
            
            Progress so far: {len(context.previous_results)} tools executed
            Successful results: {context.get_all_results_data()}
            
            Last failure: {failure_reason}
            
            Please complete the remaining work considering the failure.
            """
            
            if stream_callback:
                stream_callback("🔄 Re-planning remaining tasks...")
            
            # Get new plan for remaining work with same mode constraints
            new_plan = await self.planner.plan_tool_execution(replan_task, context.user_context, mode, agent_name, pipeline_id)
            
            if new_plan.tools:
                yield {
                    "type": "replan_created",
                    "new_tools_count": len(new_plan.tools),
                    "reasoning": new_plan.reasoning
                }
                
                # Execute new plan
                for tool_index, tool_plan in enumerate(new_plan.tools):
                    if stream_callback:
                        stream_callback(f"🔧 Re-planned: {tool_plan.tool_name}")
                    
                    yield {
                        "type": "replan_tool_start",
                        "tool_name": tool_plan.tool_name,
                        "purpose": tool_plan.purpose
                    }
                    
                    resolved_params = self._resolve_parameters(tool_plan, context)
                    tool_result = await self._execute_single_tool(
                        tool_plan.tool_name, 
                        resolved_params, 
                        context, 
                        parent_span=self.execution_span
                    )
                    context.previous_results.append(tool_result)
                    
                    yield {
                        "type": "replan_tool_complete",
                        "tool_name": tool_plan.tool_name,
                        "success": tool_result.success,
                        "result": tool_result.result if tool_result.success else None,
                        "error": tool_result.error
                    }
            else:
                yield {
                    "type": "replan_failed",
                    "reason": "No suitable alternative tools found"
                }
                
        except Exception as e:
            yield {
                "type": "replan_error",
                "error": str(e)
            }

# Convenience function for easy integration
async def execute_task_with_intelligent_tools(
    task: str, 
    context: Dict[str, Any] = None, 
    trace=None,
    stream_callback=None,
    mode: str = "standard",
    agent_name: str = None,
    pipeline_id: int = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to execute a task using intelligent tool planning
    
    Args:
        task: The task to accomplish
        context: Additional context
        trace: Langfuse trace for observability
        stream_callback: Optional streaming callback
        
    Returns:
        List of execution events
    """
    executor = IntelligentToolExecutor(trace=trace)
    
    events = []
    async for event in executor.execute_task_intelligently(task, context, stream_callback, mode, agent_name, pipeline_id):
        events.append(event)
    
    return events