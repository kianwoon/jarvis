"""
Langfuse Integration Module
Provides centralized Langfuse tracing for LLM operations
"""

from typing import Optional, Dict, Any, List
from langfuse import Langfuse
from functools import wraps
import logging
import time

from app.core.langfuse_settings_cache import get_langfuse_settings

logger = logging.getLogger(__name__)

class LangfuseTracer:
    """Centralized Langfuse tracing manager"""
    
    def __init__(self):
        self._client: Optional[Langfuse] = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize Langfuse client with current settings"""
        try:
            config = get_langfuse_settings()
            
            if not config.get('enabled', False):
                logger.info("Langfuse tracing is disabled")
                return False
            
            # Determine the correct host for Docker environment
            default_host = 'http://localhost:3000'
            # Check if we're running in Docker and should use container name
            import os
            if os.path.exists('/.dockerenv') or os.environ.get('DOCKER_ENV'):
                default_host = 'http://jarvis-langfuse-web:3000'
            
            langfuse_host = config.get('host') or default_host
            
            self._client = Langfuse(
                public_key=config.get('public_key', ''),
                secret_key=config.get('secret_key', ''),
                host=langfuse_host,
                debug=config.get('debug_mode', False)
            )
            
            logger.info(f"Langfuse client initialized with host: {langfuse_host}")
            
            self._initialized = True
            logger.info("Langfuse tracing initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self._initialized = False
            return False
    
    @property
    def client(self) -> Optional[Langfuse]:
        """Get Langfuse client, initializing if needed"""
        if not self._initialized:
            self.initialize()
        return self._client
    
    def is_enabled(self) -> bool:
        """Check if Langfuse tracing is enabled and working"""
        return self._initialized and self._client is not None
    
    def create_trace(self, name: str, **kwargs) -> Optional[Any]:
        """Create a new trace"""
        if not self.is_enabled():
            return None
        try:
            return self.client.trace(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create trace: {e}")
            return None
    
    def create_generation(self, trace, name: str, **kwargs) -> Optional[Any]:
        """Create a generation within a trace"""
        if not self.is_enabled() or not trace:
            return None
        try:
            return trace.generation(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create generation: {e}")
            return None
    
    def create_generation_with_usage(self, trace, name: str, model: str, input_text: str, 
                                   output_text: str = None, usage: Dict[str, Any] = None, parent_span=None, **kwargs) -> Optional[Any]:
        """Create a generation with usage tracking for cost calculation"""
        if not self.is_enabled() or not trace:
            return None
        try:
            # Extract parent_span from kwargs if not passed as parameter
            if parent_span is None:
                parent_span = kwargs.pop('parent_span', None)
            
            generation_data = {
                "name": name,
                "model": model,
                "input": input_text,
                **kwargs
            }
            
            # Add output if provided
            if output_text:
                generation_data["output"] = output_text
            
            # Add usage information if provided
            if usage:
                generation_data["usage"] = usage
            
            logger.info(f"[LANGFUSE DEBUG] Creating generation with model: '{model}', parent_span: {parent_span is not None}, usage: {usage}")
            
            # Create generation under parent span if provided, otherwise under trace
            if parent_span:
                return parent_span.generation(**generation_data)
            else:
                return trace.generation(**generation_data)
        except Exception as e:
            logger.error(f"Failed to create generation with usage: {e}")
            return None
    
    def create_span(self, trace, name: str, parent_span=None, **kwargs) -> Optional[Any]:
        """Create a span within a trace or under a parent span"""
        if not self.is_enabled() or not trace:
            return None
        try:
            # Extract parent_span from kwargs if not passed as parameter
            if parent_span is None:
                parent_span = kwargs.pop('parent_span', None)
            
            logger.debug(f"[LANGFUSE DEBUG] Creating span '{name}' with parent_span: {parent_span is not None}")
            
            # Create span under parent if provided, otherwise under trace
            if parent_span:
                return parent_span.span(name=name, **kwargs)
            else:
                return trace.span(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
            return None
    
    def create_nested_span(self, parent_observation, name: str, **kwargs) -> Optional[Any]:
        """Create a nested span within a parent observation (span or generation)"""
        if not self.is_enabled() or not parent_observation:
            return None
        try:
            return parent_observation.span(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create nested span: {e}")
            return None
    
    def score(self, trace_id: str, name: str, value: float, **kwargs) -> bool:
        """Add a score to a trace"""
        if not self.is_enabled():
            return False
        try:
            self.client.score(name=name, value=value, trace_id=trace_id, **kwargs)
            return True
        except Exception as e:
            logger.error(f"Failed to add score: {e}")
            return False
    
    def flush(self):
        """Flush pending traces"""
        if self.is_enabled():
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Failed to flush traces: {e}")
    
    def create_tool_span(self, trace_or_parent, tool_name: str, parameters: Dict[str, Any], parent_span=None) -> Optional[Any]:
        """
        Create a span specifically for tool execution with proper parent-child hierarchy
        
        Args:
            trace_or_parent: Either a trace (for top-level tools) or parent span (for nested tools)
            tool_name: Name of the tool being executed
            parameters: Tool parameters
            parent_span: Optional parent span for nested hierarchy
        """
        if not self.is_enabled() or not trace_or_parent:
            return None
        try:
            # Sanitize parameters for Langfuse
            safe_parameters = self._sanitize_parameters(parameters)
            safe_tool_name = str(tool_name)[:100] if tool_name else "unknown_tool"
            
            # Create span either nested in parent or as direct child of trace
            if parent_span:
                # Create as nested span under parent
                tool_span = parent_span.span(
                    name=f"tool-{safe_tool_name}",
                    input=safe_parameters,
                    metadata={
                        "tool_name": safe_tool_name,
                        "tool_type": "mcp_tool",
                        "operation": "tool_execution",
                        "nesting_level": "nested"
                    }
                )
            else:
                # Create as direct child of trace
                tool_span = trace_or_parent.span(
                    name=f"tool-{safe_tool_name}",
                    input=safe_parameters,
                    metadata={
                        "tool_name": safe_tool_name,
                        "tool_type": "mcp_tool",
                        "operation": "tool_execution",
                        "nesting_level": "direct"
                    }
                )
            
            return tool_span
        except Exception as e:
            logger.error(f"Failed to create tool span: {e}")
            return None
    
    def create_rag_span(self, trace_or_parent, query: str, collections: List[str] = None) -> Optional[Any]:
        """Create a span specifically for RAG operations"""
        if not self.is_enabled() or not trace_or_parent:
            return None
        try:
            # Check if trace_or_parent is a span (has .span method) or a trace
            if hasattr(trace_or_parent, 'span') and callable(getattr(trace_or_parent, 'span')):
                # Create span under parent
                return trace_or_parent.span(
                    name="rag-search",
                    input={"query": query, "collections": collections},
                    metadata={
                        "operation": "rag_search",
                        "query_length": len(query),
                        "collections": collections or [],
                        "nesting_level": "nested"
                    }
                )
            else:
                # Fallback to creating directly under trace
                return trace_or_parent.span(
                    name="rag-search",
                    input={"query": query, "collections": collections},
                    metadata={
                        "operation": "rag_search",
                        "query_length": len(query),
                        "collections": collections or [],
                        "nesting_level": "direct"
                    }
                )
        except Exception as e:
            logger.error(f"Failed to create RAG span: {e}")
            return None
    
    def end_span_with_result(self, span, result: Any, success: bool = True, error: str = None):
        """End a span with result and metadata"""
        if not span:
            return
        try:
            # Sanitize metadata
            metadata = {
                "success": success,
                "result_type": type(result).__name__
            }
            if error:
                metadata["error"] = str(error)[:500]  # Truncate error messages
            
            # Sanitize output
            if success:
                if result is None:
                    output = "null"
                elif isinstance(result, (str, int, float, bool)):
                    output = str(result)[:1000]  # Truncate long strings
                elif isinstance(result, dict):
                    output = self._sanitize_parameters(result)
                else:
                    output = str(type(result).__name__)
            else:
                output = f"Error: {str(error)[:500] if error else 'Unknown error'}"
            
            span.end(output=output, metadata=metadata)
        except Exception as e:
            logger.error(f"Failed to end span: {e}")
    
    def _sanitize_parameters(self, parameters: Any) -> Dict[str, Any]:
        """Sanitize parameters for safe Langfuse submission"""
        if parameters is None:
            return {}
        
        try:
            if isinstance(parameters, dict):
                safe_params = {}
                for key, value in parameters.items():
                    # Sanitize key
                    safe_key = str(key)[:100] if key is not None else "unknown"
                    
                    # Sanitize value
                    if value is None:
                        safe_value = "null"
                    elif isinstance(value, (str, int, float, bool)):
                        safe_value = str(value)[:500] if isinstance(value, str) else value
                    elif isinstance(value, (list, tuple)):
                        safe_value = [str(item)[:100] for item in value[:10]]  # Limit list size and item length
                    elif isinstance(value, dict):
                        safe_value = self._sanitize_parameters(value)  # Recursive sanitization
                    else:
                        safe_value = str(type(value).__name__)
                    
                    safe_params[safe_key] = safe_value
                return safe_params
            elif isinstance(parameters, (list, tuple)):
                return {"items": [str(item)[:100] for item in parameters[:10]]}
            else:
                return {"value": str(parameters)[:500]}
        except Exception as e:
            logger.warning(f"Failed to sanitize parameters: {e}")
            return {"error": "sanitization_failed"}
    
    def estimate_token_usage(self, input_text: str, output_text: str = "") -> Dict[str, int]:
        """Estimate token usage for cost calculation (simple approximation)"""
        try:
            # Simple approximation: ~4 characters per token for most languages
            input_tokens = max(1, len(input_text) // 4)
            output_tokens = max(1, len(output_text) // 4) if output_text else 0
            
            usage = {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            }
            
            logger.info(f"[LANGFUSE DEBUG] Estimated usage: {usage}")
            return usage
        except Exception as e:
            logger.warning(f"Failed to estimate tokens: {e}")
            return {"input": 0, "output": 0, "total": 0}
    
    def create_agent_span(self, trace, agent_name: str, query: str, metadata: Dict[str, Any] = None) -> Optional[Any]:
        """Create a span specifically for individual agent execution"""
        if not self.is_enabled() or not trace:
            return None
        try:
            # Sanitize inputs
            safe_agent_name = str(agent_name)[:100] if agent_name else "unknown_agent"
            safe_query = str(query)[:1000] if query else ""
            
            span_metadata = {
                "agent_name": safe_agent_name,
                "operation": "agent_execution",
                "query_length": len(safe_query)
            }
            
            if metadata:
                # Sanitize metadata
                safe_metadata = self._sanitize_parameters(metadata)
                span_metadata.update(safe_metadata)
                
            return trace.span(
                name=f"agent-{safe_agent_name}",
                input={"query": safe_query, "agent": safe_agent_name},
                metadata=span_metadata
            )
        except Exception as e:
            logger.error(f"Failed to create agent span: {e}")
            return None
    
    def create_multi_agent_workflow_span(self, trace, workflow_type: str, selected_agents: List[str]) -> Optional[Any]:
        """Create a span for the overall multi-agent workflow"""
        if not self.is_enabled() or not trace:
            return None
        try:
            # Sanitize inputs
            safe_workflow_type = str(workflow_type)[:100] if workflow_type else "unknown"
            safe_agents = [str(agent)[:100] for agent in (selected_agents or [])[:20]]  # Limit to 20 agents
            
            return trace.span(
                name=f"multi-agent-{safe_workflow_type}",
                input={"selected_agents": safe_agents},
                metadata={
                    "workflow_type": safe_workflow_type,
                    "agent_count": len(safe_agents),
                    "operation": "multi_agent_workflow"
                }
            )
        except Exception as e:
            logger.error(f"Failed to create multi-agent workflow span: {e}")
            return None
    
    def create_large_generation_span(self, trace, task_description: str, target_count: int, chunk_size: int) -> Optional[Any]:
        """Create a span for large generation tasks"""
        if not self.is_enabled() or not trace:
            return None
        try:
            return trace.span(
                name="large-generation-workflow",
                input={"task_description": task_description, "target_count": target_count},
                metadata={
                    "target_count": target_count,
                    "chunk_size": chunk_size,
                    "operation": "large_generation",
                    "task_length": len(task_description)
                }
            )
        except Exception as e:
            logger.error(f"Failed to create large generation span: {e}")
            return None
    
    # Mode-specific span creation methods for proper hierarchy
    
    def create_standard_chat_span(self, trace, query: str, thinking: bool = False) -> Optional[Any]:
        """Create a span for standard chat mode with proper hierarchy"""
        if not self.is_enabled() or not trace:
            return None
        try:
            return trace.span(
                name="standard-chat",
                input={"query": query, "thinking": thinking},
                metadata={
                    "mode": "standard",
                    "operation": "chat_processing",
                    "thinking_enabled": thinking,
                    "query_length": len(query)
                }
            )
        except Exception as e:
            logger.error(f"Failed to create standard chat span: {e}")
            return None
    
    def create_intelligent_tool_planning_span(self, parent_span, task: str, mode: str, available_tools_count: int) -> Optional[Any]:
        """Create a span for intelligent tool planning phase"""
        if not self.is_enabled() or not parent_span:
            return None
        try:
            return parent_span.span(
                name="intelligent-tool-planning",
                input={"task": task, "mode": mode},
                metadata={
                    "operation": "tool_planning",
                    "mode": mode,
                    "available_tools_count": available_tools_count,
                    "task_length": len(task)
                }
            )
        except Exception as e:
            logger.error(f"Failed to create tool planning span: {e}")
            return None
    
    def create_tool_execution_workflow_span(self, parent_span, plan_tools_count: int, mode: str) -> Optional[Any]:
        """Create a span for tool execution workflow"""
        if not self.is_enabled() or not parent_span:
            return None
        try:
            return parent_span.span(
                name="tool-execution-workflow",
                input={"planned_tools": plan_tools_count, "mode": mode},
                metadata={
                    "operation": "tool_execution_workflow",
                    "mode": mode,
                    "planned_tools_count": plan_tools_count
                }
            )
        except Exception as e:
            logger.error(f"Failed to create tool execution workflow span: {e}")
            return None
    
    def create_pipeline_execution_span(self, trace, pipeline_id: int, mode: str, agents: List[str]) -> Optional[Any]:
        """Create a span for agentic pipeline execution"""
        if not self.is_enabled() or not trace:
            return None
        try:
            safe_agents = [str(agent)[:100] for agent in (agents or [])[:20]]
            return trace.span(
                name=f"pipeline-execution-{mode}",
                input={"pipeline_id": pipeline_id, "agents": safe_agents, "mode": mode},
                metadata={
                    "operation": "pipeline_execution",
                    "pipeline_id": pipeline_id,
                    "mode": mode,
                    "agent_count": len(safe_agents),
                    "pipeline_type": "agentic"
                }
            )
        except Exception as e:
            logger.error(f"Failed to create pipeline execution span: {e}")
            return None
    
    def create_agent_execution_span(self, parent_span, agent_name: str, query: str, agent_metadata: Dict[str, Any] = None) -> Optional[Any]:
        """Create a span for individual agent execution within a workflow"""
        if not self.is_enabled() or not parent_span:
            return None
        try:
            safe_agent_name = str(agent_name)[:100] if agent_name else "unknown_agent"
            safe_query = str(query)[:1000] if query else ""
            
            span_metadata = {
                "operation": "agent_execution",
                "agent_name": safe_agent_name,
                "query_length": len(safe_query),
                "nesting_level": "agent"
            }
            
            if agent_metadata:
                safe_metadata = self._sanitize_parameters(agent_metadata)
                span_metadata.update(safe_metadata)
            
            return parent_span.span(
                name=f"agent-{safe_agent_name}",
                input={"query": safe_query, "agent": safe_agent_name},
                metadata=span_metadata
            )
        except Exception as e:
            logger.error(f"Failed to create agent execution span: {e}")
            return None
    
    def create_llm_generation_span(self, parent_span, model: str, prompt: str, operation: str = "generation") -> Optional[Any]:
        """Create a span for LLM generation within a workflow"""
        if not self.is_enabled() or not parent_span:
            return None
        try:
            safe_model = str(model)[:100] if model else "unknown_model"
            safe_prompt = str(prompt)[:1000] if prompt else ""
            
            return parent_span.generation(
                name=f"llm-{operation}",
                model=safe_model,
                input=safe_prompt,
                metadata={
                    "operation": f"llm_{operation}",
                    "model": safe_model,
                    "prompt_length": len(safe_prompt),
                    "nesting_level": "generation"
                }
            )
        except Exception as e:
            logger.error(f"Failed to create LLM generation span: {e}")
            return None

# Global tracer instance
tracer = LangfuseTracer()

def trace_llm_call(name: str = "llm-call"):
    """Decorator to trace LLM function calls"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not tracer.is_enabled():
                return await func(*args, **kwargs)
            
            trace = tracer.trace(name=name)
            if not trace:
                return await func(*args, **kwargs)
            
            try:
                # Extract relevant parameters for tracing
                trace_data = {
                    "input": kwargs.get('prompt') or kwargs.get('query') or str(args[:2]),
                    "metadata": {
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                }
                trace.update(**trace_data)
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Update trace with result
                trace.update(output=str(result)[:1000])  # Limit output size
                
                return result
                
            except Exception as e:
                if trace:
                    trace.update(
                        output=f"Error: {str(e)}",
                        level="ERROR"
                    )
                raise
            finally:
                tracer.flush()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not tracer.is_enabled():
                return func(*args, **kwargs)
            
            trace = tracer.trace(name=name)
            if not trace:
                return func(*args, **kwargs)
            
            try:
                # Extract relevant parameters for tracing
                trace_data = {
                    "input": kwargs.get('prompt') or kwargs.get('query') or str(args[:2]),
                    "metadata": {
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                }
                trace.update(**trace_data)
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Update trace with result
                trace.update(output=str(result)[:1000])  # Limit output size
                
                return result
                
            except Exception as e:
                if trace:
                    trace.update(
                        output=f"Error: {str(e)}",
                        level="ERROR"
                    )
                raise
            finally:
                tracer.flush()
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def get_tracer() -> LangfuseTracer:
    """Get the global Langfuse tracer instance"""
    return tracer