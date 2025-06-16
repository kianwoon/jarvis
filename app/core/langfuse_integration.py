"""
Langfuse Integration Module
Provides centralized Langfuse tracing for LLM operations
"""

from typing import Optional, Dict, Any, List
from langfuse import Langfuse
from functools import wraps
import logging

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
            
            self._client = Langfuse(
                public_key=config.get('public_key', ''),
                secret_key=config.get('secret_key', ''),
                host=config.get('host', 'http://localhost:3000'),
                debug=config.get('debug_mode', False)
            )
            
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
    
    def create_span(self, trace, name: str, **kwargs) -> Optional[Any]:
        """Create a span within a trace"""
        if not self.is_enabled() or not trace:
            return None
        try:
            return trace.span(name=name, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create span: {e}")
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